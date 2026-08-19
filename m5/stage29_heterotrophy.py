# -*- coding: utf-8 -*-
"""
M5 阶段 29：异养涌现（进化路线阶段 B）——混合营养连续谱（现实研究支撑）

研究依据：
  Ward & Follows：单轴 phototrophy-phagotrophy 捕捉生态变异——混合营养连续谱——
    严格权衡 φ≈1.8（50% 光合 → 仅 29% 摄食）
  Tilman 资源比率：光受限（光斑拥挤）→ 吞噬有利；猎物受限 → 自养有利
  Myxococcota：捕食菌含光合基因簇（捕食-光合嵌合体真实存在）

设计（用户：神经=细胞——τ 是网络输出——行为训练产物——两层时间尺度）：
  基因（慢）：NEAT 结构/权重——进化塑造网络先天
  神经（快）：网络输出实时决定营养策略 τ——行为训练

网络输出 3 维：[移动方向 sin, cos, 营养策略 τ（sigmoid 0-1）]
  光合吸收 = 光强 × 0.4 × (1-τ)^1.8 ÷ 拥挤共享（φ≈1.8 权衡——研究 1）
  捕食（接触吞噬）：邻居接触且 τ 高者吞 τ 低者——能量 = 邻居能量 × τ^1.8
  ——能量流守恒：光合积累（被吞者）→ 异养者（研究 3 嵌合体）

环境：光斑（stage28 延续——光受限触发异养——研究 2）+ 邻居细胞（猎物）
  初始：纯自养（τ 输出偏置 0——光合先验——网络学何时转向捕食）
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(109)
W, H = 100.0, 100.0
# 光受限压力（研究 2：光不足 → 吞噬有利——纯异养分化条件）：
# 光斑 6→4 + 光强 0.4→0.28——光合不足——吞噬补足——τ 分化
LIGHTS = [(25, 25), (75, 75), (50, 50), (15, 75)]
LIGHT_SIGMA = 15.0
ABS_RATE = 0.35
BASE_COST = 0.05
MOVE_COST = 0.1
SPEED = 1.0
REPRO_E = 100.0
CONTACT = 2.5          # 吞噬接触距离
PHI = 1.8              # 权衡指数（研究：50% 光合 → 29% 摄食）
N_IN = 7               # 光方向 sin/cos + 距离 + 光强 + 拥挤 + 噪声 + 常数
N_OUT = 3              # 移动 sin/cos + 营养策略 τ
P_ADD_CONN = 0.10
P_ADD_NODE = 0.05
P_DISABLE = 0.05

def light_intensity(x, y):
    best = 0.0
    for lx, ly in LIGHTS:
        d = np.hypot(x - lx, y - ly)
        best = max(best, np.exp(-(d / LIGHT_SIGMA) ** 2))
    return best

def nearest_light(x, y):
    best_d, best_ang = 1e9, 0.0
    for lx, ly in LIGHTS:
        d = np.hypot(lx - x, ly - y)
        if d < best_d:
            best_d = d
            best_ang = np.arctan2(ly - y, lx - x)
    return best_ang, best_d

def init_genome(rng):
    """光合先验：移动≈光方向（趋光）+ τ 输出偏置 0（初始纯自养——网络学何时转向捕食）"""
    conns = []
    for i in range(N_IN):
        for o in range(N_OUT):
            if o < 2:
                w = 1.5 if (i == o) else 0.0
            else:
                # 常数输入 → τ 偏置 -0.5（线性区——-2 时 tanh 饱和——变异 ±20% 权重只改饱和点——
                # τ 变异被压缩到 0.009 < 吞噬门槛 0.02——永远无法触发——线性区变异可见 0.04）
                w = -0.5 if i == N_IN - 1 else 0.0
            conns.append([i, N_IN + o, w + rng.normal(0, 0.05), 1.0])
    return np.array(conns)

def forward(genome, x):
    max_node = int(genome[:, :2].max())
    n_nodes = max_node + 1
    vals = np.zeros(n_nodes)
    vals[:N_IN] = x
    enabled = genome[:, 3] > 0.5
    conns = genome[enabled]
    for _ in range(n_nodes):
        for (f, t, w, _e) in conns:
            f, t = int(f), int(t)
            if t >= N_IN:
                vals[t] += w * vals[f]
        for t in range(N_IN, n_nodes):
            vals[t] = np.tanh(vals[t])
    out = vals[N_IN:N_IN + N_OUT]
    tau = 1.0 / (1.0 + np.exp(-out[2]))   # sigmoid：营养策略 0-1（0=纯自养 1=纯异养）
    return out[0], out[1], tau

def mutate_genome(genome, rng):
    g = genome.copy()
    w_idx = np.arange(len(g))
    g[w_idx, 2] += rng.normal(0, 0.2, len(g)) * np.abs(g[w_idx, 2]) + 0.03
    if len(g) > N_IN * N_OUT:
        for i in range(len(g)):
            if g[i, 3] > 0.5 and rng.random() < P_DISABLE / len(g):
                g[i, 3] = 0.0
    if rng.random() < P_ADD_NODE:
        on = np.where(g[:, 3] > 0.5)[0]
        if len(on):
            ci = int(rng.choice(on))
            f, t, w = int(g[ci, 0]), int(g[ci, 1]), g[ci, 2]
            new_id = int(g[:, :2].max()) + 1
            g[ci, 3] = 0.0
            g = np.vstack([g, [f, new_id, w, 1.0], [new_id, t, 1.0, 1.0]])
    if rng.random() < P_ADD_CONN:
        max_id = int(g[:, :2].max())
        pairs = [(a, b) for a in range(N_IN, max_id + 1) for b in range(N_IN, max_id + 1) if a < b]
        pairs += [(a, b) for a in range(N_IN) for b in range(N_IN, max_id + 1)]
        existing = set((int(a), int(b)) for a, b in g[:, :2])
        cand = [p for p in pairs if p not in existing]
        if cand:
            f, t = cand[int(rng.integers(len(cand)))]
            g = np.vstack([g, [f, t, rng.normal(0, 0.5), 1.0]])
    return g

class HeteroWorld:
    def __init__(self, workdir="stage29_evo", n0=60, seed=109):
        self.workdir = workdir
        os.makedirs(workdir, exist_ok=True)
        for f in os.listdir(workdir):
            os.remove(os.path.join(workdir, f))
        global RNG
        RNG = np.random.default_rng(seed)
        self.bios = {}
        self.genomes = {}
        self.next_id = 0
        for _ in range(n0):
            self._spawn_new()
        self.history = []
        self.kills = 0

    def _spawn_new(self):
        bid = self.next_id
        self.next_id += 1
        g = init_genome(RNG)
        self.genomes[bid] = g
        np.save(f"{self.workdir}/bio_{bid}.npy", g)
        self.bios[bid] = [RNG.uniform(0, W), RNG.uniform(0, H), RNG.uniform(30, 60), 0.0]
        return bid

    def _reproduce(self, parent_id):
        bid = self.next_id
        self.next_id += 1
        g = mutate_genome(self.genomes[parent_id], RNG)
        self.genomes[bid] = g
        np.save(f"{self.workdir}/bio_{bid}.npy", g)
        px, py, pe, _ptau = self.bios[parent_id]
        self.bios[bid] = [np.clip(px + RNG.uniform(-5, 5), 0, W),
                          np.clip(py + RNG.uniform(-5, 5), 0, H), pe / 2.0, 0.0]
        self.bios[parent_id][2] = pe / 2.0
        return bid

    def _kill(self, bid):
        del self.genomes[bid]
        del self.bios[bid]
        try:
            os.remove(f"{self.workdir}/bio_{bid}.npy")
        except OSError:
            pass

    def step(self):
        if len(self.bios) == 0:
            self.history.append((0, 0, 0, 0, 0))
            return
        # 感知+决策（网络实时决定移动与营养策略 τ）
        for bid in list(self.bios.keys()):
            x, y, e, _tau0 = self.bios[bid]
            ang, d_light = nearest_light(x, y)
            intensity = light_intensity(x, y)
            crowd = 0
            for oid, (ox, oy, oe, otau) in self.bios.items():
                if oid != bid and np.hypot(ox - x, oy - y) < 3.0:
                    crowd += 1
            inp = np.array([np.sin(ang), np.cos(ang), d_light / 70.0,
                            intensity, crowd / 20.0, RNG.uniform(-1, 1) * 0.1, 1.0])
            dx_out, dy_out, tau = forward(self.genomes[bid], inp)
            norm = max(np.hypot(dx_out, dy_out), 1e-6)
            spd = SPEED * np.clip(norm, 0.1, 1.5)
            nx = np.clip(x + dy_out / norm * spd, 0, W)
            ny = np.clip(y + dx_out / norm * spd, 0, H)
            # 能量：光合（×(1-τ)^φ 权衡）+ 消耗
            # 拥挤共享减轻（0.5→0.15：原共享让光斑承载力仅 1-2 只——细胞分散从不接触——
            # 吞噬无机会——0.15 让光斑养 5-8 只——接触频繁——异养可行）
            absorb = light_intensity(nx, ny) * ABS_RATE * (1.0 - tau) ** PHI / (1.0 + crowd * 0.15)
            e += absorb - BASE_COST - MOVE_COST * spd
            self.bios[bid] = [nx, ny, e, tau]
        # 吞噬（接触：τ 高者吞 τ 低者——能量 = 被吞者能量 × τ^φ——能量流守恒）
        for bid in list(self.bios.keys()):
            if bid not in self.bios:
                continue
            x, y, e, tau = self.bios[bid]
            for oid in list(self.bios.keys()):
                if oid == bid or oid not in self.bios:
                    continue
                ox, oy, oe, otau = self.bios[oid]
                # 饿才吞（e < 80——饱了不吞——消化——猎物有喘息——共存条件——
                # 否则捕食者吃光猎物双灭绝——像 stage25 饱食 cap）
                if np.hypot(ox - x, oy - y) < CONTACT and tau > otau + 0.02 and e < 80.0:
                    # 吞噬成功（τ 优势即可——0.15 门槛造成先有鸡问题：变异出高 τ 者光合先亏死）
                    # 收益 = 被吞者能量 × (0.3 + τ^φ)——吃掉 = 大餐（能量流守恒）
                    gain = oe * (0.3 + tau ** PHI)
                    self.bios[bid][2] += gain
                    self._kill(oid)
                    self.kills += 1
                    break
        # 死亡/繁殖
        for bid in list(self.bios.keys()):
            if self.bios[bid][2] <= 0:
                self._kill(bid)
            elif self.bios[bid][2] > REPRO_E:
                self._reproduce(bid)
        # 统计
        if len(self.bios):
            taus = [b[3] for b in self.bios.values()]
            self.history.append((len(self.bios), np.mean(taus),
                                 np.sum(np.array(taus) > 0.5), np.mean([b[2] for b in self.bios.values()]),
                                 self.kills))
        else:
            self.history.append((0, 0, 0, 0, self.kills))

    def run(self, T):
        for _ in range(T):
            self.step()
        return np.array(self.history)

def run():
    print("=== M5 阶段 29：异养涌现（阶段 B——混合营养连续谱） ===\n")
    print("研究：φ=1.8 权衡（Ward）| 资源比率（Tilman）| 捕食-光合嵌合（Myxococcota）")
    print("τ=网络输出（神经=细胞——行为训练）——基因只塑造网络结构\n")

    w = HeteroWorld(workdir="stage29_evo", n0=60)
    h = w.run(2500)
    for t in range(0, 2500, 250):
        print(f"t={t:4d} 细胞={h[t,0]:3.0f} 平均τ={h[t,1]:.2f} 异养(τ>0.5)={h[t,2]:3.0f} "
              f"平均能量={h[t,3]:5.1f} 累计吞噬={h[t,4]:3.0f}")
    live = np.mean(h[500:, 0] > 5)
    het_final = h[-1, 2]
    print(f"\n[结果] 种群自持: {100*live:.0f}% | 末代异养者(τ>0.5): {het_final:.0f}"
          f"（{'✓ 异养涌现——混合营养谱分化' if het_final > 3 else '异养未涌现——纯自养稳定'}）")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(h[:, 0], label="cells")
    axes[0].set_title("Population")
    axes[0].legend(fontsize=8)
    axes[1].plot(h[:, 1], label="avg tau", color='orange')
    axes[1].plot(h[:, 2], label="heterotrophs", color='red')
    axes[1].set_title("Trophic strategy evolution")
    axes[1].legend(fontsize=8)
    axes[2].plot(h[:, 4], label="cumulative kills")
    axes[2].set_title("Predation events")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("fig_stage29.png", dpi=110)
    print("\n[plot] saved fig_stage29.png")
    print("[done] stage29 heterotrophy")

if __name__ == "__main__":
    run()
