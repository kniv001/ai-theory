# -*- coding: utf-8 -*-
"""
M5 阶段 30：细胞多维机制（阶段 B 扩展——防御/体积/威胁逃逸——生态位三维）

用户："对于细胞加入更多的机制，比如防御等等参数，目前的结论还是不够"

单细胞真实对应：
  细胞壁防御 d（细菌肽聚糖壁——防吞噬）：被吞概率 = 1/(1+exp(6×(d_i-d_j)))
    ——代价：代谢 +0.1×d（壁合成能耗）——厚壁者安全吃肉（异养专精者防反噬）
  体积 s（大小——大难吞/光合表面积比降）：
    被吞概率 ×(s_j/s_i)（大难吞）——光合 ×(0.5+0.5×s_min/s)（表面积/体积比）
    吞噬收益 ×s_j（大猎物能量多——能量流守恒）
  威胁感知+逃逸（行为——网络学）：输入加"最近 τ 高邻居方向/距离"——学远离捕食者

生态位三维空间：τ（行为营养）× d（防御身体）× s（体积身体）——分化丰富
两层时间尺度：d/s = 基因（身体——慢）| τ/逃逸 = 网络（行为——快）
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(109)
W, H = 100.0, 100.0
LIGHTS = [(25, 25), (75, 75), (50, 50), (15, 75)]
LIGHT_SIGMA = 15.0
ABS_RATE = 0.7    # 光斑承载力 8-10 只（猎物密度高——吞噬持续——军备竞赛活跃）
BASE_COST = 0.05
MOVE_COST = 0.1
SPEED = 1.0
REPRO_E = 80.0   # 100→80（繁殖更多——密度↑——接触机会↑）
CONTACT = 2.5
PHI = 1.8
S_MIN = 0.3
N_IN = 9    # 光 sin/cos + 光距 + 光强 + 拥挤 + 威胁 sin/cos + 噪声 + 常数
N_OUT = 3
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

def prey_prob(def_i, def_j, s_i, s_j):
    """被吞概率：防御差（细胞壁）+ 体积比（大难吞）"""
    p_wall = 1.0 / (1.0 + np.exp(6.0 * (def_i - def_j)))
    p_size = s_j / max(s_i, 1e-6)
    return np.clip(p_wall * p_size, 0.0, 0.95)

def photo_eff(s):
    """光合效率：表面积/体积比（大体积效率降——0.5+0.5×min/s 太狠——降 32% 收支转负——
    温和化 0.75+0.25×min/s——s=0.82→0.84——体积分化仍有空间）"""
    return 0.75 + 0.25 * S_MIN / s

def init_genome(rng):
    conns = []
    for i in range(N_IN):
        for o in range(N_OUT):
            if o < 2:
                w = 1.5 if (i == o) else 0.0
            else:
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
    tau = 1.0 / (1.0 + np.exp(-out[2]))
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

class CellWorld:
    def __init__(self, workdir="stage30_evo", n0=60, seed=109):
        self.workdir = workdir
        os.makedirs(workdir, exist_ok=True)
        for f in os.listdir(workdir):
            os.remove(os.path.join(workdir, f))
        global RNG
        RNG = np.random.default_rng(seed)
        self.bios = {}       # id -> [x, y, e, tau, d(防御), s(体积)]
        self.genomes = {}
        self.next_id = 0
        for _ in range(n0):
            self._spawn_new()
        self.history = []
        self.kills = 0
        self.attacks = 0

    def _spawn_new(self):
        bid = self.next_id
        self.next_id += 1
        g = init_genome(RNG)
        self.genomes[bid] = g
        np.save(f"{self.workdir}/bio_{bid}.npy", g)
        self.bios[bid] = [RNG.uniform(0, W), RNG.uniform(0, H), RNG.uniform(30, 60),
                          0.0, RNG.uniform(0.1, 0.4), RNG.uniform(0.5, 1.2)]
        return bid

    def _reproduce(self, parent_id):
        bid = self.next_id
        self.next_id += 1
        g = mutate_genome(self.genomes[parent_id], RNG)
        self.genomes[bid] = g
        np.save(f"{self.workdir}/bio_{bid}.npy", g)
        px, py, pe, pt, pd, ps = self.bios[parent_id]
        # d/s 变异（身体基因——±10% 小 + 5% 大 ±0.3）
        if RNG.random() < 0.05:
            nd = np.clip(pd + RNG.uniform(-0.3, 0.3), 0.0, 1.0)
        else:
            nd = np.clip(pd * RNG.uniform(0.9, 1.1), 0.0, 1.0)
        ns = np.clip(ps * RNG.uniform(0.9, 1.1), S_MIN, 1.7)
        self.bios[bid] = [np.clip(px + RNG.uniform(-5, 5), 0, W),
                          np.clip(py + RNG.uniform(-5, 5), 0, H), pe / 2.0, 0.0, nd, ns]
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
            self.history.append((0, 0, 0, 0, 0, 0, 0, 0))
            return
        # 感知+决策
        for bid in list(self.bios.keys()):
            x, y, e, _t, d, s = self.bios[bid]
            ang, d_light = nearest_light(x, y)
            intensity = light_intensity(x, y)
            crowd = 0
            threat_ang, threat_d = 0.0, 50.0
            for oid, (ox, oy, oe, otau, od, os_) in self.bios.items():
                if oid == bid:
                    continue
                if np.hypot(ox - x, oy - y) < 3.0:
                    crowd += 1
                # 威胁：τ 显著高者（潜在捕食者）——最近者
                if otau > _t + 0.1:
                    td = np.hypot(ox - x, oy - y)
                    if td < threat_d:
                        threat_d = td
                        threat_ang = np.arctan2(oy - y, ox - x)
            inp = np.array([np.sin(ang), np.cos(ang), d_light / 70.0,
                            intensity, crowd / 20.0,
                            np.sin(threat_ang) if threat_d < 50 else 0.0,
                            np.cos(threat_ang) if threat_d < 50 else 0.0,
                            RNG.uniform(-1, 1) * 0.1, 1.0])
            dx_out, dy_out, tau = forward(self.genomes[bid], inp)
            norm = max(np.hypot(dx_out, dy_out), 1e-6)
            spd = SPEED * np.clip(norm, 0.1, 1.5)
            nx = np.clip(x + dy_out / norm * spd, 0, W)
            ny = np.clip(y + dx_out / norm * spd, 0, H)
            # 能量：光合（×体积效率 ×(1-τ)^φ ÷拥挤）+ 壁代价 - 消耗
            absorb = intensity * ABS_RATE * photo_eff(s) * (1.0 - tau) ** PHI / (1.0 + crowd * 0.08)
            e += absorb - BASE_COST - MOVE_COST * spd - 0.1 * d   # 细胞壁合成代价
            self.bios[bid] = [nx, ny, e, tau, d, s]
        # 吞噬（τ 优势 + 防御差概率 + 饿才吞 + 体积收益）
        for bid in list(self.bios.keys()):
            if bid not in self.bios:
                continue
            x, y, e, tau, d, s = self.bios[bid]
            for oid in list(self.bios.keys()):
                if oid == bid or oid not in self.bios:
                    continue
                ox, oy, oe, otau, od, os_ = self.bios[oid]
                if np.hypot(ox - x, oy - y) < CONTACT and tau > otau + 0.02 and e < 80.0:
                    self.attacks += 1
                    p = prey_prob(d, od, s, os_)
                    if RNG.random() < p:
                        gain = oe * (0.3 + tau ** PHI) * os_   # 大猎物能量多
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
            ds = [b[4] for b in self.bios.values()]
            ss = [b[5] for b in self.bios.values()]
            self.history.append((len(self.bios), np.mean(taus), np.mean(ds), np.mean(ss),
                                 np.sum(np.array(taus) > 0.5), self.kills, self.attacks,
                                 np.mean([b[2] for b in self.bios.values()])))
        else:
            self.history.append((0, 0, 0, 0, 0, self.kills, self.attacks, 0))

    def run(self, T):
        for _ in range(T):
            self.step()
        return np.array(self.history)

def run():
    print("=== M5 阶段 30：细胞多维机制（防御/体积/威胁逃逸——生态位三维） ===\n")
    print("τ=网络输出（行为）| d 防御/细胞壁（基因）| s 体积（基因）——两层时间尺度\n")

    w = CellWorld(workdir="stage30_evo", n0=60)
    h = w.run(2500)
    for t in range(0, 2500, 250):
        print(f"t={t:4d} 细胞={h[t,0]:3.0f} τ={h[t,1]:.2f} 防御d={h[t,2]:.2f} 体积s={h[t,3]:.2f} "
              f"异养={h[t,4]:2.0f} 吞噬成功={h[t,5]:3.0f}/{h[t,6]:3.0f}")
    live = np.mean(h[500:, 0] > 5)
    het = h[-1, 4]
    print(f"\n[结果] 种群自持: {100*live:.0f}% | 末代异养(τ>0.5): {het:.0f}")

    # 生态位三维散点（末代）
    fig = plt.figure(figsize=(13, 4))
    ax1 = fig.add_subplot(131)
    ax1.scatter([b[3] for b in w.bios.values()], [b[4] for b in w.bios.values()], s=20)
    ax1.set_xlabel("tau (behavior)"); ax1.set_ylabel("defense d (gene)")
    ax1.set_title("Niche: tau x defense")
    ax2 = fig.add_subplot(132)
    ax2.scatter([b[3] for b in w.bios.values()], [b[5] for b in w.bios.values()], s=20)
    ax2.set_xlabel("tau"); ax2.set_ylabel("size s")
    ax2.set_title("Niche: tau x size")
    ax3 = fig.add_subplot(133)
    ax3.plot(h[:, 0], label="cells")
    ax3.plot(h[:, 5], label="kills", color='red')
    ax3.set_title("Population & predation")
    ax3.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("fig_stage30.png", dpi=110)
    print("\n[plot] saved fig_stage30.png")
    print("[done] stage30 cell mechanisms")

if __name__ == "__main__":
    run()
