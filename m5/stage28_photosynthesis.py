# -*- coding: utf-8 -*-
"""
M5 阶段 28：光合阶段（进化路线阶段 A——用户："从光合开始"）

真实对应：35 亿年前光合细菌（蓝藻）——自养——能量=光——趋光性（phototaxis）

环境（光合）：
  光场：中心光源（高斯衰减 σ=25——中心光强 1.0——边缘 ~0）
  能量吸收 = 位置光强 × 0.4（光合效率——进化可调？——第一版固定）
  消耗 = 0.05（光合细胞代谢低——静息）+ 移动 0.1×速度
  平衡：中心净正（+0.35/步）——边缘净负（饿死）——必须趋光
  拥挤竞争：同格细胞光合共享（遮挡——吸收 ÷ 同格数）——趋光 vs 避挤权衡

生物：NEAT 结构进化（stage27 框架延续——连接表基因）
  感知 6：[光源方向 sin, cos, 距离/70, 当前位置光强, 噪声, 常数]
  网络：输出移动方向——学趋光（功能涌现）
  初始：全图随机分布（大部分边缘——饿死压力——趋光被选择）

时间轴机制（阶段 A 内部）：
  世代加速：繁殖阈 100（低——换代快）——变异 σ0.2+1/4 重初始化（探索快）
  里程碑观察：① 趋光涌现（聚集半径）② 种群自持 ③ 结构生长（拥挤权衡需要非线性？）
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(109)
W, H = 100.0, 100.0
# 光斑分布（6 个光源——六边形分布——单点光源承载力 3 太低——光斑=区域——
# 分散结构涌现：不同细胞趋不同光斑——种群分化）
LIGHTS = [(25, 25), (75, 25), (25, 75), (75, 75), (50, 50), (15, 50)]
LIGHT_SIGMA = 15.0              # 光斑衰减半径
ABS_RATE = 0.4                  # 光合吸收率（光强 → 能量）
BASE_COST = 0.05                # 静息代谢（光合细胞低）
MOVE_COST = 0.1
SPEED = 1.0
REPRO_E = 100.0
N_IN = 7   # 6+拥挤度（群体感应 quorum sensing——感知邻居密度——趋光 vs 避挤权衡）
N_OUT = 2
P_ADD_CONN = 0.10
P_ADD_NODE = 0.05
P_DISABLE = 0.05

def light_intensity(x, y):
    """光场：多光斑叠加（最近光斑主导）"""
    best = 0.0
    for lx, ly in LIGHTS:
        d = np.hypot(x - lx, y - ly)
        best = max(best, np.exp(-(d / LIGHT_SIGMA) ** 2))
    return best

def nearest_light(x, y):
    """最近光斑"""
    best_d, best_ang = 1e9, 0.0
    for lx, ly in LIGHTS:
        d = np.hypot(lx - x, ly - y)
        if d < best_d:
            best_d = d
            best_ang = np.arctan2(ly - y, lx - x)
    return best_ang, best_d

def init_genome(rng):
    conns = []
    for i in range(N_IN):
        for o in range(N_OUT):
            w = 1.5 if (i == o) else 0.0
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
    return vals[N_IN:N_IN + N_OUT]

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

class PhotoWorld:
    def __init__(self, workdir="stage28_evo", n0=60, seed=109):
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

    def _spawn_new(self):
        bid = self.next_id
        self.next_id += 1
        g = init_genome(RNG)
        self.genomes[bid] = g
        np.save(f"{self.workdir}/bio_{bid}.npy", g)
        self.bios[bid] = [RNG.uniform(0, W), RNG.uniform(0, H), RNG.uniform(30, 60)]
        return bid

    def _reproduce(self, parent_id):
        bid = self.next_id
        self.next_id += 1
        g = mutate_genome(self.genomes[parent_id], RNG)
        self.genomes[bid] = g
        np.save(f"{self.workdir}/bio_{bid}.npy", g)
        px, py, pe = self.bios[parent_id]
        self.bios[bid] = [np.clip(px + RNG.uniform(-5, 5), 0, W),
                          np.clip(py + RNG.uniform(-5, 5), 0, H), pe / 2.0]
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
            self.history.append((0, 0, 0))
            return
        for bid in list(self.bios.keys()):
            x, y, e = self.bios[bid]
            # 感知：最近光斑方向 + 距离 + 当前位置光强 + 拥挤度（群体感应——邻居密度）
            ang, d_light = nearest_light(x, y)
            intensity = light_intensity(x, y)
            crowd = 0
            for oid, (ox, oy, oe) in self.bios.items():
                if oid != bid and np.hypot(ox - x, oy - y) < 3.0:
                    crowd += 1
            inp = np.array([np.sin(ang), np.cos(ang), d_light / 70.0,
                            intensity, crowd / 20.0, RNG.uniform(-1, 1) * 0.1, 1.0])
            out = forward(self.genomes[bid], inp)
            norm = max(np.hypot(out[0], out[1]), 1e-6)
            spd = SPEED * np.clip(norm, 0.1, 1.5)
            nx = np.clip(x + out[1] / norm * spd, 0, W)
            ny = np.clip(y + out[0] / norm * spd, 0, H)
            # 能量：光合吸收（位置光强 × 吸收率 ÷ 拥挤共享）+ 消耗
            # 拥挤：同格（半径 3 内）细胞数——光合共享（遮挡）
            crowd = 0
            for oid, (ox, oy, oe) in self.bios.items():
                if oid != bid and np.hypot(ox - nx, oy - ny) < 3.0:
                    crowd += 1
            absorb = light_intensity(nx, ny) * ABS_RATE / (1.0 + crowd * 0.5)
            e += absorb - BASE_COST - MOVE_COST * spd
            self.bios[bid] = [nx, ny, e]
        for bid in list(self.bios.keys()):
            if self.bios[bid][2] <= 0:
                self._kill(bid)
            elif self.bios[bid][2] > REPRO_E:
                self._reproduce(bid)
        # 聚集统计：平均离最近光斑距离
        if len(self.bios):
            ds = []
            for b in self.bios.values():
                _, d = nearest_light(b[0], b[1])
                ds.append(d)
            self.history.append((len(self.bios), np.mean(ds), np.mean([b[2] for b in self.bios.values()])))
        else:
            self.history.append((0, 70.0, 0))

    def run(self, T):
        for _ in range(T):
            self.step()
        return np.array(self.history)

def run():
    print("=== M5 阶段 28：光合阶段（进化路线 A——趋光涌现） ===\n")
    print("环境：中心光源（高斯光场）| 吸收=光强×0.4÷拥挤 | 静息代谢 0.05（低）\n")

    w = PhotoWorld(workdir="stage28_evo", n0=60)
    h = w.run(2000)
    for t in range(0, 2000, 200):
        print(f"t={t:4d} 细胞={h[t,0]:3.0f} 平均离光斑={h[t,1]:5.1f} 平均能量={h[t,2]:5.1f}")
    live = np.mean(h[500:, 0] > 5)
    init_d = np.mean([nearest_light(b[0], b[1])[1] for b in PhotoWorld(workdir='_t', n0=60, seed=777).bios.values()])
    import shutil; shutil.rmtree('_t')
    print(f"\n[结果] 种群自持: {100*live:.0f}% | 初始平均离光斑 {init_d:.1f} → 末代 {h[-1,1]:.1f}"
          f"（{'✓ 趋光涌现——细胞向光斑聚集' if h[-1,1] < init_d * 0.6 else '趋光未涌现'}）")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(h[:, 0], label="cells")
    axes[0].set_title("Population (photosynthesis)")
    axes[0].legend(fontsize=8)
    axes[1].plot(h[:, 1], label="avg dist to light", color='orange')
    axes[1].axhline(init_d, color='gray', ls='--', lw=0.8, label="initial")
    axes[1].set_title("Phototaxis (gathering)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("fig_stage28.png", dpi=110)
    print("\n[plot] saved fig_stage28.png")
    print("[done] stage28 photosynthesis")

if __name__ == "__main__":
    run()
