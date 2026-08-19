# -*- coding: utf-8 -*-
"""
M5 阶段 27：结构进化（NEAT-lite）——"有了功能才会有行为，行为是训练出来的"

用户设计哲学：结构 = 基因（拓扑也进化）→ 功能涌现（隐藏单元学会角色）→ 行为 = 训练产物

与 stage26 的差异：
  stage26：固定结构 6→16→2（手工定的拓扑）——只进化权重
  stage27：结构也进化——连接表基因——从最小结构开始生长：
    初始：6 输入 → 2 输出（全连接——趋化先验权重——无隐藏层）
    变异：
      权重扰动（σ0.2 比例 + 重初始化）
      加连接（0.1 概率——任意未连接节点对——保证无环）
      加节点（0.05 概率——拆一个连接——中间插隐藏单元——结构生长！）
      禁用连接（0.05 概率——剪枝）
    结构可遗传（繁殖时结构变异——子代可能更大/更小/不同）

功能涌现观察：
  隐藏节点数随时间（结构生长曲线）
  各隐藏单元的"角色"（输入权重分布——感知什么）
  行为 = 结构与权重的训练产物（无手工行为设计）

文件格式：bio_N.npy 保存连接表 [[from, to, weight, enabled], ...]——节点池约定：
  0-5 输入（同 stage26）/ 6-7 输出 / 8+ 隐藏（动态添加）
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(109)
W, H = 100.0, 100.0
FOOD_R = 2.0
FOOD_ENERGY = 20.0
FOOD_N0 = 100
FOOD_SPAWN = 0.25
MAX_FOOD = 250
BASE_COST = 0.2
MOVE_COST = 0.1
SPEED = 1.0
REPRO_E = 100.0
N_IN = 6
N_OUT = 2

P_ADD_CONN = 0.10   # 加连接概率
P_ADD_NODE = 0.05   # 加节点概率（结构生长）
P_DISABLE = 0.05    # 禁用连接概率

def init_genome(rng):
    """初始基因组：6 输入 → 2 输出全连接（趋化先验权重——同 stage26——无隐藏层）"""
    conns = []
    for i in range(N_IN):
        for o in range(N_OUT):
            w = 1.5 if (i == o) else 0.0   # 输入 i → 输出 i（sin→sin, cos→cos）
            conns.append([i, N_IN + o, w + rng.normal(0, 0.05), 1.0])
    return np.array(conns)

def forward(genome, x):
    """前向传播（DAG——拓扑顺序计算）：
    节点值：输入=x，隐藏/输出 = tanh(Σ 入连接×源值)（输出层也用 tanh——已有趋化先验尺度）"""
    max_node = int(genome[:, :2].max())
    n_nodes = max_node + 1
    vals = np.zeros(n_nodes)
    vals[:N_IN] = x
    enabled = genome[:, 3] > 0.5
    conns = genome[enabled]
    # 按 from 拓扑：重复传播直到收敛（无环保证——最多 n 轮）
    for _ in range(n_nodes):
        for (f, t, w, _e) in conns:
            f, t = int(f), int(t)
            if t >= N_IN:
                vals[t] += w * vals[f]
        # 应用激活（输入节点不动——隐藏/输出 tanh）
        for t in range(N_IN, n_nodes):
            vals[t] = np.tanh(vals[t])
    return vals[N_IN:N_IN + N_OUT]

def mutate_genome(genome, rng):
    """结构变异：权重扰动 + 加连接 + 加节点 + 禁用——结构可遗传"""
    g = genome.copy()
    # 1. 权重扰动
    w_idx = np.arange(len(g))
    g[w_idx, 2] += rng.normal(0, 0.2, len(g)) * np.abs(g[w_idx, 2]) + 0.03
    # 2. 禁用连接（剪枝）
    if len(g) > N_IN * N_OUT:
        for i in range(len(g)):
            if g[i, 3] > 0.5 and rng.random() < P_DISABLE / len(g):
                g[i, 3] = 0.0
    # 3. 加节点（拆连接——结构生长！）
    if rng.random() < P_ADD_NODE:
        on = np.where(g[:, 3] > 0.5)[0]
        if len(on):
            ci = int(rng.choice(on))
            f, t, w = int(g[ci, 0]), int(g[ci, 1]), g[ci, 2]
            new_id = int(g[:, :2].max()) + 1
            g[ci, 3] = 0.0   # 原连接禁用
            g = np.vstack([g,
                           [f, new_id, w, 1.0],      # 入段（继承权重）
                           [new_id, t, 1.0, 1.0]])   # 出段
    # 4. 加连接（任意未连接节点对——无环：from < to）
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

def genome_size(genome):
    """结构度量：隐藏节点数 + 有效连接数"""
    max_id = int(genome[:, :2].max())
    hidden = max(0, max_id - N_IN - N_OUT + 1)
    conns = int(np.sum(genome[:, 3] > 0.5))
    return hidden, conns

class NeatWorld:
    def __init__(self, workdir="stage27_evo", n0=40, seed=109):
        self.workdir = workdir
        os.makedirs(workdir, exist_ok=True)
        for f in os.listdir(workdir):
            os.remove(os.path.join(workdir, f))
        global RNG
        RNG = np.random.default_rng(seed)
        self.fx = RNG.uniform(0, W, FOOD_N0)
        self.fy = RNG.uniform(0, H, FOOD_N0)
        self.bios = {}
        self.genomes = {}
        self.next_id = 0
        for _ in range(n0):
            self._spawn_new()
        self.history = []
        self.struct_hist = []   # [平均隐藏节点, 平均连接数]

    def _spawn_new(self):
        bid = self.next_id
        self.next_id += 1
        g = init_genome(RNG)
        self.genomes[bid] = g
        np.save(f"{self.workdir}/bio_{bid}.npy", g)
        self.bios[bid] = [RNG.uniform(0, W), RNG.uniform(0, H), RNG.uniform(60, 100)]
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
        if len(self.fx) < MAX_FOOD and RNG.random() < FOOD_SPAWN:
            self.fx = np.append(self.fx, RNG.uniform(0, W))
            self.fy = np.append(self.fy, RNG.uniform(0, H))
        if len(self.bios) == 0:
            self.history.append((0, len(self.fx), 0))
            self.struct_hist.append((0, 0))
            return
        for bid in list(self.bios.keys()):
            x, y, e = self.bios[bid]
            if len(self.fx):
                ds = np.hypot(self.fx - x, self.fy - y)
                idx = int(np.argmin(ds))
                fd = ds[idx]
                ang = np.arctan2(self.fy[idx] - y, self.fx[idx] - x)
            else:
                fd, ang = 70.0, RNG.uniform(0, 2*np.pi)
            inp = np.array([np.sin(ang), np.cos(ang), fd / 70.0,
                            e / 150.0, RNG.uniform(-1, 1) * 0.1, 1.0])
            out = forward(self.genomes[bid], inp)
            norm = max(np.hypot(out[0], out[1]), 1e-6)
            spd = SPEED * np.clip(norm, 0.3, 1.5)
            nx = np.clip(x + out[1] / norm * spd, 0, W)
            ny = np.clip(y + out[0] / norm * spd, 0, H)
            e -= BASE_COST + MOVE_COST * spd
            if len(self.fx):
                ds = np.hypot(self.fx - nx, self.fy - ny)
                hit = np.where(ds < FOOD_R)[0]
                if len(hit):
                    e += FOOD_ENERGY * len(hit)
                    keep = np.ones(len(self.fx), dtype=bool)
                    keep[hit] = False
                    self.fx, self.fy = self.fx[keep], self.fy[keep]
            self.bios[bid] = [nx, ny, e]
        for bid in list(self.bios.keys()):
            if self.bios[bid][2] <= 0:
                self._kill(bid)
            elif self.bios[bid][2] > REPRO_E:
                self._reproduce(bid)
        # 结构统计
        hs, cs = [], []
        for g in self.genomes.values():
            hh, cc = genome_size(g)
            hs.append(hh); cs.append(cc)
        self.struct_hist.append((np.mean(hs) if hs else 0, np.mean(cs) if cs else 0))
        self.history.append((len(self.bios), len(self.fx), 0))

    def run(self, T):
        for _ in range(T):
            self.step()
        return np.array(self.history)

def run():
    print("=== M5 阶段 27：结构进化（NEAT-lite）——功能涌现 → 行为是训练产物 ===\n")
    print("初始：6→2 全连接（趋化先验——无隐藏层）| 变异：权重/加连接/加节点/禁用\n")

    w = NeatWorld(workdir="stage27_evo", n0=40)
    h = w.run(2000)
    for t in range(0, 2000, 200):
        hh, cc = w.struct_hist[t]
        print(f"t={t:4d} 生物={h[t,0]:3.0f} 食物={h[t,1]:3.0f} 平均隐藏节点={hh:.2f} 连接数={cc:.1f}")
    live = np.mean(h[500:, 0] > 5)
    print(f"\n[结果] 种群自持: {100*live:.0f}%")

    # 结构分布（末代）
    hs = [genome_size(g)[0] for g in w.genomes.values()]
    cs = [genome_size(g)[1] for g in w.genomes.values()]
    print(f"[结构] 末代: 隐藏节点分布 {np.percentile(hs,[0,50,100]) if hs else '—'} | "
          f"连接数 {np.percentile(cs,[0,50,100]) if cs else '—'}")
    if hs:
        print(f"[结构] 最大结构: {max(hs)} 隐藏节点 / {max(cs)} 连接（结构生长数）")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(h[:, 0], label="population")
    axes[0].set_title("Population (structure evolution)")
    axes[0].legend(fontsize=8)
    sh = np.array(w.struct_hist)
    axes[1].plot(sh[:, 0], label="avg hidden nodes")
    axes[1].plot(sh[:, 1] / 10, label="avg conns/10")
    axes[1].set_title("Structure growth")
    axes[1].legend(fontsize=8)
    if hs:
        axes[2].hist(hs, bins=10)
        axes[2].set_title("Hidden node distribution (final)")
    fig.tight_layout()
    fig.savefig("fig_stage27.png", dpi=110)
    print("\n[plot] saved fig_stage27.png")
    print("[done] stage27 NEAT evolution")

if __name__ == "__main__":
    run()
