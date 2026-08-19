# -*- coding: utf-8 -*-
"""
M5 阶段 31：多细胞聚合（进化路线阶段 C——聚合体涌现）

真实对应：10 亿年前多细胞起源——分裂不完全（子细胞黏连——团藻 Volvox）——
  黏附分子（cadherin）——聚合体 = 共享防御（体积大难吞）——但内部光衰减
  （Volvox 内部细胞退化——分工雏形）

机制：
  黏附基因 a（0-1——身体基因——变异遗传）：
    繁殖时 a 高 → 子代留在母体旁（黏附——聚合体生长——不移动/固化）
             a 低 → 子代分离（自由生活——移动——觅食/吞噬）
  聚合体：空间连接簇（距离 < CONTACT 的连接组件）
    光衰减：簇内光强 × 1/(1+0.3×(簇成员-1))（内层细胞光合差——聚合体大小受光限制）
    防御：被吞概率 × 1/(1+0.2×(簇成员-1))（聚合体难吞——共享防御）
    被吞的是成员（一个细胞）——簇失去成员但整体存活
  生态位分化预测：单体（机动——光合/吞噬）vs 聚合体（静止——体积防御）——
    最佳簇大小由 防御收益 vs 光衰减 平衡涌现

基于 stage30（τ 行为 + d 防御 + s 体积保留）
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
ABS_RATE = 0.7
BASE_COST = 0.05
MOVE_COST = 0.1
SPEED = 1.0
REPRO_E = 80.0
CONTACT = 2.5
PHI = 1.8
S_MIN = 0.3
N_IN = 11   # 9+邻居方向 sin/cos（聚合体协调感知——学集体移动）
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

def prey_prob(def_i, def_j, s_i, s_j, cluster_i, cluster_j):
    """被吞概率：防御差 + 体积比 + 聚合体防御（大簇难吞——共享防御）"""
    p_wall = 1.0 / (1.0 + np.exp(6.0 * (def_i - def_j)))
    p_size = s_j / max(s_i, 1e-6)
    p_clust = 1.0 / (1.0 + 0.5 * max(0, cluster_j - 1))   # 聚合体防御（0.2→0.5：簇3降50%——
    # 真实多细胞体积防御巨大——收益要超过光衰减）
    return np.clip(p_wall * p_size * p_clust, 0.0, 0.95)

def photo_eff(s):
    return 0.75 + 0.25 * S_MIN / s

def init_genome(rng):
    conns = []
    for i in range(N_IN):
        for o in range(N_OUT):
            if o < 2:
                w = 1.5 if (i == o) else 0.0
            else:
                w = -0.3 if i == N_IN - 1 else 0.0   # 猎食压力（τ 初始更高——更多混合营养者）
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

class MultiWorld:
    def __init__(self, workdir="stage31_evo", n0=60, seed=109):
        self.workdir = workdir
        os.makedirs(workdir, exist_ok=True)
        for f in os.listdir(workdir):
            os.remove(os.path.join(workdir, f))
        global RNG
        RNG = np.random.default_rng(seed)
        self.bios = {}       # id -> [x, y, e, tau, d, s, a(黏附)]
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
        self.bios[bid] = [RNG.uniform(0, W), RNG.uniform(0, H), RNG.uniform(30, 60),
                          0.0, RNG.uniform(0.1, 0.4), RNG.uniform(0.5, 1.2),
                          RNG.uniform(0.1, 0.4)]
        return bid

    def _reproduce(self, parent_id):
        bid = self.next_id
        self.next_id += 1
        g = mutate_genome(self.genomes[parent_id], RNG)
        self.genomes[bid] = g
        np.save(f"{self.workdir}/bio_{bid}.npy", g)
        px, py, pe, pt, pd, ps, pa = self.bios[parent_id]
        if RNG.random() < 0.05:
            nd = np.clip(pd + RNG.uniform(-0.3, 0.3), 0.0, 1.0)
        else:
            nd = np.clip(pd * RNG.uniform(0.9, 1.1), 0.0, 1.0)
        ns = np.clip(ps * RNG.uniform(0.9, 1.1), S_MIN, 1.7)
        na = np.clip(pa * RNG.uniform(0.9, 1.1), 0.0, 1.0)
        # 黏附连续化（先有鸡问题：阈值 0.5 时初始 0.24 爬不到——黏附者从不出现——
        # off = ±6×(1-a)——a 高紧黏附 a 低分离——选择连续——任何 a 都有倾向）
        off = RNG.uniform(-6, 6) * (1.0 - pa)
        self.bios[bid] = [np.clip(px + off, 0, W), np.clip(py + off, 0, H),
                          pe / 2.0, 0.0, nd, ns, na]
        self.bios[parent_id][2] = pe / 2.0
        return bid

    def _kill(self, bid):
        del self.genomes[bid]
        del self.bios[bid]
        try:
            os.remove(f"{self.workdir}/bio_{bid}.npy")
        except OSError:
            pass

    def clusters(self):
        """空间连接簇：距离 < CONTACT 的连接组件——返回 {bid: 簇大小}"""
        bids = list(self.bios.keys())
        parent = {b: b for b in bids}
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for i in range(len(bids)):
            for j in range(i + 1, len(bids)):
                a, b = bids[i], bids[j]
                if np.hypot(self.bios[a][0] - self.bios[b][0],
                            self.bios[a][1] - self.bios[b][1]) < CONTACT:
                    parent[find(a)] = find(b)
        size = {}
        for b in bids:
            r = find(b)
            size[r] = size.get(r, 0) + 1
        return {b: size[find(b)] for b in bids}

    def step(self):
        if len(self.bios) == 0:
            self.history.append((0, 0, 0, 0, 0, 0, 0))
            return
        clust = self.clusters()
        # 感知+决策（黏附细胞固化不移动——a>0.5 时仍可微动？——第一版：黏附者不动）
        for bid in list(self.bios.keys()):
            x, y, e, _t, d, s, a = self.bios[bid]
            ang, d_light = nearest_light(x, y)
            intensity = light_intensity(x, y)
            crowd = 0
            threat_ang, threat_d = 0.0, 50.0
            for oid, (ox, oy, oe, otau, od, os_, oa) in self.bios.items():
                if oid == bid:
                    continue
                if np.hypot(ox - x, oy - y) < 3.0:
                    crowd += 1
                if otau > _t + 0.1:
                    td = np.hypot(ox - x, oy - y)
                    if td < threat_d:
                        threat_d = td
                        threat_ang = np.arctan2(oy - y, ox - x)
            # 邻居平均方向（簇成员——协调感知：学一致移动——聚合体集体行动）
            nb_ang_sum = np.zeros(2)
            for oid, (ox, oy, oe, otau, od, os_, oa) in self.bios.items():
                if oid == bid:
                    continue
                if np.hypot(ox - x, oy - y) < CONTACT * 1.5:
                    nb_ang_sum += np.array([np.sin(np.arctan2(oy - y, ox - x)),
                                            np.cos(np.arctan2(oy - y, ox - x))])
            nb_norm = max(np.hypot(*nb_ang_sum), 1e-6)
            inp = np.array([np.sin(ang), np.cos(ang), d_light / 70.0,
                            intensity, crowd / 20.0,
                            np.sin(threat_ang) if threat_d < 50 else 0.0,
                            np.cos(threat_ang) if threat_d < 50 else 0.0,
                            nb_ang_sum[0] / nb_norm, nb_ang_sum[1] / nb_norm,
                            RNG.uniform(-1, 1) * 0.1, 1.0])
            dx_out, dy_out, tau = forward(self.genomes[bid], inp)
            # 聚合体光衰减（簇内成员遮挡）+ 拥挤
            cs = clust.get(bid, 1)
            shade = 1.0 / (1.0 + 0.15 * (cs - 1))   # 光衰减减轻（0.3→0.15——配合能量共享——内层被供养）
            absorb = intensity * ABS_RATE * photo_eff(s) * (1.0 - tau) ** PHI \
                     * shade / (1.0 + crowd * 0.08)
            e += absorb - BASE_COST - MOVE_COST * 0.1 - 0.1 * d
            norm = max(np.hypot(dx_out, dy_out), 1e-6)
            spd = SPEED * np.clip(norm, 0.1, 1.5)
            nx = np.clip(x + dy_out / norm * spd, 0, W)
            ny = np.clip(y + dx_out / norm * spd, 0, H)
            e -= MOVE_COST * spd
            # 物理连接（多细胞本质：黏附分子——结构约束——聚合体稳定）：
            # 黏附细胞（a>0.5）被簇质心弹簧拉回（凝聚——聚合体=稳定结构——不是行为选择）
            if a > 0.3 and clust.get(bid, 1) > 1:   # 黏附约束（连续化后阈值降低）
                cx = np.mean([self.bios[m][0] for m in self.bios if m != bid and
                              np.hypot(self.bios[m][0]-x, self.bios[m][1]-y) < CONTACT*2])
                cy = np.mean([self.bios[m][1] for m in self.bios if m != bid and
                              np.hypot(self.bios[m][0]-x, self.bios[m][1]-y) < CONTACT*2])
                if np.isfinite(cx):
                    nx = np.clip(x + (cx - x) * 0.3, 0, W)
                    ny = np.clip(y + (cy - y) * 0.3, 0, H)
            self.bios[bid] = [nx, ny, e, tau, d, s, a]
        # 吞噬（聚合体防御——被吞成员概率降）
        for bid in list(self.bios.keys()):
            if bid not in self.bios:
                continue
            x, y, e, tau, d, s, a = self.bios[bid]
            for oid in list(self.bios.keys()):
                if oid == bid or oid not in self.bios:
                    continue
                ox, oy, oe, otau, od, os_, oa = self.bios[oid]
                if np.hypot(ox - x, oy - y) < CONTACT and tau > otau + 0.02 and e < 80.0:
                    p = prey_prob(d, od, s, os_, clust.get(bid, 1), clust.get(oid, 1))
                    if RNG.random() < p:
                        gain = oe * (0.3 + tau ** PHI) * os_
                        self.bios[bid][2] += gain
                        self._kill(oid)
                        self.kills += 1
                        break
        # 簇内能量共享（多细胞核心：聚合体=共同能量池——内部细胞被供养——
        # 光衰减不致命——分工雏形（外层光合供养内层））
        groups = {}
        for b, root in clust.items():
            if b not in self.bios:
                continue
            groups.setdefault(root, []).append(b)
        for members in groups.values():
            alive_m = [m for m in members if m in self.bios]
            if len(alive_m) > 1:
                pool = sum(self.bios[m][2] for m in alive_m) / len(alive_m)
                for m in alive_m:
                    self.bios[m][2] = pool
        # 死亡/繁殖
        for bid in list(self.bios.keys()):
            if self.bios[bid][2] <= 0:
                self._kill(bid)
            elif self.bios[bid][2] > REPRO_E:
                self._reproduce(bid)
        # 统计
        if len(self.bios):
            taus = [b[3] for b in self.bios.values()]
            as_ = [b[6] for b in self.bios.values()]
            csizes = list(clust.values())
            self.history.append((len(self.bios), np.mean(taus), np.mean(as_),
                                 max(csizes), np.mean(csizes), self.kills,
                                 np.mean([b[2] for b in self.bios.values()])))
        else:
            self.history.append((0, 0, 0, 0, 0, self.kills, 0))

    def run(self, T):
        for _ in range(T):
            self.step()
        return np.array(self.history)

def run():
    print("=== M5 阶段 31：多细胞聚合（阶段 C——聚合体涌现） ===\n")
    print("黏附 a（基因）| 聚合体防御（簇大难吞）| 光衰减（簇内遮挡）\n")

    w = MultiWorld(workdir="stage31_evo", n0=60)
    h = w.run(2500)
    for t in range(0, 2500, 250):
        print(f"t={t:4d} 细胞={h[t,0]:3.0f} τ={h[t,1]:.2f} 黏附a={h[t,2]:.2f} "
              f"最大簇={h[t,3]:2.0f} 平均簇={h[t,4]:.1f} 吞噬={h[t,5]:3.0f}")
    live = np.mean(h[500:, 0] > 5)
    a_final = h[-1, 2]
    maxc = h[-1, 3]
    print(f"\n[结果] 种群自持: {100*live:.0f}% | 末代黏附a={a_final:.2f} 最大簇={maxc:.0f}"
          f"（{'✓ 聚合体涌现——黏附被选择' if a_final > 0.5 or maxc > 3 else '单细胞稳定——聚合无优势'}）")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(h[:, 0], label="cells")
    axes[0].set_title("Population")
    axes[0].legend(fontsize=8)
    axes[1].plot(h[:, 2], label="adhesion a", color='orange')
    axes[1].plot(h[:, 3], label="max cluster", color='red')
    axes[1].set_title("Multicellularity (adhesion & cluster)")
    axes[1].legend(fontsize=8)
    axes[2].plot(h[:, 5], label="kills", color='green')
    axes[2].set_title("Predation")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("fig_stage31.png", dpi=110)
    print("\n[plot] saved fig_stage31.png")
    print("[done] stage31 multicellular")

if __name__ == "__main__":
    run()
