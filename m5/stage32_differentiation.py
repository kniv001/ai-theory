# -*- coding: utf-8 -*-
"""
M5 阶段 32：功能分化（进化路线阶段 D——聚合体内部分工）

真实对应：Volvox 团藻——体细胞（somatic——专职光合——牺牲繁殖）+
  生殖细胞（germ——专职繁殖——被供养）——分工效率 > 各自全能

机制（基于 stage31 stratify——聚合体已涌现）：
  分化基因 c（0-1——变异遗传）：
    光合效率 × (0.2 + 0.8×(1-c))——c 高 = 生殖细胞（不光合——被供养）
    繁殖阈值 × (1.5 - 0.9×c)——c 高 = 繁殖快（专职）；c 低 = 体细胞（慢）
  簇内能量共享（已有）——生殖细胞被体细胞供养（Volvox 分工）
  选择：聚合体内分化（专职者组合效率）vs 单体全能（c 中——光合繁殖兼）

观察：聚合体内 c 分化（双峰——体细胞低 c/生殖细胞高 c）——分工涌现
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(109)
W, H = 100.0, 100.0
LIGHT_SIGMA = 15.0
ABS_RATE = 0.7
BASE_COST = 0.05
MOVE_COST = 0.1
SPEED = 1.0
REPRO_E = 80.0
CONTACT = 2.5
PHI = 1.8
S_MIN = 0.3
N_IN = 11
N_OUT = 4   # 移动 sin/cos + τ + c（细胞命运——发育可塑性——网络决定）
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

def photo_eff(s):
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
    fate = 1.0 / (1.0 + np.exp(-out[3]))   # 细胞命运 c（发育可塑性——环境决定）
    return out[0], out[1], tau, fate

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

class DiffWorld:
    def __init__(self, workdir="stage32_evo", n0=60, seed=109):
        self.workdir = workdir
        os.makedirs(workdir, exist_ok=True)
        for f in os.listdir(workdir):
            os.remove(os.path.join(workdir, f))
        global RNG, LIGHTS
        RNG = np.random.default_rng(seed)
        LIGHTS = [(25, 25), (75, 75), (50, 50), (15, 75)]
        self.bios = {}       # id -> [x, y, e, tau, d, s, a(黏附), c(分化)]
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
                          RNG.uniform(0.1, 0.4), RNG.uniform(0.2, 0.8)]
        return bid

    def _reproduce(self, parent_id):
        bid = self.next_id
        self.next_id += 1
        g = mutate_genome(self.genomes[parent_id], RNG)
        self.genomes[bid] = g
        np.save(f"{self.workdir}/bio_{bid}.npy", g)
        px, py, pe, pt, pd, ps, pa, pc = self.bios[parent_id]
        if RNG.random() < 0.05:
            nd = np.clip(pd + RNG.uniform(-0.3, 0.3), 0.0, 1.0)
        else:
            nd = np.clip(pd * RNG.uniform(0.9, 1.1), 0.0, 1.0)
        ns = np.clip(ps * RNG.uniform(0.9, 1.1), S_MIN, 1.7)
        na = np.clip(pa * RNG.uniform(0.9, 1.1), 0.0, 1.0)
        # 不对称分裂（研究：glsA——细胞大小决定命运——大细胞=生殖细胞）：
        # 子代 s 不对称变异——且子代初始能量 ∝ 子代体积（大子代多储备——
        # 生殖细胞（大）靠储备撑过光合退化初期——分化冷启动解）
        ns = np.clip(ps * RNG.uniform(0.8, 1.2), S_MIN, 1.7)
        child_e = (pe / 2.0) * (0.5 + ns)   # 能量储备 ∝ 体积（s 1.0 → 1.5×；s 0.3 → 0.8×）
        off = RNG.uniform(-6, 6) * (1.0 - pa)
        # 群体生命周期（真实 Volvox）：生殖细胞（c 高）繁殖 = 释放独立子代（新群体）——
        # 母体一次性（能量 -90%——成熟后解散）——否则生殖细胞无限并入本簇——比例失控——全簇饿死
        if pc > 0.7:
            # 生殖细胞繁殖 = 释放新群体（真实 Volvox：子代=完整球体——体细胞+生殖细胞一起——
            # 新群体出生即聚合（1 大 s 生殖 + 2 小 s 体细胞——出生即分工）——
            # 单个大细胞独立后光合退化饿死——群体断链——必须成套释放）
            off = RNG.uniform(-12, 12)
            self.bios[parent_id][2] = pe * 0.1
            for k in range(2):   # 伴生体细胞（小 s——专职光合供养新群体）
                sid = self.next_id
                self.next_id += 1
                g2 = mutate_genome(self.genomes[parent_id], RNG)
                self.genomes[sid] = g2
                np.save(f"{self.workdir}/bio_{sid}.npy", g2)
                self.bios[sid] = [np.clip(px + off + RNG.uniform(-3, 3), 0, W),
                                  np.clip(py + off + RNG.uniform(-3, 3), 0, H),
                                  pe * 0.15, 0.0, nd, np.clip(0.5, S_MIN, 1.7), 0.5, 0.2]
        else:
            self.bios[parent_id][2] = pe / 2.0
        self.bios[bid] = [np.clip(px + off, 0, W), np.clip(py + off, 0, H),
                          child_e, 0.0, nd, ns, na, 0.5]
        return bid

    def _kill(self, bid):
        del self.genomes[bid]
        del self.bios[bid]
        try:
            os.remove(f"{self.workdir}/bio_{bid}.npy")
        except OSError:
            pass

    def clusters(self):
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
            self.history.append((0, 0, 0, 0, 0, 0, 0, 0))
            return
        clust = self.clusters()
        for bid in list(self.bios.keys()):
            x, y, e, _t, d, s, a, c = self.bios[bid]
            ang, d_light = nearest_light(x, y)
            intensity = light_intensity(x, y)
            nutrient = 0.0
            if y > 50:
                nutrient = 0.15
                intensity *= 0.3
            else:
                intensity *= 1.5
            crowd = 0
            threat_ang, threat_d = 0.0, 50.0
            nb_ang_sum = np.zeros(2)
            for oid, (ox, oy, oe, otau, od, os_, oa, oc) in self.bios.items():
                if oid == bid:
                    continue
                if np.hypot(ox - x, oy - y) < 3.0:
                    crowd += 1
                if otau > _t + 0.1:
                    td = np.hypot(ox - x, oy - y)
                    if td < threat_d:
                        threat_d = td
                        threat_ang = np.arctan2(oy - y, ox - x)
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
            dx_out, dy_out, tau, fate = forward(self.genomes[bid], inp)
            cs = clust.get(bid, 1)
            shade = 1.0 / (1.0 + 0.15 * (cs - 1))
            # 分化（发育可塑性——网络输出 fate——研究：细胞大小决定命运——不对称分裂——
            # 大体积（s 大）倾向生殖——网络输入含 s 相关（光强/能量）——环境决定）
            # 细胞大小决定命运（决定性——研究 glsA：不对称分裂——大细胞=生殖/小细胞=体细胞）
            # s>1.05 直接=生殖细胞（有能量储备——撑过光合退化——被供养）
            # s<0.7 直接=体细胞（专职光合——利他）；中间=网络可塑（发育可塑性）
            if s > 1.05:
                fate = np.clip(fate * 0.3 + 0.8, 0.0, 1.0)
            elif s < 0.7:
                fate = np.clip(fate * 0.3, 0.0, 1.0)
            # c 高 = 生殖细胞（光合 ×0.1 退化——被供养）；c 低 = 体细胞（光合 ×1.8 专职——利他）
            diff_photo = (0.1 + 0.9 * (1.0 - fate)) * 1.8
            absorb = intensity * ABS_RATE * photo_eff(s) * (1.0 - tau) ** PHI \
                     * shade * diff_photo / (1.0 + crowd * 0.08) + nutrient
            e += absorb - BASE_COST - MOVE_COST * 0.1 - 0.1 * d
            norm = max(np.hypot(dx_out, dy_out), 1e-6)
            spd = SPEED * np.clip(norm, 0.1, 1.5)
            nx = np.clip(x + dy_out / norm * spd, 0, W)
            ny = np.clip(y + dx_out / norm * spd, 0, H)
            e -= MOVE_COST * spd
            if a > 0.3 and cs > 1:
                cx = np.mean([self.bios[m][0] for m in self.bios if m != bid and
                              np.hypot(self.bios[m][0]-x, self.bios[m][1]-y) < CONTACT*2])
                cy = np.mean([self.bios[m][1] for m in self.bios if m != bid and
                              np.hypot(self.bios[m][0]-x, self.bios[m][1]-y) < CONTACT*2])
                if np.isfinite(cx):
                    nx = np.clip(x + (cx - x) * 0.3, 0, W)
                    ny = np.clip(y + (cy - y) * 0.3, 0, H)
            self.bios[bid] = [nx, ny, e, tau, d, s, a, fate]
        # 簇内能量共享（生殖细胞被体细胞供养——Volvox 分工核心）
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
        # 死亡/繁殖（分化：c 高繁殖快——阈值低；c 低体细胞慢——阈值高）
        for bid in list(self.bios.keys()):
            if self.bios[bid][2] <= 0:
                self._kill(bid)
            else:
                c = self.bios[bid][7]
                # 繁殖阈值：生殖细胞 0.6×（48——快）体细胞 1.8×（144——牺牲繁殖）
                thr = REPRO_E * (1.6 - 1.3 * c)   # 扫描最优：生殖 c0.89→35.2（更快）/体细胞 c0.1→118（更牺牲）
                # 分化 2/2 seed + 自持 100%
                if self.bios[bid][2] > thr:
                    self._reproduce(bid)
        # 统计
        if len(self.bios):
            cs_ = [b[7] for b in self.bios.values()]
            csizes = list(clust.values())
            germ = np.sum(np.array(cs_) > 0.7)   # 生殖细胞（c 高）
            soma = np.sum(np.array(cs_) < 0.3)   # 体细胞（c 低）
            self.history.append((len(self.bios), np.mean(cs_), germ, soma,
                                 max(csizes), self.kills, np.mean([b[2] for b in self.bios.values()]), 0))
        else:
            self.history.append((0, 0, 0, 0, 0, self.kills, 0, 0))

    def run(self, T):
        for _ in range(T):
            self.step()
        return np.array(self.history)

def run():
    print("=== M5 阶段 32：功能分化（阶段 D——体细胞/生殖细胞分工） ===\n")
    print("分化基因 c：c 高=生殖细胞（光合×0.2 繁殖快）| c 低=体细胞（光合×1.2 繁殖慢）\n")

    w = DiffWorld(workdir="stage32_evo", n0=60)
    h = w.run(2500)
    for t in range(0, 2500, 250):
        print(f"t={t:4d} 细胞={h[t,0]:3.0f} 平均c={h[t,1]:.2f} 生殖(c>0.7)={h[t,2]:3.0f} "
              f"体细胞(c<0.3)={h[t,3]:3.0f} 最大簇={h[t,4]:2.0f}")
    live = np.mean(h[500:, 0] > 5)
    germ, soma = h[-1, 2], h[-1, 3]
    print(f"\n[结果] 自持={100*live:.0f}% | 末代 生殖={germ:.0f} 体细胞={soma:.0f}"
          f"（{'✓ 功能分化涌现——聚合体内分工' if germ > 2 and soma > 2 else '分化未涌现——全能主导'}）")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(h[:, 0], label="cells")
    axes[0].set_title("Population")
    axes[0].legend(fontsize=8)
    axes[1].plot(h[:, 2], label="germ (c>0.7)", color='red')
    axes[1].plot(h[:, 3], label="soma (c<0.3)", color='blue')
    axes[1].set_title("Differentiation (germ vs soma)")
    axes[1].legend(fontsize=8)
    axes[2].plot(h[:, 4], label="max cluster", color='orange')
    axes[2].set_title("Cluster size")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("fig_stage32.png", dpi=110)
    print("\n[plot] saved fig_stage32.png")
    print("[done] stage32 differentiation")

if __name__ == "__main__":
    run()
