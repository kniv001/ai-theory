# -*- coding: utf-8 -*-
"""
M5 阶段 34：功能分化深化（阶段 D 深化——生态位驱动的稳健分化）

诊断（stage32 脆弱根因）：s（体积）绑定命运（glsA），但 s 没有生态学收益——
  s 漂移是偶然的 → 分化 = 参数临界幸运（1/5 seed）而非生态涌现。
深化（研究驱动）：
  ① 营养吸收 ∝ s（大细胞吸收强——真实：大细胞储存容量/吸收表面积大）——
     深层营养层收益 ∝ 体积 → 大 s 深层驻留可行、小 s 必须浅层光合 →
     体积-生态位耦合 → s 演化有方向 → glsA 绑定自然触发
  ② 遗传群体共享（真实 Volvox：群体 = 克隆后代——细胞质桥全身共享——
     共享不依赖空间接触）——gid 继承母体——共享池按 gid——
     生殖细胞深层吸营养也能被体细胞供养（跨层无需空间相邻）
  ③ glsA 绑定保留（s>1.05 生殖 / s<0.7 体细胞 / 中间网络可塑）
  ④ 生命周期保留（生殖释放新群体——新 gid——1 大 + 2 伴生体细胞成套释放）
预期：分化从"参数幸运"变"生态位驱动涌现"——多 seed 分化率显著提升
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
NUT_DEEP = 0.25      # 深层营养基础（y>50）——吸收量 × clip(s, 0.2, 1.6)
NUT_S_CLIP = (0.2, 1.6)
ABL_MODE = os.environ.get("STAGE34_ABL", "both")   # both/size_only/gid_only（消融）

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
    def __init__(self, workdir="stage34_evo", n0=60, seed=109):
        self.workdir = workdir
        os.makedirs(workdir, exist_ok=True)
        for f in os.listdir(workdir):
            os.remove(os.path.join(workdir, f))
        global RNG, LIGHTS
        RNG = np.random.default_rng(seed)
        # 光斑偏浅层（光合区）
        LIGHTS = [(25, 22), (75, 22), (50, 18), (15, 32)]
        self.bios = {}       # id -> [x, y, e, tau, d, s, a(黏附), c(分化), gid(遗传群体)]
        self.genomes = {}
        self.next_id = 0
        self.next_gid = 0
        for _ in range(n0):
            self._spawn_new()
        self.history = []
        self.kills = 0

    def _spawn_new(self):
        bid = self.next_id
        self.next_id += 1
        g = init_genome(RNG)
        self.genomes[bid] = g
        gid = self.next_gid
        self.next_gid += 1
        np.save(f"{self.workdir}/bio_{bid}.npy", g)
        self.bios[bid] = [RNG.uniform(0, W), RNG.uniform(0, H), RNG.uniform(30, 60),
                          0.0, RNG.uniform(0.1, 0.4), RNG.uniform(0.5, 1.2),
                          RNG.uniform(0.1, 0.4), RNG.uniform(0.2, 0.8), gid]
        return bid

    def _reproduce(self, parent_id):
        bid = self.next_id
        self.next_id += 1
        g = mutate_genome(self.genomes[parent_id], RNG)
        self.genomes[bid] = g
        np.save(f"{self.workdir}/bio_{bid}.npy", g)
        px, py, pe, pt, pd, ps, pa, pc, pgid = self.bios[parent_id]
        nd = np.clip(pd * RNG.uniform(0.9, 1.1), 0.0, 1.0)
        ns = np.clip(ps * RNG.uniform(0.8, 1.2), S_MIN, 1.7)
        na = np.clip(pa * RNG.uniform(0.9, 1.1), 0.0, 1.0)
        # 不对称分裂（glsA）：子代能量储备 ∝ 体积（大子代多储备）
        child_e = (pe / 2.0) * (0.5 + ns)
        off = RNG.uniform(-6, 6) * (1.0 - pa)
        if pc > 0.7:
            # 生殖细胞繁殖 = 释放新群体（真实 Volvox：子代 = 完整球体——
            # 新 gid——1 大 s 生殖 + 2 伴生体细胞——出生即分工——群体断链修复）
            off = RNG.uniform(-12, 12)
            child_gid = self.next_gid
            self.next_gid += 1
            self.bios[parent_id][2] = pe * 0.1   # 母体解散（-90%）
            for k in range(2):   # 伴生体细胞（小 s——专职光合供养新群体）
                sid = self.next_id
                self.next_id += 1
                g2 = mutate_genome(self.genomes[parent_id], RNG)
                self.genomes[sid] = g2
                np.save(f"{self.workdir}/bio_{sid}.npy", g2)
                self.bios[sid] = [np.clip(px + off + RNG.uniform(-3, 3), 0, W),
                                  np.clip(py + off + RNG.uniform(-3, 3), 0, H),
                                  pe * 0.15, 0.0, nd, RNG.uniform(0.5, 0.65),
                                  RNG.uniform(0.4, 0.6), 0.2, child_gid]
        else:
            child_gid = pgid       # 普通分裂 = 克隆——同群体（胞质桥共享）
            self.bios[parent_id][2] = pe / 2.0
        self.bios[bid] = [np.clip(px + off, 0, W), np.clip(py + off, 0, H),
                          child_e, 0.0, nd, ns, na, 0.5, child_gid]
        return bid

    def _kill(self, bid):
        del self.genomes[bid]
        del self.bios[bid]
        try:
            os.remove(f"{self.workdir}/bio_{bid}.npy")
        except OSError:
            pass

    def clusters(self):
        # 空间簇（接触闭包——消融对照：size_only 时共享按空间簇而非 gid）
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
        for bid in list(self.bios.keys()):
            x, y, e, _t, d, s, a, c, gid = self.bios[bid]
            ang, d_light = nearest_light(x, y)
            intensity = light_intensity(x, y)
            # 垂直分层（光浅/营养深）——营养吸收 ∝ 体积（深化①）
            if y > 50:
                if ABL_MODE == "gid_only":
                    nutrient = 0.15          # 消融：无体积收益
                else:
                    nutrient = NUT_DEEP * np.clip(s, *NUT_S_CLIP)   # 大 s 吸收强——深层驻留可行
                intensity *= 0.3
            else:
                nutrient = 0.0
                intensity *= 1.5
            crowd = 0
            threat_ang, threat_d = 0.0, 50.0
            nb_ang_sum = np.zeros(2)
            for oid, (ox, oy, oe, otau, od, os_, oa, oc, og) in self.bios.items():
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
            # glsA 绑定（体积决定命运——深化③）
            if s > 1.05:
                fate = np.clip(fate * 0.3 + 0.8, 0.0, 1.0)
            elif s < 0.7:
                fate = np.clip(fate * 0.3, 0.0, 1.0)
            # c 高 = 生殖细胞（光合退化——被供养）；c 低 = 体细胞（专职光合 ×1.8——利他）
            diff_photo = (0.1 + 0.9 * (1.0 - fate)) * 1.8
            absorb = intensity * ABS_RATE * photo_eff(s) * (1.0 - tau) ** PHI \
                     * diff_photo / (1.0 + crowd * 0.08) + nutrient
            e += absorb - BASE_COST - MOVE_COST * 0.1 - 0.1 * d
            norm = max(np.hypot(dx_out, dy_out), 1e-6)
            spd = SPEED * np.clip(norm, 0.1, 1.5)
            nx = np.clip(x + dy_out / norm * spd, 0, W)
            ny = np.clip(y + dx_out / norm * spd, 0, H)
            e -= MOVE_COST * spd
            if a > 0.3:
                g_members = [m for m in self.bios if m != bid and self.bios[m][8] == gid]
                if g_members:
                    cx = np.mean([self.bios[m][0] for m in g_members])
                    cy = np.mean([self.bios[m][1] for m in g_members])
                    if np.isfinite(cx):
                        nx = np.clip(x + (cx - x) * 0.3, 0, W)
                        ny = np.clip(y + (cy - y) * 0.3, 0, H)
            self.bios[bid] = [nx, ny, e, tau, d, s, a, fate, gid]
        # 遗传群体共享（深化②——胞质桥等效——共享不依赖空间接触——Volvox 克隆群体）
        if ABL_MODE == "size_only":
            # 消融：共享按空间簇（stage32 原版——跨层断链）
            clust = self.clusters()
            groups = {}
            for b, root in clust.items():
                if b not in self.bios:
                    continue
                groups.setdefault(root, []).append(b)
        else:
            groups = {}
            for b, bio in self.bios.items():
                groups.setdefault(bio[8], []).append(b)
        for members in groups.values():
            if len(members) > 1:
                pool = sum(self.bios[m][2] for m in members) / len(members)
                for m in members:
                    self.bios[m][2] = pool
        # 死亡/繁殖（分化：c 高繁殖快——阈值低；c 低体细胞慢——阈值高）
        for bid in list(self.bios.keys()):
            if self.bios[bid][2] <= 0:
                self._kill(bid)
            else:
                c = self.bios[bid][7]
                thr = REPRO_E * (1.6 - 1.3 * c)   # 扫描最优：生殖快/体细胞牺牲
                if self.bios[bid][2] > thr:
                    self._reproduce(bid)
        # 统计
        if len(self.bios):
            cs_ = [b[7] for b in self.bios.values()]
            ss_ = [b[5] for b in self.bios.values()]
            gids = set(b[8] for b in self.bios.values())
            germ = np.sum(np.array(cs_) > 0.7)   # 生殖细胞（c 高）
            soma = np.sum(np.array(cs_) < 0.3)   # 体细胞（c 低）
            s_big = np.sum(np.array(ss_) > 1.05)
            s_small = np.sum(np.array(ss_) < 0.7)
            # 群体规模分布（共享池成员数）
            gsizes = {}
            for b in self.bios.values():
                gsizes[b[8]] = gsizes.get(b[8], 0) + 1
            avg_gsize = np.mean(list(gsizes.values())) if gsizes else 0
            max_gsize = max(gsizes.values()) if gsizes else 0
            self.history.append((len(self.bios), np.mean(cs_), germ, soma,
                                 len(gids), avg_gsize, max_gsize, s_big - s_small))
        else:
            self.history.append((0, 0, 0, 0, 0, 0, 0, 0))

    def run(self, T):
        for _ in range(T):
            self.step()
        return np.array(self.history)


def run_seed(seed, T=2000):
    w = DiffWorld(workdir=f"stage34_evo_{seed}", n0=60, seed=seed)
    h = w.run(T)
    live = np.mean(h[500:, 0] > 5)
    germ, soma = h[-1, 2], h[-1, 3]
    return h, live, germ, soma


def run_ablation(mode):
    print(f"--- 消融 {mode} ---")
    n = 0
    for seed in [109, 307, 503]:
        w = DiffWorld(workdir=f"stage34_evo_{seed}", n0=60, seed=seed)
        h = w.run(1500)
        live = np.mean(h[500:, 0] > 5)
        germ, soma = h[-1, 2], h[-1, 3]
        diff = germ > 2 and soma > 2
        n += diff
        print(f"  seed {seed}: 自持={100*live:.0f}% 生殖={germ:.0f} 体细胞={soma:.0f} "
              f"{'✓ 分化' if diff else '✗ 全能主导'}")
    print(f"  → 分化 {n}/3 seed\n")


def run():
    if ABL_MODE != "both":
        run_ablation(ABL_MODE)
        return
    print("=== M5 阶段 34：功能分化深化（阶段 D 深化——生态位驱动的稳健分化） ===\n")
    print("机制：营养吸收 ∝ 体积（大 s 深层驻留可行）+ 遗传群体共享（gid——胞质桥）\n"
          "      + glsA 绑定（s>1.05 生殖 / s<0.7 体细胞）+ 生命周期（新 gid 成套释放）\n")
    results = []
    for seed in [109, 203, 307, 411, 503]:
        h, live, germ, soma = run_seed(seed)
        diff = "✓ 分化" if germ > 2 and soma > 2 else "✗ 全能主导"
        results.append((seed, live, germ, soma, diff))
        print(f"seed {seed}: 自持={100*live:.0f}% 末代 生殖={germ:.0f} 体细胞={soma:.0f} {diff}")
        if seed == 109:
            for t in range(0, 2000, 200):
                print(f"  t={t:4d} 细胞={h[t,0]:3.0f} 生殖={h[t,2]:3.0f} 体细胞={h[t,3]:3.0f} "
                      f"群体数={h[t,4]:3.0f} 平均群体规模={h[t,5]:.1f} 最大群体={h[t,6]:3.0f}")
    n_diff = sum(1 for r in results if "✓" in r[4])
    print(f"\n[结果] 分化 {n_diff}/5 seed——" +
          ("生态位驱动生效（vs stage32 的 1/5——参数幸运 → 生态涌现）"
           if n_diff >= 3 else "仍需调整——检查 s 演化方向/营养收益强度"))
    h, live, germ, soma = run_seed(109)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(h[:, 0], label="cells")
    axes[0].set_title("Population (seed109)")
    axes[0].legend(fontsize=8)
    axes[1].plot(h[:, 2], label="germ", color='red')
    axes[1].plot(h[:, 3], label="soma", color='blue')
    axes[1].set_title("Differentiation")
    axes[1].legend(fontsize=8)
    axes[2].plot(h[:, 4], label="groups", color='green')
    axes[2].plot(h[:, 6], label="max group size", color='orange')
    axes[2].set_title("Genetic groups")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("fig_stage34.png", dpi=110)
    print("\n[plot] saved fig_stage34.png")
    print("[done] stage34 differentiation deep")


if __name__ == "__main__":
    run()
