# -*- coding: utf-8 -*-
"""
M5 阶段 35：信息处理层深化（阶段 E 深化——协调 = 群体迁移能力——生态位驱动）

诊断（stage33b 协调不稳定的根因）：协调无生态学收益——选择压力弱——seed 波动。
深化（研究驱动——Volvox 鞭毛同步 + DVM 垂直迁移）：
  ① 资源按群体中心深度（全体成员共享同深度资源）——迁移 = 群体级行为——
     浅层光合（cy<45）/ 深层营养（cy>55）——过渡带两边弱
  ② 成员移动平均 = 群体中心位移——协调度高 → 中心迁移快（一致才快）——
     协调 = 群体级资源采集效率——直接生态收益 → 进化选择压力
  ③ 克隆群体的协调遗传基础：同 gid 成员基因组相似（变异小）→ 行为相似——
     信号共享（群体光方向广播——N_IN 13）+ 能量输入 → 网络学"低能下潜/高能上浮"
  ④ 基于 stage34 分化世界（营养×s + gid 共享 + glsA + 生命周期——5/5 稳健）
预期：协调涌现（>0.6）+ 迁移分相（低能深层/高能浅层）+ 分化保持
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
N_IN = 15
N_OUT = 4   # 移动 sin/cos + τ + c（细胞命运——发育可塑性——网络决定）
P_ADD_CONN = 0.10
P_ADD_NODE = 0.05
P_DISABLE = 0.05
NUT_DEEP = 0.25    # 0.4 太富 → 种群爆炸（seed203 上千细胞——O(n²) 拖垮）→ 0.25（与 stage34/35v 一致）
NUT_S_CLIP = (0.2, 1.6)
T_DIURNAL = 200     # 昼夜周期（步）
T_DAY = 100         # 白天（前一半——浅层光合可用）

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
    fate = 1.0 / (1.0 + np.exp(-out[3]))
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

class CoordWorld:
    def __init__(self, workdir="stage35_evo", n0=60, seed=109):
        self.workdir = workdir
        os.makedirs(workdir, exist_ok=True)
        for f in os.listdir(workdir):
            os.remove(os.path.join(workdir, f))
        global RNG, LIGHTS
        RNG = np.random.default_rng(seed)
        LIGHTS = [(25, 22), (75, 22), (50, 18), (15, 32)]
        self.bios = {}       # id -> [x, y, e, tau, d, s, a(黏附), c(分化), gid]
        self.genomes = {}
        self.next_id = 0
        self.next_gid = 0
        self.prev_center = None
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
        child_e = (pe / 2.0) * (0.5 + ns)
        off = RNG.uniform(-6, 6) * (1.0 - pa)
        if pc > 0.7:
            off = RNG.uniform(-12, 12)
            child_gid = self.next_gid
            self.next_gid += 1
            self.bios[parent_id][2] = pe * 0.1
            for k in range(2):
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
            child_gid = pgid
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

    def group_centers(self):
        groups = {}
        for b, bio in self.bios.items():
            groups.setdefault(bio[8], []).append(b)
        centers = {}
        for gid, members in groups.items():
            centers[gid] = (np.mean([self.bios[m][0] for m in members]),
                            np.mean([self.bios[m][1] for m in members]))
        return centers

    def step(self):
        if len(self.bios) == 0:
            self.history.append((0, 0, 0, 0, 0, 0, 0, 0, 0))
            return
        # 昼夜相位（真实 DVM 驱动——显式时间信号——前馈网络无记忆）
        self.phase_t = getattr(self, "phase_t", 0)
        phase = 2 * np.pi * (self.phase_t % T_DIURNAL) / T_DIURNAL
        ph_sin, ph_cos = np.sin(phase), np.cos(phase)
        night = (self.phase_t % T_DIURNAL) >= T_DAY
        self.phase_t += 1
        centers = self.group_centers()
        # 每 gid 群体信号：中心到最近光斑的方向（广播——全体成员同一信号）
        group_sig = {}
        for gid, (cx, cy) in centers.items():
            ang, _d = nearest_light(cx, cy)
            group_sig[gid] = (np.sin(ang), np.cos(ang))
        moves = {}
        for bid in list(self.bios.keys()):
            x, y, e, _t, d, s, a, c, gid = self.bios[bid]
            cx, cy = centers[gid]
            gx, gy = group_sig[gid]
            ang, d_light = nearest_light(x, y)
            intensity = light_intensity(x, y)
            # 资源按群体中心深度（深化①——迁移 = 群体级行为——全体同深度）
            if cy > 50:
                nutrient = NUT_DEEP * np.clip(s, *NUT_S_CLIP)
                intensity *= 0.3
            else:
                nutrient = 0.0
                intensity *= 1.5
            if night:                    # 夜间：浅层光合不可用（DVM 驱动力——真实）
                intensity *= 0.1
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
                            e / 150.0,            # 能量（低能下潜/高能上浮的调制信号）
                            gx, gy,               # 群体光方向（信号共享广播）
                            ph_sin, ph_cos,       # 昼夜相位（新——DVM 时间信号——N_IN 15）
                            1.0])
            dx_out, dy_out, tau, fate = forward(self.genomes[bid], inp)
            if s > 1.05:
                fate = np.clip(fate * 0.3 + 0.8, 0.0, 1.0)
            elif s < 0.7:
                fate = np.clip(fate * 0.3, 0.0, 1.0)
            diff_photo = (0.1 + 0.9 * (1.0 - fate)) * 1.8
            absorb = intensity * ABS_RATE * photo_eff(s) * (1.0 - tau) ** PHI \
                     * diff_photo / (1.0 + crowd * 0.08) + nutrient
            e += absorb - BASE_COST - MOVE_COST * 0.1 - 0.1 * d
            norm = max(np.hypot(dx_out, dy_out), 1e-6)
            spd = SPEED * np.clip(norm, 0.1, 1.5)
            nx = np.clip(x + dy_out / norm * spd, 0, W)
            ny = np.clip(y + dx_out / norm * spd, 0, H)
            e -= MOVE_COST * spd
            # 群体向心力（a>0.3——成员跟随质心——群体紧凑——中心 = 平均）
            if a > 0.3:
                g_members = [m for m in self.bios if m != bid and self.bios[m][8] == gid]
                if g_members:
                    mcx = np.mean([self.bios[m][0] for m in g_members])
                    mcy = np.mean([self.bios[m][1] for m in g_members])
                    if np.isfinite(mcx):
                        nx = np.clip(x + (mcx - x) * 0.3, 0, W)
                        ny = np.clip(y + (mcy - y) * 0.3, 0, H)
            self.bios[bid] = [nx, ny, e, tau, d, s, a, fate, gid]
            moves[bid] = (nx - x, ny - y)
        # 协调度（深化②）：成员移动方向 vs 群体中心位移方向的平均 cos
        new_centers = self.group_centers()
        coord = 0.0
        n_grp = 0
        for gid, (ncx, ncy) in new_centers.items():
            members = [b for b, bio in self.bios.items() if bio[8] == gid]
            if len(members) < 2:
                continue
            pcx, pcy = centers[gid]
            cdx, cdy = ncx - pcx, ncy - pcy
            cn = np.hypot(cdx, cdy)
            if cn < 1e-9:
                continue
            cdx, cdy = cdx / cn, cdy / cn
            g_coord = np.mean([(moves[b][0] * cdx + moves[b][1] * cdy) /
                               max(np.hypot(*moves[b]), 1e-9) for b in members if b in moves])
            coord += max(g_coord, 0.0)
            n_grp += 1
        coord = coord / max(n_grp, 1)
        # 遗传群体共享（stage34——胞质桥）
        groups = {}
        for b, bio in self.bios.items():
            groups.setdefault(bio[8], []).append(b)
        for members in groups.values():
            if len(members) > 1:
                pool = sum(self.bios[m][2] for m in members) / len(members)
                for m in members:
                    self.bios[m][2] = pool
        # 死亡/繁殖
        for bid in list(self.bios.keys()):
            if self.bios[bid][2] <= 0:
                self._kill(bid)
            else:
                c = self.bios[bid][7]
                thr = REPRO_E * (1.6 - 1.3 * c)
                if self.bios[bid][2] > thr:
                    self._reproduce(bid)
        # 统计：细胞/平均c/生殖/体细胞/群体数/最大群体/协调度/平均中心y
        if len(self.bios):
            cs_ = [b[7] for b in self.bios.values()]
            germ = np.sum(np.array(cs_) > 0.7)
            soma = np.sum(np.array(cs_) < 0.3)
            gsizes = {}
            for b in self.bios.values():
                gsizes[b[8]] = gsizes.get(b[8], 0) + 1
            cys = [self.group_centers().get(g, (0, 0))[1] for g in gsizes]
            avg_cy = np.mean(cys) if cys else 50.0
            self.history.append((len(self.bios), np.mean(cs_), germ, soma,
                                 len(gsizes), max(gsizes.values()) if gsizes else 0,
                                 coord, avg_cy))
        else:
            self.history.append((0, 0, 0, 0, 0, 0, 0, 50.0))

    def run(self, T):
        for _ in range(T):
            self.step()
        return np.array(self.history)


def run_seed(seed, T=2000):
    w = CoordWorld(workdir=f"stage35_evo_{seed}", n0=60, seed=seed)
    h = w.run(T)
    live = np.mean(h[500:, 0] > 5)
    germ, soma = h[-1, 2], h[-1, 3]
    coord = np.mean(h[500:, 6])
    return h, live, germ, soma, coord


def run():
    print("=== M5 阶段 35：信息处理层深化（阶段 E 深化——协调 = 群体迁移能力） ===\n")
    print("机制：资源按群体中心深度（迁移=群体级）+ 成员平均移动=中心位移\n"
          "      + 信号共享（群体光方向广播）+ 能量调制（低能下潜/高能上浮）\n"
          "      + 克隆群体协调遗传基础（同 gid 基因组相似）+ stage34 分化世界\n")
    results = []
    for seed in [109, 203, 307, 411, 503]:
        h, live, germ, soma, coord = run_seed(seed)
        diff = "✓ 分化" if germ > 2 and soma > 2 else "✗ 全能主导"
        print(f"seed {seed}: 自持={100*live:.0f}% 协调度={coord:.2f} 生殖={germ:.0f} "
              f"体细胞={soma:.0f} {diff}")
        results.append((seed, live, coord, germ, soma, diff))
        if seed == 109:
            for t in range(0, 2000, 250):
                print(f"  t={t:4d} 细胞={h[t,0]:3.0f} 生殖={h[t,2]:3.0f} 体细胞={h[t,3]:3.0f} "
                      f"协调={h[t,6]:.2f} 中心y={h[t,7]:.0f}")
    n_diff = sum(1 for r in results if "✓" in r[5])
    n_coord = sum(1 for r in results if r[2] > 0.5)
    print(f"\n[结果] 分化 {n_diff}/5 seed | 协调>0.5 {n_coord}/5 seed——" +
          ("协调稳定涌现（vs stage33b seed 波动）" if n_coord >= 3 else "协调仍需加强——检查迁移收益/信号共享"))
    h, live, germ, soma, coord = run_seed(109)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(h[:, 0], label="cells")
    axes[0].set_title("Population (seed109)")
    axes[0].legend(fontsize=8)
    axes[1].plot(h[:, 6], label="coordination", color='purple')
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Coordination")
    axes[1].legend(fontsize=8)
    axes[2].plot(h[:, 7], label="group center y", color='orange')
    axes[2].axhline(50, color='gray', linestyle='--', lw=0.8)
    axes[2].set_title("Vertical migration")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("fig_stage35.png", dpi=110)
    print("\n[plot] saved fig_stage35.png")
    print("[done] stage35 coordination deep")


if __name__ == "__main__":
    run()
