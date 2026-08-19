# -*- coding: utf-8 -*-
"""
M5 阶段 31b：多细胞聚合机制扫描（用户："应该不止吞噬压力——都试试"）

6 个聚合体驱动机制独立实验（+基线对照）——看哪个让聚合体涌现：
  defense ① 捕食防御（体积大难吞——已试——作为对照）
  stress  ② 抗环境压力（生物膜 EPS——周期性干燥波——单体损失 10 倍于聚合体）
  share   ③ 营养共享（能量池——内层被供养）
  stratify④ 资源分层（浅层光/深层营养——聚合体跨层双收益）
  migrate ⑤ 运动迁移（光斑周期性迁移——聚合体集体跟随）
  repro   ⑥ 繁殖效率（聚合体繁殖阈值低——批量产出）
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
N_OUT = 3
P_ADD_CONN = 0.10
P_ADD_NODE = 0.05
P_DISABLE = 0.05
STRESS_PERIOD = 200   # 干燥波周期
STRESS_DUR = 50       # 干燥波持续

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
    p_wall = 1.0 / (1.0 + np.exp(6.0 * (def_i - def_j)))
    p_size = s_j / max(s_i, 1e-6)
    p_clust = 1.0 / (1.0 + 0.5 * max(0, cluster_j - 1))
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
                w = -0.5 if i == N_IN - 1 else 0.0   # 温和 τ（-0.3 猎食过度——世界崩溃——机制无法涌现）
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

class MechWorld:
    def __init__(self, mech="baseline", workdir="stage31b_evo", n0=60, seed=109):
        self.mech = mech
        self.workdir = workdir
        os.makedirs(workdir, exist_ok=True)
        for f in os.listdir(workdir):
            os.remove(os.path.join(workdir, f))
        global RNG
        RNG = np.random.default_rng(seed)
        global LIGHTS
        LIGHTS = [(25, 25), (75, 75), (50, 50), (15, 75)]
        self.bios = {}
        self.genomes = {}
        self.next_id = 0
        for _ in range(n0):
            self._spawn_new()
        self.history = []
        self.kills = 0
        self.step_n = 0
        self.light_history = []

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
        self.step_n += 1
        if len(self.bios) == 0:
            self.history.append((0, 0, 0, 0, 0, 0, 0))
            return
        # 机制④：光斑迁移（migrate——每 300 步光斑重定位）
        if self.mech == "migrate" and self.step_n % 300 == 1:
            global LIGHTS
            LIGHTS = [(RNG.uniform(10, 90), RNG.uniform(10, 90)) for _ in range(4)]
        clust = self.clusters()
        # 干燥波（stress——周期性）
        in_stress = self.mech == "stress" and (self.step_n % STRESS_PERIOD) < STRESS_DUR
        for bid in list(self.bios.keys()):
            x, y, e, _t, d, s, a = self.bios[bid]
            ang, d_light = nearest_light(x, y)
            intensity = light_intensity(x, y)
            # 机制④资源分层：深层（y>50）营养收益（光弱但营养浓）
            nutrient = 0.0
            if self.mech == "stratify":
                if y > 50:
                    nutrient = 0.15
                    intensity *= 0.3
                else:
                    intensity *= 1.5
            crowd = 0
            threat_ang, threat_d = 0.0, 50.0
            nb_ang_sum = np.zeros(2)
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
            cs = clust.get(bid, 1)
            shade = 1.0 / (1.0 + 0.15 * (cs - 1))
            absorb = intensity * ABS_RATE * photo_eff(s) * (1.0 - tau) ** PHI \
                     * shade / (1.0 + crowd * 0.08) + nutrient
            e += absorb - BASE_COST - MOVE_COST * 0.1 - 0.1 * d
            # 干燥波损失（stress）：单体 ×0.1 能量流失——聚合体 EPS 保护 ×0.5
            if in_stress:
                if cs > 1:
                    e -= 0.5 * (BASE_COST + 0.1)
                else:
                    e -= 1.0 * (BASE_COST + 0.1)
            norm = max(np.hypot(dx_out, dy_out), 1e-6)
            spd = SPEED * np.clip(norm, 0.1, 1.5)
            nx = np.clip(x + dy_out / norm * spd, 0, W)
            ny = np.clip(y + dx_out / norm * spd, 0, H)
            e -= MOVE_COST * spd
            # 物理连接（黏附约束）
            if a > 0.3 and cs > 1:
                cx = np.mean([self.bios[m][0] for m in self.bios if m != bid and
                              np.hypot(self.bios[m][0]-x, self.bios[m][1]-y) < CONTACT*2])
                cy = np.mean([self.bios[m][1] for m in self.bios if m != bid and
                              np.hypot(self.bios[m][0]-x, self.bios[m][1]-y) < CONTACT*2])
                if np.isfinite(cx):
                    nx = np.clip(x + (cx - x) * 0.3, 0, W)
                    ny = np.clip(y + (cy - y) * 0.3, 0, H)
            self.bios[bid] = [nx, ny, e, tau, d, s, a]
        # 吞噬
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
        # 簇内能量共享（share 机制——或防御机制下也有——共享=聚合体基础）
        if self.mech in ("share", "defense", "stress", "stratify", "migrate"):
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
                # 机制⑥繁殖效率：聚合体成员批量繁殖（簇大——多生一个）
                if self.mech == "repro" and clust.get(bid, 1) > 2 and self.bios[bid][2] > 60:
                    self._reproduce(bid)
        if len(self.bios):
            as_ = [b[6] for b in self.bios.values()]
            csizes = list(clust.values())
            self.history.append((len(self.bios), np.mean(as_), max(csizes), np.mean(csizes),
                                 self.kills, np.mean([b[2] for b in self.bios.values()]), 0))
        else:
            self.history.append((0, 0, 0, 0, self.kills, 0, 0))

    def run(self, T):
        for _ in range(T):
            self.step()
        return np.array(self.history)

def run():
    print("=== M5 阶段 31b：多细胞聚合机制扫描（6 机制 + 基线） ===\n")
    mechs = [("baseline", "无机制对照"), ("defense", "①捕食防御"), ("stress", "②抗环境压力"),
             ("share", "③营养共享"), ("stratify", "④资源分层"), ("migrate", "⑤运动迁移"),
             ("repro", "⑥繁殖效率")]
    results = {}
    for mech, name in mechs:
        w = MechWorld(mech=mech, workdir="stage31b_evo", n0=60)
        h = w.run(1800)
        live = np.mean(h[500:, 0] > 5)
        a_final = h[-1, 1]
        maxc = int(h[500:, 2].max())
        results[mech] = (live, a_final, maxc)
        tag = "★聚合涌现" if (a_final > 0.45 or maxc >= 5) else ""
        print(f"{name:12s} 自持={100*live:4.0f}% 黏附a={a_final:.2f} 最大簇={maxc:2d} {tag}")
    print("\n[结果] 聚合体涌现机制：", [m for m, (l, a, c) in results.items() if a > 0.45 or c >= 5] or "无")

if __name__ == "__main__":
    run()
