# -*- coding: utf-8 -*-
"""
M5 stage35v：CPU 向量化版（用户：让训练加速——路径 A——先向量化再上 GPU）

与 stage35_coordination_deep.py 机制完全一致——仅把逐细胞循环改为 numpy 批量：
  ① 状态矩阵 (N×9) + 距离矩阵 (N×N)——邻居/接触/威胁一次算完（消除 O(n²) 循环）
  ② 批量前向：基因组 padding 为 (N, MAX_C, 4) 连接矩阵——迭代 max_nodes 轮——
     每轮 gather 批量矩阵乘（np.add.at）——全部细胞同帧算网络
  ③ 群体中心/共享池/协调度——np.bincount 聚合
  ④ 繁殖/死亡逐事件（低频——写 bio_N.npy 文件架构保留）
验证：同 seed 与逐细胞版结果一致 + 测速对比
"""
import os
import time
import numpy as np
from scipy.spatial import cKDTree
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
N_OUT = 4
MAX_C = 120          # 连接表 padding 上限（初始 60 + NEAT 变异生长余量）
MAX_NODE = 40        # 节点 padding 上限（初始 19 + 加节点变异余量）
P_ADD_CONN = 0.10
P_ADD_NODE = 0.05
P_DISABLE = 0.05
NUT_DEEP = 0.25     # 0.4 太富 → 种群爆炸（seed203 上千细胞——O(n²) 拖垮性能）→ 回 0.25（stage34 验证种群可控）
GRP_W = float(os.environ.get("STAGE35_GRPW", "1.0"))   # 群体级资源权重（0.3-1.0 扫描——分化-协调平衡）
GAP = 50            # 世代落盘间隔：文件写/删只在 GAP 倍数步执行（IO ÷50——运行中纯内存）
NIGHT_NUT = 1.5     # 夜间营养倍数（2.0 太富→种群爆炸——1.5 折衷：夜间下潜收益保留但可控）
NUT_S_CLIP = (0.2, 1.6)
T_DIURNAL = 200
T_DAY = 100


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

class CoordWorldV:
    def __init__(self, workdir="stage35v_evo", n0=60, seed=109):
        self.workdir = workdir
        os.makedirs(workdir, exist_ok=True)
        for f in os.listdir(workdir):
            os.remove(os.path.join(workdir, f))
        global RNG, LIGHTS
        RNG = np.random.default_rng(seed)
        LIGHTS = np.array([(25, 22), (75, 22), (50, 18), (15, 32)])
        self.next_id = 0
        self.next_gid = 0
        self.phase_t = 0
        self.arr = None      # (N, 10) [x, y, e, tau, d, s, a, c, gid, bid]
        self.g_mat = None    # (N, MAX_C, 4) 连接表 padding（0 = 空连接）
        self.g_len = None    # (N,) 实际连接数
        self.n_nodes = np.zeros(n0, dtype=int)   # (N,) 节点数
        self.pending = []    # 繁殖缓冲（步末合并数组）
        self.step_count = 0
        for _ in range(n0):
            self._spawn_new()
        self._flush()
        self.history = []
        self.kills = 0

    def savesnapshot(self):
        """单文件快照（用户方案）：所有生物数据存一个 bios.npz——
        删除 = 数组删行 + 批量保存（大文件内该生物字段消失）——
        IO = 每 GAP 步 1 次大文件写（vs N 个独立文件——syscall ÷N）"""
        np.savez(f"{self.workdir}/bios.npz",
                 arr=self.arr, g_mat=self.g_mat, g_len=self.g_len,
                 n_nodes=self.n_nodes, next_id=self.next_id, next_gid=self.next_gid,
                 step_count=self.step_count)

    @staticmethod
    def load(workdir, n0=60, seed=109):
        """从快照恢复世界（评估/复用——单文件读取一次恢复全种群）"""
        d = np.load(f"{workdir}/bios.npz", allow_pickle=False)
        w = CoordWorldV.__new__(CoordWorldV)
        w.workdir = workdir
        w.next_id = int(d["next_id"])
        w.next_gid = int(d["next_gid"])
        w.phase_t = int(d["step_count"])
        w.step_count = int(d["step_count"])
        w.arr = d["arr"]
        w.g_mat = d["g_mat"]
        w.g_len = d["g_len"]
        w.n_nodes = d["n_nodes"]
        w.pending = []
        w.history = []
        w.kills = 0
        return w

    def _flush(self):
        """步末合并缓冲繁殖（一次 vstack/concat——消除 O(N²) 重分配）"""
        if not self.pending:
            return
        rows = np.vstack([p[0] for p in self.pending])
        gs = np.array([self._pad(g) for _, g in self.pending])
        lens = np.array([len(g) for _, g in self.pending])
        nodes = np.array([int(g[:, :2].max()) + 1 for _, g in self.pending])
        if self.arr is None:
            self.arr = rows
            self.g_mat = gs
            self.g_len = lens
            self.n_nodes = nodes
        else:
            self.arr = np.vstack([self.arr, rows])
            self.g_mat = np.concatenate([self.g_mat, gs])
            self.g_len = np.concatenate([self.g_len, lens])
            self.n_nodes = np.concatenate([self.n_nodes, nodes])
        self.pending = []

    def _pad(self, g):
        p = np.zeros((MAX_C, 4))
        p[:len(g)] = g
        return p

    def _spawn_new(self):
        bid = self.next_id
        self.next_id += 1
        g = init_genome(RNG)
        row = np.array([[RNG.uniform(0, W), RNG.uniform(0, H), RNG.uniform(30, 60),
                         0.0, RNG.uniform(0.1, 0.4), RNG.uniform(0.5, 1.2),
                         RNG.uniform(0.1, 0.4), RNG.uniform(0.2, 0.8),
                         float(self.next_gid), float(bid)]])
        self.next_gid += 1
        self._append_cell(row, g)
        return bid

    def _append_cell(self, row, g):
        # 只缓冲——步末统一合并（避免每繁殖一次 np.vstack 的 O(N²) 重分配）
        self.pending.append((row, g))

    def _reproduce(self, idx):
        bid = self.next_id
        self.next_id += 1
        g = mutate_genome(self.g_mat[idx, :self.g_len[idx]], RNG)
        px, py, pe, pt, pd, ps, pa, pc, pgid, _bid = self.arr[idx]
        nd = np.clip(pd * RNG.uniform(0.9, 1.1), 0.0, 1.0)
        ns = np.clip(ps * RNG.uniform(0.8, 1.2), S_MIN, 1.7)
        na = np.clip(pa * RNG.uniform(0.9, 1.1), 0.0, 1.0)
        child_e = (pe / 2.0) * (0.5 + ns)
        off = RNG.uniform(-6, 6) * (1.0 - pa)
        if pc > 0.7:
            off = RNG.uniform(-12, 12)
            child_gid = self.next_gid
            self.next_gid += 1
            self.arr[idx, 2] = pe * 0.1
            for k in range(2):
                sid = self.next_id
                self.next_id += 1
                g2 = mutate_genome(self.g_mat[idx, :self.g_len[idx]], RNG)
                row2 = np.array([[np.clip(px + off + RNG.uniform(-3, 3), 0, W),
                                  np.clip(py + off + RNG.uniform(-3, 3), 0, H),
                                  pe * 0.15, 0.0, nd, RNG.uniform(0.5, 0.65),
                                  RNG.uniform(0.4, 0.6), 0.2, float(child_gid), float(sid)]])
                self._append_cell(row2, g2)
        else:
            child_gid = pgid
            self.arr[idx, 2] = pe / 2.0
        row = np.array([[np.clip(px + off, 0, W), np.clip(py + off, 0, H),
                         child_e, 0.0, nd, ns, na, 0.5, child_gid, float(bid)]])
        self._append_cell(row, g)

    def _kill(self, idx):
        keep = np.ones(len(self.arr), dtype=bool)
        keep[idx] = False
        self.arr = self.arr[keep]
        self.g_mat = self.g_mat[keep]
        self.g_len = self.g_len[keep]
        self.n_nodes = self.n_nodes[keep]

    def group_centers(self):
        gids = self.arr[:, 8].astype(int)
        m = gids.max() + 1
        sx = np.bincount(gids, weights=self.arr[:, 0], minlength=m)
        sy = np.bincount(gids, weights=self.arr[:, 1], minlength=m)
        cnt = np.bincount(gids, minlength=m)
        uni = np.where(cnt > 0)[0]
        centers = np.column_stack([sx[uni] / cnt[uni], sy[uni] / cnt[uni]])
        return uni, centers

    def forward_batch(self, X):
        """批量前向：全部细胞同帧——迭代 max_nodes 轮——bincount 累加（快于 np.add.at 10-100×）"""
        N = len(X)
        f = self.g_mat[:, :, 0].astype(int)     # (N, MAX_C)
        t = self.g_mat[:, :, 1].astype(int)
        w = self.g_mat[:, :, 2]
        en = self.g_mat[:, :, 3] > 0.5
        max_nn = int(self.n_nodes.max())
        vals = np.zeros((N, MAX_NODE))
        vals[:, :N_IN] = X
        row_idx = np.repeat(np.arange(N), MAX_C)          # 预计算索引
        t_flat = t.ravel()
        keys = row_idx * MAX_NODE + t_flat
        nb = N * MAX_NODE
        src = vals[np.arange(N)[:, None], f]              # (N, MAX_C) gather
        contrib = np.where(en, src * w, 0.0)
        acc = np.bincount(keys, weights=contrib.ravel(), minlength=nb).reshape(N, MAX_NODE)
        vals += acc
        for _ in range(max_nn - 1):
            vals[:, N_IN:max_nn] = np.tanh(vals[:, N_IN:max_nn])
            src = vals[np.arange(N)[:, None], f]
            contrib = np.where(en, src * w, 0.0)
            acc = np.bincount(keys, weights=contrib.ravel(), minlength=nb).reshape(N, MAX_NODE)
            vals += acc
        vals[:, N_IN:max_nn] = np.tanh(vals[:, N_IN:max_nn])
        out = vals[:, N_IN:N_IN + N_OUT]
        tau = 1.0 / (1.0 + np.exp(-out[:, 2]))
        fate = 1.0 / (1.0 + np.exp(-out[:, 3]))
        return out[:, 0], out[:, 1], tau, fate

    def step(self):
        N = len(self.arr)
        if N == 0:
            self.history.append((0, 0, 0, 0, 0, 0, 0, 0))
            return
        phase = 2 * np.pi * (self.phase_t % T_DIURNAL) / T_DIURNAL
        ph_sin, ph_cos = np.sin(phase), np.cos(phase)
        night = (self.phase_t % T_DIURNAL) >= T_DAY
        self.phase_t += 1
        x, y, e = self.arr[:, 0], self.arr[:, 1], self.arr[:, 2]
        s = self.arr[:, 5]
        tau_v = self.arr[:, 3]
        gids = self.arr[:, 8].astype(int)
        uni, centers = self.group_centers()
        # 感知：cKDTree 邻居查询（O(n log n)——替代 O(n²) 距离矩阵——大种群的根本优化）
        tree = cKDTree(np.column_stack([x, y]))
        crowd = np.zeros(N)
        nb_sum_x = np.zeros(N)
        nb_sum_y = np.zeros(N)
        for a, b in tree.query_pairs(CONTACT * 1.5):
            dx, dy = x[b] - x[a], y[b] - y[a]
            ang = np.arctan2(dy, dx)
            nb_sum_x[a] += np.sin(ang); nb_sum_y[a] += np.cos(ang)
            nb_sum_x[b] += np.sin(ang + np.pi); nb_sum_y[b] += np.cos(ang + np.pi)
            if dx * dx + dy * dy < 9.0:
                crowd[a] += 1; crowd[b] += 1
        nb_norm = np.maximum(np.hypot(nb_sum_x, nb_sum_y), 1e-6)
        # 威胁：k 近邻（τ 高者——取最近——从近到远、每细胞第一个命中）
        t_sin = np.zeros(N); t_cos = np.zeros(N)
        dists, idxs = tree.query(np.column_stack([x, y]), k=min(8, N))
        dists = np.atleast_2d(dists); idxs = np.atleast_2d(idxs)
        found = np.zeros(N, dtype=bool)
        for k in range(1 if dists.shape[1] > 1 else 0, dists.shape[1]):
            mask = (tau_v[idxs[:, k]] > tau_v + 0.1) & (dists[:, k] < 50) & ~found
            ang = np.arctan2(y[idxs[:, k]] - y, x[idxs[:, k]] - x)
            t_sin = np.where(mask, np.sin(ang), t_sin)
            t_cos = np.where(mask, np.cos(ang), t_cos)
            found |= mask
        # 光斑（N×4 距离——取 min）
        dl = np.hypot(LIGHTS[:, 0][None, :] - x[:, None],
                      LIGHTS[:, 1][None, :] - y[:, None])
        l_best = dl.argmin(axis=1)
        l_d = dl[np.arange(N), l_best]
        l_ang = np.arctan2(LIGHTS[l_best, 1] - y, LIGHTS[l_best, 0] - x)
        intensity = np.exp(-(dl / LIGHT_SIGMA) ** 2).max(axis=1)
        # 群体中心 → 光斑方向（广播信号）
        g_idx = {g: i for i, g in enumerate(uni)}
        c_of = np.array([g_idx[g] for g in gids])
        cx, cy = centers[c_of, 0], centers[c_of, 1]
        dlc = np.hypot(LIGHTS[:, 0][None, :] - cx[:, None],
                       LIGHTS[:, 1][None, :] - cy[:, None])
        lc_best = dlc.argmin(axis=1)
        g_sin = np.sin(np.arctan2(LIGHTS[lc_best, 1] - cy, LIGHTS[lc_best, 0] - cx))
        g_cos = np.cos(np.arctan2(LIGHTS[lc_best, 1] - cy, LIGHTS[lc_best, 0] - cx))
        # 资源混合（GRP_W 群体级 + (1-GRP_W) 个体级——分化-协调平衡）：
        #   群体级 = 按群体中心深度（协调收益——迁移=全体资源切换）
        #   个体级 = 按个体深度（体积生态位——大 s 深层独立吸营养——分化保持）
        deep_g = cy > 50
        deep_i = y > 50
        nut_g = np.where(deep_g, NUT_DEEP * np.clip(s, *NUT_S_CLIP), 0.0)
        nut_i = np.where(deep_i, NUT_DEEP * np.clip(s, *NUT_S_CLIP), 0.0)
        nutrient = GRP_W * nut_g + (1.0 - GRP_W) * nut_i
        int_g = np.where(deep_g, intensity * 0.3, intensity * 1.5)
        int_i = np.where(deep_i, intensity * 0.3, intensity * 1.5)
        intensity = GRP_W * int_g + (1.0 - GRP_W) * int_i
        if night:
            intensity *= 0.1
            nutrient *= NIGHT_NUT   # 夜间深层营养更强（真实 DVM：夜间下潜就是为了吃营养）
        # 输入矩阵（N×15——顺序与逐细胞版一致）
        inp = np.column_stack([np.sin(l_ang), np.cos(l_ang), l_d / 70.0,
                               intensity, crowd / 20.0, t_sin, t_cos,
                               nb_sum_x / nb_norm, nb_sum_y / nb_norm,
                               e / 150.0, g_sin, g_cos,
                               np.full(N, ph_sin), np.full(N, ph_cos),
                               np.ones(N)])
        dx_out, dy_out, tau_v, fate = self.forward_batch(inp)
        # glsA 绑定
        fate = np.where(s > 1.05, np.clip(fate * 0.3 + 0.8, 0, 1), fate)
        fate = np.where(s < 0.7, np.clip(fate * 0.3, 0, 1), fate)
        diff_photo = (0.1 + 0.9 * (1.0 - fate)) * 1.8
        absorb = intensity * ABS_RATE * (0.75 + 0.25 * S_MIN / s) * (1.0 - tau_v) ** PHI \
                 * diff_photo / (1.0 + crowd * 0.08) + nutrient
        e = e + absorb - BASE_COST - MOVE_COST * 0.1 - 0.1 * self.arr[:, 4]
        norm = np.maximum(np.hypot(dx_out, dy_out), 1e-6)
        spd = SPEED * np.clip(norm, 0.1, 1.5)
        nx = np.clip(x + dy_out / norm * spd, 0, W)
        ny = np.clip(y + dx_out / norm * spd, 0, H)
        e = e - MOVE_COST * spd
        # 向心力（a>0.3——跟随 gid 质心）
        a_v = self.arr[:, 6]
        follow = a_v > 0.3
        nx = np.where(follow, np.clip(x + (cx - x) * 0.3, 0, W), nx)
        ny = np.where(follow, np.clip(y + (cy - y) * 0.3, 0, H), ny)
        mvx, mvy = nx - x, ny - y
        self.arr[:, 0], self.arr[:, 1], self.arr[:, 2] = nx, ny, e
        self.arr[:, 3] = tau_v
        self.arr[:, 7] = fate
        # 协调度：成员移动 vs 群体中心位移（批量）
        n_centers = self.group_centers()[1]
        cdx, cdy = n_centers[c_of, 0] - cx, n_centers[c_of, 1] - cy
        cn = np.hypot(cdx, cdy)
        ok = (cn > 1e-9)
        cdx, cdy = np.where(ok, cdx / np.maximum(cn, 1e-9), 0.0), np.where(ok, cdy / np.maximum(cn, 1e-9), 0.0)
        mv_norm = np.maximum(np.hypot(mvx, mvy), 1e-9)
        per_coord = np.where(ok, np.maximum((mvx * cdx + mvy * cdy) / mv_norm, 0.0), 0.0)
        coord = per_coord.mean()
        # 共享池（按 gid——bincount 平均）
        if len(uni) > 1:
            sums = np.bincount(gids, weights=e, minlength=gids.max() + 1)
            cnts = np.bincount(gids, minlength=gids.max() + 1)
            pool = sums / np.maximum(cnts, 1)
            e = pool[gids]
            self.arr[:, 2] = e
        # 死亡/繁殖（批量 kill——布尔掩码替代逐删 O(N²) 复制）
        dead_mask = self.arr[:, 2] <= 0
        alive_c = self.arr[:, 7]
        thr = REPRO_E * (1.6 - 1.3 * alive_c)
        reps = np.where((self.arr[:, 2] > thr) & ~dead_mask)[0]
        for i in reps[::-1]:
            self._reproduce(i)
        if dead_mask.any():
            keep = ~dead_mask   # 删除 = 数组删行（单文件保存时该生物字段自动消失——用户方案）
            self.arr = self.arr[keep]
            self.g_mat = self.g_mat[keep]
            self.g_len = self.g_len[keep]
            self.n_nodes = self.n_nodes[keep]
        self._flush()   # 步末合并繁殖缓冲
        self.step_count += 1
        if self.step_count % GAP == 0:
            self.savesnapshot()   # 单文件快照（每 GAP 步 1 次大文件写）
        # 统计
        if len(self.arr):
            cs_ = self.arr[:, 7]
            germ = np.sum(cs_ > 0.7)
            soma = np.sum(cs_ < 0.3)
            gsizes = {}
            for g in np.unique(self.arr[:, 8].astype(int)):
                gsizes[g] = np.sum(self.arr[:, 8].astype(int) == g)
            avg_cy = np.mean(centers[:, 1]) if len(centers) else 50.0
            self.history.append((len(self.arr), np.mean(cs_), germ, soma,
                                 len(gsizes), max(gsizes.values()) if gsizes else 0,
                                 coord, avg_cy))
        else:
            self.history.append((0, 0, 0, 0, 0, 0, 0, 50.0))

    def run(self, T):
        for _ in range(T):
            self.step()
        return np.array(self.history)


def run_seed(seed, T=2000):
    w = CoordWorldV(workdir=f"stage35v_evo_{seed}", n0=60, seed=seed)
    h = w.run(T)
    live = np.mean(h[500:, 0] > 5)
    germ, soma = h[-1, 2], h[-1, 3]
    coord = np.mean(h[500:, 6])
    return h, live, germ, soma, coord


def run_scan():
    """资源混合扫描：GRP_W ∈ {1.0, 0.7, 0.5, 0.3} × 5 seed——分化-协调平衡点"""
    global GRP_W
    print("=== 资源混合扫描（分化-协调平衡——GRP_W 群体级权重） ===\n")
    for gw in [1.0, 0.7, 0.5, 0.3]:
        GRP_W = gw
        n_diff = 0; n_coord = 0; n_live = 0
        coords = []
        for seed in [109, 203, 307, 411, 503]:
            w = CoordWorldV(workdir=f"stage35v_evo_{seed}", n0=60, seed=seed)
            h = w.run(1500)
            live = np.mean(h[500:, 0] > 5)
            germ, soma = h[-1, 2], h[-1, 3]
            coord = np.mean(h[500:, 6])
            n_live += live >= 0.5
            n_diff += (germ > 2 and soma > 2)
            n_coord += coord > 0.5
            coords.append(coord)
        print(f"GRP_W={gw}: 分化 {n_diff}/5 | 协调>0.5 {n_coord}/5 | 自持 {n_live}/5 | "
              f"协调均值 {np.mean(coords):.2f}")
    print("[done] stage35v resource mix scan")


def run():
    print("=== stage35v：CPU 向量化版（机制同 stage35——逐细胞循环 → numpy 批量） ===\n")
    results = []
    for seed in [109, 203, 307, 411, 503]:
        t0 = time.time()
        h, live, germ, soma, coord = run_seed(seed)
        dt = time.time() - t0
        diff = "✓ 分化" if germ > 2 and soma > 2 else "✗ 全能主导"
        print(f"seed {seed}: 自持={100*live:.0f}% 协调={coord:.2f} 生殖={germ:.0f} "
              f"体细胞={soma:.0f} {diff} ({dt:.0f}s)")
        results.append((seed, live, coord, germ, soma, diff, dt))
    n_diff = sum(1 for r in results if "✓" in r[5])
    n_coord = sum(1 for r in results if r[2] > 0.5)
    avg_t = np.mean([r[6] for r in results])
    print(f"\n[结果] 分化 {n_diff}/5 | 协调>0.5 {n_coord}/5 | 平均 {avg_t:.0f}s/seed")
    print("[done] stage35v vectorized")


if __name__ == "__main__":
    if os.environ.get("STAGE35_SCAN") == "1":
        run_scan()
    else:
        run()
