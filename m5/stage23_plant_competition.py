# -*- coding: utf-8 -*-
"""
M5 阶段 23：用户的进化实验扩展 2——植物竞争（密度过高生长慢——资源竞争）
stage22 升级：繁殖概率 ∝ 局部密度（光/水/空间——竞争压力）

竞争机制：每株成熟植物的局部密度（半径 15 内其他植物数）
  复制概率 = base / (1 + 0.2 × 局部密度)（拥挤 → 繁殖慢）
  传播概率 = base / (1 + 0.1 × 局部密度)（传播受竞争影响小——远处种子）

实验 1：竞争-繁殖关系（高密度区繁殖慢 vs 低密度快——机制验证）
实验 2：分布均匀化（竞争 vs 无竞争——植物空间分布——最近邻距离）
实验 3：生态平衡（竞争下植物-生物共存）
实验 4：空间模式（竞争 → 分散 vs 无竞争 → 聚集）
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(103)
W, H = 100.0, 100.0
CONTACT = 2.0
PLANT_REWARD = 8.0
PLANT_MATURE = 10
PLANT_REPRO = 0.12
PLANT_SPREAD = 0.05
MAX_PLANTS = 400
COMP_RADIUS = 15.0
REPRO_TH = 120.0
MAX_POP = 500

def tradeoff(volume):
    return np.clip(1.5 - 0.8 * volume, 0.1, 1.5)

def decay_rate(volume):
    return 0.4 + 0.35 * volume

class CompWorld:
    def __init__(self, n0=30, plants0=50, competition=True):
        self.competition = competition
        self.n = n0
        self.x = RNG.uniform(0, W, n0)
        self.y = RNG.uniform(0, H, n0)
        self.volume = np.full(n0, 0.9) * RNG.uniform(0.9, 1.1, n0)
        self.speed = tradeoff(self.volume)
        self.satiety = np.full(n0, 60.0)
        self.alive = np.ones(n0, dtype=bool)
        self.px = RNG.uniform(0, W, plants0)
        self.py = RNG.uniform(0, H, plants0)
        self.page = np.zeros(plants0)
        self.history = []

    def local_density(self, idx):
        """半径 COMP_RADIUS 内的其他植物数"""
        if len(self.px) <= 1:
            return 0.0
        ds = np.hypot(self.px - self.px[idx], self.py - self.py[idx])
        return np.sum((ds < COMP_RADIUS) & (ds > 1e-6))

    def plant_grow(self):
        if len(self.px) == 0:
            return
        self.page += 1
        mature = np.where(self.page > PLANT_MATURE)[0]
        new_x, new_y, new_age = [], [], []
        for idx in mature:
            # 竞争压力（局部密度）
            if self.competition:
                dens = self.local_density(idx)
                repro_p = PLANT_REPRO / (1 + 0.2 * dens)
                spread_p = PLANT_SPREAD / (1 + 0.1 * dens)
            else:
                repro_p, spread_p = PLANT_REPRO, PLANT_SPREAD
            if RNG.random() < repro_p:
                new_x.append(np.clip(self.px[idx] + RNG.uniform(-10, 10), 0, W))
                new_y.append(np.clip(self.py[idx] + RNG.uniform(-10, 10), 0, H))
                new_age.append(0)
            if RNG.random() < spread_p:
                ang = RNG.uniform(0, 2*np.pi)
                dist = RNG.uniform(10, 50)
                new_x.append(np.clip(self.px[idx] + dist*np.cos(ang), 0, W))
                new_y.append(np.clip(self.py[idx] + dist*np.sin(ang), 0, H))
                new_age.append(0)
        if new_x:
            room = MAX_PLANTS - len(self.px)
            if room > 0:
                new_x, new_y, new_age = new_x[:room], new_y[:room], new_age[:room]
                self.px = np.append(self.px, new_x)
                self.py = np.append(self.py, new_y)
                self.page = np.append(self.page, new_age)

    def step(self):
        alive = np.where(self.alive)[0]
        if len(alive) == 0:
            self.plant_grow()
            self.history.append((0, len(self.px), 0, 0))
            return
        urge_full = np.zeros(len(self.satiety))
        urge_full[alive] = 1.0 + (1.0 - np.clip(self.satiety[alive] / 100, 0, 1)) * 2.0
        targets = []
        for fx, fy in zip(self.px, self.py):
            for i in alive:
                d = np.hypot(fx - self.x[i], fy - self.y[i])
                if d < 1e-6:
                    d = 1e-6
                targets.append((i, fx, fy, PLANT_REWARD / d * urge_full[i], "food"))
        for i in alive:
            for j in alive:
                if i == j:
                    continue
                if self.volume[j] < self.volume[i] * 0.9:
                    d = np.hypot(self.x[j] - self.x[i], self.y[j] - self.y[i])
                    if d < 1e-6:
                        d = 1e-6
                    targets.append((i, self.x[j], self.y[j], self.volume[j]*10.0 / d * urge_full[i], "prey", j))
        best = {}
        for t in targets:
            if t[0] not in best or t[3] > best[t[0]][3]:
                best[t[0]] = t
        for i, (i0, tx, ty, val, kind, *rest) in best.items():
            dx, dy = tx - self.x[i], ty - self.y[i]
            d = np.hypot(dx, dy)
            if d > 1e-6:
                step = self.speed[i]
                self.x[i] = np.clip(self.x[i] + dx/d*step, 0, W)
                self.y[i] = np.clip(self.y[i] + dy/d*step, 0, H)
            if kind == "food" and d < CONTACT and len(self.px) > 0:
                self.satiety[i] += PLANT_REWARD
                ds = np.hypot(self.px - self.x[i], self.py - self.y[i])
                idx = int(np.argmin(ds))
                if ds[idx] < CONTACT:
                    self.px = np.delete(self.px, idx)
                    self.py = np.delete(self.py, idx)
                    self.page = np.delete(self.page, idx)
            elif kind == "prey" and d < CONTACT:
                j = rest[0]
                if self.alive[j]:
                    self.satiety[i] += self.volume[j] * 10.0
                    self.alive[j] = False
        self.satiety[alive] -= decay_rate(self.volume[alive])
        self.alive[np.where(self.satiety <= 0)[0]] = False
        alive = np.where(self.alive)[0]
        new = []
        for i in alive:
            if self.satiety[i] > REPRO_TH and len(alive) + len(new) < MAX_POP:
                v = np.clip(self.volume[i] * RNG.uniform(0.9, 1.1), 0.3, 1.7)
                new.append((self.x[i] + RNG.uniform(-3, 3), self.y[i] + RNG.uniform(-3, 3),
                            v, tradeoff(v), 40.0))
                self.satiety[i] -= 60.0
        for nx, ny, nv, ns, nsat in new:
            self.x = np.append(self.x, np.clip(nx, 0, W))
            self.y = np.append(self.y, np.clip(ny, 0, H))
            self.volume = np.append(self.volume, nv)
            self.speed = np.append(self.speed, ns)
            self.satiety = np.append(self.satiety, nsat)
            self.alive = np.append(self.alive, True)
        self.plant_grow()
        self.history.append((np.sum(self.alive), len(self.px),
                             np.mean(self.volume[self.alive]) if np.any(self.alive) else 0,
                             np.mean(self.speed[self.alive]) if np.any(self.alive) else 0))

    def nn_dist_std(self):
        """最近邻距离标准差（分布均匀度——小 = 均匀）"""
        if len(self.px) < 3:
            return 0.0
        nns = []
        for i in range(len(self.px)):
            ds = np.hypot(self.px - self.px[i], self.py - self.py[i])
            ds = ds[ds > 1e-6]
            if len(ds):
                nns.append(np.min(ds))
        return np.std(nns)

    def run(self, T):
        for _ in range(T):
            self.step()
        return np.array(self.history)

def run():
    print("=== 植物竞争（密度过高生长慢——资源竞争） ===\n")

    # ---- 实验 1：竞争-繁殖关系 ----
    print("[exp1] 竞争-繁殖关系（机制）:")
    w = CompWorld(competition=True)
    w.px = np.array([50.0, 51.0, 52.0, 53.0, 54.0, 80.0])   # 拥挤区(50-54) + 孤立区(80)
    w.py = np.array([50.0, 50.0, 50.0, 50.0, 50.0, 50.0])
    w.page = np.full(6, 20.0)   # 全部成熟
    d_crowd = w.local_density(0)
    d_lone = w.local_density(5)
    p_crowd = PLANT_REPRO / (1 + 0.2 * d_crowd)
    p_lone = PLANT_REPRO / (1 + 0.2 * d_lone)
    print(f"  拥挤区（密度 {d_crowd:.0f}）: 繁殖概率 {p_crowd:.3f} | 孤立区（密度 {d_lone:.0f}）: {p_lone:.3f}")
    drop = 100 * (1 - p_crowd / p_lone)
    verdict = f"✓ 竞争机制（拥挤 → 繁殖慢——资源竞争——降低 {drop:.0f}%）" if p_crowd < p_lone * 0.75 else "✗"
    print(f"  = {verdict}")

    # ---- 实验 2/4：分布均匀化 ----
    print("\n[exp2/4] 分布模式（竞争 vs 无竞争——500 步后最近邻分布）:")
    wc = CompWorld(plants0=40, competition=True)
    wc.run(500)
    wn = CompWorld(plants0=40, competition=False)
    wn.run(500)
    std_c, std_n = wc.nn_dist_std(), wn.nn_dist_std()
    print(f"  竞争: 植物 {len(wc.px)} 株 | 最近邻标准差 {std_c:.2f}"
          f" | 无竞争: 植物 {len(wn.px)} 株 | 标准差 {std_n:.2f}")
    print(f"  = {'✓ 竞争 → 更均匀分布（分散繁殖——避开拥挤区——空间资源竞争）' if std_c < std_n * 0.95 else '需检查'}")

    # ---- 实验 3：生态平衡 ----
    print("\n[exp3] 竞争下生态平衡（500 步）:")
    w3 = CompWorld(plants0=50, competition=True)
    h3 = w3.run(500)
    print(f"  生物 {h3[-1,0]:.0f} | 植物 {h3[-1,1]:.0f}"
          f" = {'✅ 竞争下共存（生态平衡保持）' if h3[-1,0] > 10 and h3[-1,1] > 50 else '✗'}")

    # ---- 图 ----
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].scatter(wc.px, wc.py, s=8, label="competition")
    axes[0].scatter(wn.px, wn.py, s=8, marker="x", label="no competition")
    axes[0].set_title("Exp2: spatial distribution")
    axes[0].legend(fontsize=8)
    axes[1].bar(["comp", "no-comp"], [std_c, std_n])
    axes[1].set_title("Exp4: NN-dist std (uniformity)")
    axes[2].plot(h3[:, 0], label="animals")
    axes[2].plot(h3[:, 1], "--", label="plants")
    axes[2].set_title("Exp3: coexistence under competition")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("fig_stage23.png", dpi=110)
    print("\n[plot] saved fig_stage23.png")
    print("[done] stage23 plant competition complete")

if __name__ == "__main__":
    run()
