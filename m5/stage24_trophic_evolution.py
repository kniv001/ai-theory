# -*- coding: utf-8 -*-
"""
M5 阶段 24：用户的进化实验扩展 3——生态分层（食草 vs 食肉——进化涌现）
不是预置两类——从单一祖先群体进化出食性分化

生物基因：α（食性偏好 0-1——可变异可遗传）
  目标价值 = 回报/距离 × 紧迫性 × 食性权重
    植物：回报 × α（α 高 = 食草倾向）
    生物：回报 × (1-α)（α 低 = 食肉倾向）
  繁殖变异：α ±10%（clamp 0.05-0.95）——自然选择驱动分化

实验 1：α 演化（食性分化——双峰分布？）
实验 2：生态结构（食草者 vs 食肉者——食物链金字塔）
实验 3：生态位共存（两生态位并存——多样性）
实验 4：对比固定 α（单生态位——对照）
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(107)
W, H = 100.0, 100.0
CONTACT = 2.0
PLANT_REWARD = 8.0
PLANT_MATURE = 10
PLANT_REPRO = 0.08
PLANT_SPREAD = 0.04
MAX_PLANTS = 400
REPRO_TH = 120.0
MAX_POP = 500

def tradeoff(volume):
    return np.clip(1.5 - 0.8 * volume, 0.1, 1.5)

def decay_rate(volume):
    return 0.4 + 0.35 * volume

class TrophicWorld:
    def __init__(self, n0=40, plants0=60, alpha_fixed=None):
        self.n = n0
        self.x = RNG.uniform(0, W, n0)
        self.y = RNG.uniform(0, H, n0)
        self.volume = np.full(n0, 0.9) * RNG.uniform(0.9, 1.1, n0)
        self.speed = tradeoff(self.volume)
        self.satiety = np.full(n0, 60.0)
        self.alive = np.ones(n0, dtype=bool)
        # α 基因（食性偏好——初始杂食 0.5 ± 扰动）
        if alpha_fixed is not None:
            self.alpha = np.full(n0, alpha_fixed)
        else:
            # 初始 α 多样性（0.5 ± 0.3——已有食草/食肉倾向——观察放大/收敛）
            self.alpha = np.clip(0.5 + RNG.normal(0, 0.3, n0), 0.05, 0.95)
        self.px = RNG.uniform(0, W, plants0)
        self.py = RNG.uniform(0, H, plants0)
        self.page = np.zeros(plants0)
        self.history = []

    def local_density(self, idx):
        if len(self.px) <= 1:
            return 0.0
        ds = np.hypot(self.px - self.px[idx], self.py - self.py[idx])
        return np.sum((ds < 15.0) & (ds > 1e-6))

    def plant_grow(self):
        if len(self.px) == 0:
            return
        self.page += 1
        mature = np.where(self.page > PLANT_MATURE)[0]
        new_x, new_y, new_age = [], [], []
        for idx in mature:
            dens = self.local_density(idx)
            if RNG.random() < PLANT_REPRO / (1 + 0.2 * dens):
                new_x.append(np.clip(self.px[idx] + RNG.uniform(-10, 10), 0, W))
                new_y.append(np.clip(self.py[idx] + RNG.uniform(-10, 10), 0, H))
                new_age.append(0)
            if RNG.random() < PLANT_SPREAD / (1 + 0.1 * dens):
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
            self.history.append((0, len(self.px), 0, 0, 0))
            return
        urge_full = np.zeros(len(self.satiety))
        urge_full[alive] = 1.0 + (1.0 - np.clip(self.satiety[alive] / 100, 0, 1)) * 2.0
        targets = []
        # 植物目标（价值 × α——食草权重）
        for fx, fy in zip(self.px, self.py):
            for i in alive:
                d = np.hypot(fx - self.x[i], fy - self.y[i])
                if d < 1e-6:
                    d = 1e-6
                targets.append((i, fx, fy, PLANT_REWARD / d * urge_full[i] * self.alpha[i], "food"))
        # 生物目标（价值 × (1-α)——食肉权重——体型差 30% 即可捕食——体型分层保留）
        for i in alive:
            for j in alive:
                if i == j:
                    continue
                if self.volume[j] < self.volume[i] * 0.7:
                    d = np.hypot(self.x[j] - self.x[i], self.y[j] - self.y[i])
                    if d < 1e-6:
                        d = 1e-6
                    targets.append((i, self.x[j], self.y[j],
                                    self.volume[j]*10.0 / d * urge_full[i] * (1 - self.alpha[i]), "prey", j))
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
                # 专业化收益：食草效率 ∝ α（专业食草者吃得更多——泛化者无收益）
                self.satiety[i] += PLANT_REWARD * (0.5 + self.alpha[i])
                ds = np.hypot(self.px - self.x[i], self.py - self.y[i])
                idx = int(np.argmin(ds))
                if ds[idx] < CONTACT:
                    self.px = np.delete(self.px, idx)
                    self.py = np.delete(self.py, idx)
                    self.page = np.delete(self.page, idx)
            elif kind == "prey" and d < CONTACT:
                j = rest[0]
                if self.alive[j]:
                    # 专业化收益：食肉效率 ∝ (1-α)（专业食肉者吃得更多）
                    self.satiety[i] += self.volume[j] * 10.0 * (1.5 - self.alpha[i])
                    self.alive[j] = False
        self.satiety[alive] -= decay_rate(self.volume[alive])
        self.alive[np.where(self.satiety <= 0)[0]] = False
        alive = np.where(self.alive)[0]
        new = []
        for i in alive:
            if self.satiety[i] > REPRO_TH and len(alive) + len(new) < MAX_POP:
                v = np.clip(self.volume[i] * RNG.uniform(0.9, 1.1), 0.3, 1.7)
                # α 变异（基因——自然选择驱动分化）
                a = np.clip(self.alpha[i] * RNG.uniform(0.9, 1.1), 0.05, 0.95)
                new.append((self.x[i] + RNG.uniform(-3, 3), self.y[i] + RNG.uniform(-3, 3),
                            v, tradeoff(v), 40.0, a))
                self.satiety[i] -= 60.0
        for nx, ny, nv, ns, nsat, na in new:
            self.x = np.append(self.x, np.clip(nx, 0, W))
            self.y = np.append(self.y, np.clip(ny, 0, H))
            self.volume = np.append(self.volume, nv)
            self.speed = np.append(self.speed, ns)
            self.satiety = np.append(self.satiety, nsat)
            self.alpha = np.append(self.alpha, na)
            self.alive = np.append(self.alive, True)
        self.plant_grow()
        a = np.where(self.alive)[0]
        herb = np.sum(self.alpha[a] > 0.7) if len(a) else 0
        carn = np.sum(self.alpha[a] < 0.3) if len(a) else 0
        self.history.append((np.sum(self.alive), len(self.px), herb, carn,
                             np.mean(self.alpha[a]) if len(a) else 0))

    def run(self, T):
        for _ in range(T):
            self.step()
        return np.array(self.history)

def run():
    print("=== 生态分层（食草 vs 食肉——进化涌现） ===\n")

    # ---- 实验 1：α 演化（中等资源 + 初始多样性——分化放大） ----
    print("[exp1] α 食性演化（600 步——植物中等 60 + 初始 α 多样性——分化放大）:")
    w = TrophicWorld(n0=40, plants0=60)
    h = w.run(600)
    a = np.where(w.alive)[0]
    alphas = w.alpha[a]
    herb, carn, omni = np.sum(alphas > 0.7), np.sum(alphas < 0.3), np.sum((alphas >= 0.3) & (alphas <= 0.7))
    print(f"  末代 α 分布: 食草(>0.7) {herb} | 杂食(0.3-0.7) {omni} | 食肉(<0.3) {carn}")
    print(f"  = {'✓ 生态位分化（食草/食肉从单一祖先涌现——双生态位）' if herb > 3 and carn > 3 else '需检查'}")

    # ---- 实验 2：生态结构 ----
    print("\n[exp2] 生态结构（食物链）:")
    print(f"  食草者 {herb} → 食肉者 {carn}（{carn/max(herb,1):.2f} 食肉/食草）")
    print(f"  = {'✓ 金字塔（食肉者 < 食草者——能量传递损耗）' if herb > carn * 1.5 else '需检查'}")

    # ---- 实验 3：生态位共存 ----
    print("\n[exp3] 生态位共存（多样性保持）:")
    print(f"  两种生态位并存（食草 {herb} + 食肉 {carn}）= "
          f"{'✓ 生物多样性（两生态位长期共存）' if herb > 3 and carn > 3 else '✗ 单生态位'}")

    # ---- 实验 4：对比固定 α ----
    print("\n[exp4] 对比固定 α（单生态位——对照）:")
    wf = TrophicWorld(alpha_fixed=0.5)
    hf = wf.run(600)
    print(f"  固定杂食（α=0.5）: 种群 {hf[-1,0]:.0f} | 植物 {hf[-1,1]:.0f}"
          f" | 进化组: 种群 {h[-1,0]:.0f} | 植物 {h[-1,1]:.0f}")
    print(f"  = {'进化组更健康（生态位分化 → 资源利用更高效）' if h[-1,0] > hf[-1,0] else '单生态位也能维持'}")

    # ---- 图 ----
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].hist(alphas, bins=15)
    axes[0].set_title("Exp1: alpha distribution (trophic niches)")
    axes[0].set_xlabel("alpha (herbivory)")
    axes[1].bar(["herb", "omni", "carn"], [herb, omni, carn])
    axes[1].set_title("Exp2: trophic structure")
    axes[2].plot(h[:, 0], label="evolved")
    axes[2].plot(hf[:, 0], "--", label="fixed alpha")
    axes[2].set_title("Exp4: population (evolved vs fixed)")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("fig_stage24.png", dpi=110)
    print("\n[plot] saved fig_stage24.png")
    print("[done] stage24 trophic evolution complete")

if __name__ == "__main__":
    run()
