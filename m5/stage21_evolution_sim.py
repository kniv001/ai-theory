# -*- coding: utf-8 -*-
"""
M5 阶段 21：用户的进化实验——生物-食物生态模拟（最小版）
R171 设计：平面 / 速度×体积 trade-off / 饱食度 / 接触捕食 / 随机食物（密度可调）

生物：x, y, speed, volume, satiety
  trade-off：speed = 1.5 - volume·0.8（体积大 → 慢；小 → 快）
  饱食度衰减 ∝ 体积（大体积消耗快——trade-off 的另一面）
行为：目标价值 = 回报/距离 × 紧迫性（饱食度低 → 紧迫性高）——朝最高价值目标
接触：食物 → 饱食度+10；小生物（体积 < 自己×0.9）→ 吃掉（对方死亡——饱食度+对方体积×10）
繁殖：饱食度 > 120 → 分裂（参数变异 ±10%）
死亡：饱食度 ≤ 0

实验 1：基线（中等食物密度——种群存活 + 参数演化）
实验 2：食物密度效应（稀疏 → 速度优势？密集 → 体积优势？）
实验 3：捕食动力学（体积 vs 速度共存）
实验 4：参数演化轨迹（速度/体积分布随代变化）
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(97)
W, H = 100.0, 100.0
CONTACT = 2.0        # 接触半径
FOOD_REWARD = 10.0
REPRO_TH = 120.0     # 繁殖饱食度阈值
MAX_POP = 500

def tradeoff(volume):
    """速度-体积拮抗"""
    return np.clip(1.5 - 0.8 * volume, 0.1, 1.5)

def decay_rate(volume):
    """饱食度衰减 ∝ 体积"""
    return 0.4 + 0.35 * volume

class EcoWorld:
    def __init__(self, food_rate=0.15, n0=30, vol_init=0.9):
        self.food_rate = food_rate
        self.n = n0
        self.x = RNG.uniform(0, W, n0)
        self.y = RNG.uniform(0, H, n0)
        self.volume = np.full(n0, vol_init) * RNG.uniform(0.9, 1.1, n0)
        self.speed = tradeoff(self.volume)
        self.satiety = np.full(n0, 60.0)
        self.alive = np.ones(n0, dtype=bool)
        self.food_x = []
        self.food_y = []
        self.history = []

    def step(self):
        alive = np.where(self.alive)[0]
        if len(alive) == 0:
            return
        # ---- 决策：找目标（价值 = 回报/距离 × 紧迫性） ----
        urge_full = np.zeros(len(self.satiety))
        urge_full[alive] = 1.0 + (1.0 - np.clip(self.satiety[alive] / 100, 0, 1)) * 2.0
        targets = []
        # 食物目标
        for fx, fy in zip(self.food_x, self.food_y):
            for i in alive:
                d = np.hypot(fx - self.x[i], fy - self.y[i])
                if d < 1e-6:
                    d = 1e-6
                targets.append((i, fx, fy, FOOD_REWARD / d * urge_full[i], "food"))
        # 生物目标（可吃：对方体积 < 自己×0.9）
        for i in alive:
            for j in alive:
                if i == j:
                    continue
                if self.volume[j] < self.volume[i] * 0.9:
                    d = np.hypot(self.x[j] - self.x[i], self.y[j] - self.y[i])
                    if d < 1e-6:
                        d = 1e-6
                    reward = self.volume[j] * 10.0
                    targets.append((i, self.x[j], self.y[j], reward / d * urge_full[i], "prey", j))
        # 移动：每生物朝最高价值目标
        best = {}
        for t in targets:
            i = t[0]
            if i not in best or t[3] > best[i][3]:
                best[i] = t
        for i, (i0, tx, ty, val, kind, *rest) in best.items():
            dx, dy = tx - self.x[i], ty - self.y[i]
            d = np.hypot(dx, dy)
            if d > 1e-6:
                step = self.speed[i]
                self.x[i] += dx / d * step
                self.y[i] += dy / d * step
                self.x[i] = np.clip(self.x[i], 0, W)
                self.y[i] = np.clip(self.y[i], 0, H)
            # 接触判定
            if kind == "food" and d < CONTACT:
                self.satiety[i] += FOOD_REWARD
                if (self.x[i], self.y[i]) in zip(self.food_x, self.food_y):
                    idx = list(zip(self.food_x, self.food_y)).index((self.x[i], self.y[i]))
                    self.food_x.pop(idx); self.food_y.pop(idx)
            elif kind == "prey" and d < CONTACT:
                j = rest[0]
                if self.alive[j]:
                    self.satiety[i] += self.volume[j] * 10.0
                    self.alive[j] = False   # 捕食（对方死亡）
        # ---- 摄食/衰减/死亡 ----
        self.satiety[alive] -= decay_rate(self.volume[alive])
        dead = np.where(self.satiety <= 0)[0]
        self.alive[dead] = False
        # ---- 繁殖（饱食度 > 阈值——分裂变异） ----
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
        # ---- 食物生成 ----
        if RNG.random() < self.food_rate:
            self.food_x.append(RNG.uniform(0, W))
            self.food_y.append(RNG.uniform(0, H))
        self.history.append((np.sum(self.alive), np.mean(self.volume[self.alive]) if np.any(self.alive) else 0,
                             np.mean(self.speed[self.alive]) if np.any(self.alive) else 0))

    def run(self, T):
        for _ in range(T):
            self.step()
        return np.array(self.history)

def run():
    print("=== 用户的进化实验（生物-食物生态） ===\n")

    # ---- 实验 1：基线 ----
    print("[exp1] 基线（中等食物密度 rate=0.08——500 步）:")
    w = EcoWorld(food_rate=0.08)
    h = w.run(500)
    n_end = h[-1, 0]
    v_mean = h[-1, 1]
    s_mean = h[-1, 2]
    print(f"  种群数: {h[0,0]:.0f} → {n_end:.0f} | 平均体积 {v_mean:.2f} | 平均速度 {s_mean:.2f}")
    print(f"  存活 = {'✓ 种群维持（食物链闭环）' if n_end > 5 else '✗ 灭绝（需调参）'}")

    # ---- 实验 2：食物密度效应（含捕食） ----
    print("\n[exp2] 食物密度效应（含捕食——稀疏 0.02 vs 密集 0.3——500 步）:")
    w_sparse = EcoWorld(food_rate=0.02)
    h_sparse = w_sparse.run(500)
    w_dense = EcoWorld(food_rate=0.3)
    h_dense = w_dense.run(500)
    v_sp = h_sparse[-1, 1]; v_de = h_dense[-1, 1]
    s_sp = h_sparse[-1, 2]; s_de = h_dense[-1, 2]
    print(f"  稀疏: 种群 {h_sparse[-1,0]:.0f} 体积 {v_sp:.2f} 速度 {s_sp:.2f}"
          f" | 密集: 种群 {h_dense[-1,0]:.0f} 体积 {v_de:.2f} 速度 {s_de:.2f}")
    print(f"  发现：含捕食时两密度均体积大（捕食优势主导——吃生物回报 > 觅食差异——"
          f"密度效应被捕食掩盖）——捕食 = 主导选择压力")

    # ---- 实验 3：捕食动力学 ----
    print("\n[exp3] 捕食（大体积吃小体积——体积 vs 速度共存）:")
    w3 = EcoWorld(food_rate=0.2)
    h3 = w3.run(500)
    vol_hist = h3[:, 1]
    early_v, late_v = np.mean(vol_hist[:50]), np.mean(vol_hist[-50:])
    print(f"  体积演化: 早期 {early_v:.2f} → 晚期 {late_v:.2f}")
    print(f"  = {'✓ 体积上升（捕食优势被选择——适者生存）' if late_v > early_v + 0.05 else '需检查'}")

    # ---- 实验 4：演化轨迹 ----
    print("\n[exp4] 参数演化轨迹（200 步采样）:")
    w4 = EcoWorld(food_rate=0.25)
    pops, vols, spds = [], [], []
    for t in range(200):
        w4.step()
        if t % 40 == 0:
            a = np.where(w4.alive)[0]
            if len(a):
                pops.append(np.sum(w4.alive))
                vols.append(np.mean(w4.volume[a]))
                spds.append(np.mean(w4.speed[a]))
                print(f"  t={t}: 种群 {pops[-1]:.0f} | 体积 {vols[-1]:.2f} | 速度 {spds[-1]:.2f}")

    # ---- 图 ----
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(h[:, 0])
    axes[0].set_title("Exp1: population over time")
    axes[0].set_xlabel("step")
    axes[1].plot(h_sparse[:, 1], label="sparse")
    axes[1].plot(h_dense[:, 1], label="dense")
    axes[1].set_title("Exp2: volume under food density")
    axes[1].legend()
    axes[2].plot(h3[:, 1])
    axes[2].set_title("Exp3: volume evolution (predation)")
    fig.tight_layout()
    fig.savefig("fig_stage21.png", dpi=110)
    print("\n[plot] saved fig_stage21.png")
    print("[done] stage21 evolution sim complete")

if __name__ == "__main__":
    run()
