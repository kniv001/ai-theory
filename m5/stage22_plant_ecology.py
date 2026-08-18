# -*- coding: utf-8 -*-
"""
M5 阶段 22：用户的进化实验扩展——食物 → 植物（复制性 + 传播性-非接触复制）
stage21 升级：随机食物 → 自组织植物生态

植物：位置 + 年龄
  复制性：成熟后（age>阈值）在附近产生新植物（局部扩散）
  传播性：成熟后定期"播撒种子"到远处（非接触复制——风传播——远处殖民）
  被吃：生物接触 → 吃掉（饱食度+）→ 植物消失
生物：同 stage21（速度×体积 trade-off / 饱食度 / 捕食）

实验 1：植物生态基线（植物-生物共存——植物不被吃光——自持）
实验 2：植物扩散（传播性 → 空间分布——远处殖民）
实验 3：植物-生物动力学（捕食者-猎物振荡？Lotka-Volterra 类）
实验 4：对比随机食物（同初始量——自持 vs 注入）
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(101)
W, H = 100.0, 100.0
CONTACT = 2.0
PLANT_REWARD = 8.0
PLANT_MATURE = 10      # 成熟年龄
PLANT_REPRO = 0.08     # 附近复制概率（成熟后每步）
PLANT_SPREAD = 0.04    # 远处传播概率（非接触复制）
MAX_PLANTS = 400       # 植物数量上限（防爆炸——生态容量）
REPRO_TH = 120.0
MAX_POP = 500

def tradeoff(volume):
    return np.clip(1.5 - 0.8 * volume, 0.1, 1.5)

def decay_rate(volume):
    return 0.4 + 0.35 * volume

class PlantWorld:
    def __init__(self, n0=30, plants0=50, random_food=False, food_rate=0.08):
        self.random_food = random_food
        self.food_rate = food_rate
        self.n = n0
        self.x = RNG.uniform(0, W, n0)
        self.y = RNG.uniform(0, H, n0)
        self.volume = np.full(n0, 0.9) * RNG.uniform(0.9, 1.1, n0)
        self.speed = tradeoff(self.volume)
        self.satiety = np.full(n0, 60.0)
        self.alive = np.ones(n0, dtype=bool)
        # 植物（或随机食物）
        self.px = RNG.uniform(0, W, plants0) if not random_food else np.array([])
        self.py = RNG.uniform(0, H, plants0) if not random_food else np.array([])
        self.page = np.zeros(len(self.px)) if not random_food else np.array([])
        self.fx, self.fy = ([], []) if random_food else (None, None)
        self.history = []

    def step(self):
        alive = np.where(self.alive)[0]
        if len(alive) == 0:
            # 植物继续生长
            if not self.random_food:
                self.page += 1
                self.plant_grow()
            return
        # ---- 决策（同 stage21） ----
        urge_full = np.zeros(len(self.satiety))
        urge_full[alive] = 1.0 + (1.0 - np.clip(self.satiety[alive] / 100, 0, 1)) * 2.0
        targets = []
        # 植物目标（或随机食物）
        if self.random_food:
            foods = list(zip(self.fx, self.fy))
        else:
            foods = list(zip(self.px, self.py))
        for fx, fy in foods:
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
                    reward = self.volume[j] * 10.0
                    targets.append((i, self.x[j], self.y[j], reward / d * urge_full[i], "prey", j))
        best = {}
        for t in targets:
            if t[0] not in best or t[3] > best[t[0]][3]:
                best[t[0]] = t
        eaten = set()
        for i, (i0, tx, ty, val, kind, *rest) in best.items():
            dx, dy = tx - self.x[i], ty - self.y[i]
            d = np.hypot(dx, dy)
            if d > 1e-6:
                step = self.speed[i]
                self.x[i] = np.clip(self.x[i] + dx / d * step, 0, W)
                self.y[i] = np.clip(self.y[i] + dy / d * step, 0, H)
            if kind == "food" and d < CONTACT:
                self.satiety[i] += PLANT_REWARD
                if self.random_food:
                    if (self.x[i], self.y[i]) in zip(self.fx, self.fy):
                        idx = list(zip(self.fx, self.fy)).index((self.x[i], self.y[i]))
                        self.fx.pop(idx); self.fy.pop(idx)
                else:
                    # 吃植物（移除——按最近匹配）
                    if len(self.px) > 0:
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
        # ---- 衰减/死亡/繁殖 ----
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
        # ---- 植物生长（复制 + 传播——非接触复制） ----
        if not self.random_food:
            self.plant_grow()
        else:
            if RNG.random() < self.food_rate:
                self.fx.append(RNG.uniform(0, W))
                self.fy.append(RNG.uniform(0, H))
        self.history.append((np.sum(self.alive), len(self.px) if not self.random_food else len(self.fx),
                             np.mean(self.volume[self.alive]) if np.any(self.alive) else 0,
                             np.mean(self.speed[self.alive]) if np.any(self.alive) else 0))

    def plant_grow(self):
        """植物：成熟 → 复制（附近）+ 传播（远处——非接触复制）"""
        if len(self.px) == 0:
            return
        self.page += 1
        mature = np.where(self.page > PLANT_MATURE)[0]
        new_x, new_y, new_age = [], [], []
        for idx in mature:
            # 复制性：附近产生（概率）
            if RNG.random() < PLANT_REPRO:
                new_x.append(np.clip(self.px[idx] + RNG.uniform(-10, 10), 0, W))
                new_y.append(np.clip(self.py[idx] + RNG.uniform(-10, 10), 0, H))
                new_age.append(0)
            # 传播性：远处产生（非接触复制——种子风传播）
            if RNG.random() < PLANT_SPREAD:
                ang = RNG.uniform(0, 2*np.pi)
                dist = RNG.uniform(10, 50)
                new_x.append(np.clip(self.px[idx] + dist*np.cos(ang), 0, W))
                new_y.append(np.clip(self.py[idx] + dist*np.sin(ang), 0, H))
                new_age.append(0)
        if new_x:
            # 容量上限（生态容量——防指数爆炸）
            room = MAX_PLANTS - len(self.px)
            if room > 0:
                new_x, new_y, new_age = new_x[:room], new_y[:room], new_age[:room]
                self.px = np.append(self.px, new_x)
                self.py = np.append(self.py, new_y)
                self.page = np.append(self.page, new_age)

    def plant_spread_radius(self):
        """植物分布半径（空间扩散度量——传播性）"""
        if len(self.px) == 0:
            return 0.0
        cx, cy = np.mean(self.px), np.mean(self.py)
        return np.mean(np.hypot(self.px - cx, self.py - cy))

    def run(self, T):
        for _ in range(T):
            self.step()
        return np.array(self.history)

def run():
    print("=== 植物生态（复制性 + 传播性——非接触复制） ===\n")

    # ---- 实验 1：植物-生物共存 ----
    print("[exp1] 植物生态基线（500 步——植物不被吃光？）:")
    w = PlantWorld(plants0=50)
    h = w.run(500)
    bio, plants = h[-1, 0], h[-1, 1]
    p_hist = h[:, 1]
    print(f"  500 步后: 生物 {bio:.0f} | 植物 {plants:.0f}（初始 50）")
    print(f"  植物存活 = {'✓ 植物生态自持（复制+传播 > 被吃——食物链平衡）' if plants > 20 else '✗ 植物被吃光（灭绝）'}")

    # ---- 实验 2：植物扩散（传播性） ----
    print("\n[exp2] 植物扩散（传播性——空间殖民）:")
    w2 = PlantWorld(plants0=20)
    radii = []
    for t in range(400):
        w2.step()
        if t % 80 == 0:
            r = w2.plant_spread_radius()
            radii.append(r)
            print(f"  t={t}: 植物 {len(w2.px)} 株 | 扩散半径 {r:.1f}")
    spread = radii[-1] > radii[0] * 1.5
    print(f"  = {'✓ 传播性生效（非接触复制 → 远处殖民——空间扩散）' if spread else '需检查'}")

    # ---- 实验 3：植物-生物动力学 ----
    print("\n[exp3] 植物-生物动力学（捕食者-猎物振荡？）:")
    w3 = PlantWorld(plants0=80, n0=40)
    h3 = w3.run(600)
    bio3, plant3 = h3[:, 0], h3[:, 1]
    # 振荡检测（末段标准差 vs 均值——动态性）
    std_b = np.std(bio3[-300:])
    mean_b = np.mean(bio3[-300:])
    print(f"  生物末段: 均值 {mean_b:.0f} 标准差 {std_b:.0f}"
          f"（{'✓ 动态波动（捕食-资源反馈——Lotka-Volterra 类）' if std_b > mean_b * 0.2 else '稳定（平衡）'}）")
    print(f"  植物末段: 均值 {np.mean(plant3[-300:]):.0f}（{'共存' if np.mean(plant3[-300:]) > 10 else '近灭绝'}）")

    # ---- 实验 4：对比随机食物 ----
    print("\n[exp4] 植物 vs 随机食物（同初始 50——400 步）:")
    wp = PlantWorld(plants0=50)
    hp = wp.run(400)
    wr = PlantWorld(plants0=0, random_food=True, food_rate=0.15)
    hr = wr.run(400)
    print(f"  植物生态: 生物 {hp[-1,0]:.0f} | 食物源 {hp[-1,1]:.0f}（自持——内生）")
    print(f"  随机食物: 生物 {hr[-1,0]:.0f} | 食物 {hr[-1,1]:.0f}（注入——外生）")
    verdict = "✓ 植物生态是可持续的（食物供应由生态自身维持）" if hp[-1, 1] > 10 else "✗"
    print(f"  = 植物 = 内生食物链（自组织）vs 随机 = 外生供给（依赖注入）——{verdict}")

    # ---- 图 ----
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(h[:, 0], label="animals")
    axes[0].plot(h[:, 1], "--", label="plants")
    axes[0].set_title("Exp1: plant-animal coexistence")
    axes[0].legend(fontsize=8)
    axes[1].plot(radii, "o-")
    axes[1].set_title("Exp2: plant spread (non-contact)")
    axes[1].set_xlabel("time")
    axes[2].plot(h3[:, 0], label="animals")
    axes[2].plot(h3[:, 1], "--", label="plants")
    axes[2].set_title("Exp3: predator-prey dynamics")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("fig_stage22.png", dpi=110)
    print("\n[plot] saved fig_stage22.png")
    print("[done] stage22 plant ecology complete")

if __name__ == "__main__":
    run()
