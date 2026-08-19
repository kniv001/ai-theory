# -*- coding: utf-8 -*-
"""
M5 阶段 26（用户路线重构）：单细胞进化训练——环境简单化 + 每生物独立神经文件

用户架构（"像单细胞生物"）：
  每生物 = 独立神经文件（bio_N.npy——网络权重档案）
  训练脚本 = 环境（简单——食物+能量+繁殖） + 演化（变异/选择/文件管理）
  决策 = 生物文件里的网络前向（训练脚本不做监督——只有环境和演化）

环境（单细胞级）：
  平面 100×100 / 食物能量点（随机生成+保底）/ 生物游动（速度=网络输出）
  生物：位置/能量/朝向（基础参数）——无基因参数（网络权重就是一切）
  能量：食物碰撞 +20 / 消耗 0.3/步 + 移动 0.1×speed / 繁殖阈值 150（分裂：能量减半）
  死亡：能量 ≤ 0

网络（趋化控制器）：
  输入 6：[最近食物方向 sin, cos, 距离/70, 能量/150, 威胁方向 sin（可无）, 噪声]
  隐藏 16（tanh）→ 输出 2（移动方向 sin/cos——速度 0.8）
  权重 6×16+16+16×2+2 = 146

演化：
  繁殖：变异（高斯扰动 σ=0.15 权重 ±10% 比例扰动）→ 写新文件 bio_N.npy
  死亡：删文件
  训练脚本只做这些——网络内部（学习）是进化的黑盒
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(109)
W, H = 100.0, 100.0
FOOD_R = 2.0          # 食物半径（碰撞距离）
FOOD_ENERGY = 20.0    # 食物能量
FOOD_N0 = 100         # 初始食物（60→100——相遇率高——碰巧率↑——进化有梯度）
FOOD_SPAWN = 0.25     # 每步保底生成概率
MAX_FOOD = 250
BASE_COST = 0.2       # 基础消耗/步（0.3→0.2——活得久——繁殖机会多）
MOVE_COST = 0.1       # 移动消耗系数
SPEED = 1.0           # 移动速度（0.8→1.0）
REPRO_E = 100.0       # 繁殖阈值（150→100——繁殖快——世代多——变异积累快）
N_IN = 6
N_HID = 16

def init_weights(rng):
    """趋化先验初始化（"像单细胞生物"）：单细胞天生会趋化（生化本能——受体-鞭毛耦合）——
    进化优化的不是'会不会趋化'而是趋化细节（灵敏度/远近权衡/能量状态调制）。
    结构：隐藏单元 0 读 sin(食物方向)、单元 1 读 cos → 输出复制为移动方向——
    其余权重小噪声（σ 0.05）——纯随机冷启动在高维空间无梯度（146 权重随机搜索命中率≈0）"""
    w = np.zeros(N_IN * N_HID + N_HID + N_HID * 2 + 2)
    w[0 * N_HID + 0] = 1.5      # 输入 sin → 隐藏 0
    w[1 * N_HID + 1] = 1.5      # 输入 cos → 隐藏 1
    w[N_IN * N_HID + N_HID + 0 * 2 + 0] = 1.5   # 隐藏 0 → 输出 sin
    w[N_IN * N_HID + N_HID + 1 * 2 + 1] = 1.5   # 隐藏 1 → 输出 cos
    w += rng.normal(0, 0.05, len(w))   # 小噪声（趋化细节的进化起点）
    return w

def forward(w, x):
    W1 = w[:N_IN * N_HID].reshape(N_IN, N_HID)
    b1 = w[N_IN * N_HID:N_IN * N_HID + N_HID]
    W2 = w[N_IN * N_HID + N_HID:-2].reshape(N_HID, 2)
    b2 = w[-2:]
    h = np.tanh(x @ W1 + b1)
    return h @ W2 + b2

def mutate(w, rng):
    """繁殖变异（进化学习）：比例扰动 + 重初始化（探索更激进——随机权重时代需要大步探索）"""
    out = w.copy()
    out += rng.normal(0, 0.2, len(w)) * np.abs(w) + 0.03
    if rng.random() < 0.1:
        idx = rng.choice(len(w), max(1, len(w) // 4), replace=False)
        out[idx] = rng.normal(0, 0.6, len(idx))
    return out

class EvoWorld:
    """训练脚本：环境 + 演化——生物是文件"""
    def __init__(self, workdir="stage26_evo", n0=40, seed=109):
        self.workdir = workdir
        os.makedirs(workdir, exist_ok=True)
        # 清理旧文件
        for f in os.listdir(workdir):
            os.remove(os.path.join(workdir, f))
        global RNG
        RNG = np.random.default_rng(seed)
        self.fx = RNG.uniform(0, W, FOOD_N0)
        self.fy = RNG.uniform(0, H, FOOD_N0)
        self.bios = {}     # id -> [x, y, energy]
        self.weights = {}  # id -> 权重（内存缓存——文件是持久化档案）
        self.next_id = 0
        for _ in range(n0):
            self._spawn_new()
        print(f"  初始: {n0} 生物 / {FOOD_N0} 食物 / 繁殖阈 {REPRO_E:.0f} / 变异 σ0.2+1/4重初始化")
        self.history = []
        self.eats = 0
        self.repros = 0

    def _spawn_new(self):
        """新生物（初始或繁殖）——写文件"""
        bid = self.next_id
        self.next_id += 1
        w = init_weights(RNG)
        self.weights[bid] = w
        np.save(f"{self.workdir}/bio_{bid}.npy", w)
        self.bios[bid] = [RNG.uniform(0, W), RNG.uniform(0, H),
                          RNG.uniform(60, 100)]
        return bid

    def _reproduce(self, parent_id):
        """繁殖（分裂）：变异父权重 → 新文件——能量减半"""
        bid = self.next_id
        self.next_id += 1
        w = mutate(self.weights[parent_id], RNG)
        self.weights[bid] = w
        np.save(f"{self.workdir}/bio_{bid}.npy", w)
        px, py, pe = self.bios[parent_id]
        self.bios[bid] = [np.clip(px + RNG.uniform(-5, 5), 0, W),
                          np.clip(py + RNG.uniform(-5, 5), 0, H), pe / 2.0]
        self.bios[parent_id][2] = pe / 2.0
        self.repros += 1
        return bid

    def _kill(self, bid):
        """死亡：删文件"""
        del self.weights[bid]
        del self.bios[bid]
        try:
            os.remove(f"{self.workdir}/bio_{bid}.npy")
        except OSError:
            pass

    def step(self):
        # ---- 环境：食物生成（保底） ----
        if len(self.fx) < MAX_FOOD and RNG.random() < FOOD_SPAWN:
            self.fx = np.append(self.fx, RNG.uniform(0, W))
            self.fy = np.append(self.fy, RNG.uniform(0, H))
        if len(self.bios) == 0:
            self.history.append((0, len(self.fx), 0))
            return
        # ---- 感知 → 网络 → 移动（每生物独立决策） ----
        for bid in list(self.bios.keys()):
            x, y, e = self.bios[bid]
            # 感知：最近食物
            if len(self.fx):
                ds = np.hypot(self.fx - x, self.fy - y)
                idx = int(np.argmin(ds))
                fd = ds[idx]
                ang = np.arctan2(self.fy[idx] - y, self.fx[idx] - x)
            else:
                fd, ang = 70.0, RNG.uniform(0, 2*np.pi)
            inp = np.array([np.sin(ang), np.cos(ang), fd / 70.0,
                            e / 150.0, RNG.uniform(-1, 1) * 0.1, 1.0])
            out = forward(self.weights[bid], inp)
            # 移动（速度由网络输出调制）——out[0]=sin(y 分量)/out[1]=cos(x 分量)
            norm = max(np.hypot(out[0], out[1]), 1e-6)
            spd = SPEED * np.clip(norm, 0.3, 1.5)
            nx = np.clip(x + out[1] / norm * spd, 0, W)
            ny = np.clip(y + out[0] / norm * spd, 0, H)
            # 能量
            e -= BASE_COST + MOVE_COST * spd
            # 吃食物（碰撞）
            if len(self.fx):
                ds = np.hypot(self.fx - nx, self.fy - ny)
                hit = np.where(ds < FOOD_R)[0]
                if len(hit):
                    e += FOOD_ENERGY * len(hit)
                    self.eats += len(hit)
                    keep = np.ones(len(self.fx), dtype=bool)
                    keep[hit] = False
                    self.fx, self.fy = self.fx[keep], self.fy[keep]
            self.bios[bid] = [nx, ny, e]
        # ---- 演化：死亡 / 繁殖 ----
        for bid in list(self.bios.keys()):
            if self.bios[bid][2] <= 0:
                self._kill(bid)
            elif self.bios[bid][2] > REPRO_E:
                self._reproduce(bid)
        self.history.append((len(self.bios), len(self.fx), self.eats))

    def run(self, T):
        for _ in range(T):
            self.step()
        return np.array(self.history)

def run():
    print("=== M5 阶段 26：单细胞进化训练（环境+演化——生物=神经文件） ===\n")
    print("架构：训练脚本=环境+演化 | 每生物=bio_N.npy（网络权重档案） | 决策=文件内网络\n")

    # ---- 主实验：进化训练 1500 步 ----
    w = EvoWorld(workdir="stage26_evo", n0=40)
    h = w.run(1500)
    for t in range(0, 1500, 150):
        print(f"t={t:4d} 生物={h[t,0]:3.0f} 食物={h[t,1]:3.0f} 累计吃={h[t,2]:4.0f}")
    live = np.mean(h[500:, 0] > 5)
    print(f"\n[结果] 种群自持（500 步后 >5 只比例）: {100*live:.0f}%")
    print(f"[文件] 存活生物权重档案: {len([f for f in os.listdir('stage26_evo') if f.endswith('.npy')])} 个 bio_*.npy")

    # ---- 图 ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(h[:, 0], label="population")
    axes[0].plot(h[:, 1] / 10, label="food/10")
    axes[0].set_title("Evolution training (single-cell)")
    axes[0].legend(fontsize=8)
    axes[1].plot(np.diff(h[:, 2]), label="eats/step")
    axes[1].set_title("Food intake rate")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("fig_stage26.png", dpi=110)
    print("\n[plot] saved fig_stage26.png")
    print("[done] stage26 evo training")

if __name__ == "__main__":
    run()
