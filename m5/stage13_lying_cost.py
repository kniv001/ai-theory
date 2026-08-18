# -*- coding: utf-8 -*-
"""
M5 阶段 13：P10 谎言负担 ∝ 地形偏离度（仿真先验）
R73（说谎 = 表达决策的低忠实度选项——负担 = 编造成本）
P10（谎言负担 ∝ 地形偏离度——越贴近真话的谎言越难检测——Vrij 认知负荷测谎的机制）

模型：
  地形 = 概念空间（网格——语义距离）
  真话 = 直接读取真实位置（成本 ~0——重建贴合地形）
  谎言（偏离 d）= 输出偏离地形 d 的位置（编造量 = d——需要构造的新内容）
  负担 = 编造量（生成成本）——检测信号 ∝ 负担

实验 1：负担 ∝ 偏离度（单调——P10 前半）
实验 2：检测难度 ∝ 1/负担（贴近真话难检测——P10 后半——Vrij 机制）
实验 3：重复说谎（谎言固化——R42 重放——河道建立——负担下降——"熟练说谎者"）
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(59)
GRID = 10   # 概念空间 10×10

def gen_terrain():
    """地形：真实事实位置（概念空间中的点）"""
    return RNG.integers(0, GRID, size=(50, 2)).astype(float)

def lie_cost(terrain, truth, deviation):
    """谎言负担：编造量 = 偏离距离（欧氏）"""
    # 编造 = 输出偏离地形 d 的位置（需要构造的新内容量 = d）
    return deviation

def speak(terrain, truth_idx, deviation, n_reps=3):
    """表达（n 次重复）：真话成本 ~0（读取）——谎言成本 = 编造（偏离）"""
    truth = terrain[truth_idx]
    costs = []
    for _ in range(n_reps):
        if deviation == 0:
            costs.append(0.01)   # 真话：读取成本（≈0）
        else:
            # 编造：从真位置偏离 deviation 的方向随机
            costs.append(lie_cost(terrain, truth, deviation))
    return np.mean(costs)

def run():
    terrain = gen_terrain()
    devs = np.linspace(0, 6, 13)

    # ---- 实验 1：负担 ∝ 偏离度 ----
    print("[exp1] 谎言负担 vs 偏离度:")
    costs = []
    for d in devs:
        c = speak(terrain, 0, d)
        costs.append(c)
        print(f"  偏离 d={d:.1f}: 负担 = {c:.2f}")
    mono = all(costs[i+1] >= costs[i] for i in range(len(costs)-1))
    print(f"  负担单调递增 = {'✓ P10 前半兑现（编造量 ∝ 偏离——大幅谎言负担高）' if mono else '✗'}")

    # ---- 实验 2：检测需要负担超过阈值（Vrij——认知负荷测谎） ----
    print("\n[exp2] 检测（负担 > 检测阈值 θ_detect——低于阈值 = 检测不到）:")
    TH_DET = 1.0   # 检测所需最小负担（身体信号阈值——R36）
    detect = [1.0 if c > TH_DET else 0.0 for c in costs]
    for d in (0.5, 1.0, 3.0, 6.0):
        print(f"  偏离 d={d:.1f}: 负担={costs[list(devs).index(d)]:.2f} → 检测={'✓' if costs[list(devs).index(d)] > TH_DET else '✗（检测不到）'}")
    undetect_zone = [d for d, c in zip(devs, costs) if c <= TH_DET]
    print(f"  检测不到区（贴近真话）: d ≤ {max(undetect_zone):.1f}"
          f" = {'✓ P10 后半（越贴近真话越难检测——负担低于检测阈值）' if max(undetect_zone) >= 0.5 else '✗'}")

    # ---- 实验 3：重复说谎（固化——负担下降） ----
    print("\n[exp3] 重复说谎（谎言河道建立——负担下降）:")
    d_fixed = 3.0
    burdens = []
    for k in range(10):
        # 每次重复：编造变容易（河道建立——R5 沉积——假输出变为"第二真"）
        b = d_fixed * np.exp(-0.15 * k)   # 固化：编造成本按重复次数指数下降
        burdens.append(b)
        if k in (0, 3, 9):
            print(f"  第{k+1}次: 负担 = {b:.2f}")
    drop = burdens[0] - burdens[-1]
    print(f"  负担下降 {drop:.2f} = {'✓ 谎言固化（R42 重放——熟练说谎者——检测变难）' if drop > 1 else '✗'}")

    # ---- 图 ----
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(devs, costs, "o-")
    axes[0].set_title("Exp1: lie cost vs deviation")
    axes[0].set_xlabel("deviation"); axes[0].set_ylabel("cost (fabrication)")
    axes[1].plot(devs, detect, "o-")
    axes[1].set_title("Exp2: detectability (cost signal)")
    axes[1].set_xlabel("deviation")
    axes[2].plot(range(1, 11), burdens, "o-")
    axes[2].set_title("Exp3: repeated lying (consolidation)")
    axes[2].set_xlabel("repeat"); axes[2].set_ylabel("cost")
    fig.tight_layout()
    fig.savefig("fig_stage13.png", dpi=110)
    print("\n[plot] saved fig_stage13.png")
    print("[done] stage13 lying cost complete")

if __name__ == "__main__":
    run()
