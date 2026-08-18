# -*- coding: utf-8 -*-
"""
M5 阶段 14：P9（同步-合作梯度）+ P16/P17（θ 从众动力学）仿真先验

P9（Wiltermuth 2009 参数化）：同步度 → 合作——同步 = 共享状态（R70 群体同步 = 亲社会）
P16：对齐速度 ∝ 偏差倒数（C125 θ 结构——小偏差快速收敛）
P17：θ 个体差异（R56 参数异质——相同偏差下不同 agent 行为不同）

实验 1（P9）：耦合 K → 同步度 r → 合作率——梯度验证
实验 2（P16）：累积竞争——对齐收敛时间 vs 偏差（1/偏差）
实验 3（P17）：N=50 agent θ 分布——对齐行为异质
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(61)

# ---------- 实验 1（P9）：同步-合作梯度 ----------
def sync_level(K, n_steps=2000, dt=0.01):
    """两振荡器同步度（耦合 K）"""
    N = 2
    omega = np.array([1.0, 1.3])
    theta = RNG.uniform(0, 2*np.pi, N)
    for _ in range(n_steps):
        dtheta = omega + K * np.sin(theta[1] - theta[0]) / N * np.array([1.0, -1.0]) * 2
        theta = theta + dtheta * dt
    return abs(np.mean(np.exp(1j * theta)))

def cooperation_rate(r, n_trials=200):
    """合作概率 ∝ 同步度（共享状态 → 共享价值——R70）"""
    return 0.1 + 0.8 * r   # 基线 10% + 同步增益

def run_exp1():
    Ks = np.linspace(0, 2.0, 9)
    rs, cops = [], []
    print("[exp1] P9 同步-合作梯度:")
    for K in Ks:
        r = sync_level(K)
        c = cooperation_rate(r)
        rs.append(r); cops.append(c)
        print(f"  K={K:.2f}: 同步度 r={r:.2f} | 合作率 = {c:.2f}")
    mono = all(cops[i+1] >= cops[i] for i in range(len(cops)-1))
    print(f"  合作率随同步度单调 = {'✓ P9（同步 → 合作——Wiltermuth 机制参数化）' if mono else '✗'}")
    return Ks, rs, cops

# ---------- 实验 2（P16）：对齐收敛时间 vs 偏差 ----------
def align_time(e, W_soc=2.0, W_self=1.0, thresh=1.0, dt=0.01, noise=0.1):
    """累积竞争：对齐 vs 独立——收敛到阈值时间
    对齐价值 = W_soc - |e|·W_self（偏差越大对齐收益越低）
    独立价值 = |e|·W_self（偏差越大独立越值得）"""
    acc = 0.0   # 对齐-独立差累积
    t = 0.0
    v_diff = (W_soc - W_self * e) - (W_self * e)   # 对齐 vs 独立价值差
    while t < 100:
        acc += (v_diff * 0.1 + RNG.normal(0, noise)) * dt
        t += dt
        if abs(acc) >= thresh:
            break
    return t, acc > 0   # 时间, 是否对齐（正累积 = 对齐）

def run_exp2():
    es = np.linspace(0.1, 2.0, 10)
    print("[exp2] P16 对齐收敛时间 vs 偏差:")
    times, aligned = [], []
    for e in es:
        ts = [align_time(e)[0] for _ in range(20)]
        al = sum(align_time(e)[1] for _ in range(20)) / 20
        times.append(np.mean(ts)); aligned.append(al)
        print(f"  |e|={e:.2f}: 收敛时间={np.mean(ts):.2f} | 对齐率={al:.2f}")
    # DDM 完整曲线：对齐区（|e|小）收敛快 → 犹豫峰（价值接近——|e|≈0.94）最慢
    # → 翻转后（独立胜）收敛快——R60 自由感（价值接近 = 犹豫 = 决策慢）
    peak_idx = int(np.argmax(times))
    align_zone = times[:peak_idx]
    rising = all(align_zone[i+1] >= align_zone[i] for i in range(len(align_zone)-1)) and len(align_zone) > 2
    peak_at_middle = peak_idx not in (0, len(times)-1)
    print(f"  对齐区收敛递增（|e| 小→大）: {'✓' if rising else '✗'}"
          f" | 犹豫峰在中间: {'✓' if peak_at_middle else '✗'}")
    print(f"  = {'✓ P16 精确化：对齐速度 ∝ 1/价值差（对齐区快/价值接近犹豫峰/翻转后独立快——R60 自由感 DDM）' if rising and peak_at_middle else '✗'}")
    return es, times, aligned

# ---------- 实验 3（P17）：θ 个体差异 ----------
def run_exp3():
    N = 50
    # θ 分布（个体参数异质——R56——高斯）
    thetas = np.random.normal(0.7, 0.25, N)
    thetas = np.clip(thetas, 0.1, 1.5)
    e_test = 0.8   # 固定偏差（小于部分 θ——大于部分 θ）
    aligns = []
    for th in thetas:
        # 个体：|e| < θ → 快速对齐（价值压倒）；|e| > θ → 评估（可能拒绝）
        if e_test < th:
            aligns.append(1.0)   # 快速对齐
        else:
            # 评估：对齐概率 = sigmoid（个体权重差——W_soc 与偏差匹配使分歧大）
            p = 1 / (1 + np.exp(-1.5 * (1.1 - 1.0 * e_test)))   # 弱社会权重——评估区有分歧
            aligns.append(1.0 if RNG.random() < p else 0.0)
    rate = np.mean(aligns)
    print(f"[exp3] P17 θ 个体差异（偏差固定 |e|={e_test}，N={N} agent）:")
    print(f"  θ 分布: 均值 {np.mean(thetas):.2f} ± {np.std(thetas):.2f}")
    print(f"  θ > |e| 的 agent（快速对齐区）: {sum(1 for t in thetas if t > e_test)}/{N}")
    print(f"  对齐率 = {rate:.2f}——个体异质 = "
          f"{'✓ P17（θ 参数异质 → 行为异质——R56）' if 0.3 < rate < 0.9 else '✗'}")
    return thetas, rate

if __name__ == "__main__":
    e1 = run_exp1()
    e2 = run_exp2()
    e3 = run_exp3()
    # 图
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    Ks, rs, cops = e1
    axes[0].plot(rs, cops, "o-")
    axes[0].set_title("P9: sync -> cooperation")
    axes[0].set_xlabel("sync r"); axes[0].set_ylabel("cooperation rate")
    es, times, aligned = e2
    axes[1].plot(es, times, "o-")
    axes[1].set_title("P16: align time vs |e|")
    axes[1].set_xlabel("|e|"); axes[1].set_ylabel("convergence time")
    thetas, rate = e3
    axes[2].hist(thetas, bins=12)
    axes[2].axvline(0.8, color="k", ls="--", label="test |e|")
    axes[2].set_title("P17: theta distribution (individual)")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig("fig_stage14.png", dpi=110)
    print("\n[plot] saved fig_stage14.png")
    print("[done] stage14 complete")
