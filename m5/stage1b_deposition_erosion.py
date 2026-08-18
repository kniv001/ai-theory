# -*- coding: utf-8 -*-
"""
M5 阶段 1b：沉积-侵蚀仿真（缓存-写回模型）
验证 C96/C97（软-硬耦合动力学）与 P1（遗忘曲线三阶段）

模型（C97）：
  软区（缓存 S）：dS/dt = -λ_s·S + 学习输入（学习/复习事件 S += ΔS）
  硬区（结构 H）：dH/dt = k·S - λ_h·H （写回速率 ∝ 软区存量；慢蚀）
  总记忆强度 M = S + H

实验 1：一次学习后的遗忘曲线 —— 验证 P1 三阶段
实验 2：间隔 vs 集中复习 —— 验证间隔效应（S 满时写回饱和）
实验 3：写回速率 ∝ 软区存量 —— 验证 dH/dt = k·S
实验 4：复习时机（S 降阈值）—— Anki/SM-2 原理验证

输出：控制台摘要 + fig_stage1b.png + npz
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAM_S = 0.5      # 软区衰减（/天——半衰期 ~1.4 天——艾宾浩斯快段）
LAM_H = 0.01     # 硬区慢蚀（/天——结构冻结 R40——半衰期 ~69 天）
K_WB = 0.30      # 写回速率（/天）
DELTA_S = 1.0    # 单次学习/复习写入
DT = 0.01
T_MAX = 40.0

def simulate(learning_events, t_max=T_MAX, dt=DT, lam_s=LAM_S, lam_h=LAM_H, k=K_WB, delta=DELTA_S):
    """学习事件列表 [(t, ΔS), ...] → S/H/M 轨迹"""
    n = int(t_max / dt) + 1
    t = np.linspace(0, t_max, n)
    S = np.zeros(n); H = np.zeros(n)
    ev_idx = 0
    # t=0 事件（起点生效）
    while ev_idx < len(learning_events) and learning_events[ev_idx][0] <= t[0]:
        S[0] += learning_events[ev_idx][1]
        ev_idx += 1
    for i in range(1, n):
        # 学习事件（该时刻写入）
        while ev_idx < len(learning_events) and learning_events[ev_idx][0] <= t[i]:
            S[i] += learning_events[ev_idx][1]
            ev_idx += 1
        # 动力学
        S[i] += (-lam_s * S[i-1]) * dt
        H[i] += (k * S[i-1] - lam_h * H[i-1]) * dt
        S[i] += S[i-1]; H[i] += H[i-1]
    return t, S, H, S + H

# ---------- 实验 1：一次学习后的遗忘曲线 ----------
def run_exp1():
    t, S, H, M = simulate([(0.0, DELTA_S)])
    # 三阶段斜率（半对数：log M vs t 的分段斜率）
    logM = np.log(np.maximum(M, 1e-9))
    # 分三段拟合斜率：早期(0-2d) 中期(2-10d) 后期(10-40d)
    def slope(ta, tb):
        m = (t >= ta) & (t <= tb)
        if m.sum() < 2: return float("nan")
        return np.polyfit(t[m], logM[m], 1)[0]
    s1, s2, s3 = slope(0, 2), slope(2, 10), slope(10, 40)
    # 硬区占比
    h_frac = H[-1] / max(M[-1], 1e-12)
    print(f"[exp1] 遗忘三阶段斜率(半对数): 早期={s1:.4f}/中期={s2:.4f}/后期={s3:.4f} (/天)")
    print(f"       |s1|>|s2|>|s3| 递减 = {'✓ 三阶段成立' if abs(s1) > abs(s2) > abs(s3) else '✗ 需检查'}")
    print(f"       H 占比(40天)={h_frac:.2%} | M(40天)={M[-1]:.3f}")
    return t, S, H, M

# ---------- 实验 2：间隔 vs 集中复习 ----------
def run_exp2():
    # 集中：连续三天复习（S 满时——写回饱和）
    massed = [(0.0, DELTA_S), (0.5, DELTA_S), (1.0, DELTA_S), (1.5, DELTA_S)]
    # 间隔：S 降后复习
    spaced = [(0.0, DELTA_S), (3.0, DELTA_S), (7.0, DELTA_S), (12.0, DELTA_S)]
    t, S1, H1, M1 = simulate(massed)
    t, S2, H2, M2 = simulate(spaced)
    print(f"[exp2] 间隔效应: 集中 H(40d)={H1[-1]:.3f} | 间隔 H(40d)={H2[-1]:.3f} | 间隔/集中={H2[-1]/H1[-1]:.2f}x")
    print(f"       间隔 > 集中 = {'✓' if H2[-1] > H1[-1] else '✗'}")
    return t, S1, H1, S2, H2

# ---------- 实验 3：写回速率 ∝ 软区存量 ----------
def run_exp3():
    # 稳态 S 水平不同 → H 增长率不同（固定 S 维持：用连续小复习维持 S≈常数）
    results = {}
    for s_level in (0.5, 1.0, 2.0):
        # 周期补充维持 S 近似恒定
        events = []
        tt = 0.0
        while tt < 20.0:
            events.append((tt, s_level * (1 - np.exp(-LAM_S * 2.0))))  # 每 2 天补到 s_level 附近
            tt += 2.0
        t, S, H, M = simulate(events)
        # 稳态 H 增长率（线性拟合 5-20 天）
        m = (t >= 5) & (t <= 20)
        slope = np.polyfit(t[m], H[m], 1)[0]
        results[s_level] = (slope, np.mean(S[m]))
        print(f"[exp3] S≈{s_level:.1f}: H增长率={slope:.4f}/天 | 比值(slope/S)={slope/s_level:.3f} (≈k={K_WB}?)")
    ratio = [results[s][0] / results[s][1] for s in results]
    print(f"       slope/S 一致性 = {'✓ k≈常数（dH/dt=k·S 成立）' if max(ratio) - min(ratio) < 0.05 else '✗ 需检查'}")
    return results

# ---------- 实验 4：复习时机（S 降阈值） ----------
def run_exp4():
    # 阈值复习：S 降到 0.3 时复习 vs 固定间隔复习（同次数）
    def threshold_events():
        evs = [(0.0, DELTA_S)]
        # 每次复习后模拟直到 S 降到阈值
        s_now = DELTA_S
        tt = 0.0
        while tt < 25.0:
            # S 衰减到 0.3 的时间
            dt_ = np.log(s_now / 0.3) / LAM_S
            tt += dt_
            if tt > 25.0: break
            evs.append((tt, DELTA_S))
            s_now = DELTA_S
        return evs

    thr_evs = threshold_events()
    # 固定间隔（同复习次数——取阈值方案的次数，均匀分布）
    n_rev = len(thr_evs) - 1
    fixed_evs = [(0.0, DELTA_S)] + [((i + 0.5) * (25.0 / n_rev), DELTA_S)
                                    for i in range(n_rev) if (i + 0.5) * (25.0 / n_rev) < 25.0]
    t, S1, H1, M1 = simulate(thr_evs)
    t, S2, H2, M2 = simulate(fixed_evs)
    print(f"[exp4] 复习时机: 阈值复习({len(thr_evs)-1}次) H(25d)={H1[-1]:.3f} | 固定间隔({len(fixed_evs)-1}次) H(25d)={H2[-1]:.3f}")
    print(f"       阈值 > 固定 = {'✓（Anki 原理验证）' if H1[-1] > H2[-1] else '✗'}")
    return t, S1, H1, S2, H2

# ---------- 绘图 ----------
def plot_all(r1, r2, r4):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    t, S, H, M = r1
    axes[0, 0].semilogy(t, M, label="M=S+H (total)")
    axes[0, 0].semilogy(t, S, "--", label="S (cache)")
    axes[0, 0].semilogy(t, H, ":", label="H (structure)")
    axes[0, 0].set_title("Exp1: forgetting curve (3 stages)")
    axes[0, 0].legend()

    t, S1, H1, S2, H2 = r2
    axes[0, 1].plot(t, H1, label="massed (H)")
    axes[0, 1].plot(t, H2, label="spaced (H)")
    axes[0, 1].set_title("Exp2: spacing effect (H)")
    axes[0, 1].legend()

    t, S1, H1, S2, H2 = r4
    axes[1, 0].plot(t, H1, label="threshold review (H)")
    axes[1, 0].plot(t, H2, label="fixed review (H)")
    axes[1, 0].set_title("Exp4: review timing (H)")
    axes[1, 0].legend()

    axes[1, 1].axis("off")
    axes[1, 1].text(0.1, 0.5, "Stage 1b: deposition-erosion\n(cache-writeback model C96/C97)",
                    fontsize=12, va="center")
    fig.tight_layout()
    fig.savefig("fig_stage1b.png", dpi=110)
    print("[plot] saved fig_stage1b.png")

if __name__ == "__main__":
    r1 = run_exp1()
    r2 = run_exp2()
    r3 = run_exp3()
    r4 = run_exp4()
    plot_all(r1, r2, r4)
    np.savez("stage1b_data.npz", t1=r1[0], S1=r1[1], H1=r1[2], M1=r1[3])
    print("[done] stage1b complete")
