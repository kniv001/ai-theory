# -*- coding: utf-8 -*-
"""
M5 阶段 1a：复值单元最小仿真
验证原语方程：ż = -(γ + iω)z + D(t),  z ∈ ℂ  (C1-01/C1-03)

实验 1：无驱动衰减 —— 幅度指数衰减(γ) + 相位旋转(ω)（解析对照）
实验 2：脉冲驱动 —— RPE 式脉冲 → 幅度跳变 + 相位重置（相位携带时序）
实验 3：双单元耦合 —— Kuramoto 型相位锁定（C5-02 湖的预演）—— 扫描耦合强度 K
实验 4：耦合相位差 —— W 相位 = 传导延迟的载体（C1-03 延迟内建）

输出：控制台数值摘要 + m5/fig_stage1a_*.png + 数据 npz
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "."

# ---------- 实验 1：无驱动衰减（解析对照） ----------
def run_exp1():
    gamma, omega = 0.5, 2.0 * np.pi * 1.0   # 1 Hz 本征频率
    z0 = 1.0 + 0.0j
    t = np.linspace(0, 8.0, 2001)

    def rhs(t, y):
        return [-(gamma + 1j * omega) * (y[0] + 1j * y[1])]
    # 用实部/虚部分量
    def rhs_r(t, y):
        z = y[0] + 1j * y[1]
        dz = -(gamma + 1j * omega) * z
        return [dz.real, dz.imag]

    sol = solve_ivp(rhs_r, [0, 8.0], [z0.real, z0.imag], t_eval=t, rtol=1e-9, atol=1e-12)
    z_num = sol.y[0] + 1j * sol.y[1]
    z_ana = z0 * np.exp(-(gamma + 1j * omega) * t)

    err = np.max(np.abs(z_num - z_ana))
    # 幅度半衰期
    t_half = np.log(2) / gamma
    print(f"[exp1] 无驱动衰减: 数值-解析最大误差={err:.2e} | 幅度半衰期={t_half:.3f}s | 相位旋转频率={omega/(2*np.pi):.2f}Hz")
    return t, z_num

# ---------- 实验 2：脉冲驱动（RPE 式） ----------
def run_exp2():
    gamma, omega = 0.5, 2.0 * np.pi * 0.8
    t = np.linspace(0, 10.0, 4001)
    # RPE 式脉冲：t=2, 5, 8 处强度 2.0 的 δ 近似（高斯——max_step 强制不跳过）
    def drive(t):
        return 2.0 * np.exp(-((t - 2.0) ** 2) / (2 * 0.05 ** 2)) \
             + 2.0 * np.exp(-((t - 5.0) ** 2) / (2 * 0.05 ** 2)) \
             + 2.0 * np.exp(-((t - 8.0) ** 2) / (2 * 0.05 ** 2))

    def rhs_r(t, y):
        z = y[0] + 1j * y[1]
        dz = -(gamma + 1j * omega) * z + drive(t)
        return [dz.real, dz.imag]

    sol = solve_ivp(rhs_r, [0, 10.0], [0.0, 0.0], t_eval=t, rtol=1e-9, atol=1e-12, max_step=0.005)
    z = sol.y[0] + 1j * sol.y[1]
    phase = np.unwrap(np.angle(z))
    # 每个脉冲后：幅度被推离零点（响应）→ 衰减回零（恢复）
    pulse_peaks = []
    for tp in (2.0, 5.0, 8.0):
        idx = np.where((t > tp - 0.3) & (t < tp + 0.3))[0]
        if len(idx):
            pulse_peaks.append(np.max(np.abs(z[idx])))
    tail = np.max(np.abs(z[-500:]))
    print(f"[exp2] 脉冲驱动: 各脉冲响应峰值={np.round(pulse_peaks, 3)} | 脉冲后恢复(幅度回零)={tail:.4f}")
    return t, z

# ---------- 实验 3：双单元耦合锁定（Kuramoto 型） ----------
def run_exp3():
    gamma = 0.3
    omega1, omega2 = 2.0 * np.pi * 1.0, 2.0 * np.pi * 1.3   # 频率失谐
    t = np.linspace(0, 40.0, 4001)

    def make_rhs(K):
        def rhs_r(t, y):
            z1 = y[0] + 1j * y[1]
            z2 = y[2] + 1j * y[3]
            dz1 = -(gamma + 1j * omega1) * z1 + K * (z2 - z1)
            dz2 = -(gamma + 1j * omega2) * z2 + K * (z1 - z2)
            return [dz1.real, dz1.imag, dz2.real, dz2.imag]
        return rhs_r

    # 扫描 K：找到锁定临界
    results = {}
    for K in [0.0, 0.5, 1.0, 2.0, 4.0]:
        sol = solve_ivp(make_rhs(K), [0, 40.0], [1.0, 0.0, 1.0, 0.0], t_eval=t, rtol=1e-9, atol=1e-12)
        z1 = sol.y[0] + 1j * sol.y[1]
        z2 = sol.y[2] + 1j * sol.y[3]
        dphi = np.unwrap(np.angle(z1) - np.angle(z2))
        # 末段相位差斜率 = 是否锁定
        tail = dphi[-1000:]
        slope = np.polyfit(t[-1000:], tail, 1)[0]
        results[K] = (dphi, slope)
        print(f"[exp3] K={K:.1f}: 末段相位差斜率={slope:.4f} rad/s ({'锁定' if abs(slope) < 0.01 else '未锁定'}) | 末段相位差={np.angle(z1[-1])-np.angle(z2[-1]):.3f} rad")

    # 锁定临界估计（斜率穿越 0 的 K）
    return t, results

# ---------- 实验 4：耦合相位差 = 传导延迟载体 ----------
def run_exp4():
    gamma = 0.3
    omega0 = 2.0 * np.pi * 1.0
    K = 3.0
    t = np.linspace(0, 30.0, 3001)

    def rhs_r(t, y):
        z1 = y[0] + 1j * y[1]
        z2 = y[2] + 1j * y[3]
        # 非对称耦合：z1 领先驱动 z2（相位差 = 传导延迟的编码）
        dz1 = -(gamma + 1j * omega0) * z1 + K * (z2 - z1)
        dz2 = -(gamma + 1j * omega0) * z2 + K * (z1 * np.exp(-1j * 0.3) - z2)  # 0.3 rad 延迟
        return [dz1.real, dz1.imag, dz2.real, dz2.imag]

    sol = solve_ivp(rhs_r, [0, 30.0], [1.0, 0.0, 1.0, 0.0], t_eval=t, rtol=1e-9, atol=1e-12)
    z1 = sol.y[0] + 1j * sol.y[1]
    z2 = sol.y[2] + 1j * sol.y[3]
    dphi = np.angle(z1[-1]) - np.angle(z2[-1])
    print(f"[exp4] 耦合延迟0.3rad: 稳态相位差={dphi:.3f} rad (延迟编码{'一致' if abs(dphi - 0.3) < 0.05 else '待查'})")
    return t, z1, z2

# ---------- 绘图 ----------
def plot_all(r1, r2, r3, r4):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    t1, z1 = r1
    axes[0, 0].plot(t1, np.abs(z1), label="|z| (numeric)")
    axes[0, 0].plot(t1, 1.0 * np.exp(-0.5 * t1), "k--", label="|z| (analytic e^-γt)")
    axes[0, 0].set_title("Exp1: free decay (gamma=0.5, omega=1Hz)")
    axes[0, 0].legend(); axes[0, 0].set_xlabel("t (s)")

    t2, z2 = r2
    axes[0, 1].plot(t2, np.abs(z2))
    axes[0, 1].set_title("Exp2: RPE-like pulse drive")
    axes[0, 1].set_xlabel("t (s)")

    t3, res3 = r3
    for K, (dphi, slope) in res3.items():
        if K in (0.0, 1.0, 4.0):
            axes[1, 0].plot(t3, dphi, label=f"K={K} ({'locked' if abs(slope)<0.01 else 'drift'})")
    axes[1, 0].set_title("Exp3: phase diff under coupling (omega mismatch)")
    axes[1, 0].legend(); axes[1, 0].set_xlabel("t (s)")

    t4, a4, b4 = r4
    axes[1, 1].plot(t4, np.unwrap(np.angle(a4) - np.angle(b4)))
    axes[1, 1].set_title("Exp4: coupling delay -> phase difference")
    axes[1, 1].set_xlabel("t (s)")

    fig.tight_layout()
    fig.savefig("fig_stage1a.png", dpi=110)
    print("[plot] saved fig_stage1a.png")

if __name__ == "__main__":
    r1 = run_exp1()
    r2 = run_exp2()
    r3 = run_exp3()
    r4 = run_exp4()
    plot_all(r1, r2, r3, r4)
    np.savez("stage1a_data.npz",
             t1=r1[0], z1=r1[1], t2=r2[0], z2=r2[1], t3=r3[0], t4=r4[0])
    print("[done] stage1a complete")
