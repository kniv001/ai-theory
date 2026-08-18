# -*- coding: utf-8 -*-
"""
M5 阶段 1c：多单元同步域仿真（Kuramoto 型）
验证 C5-02（湖 = 相位锁定群体；分裂=退锁/合并=同步）+ 工程 4 问（规模临界/噪声容限）

模型：dθ_i/dt = ω_i + Σ_j K_ij·sin(θ_j − θ_i)
序参量：r = |⟨e^{iθ}⟩|（同步度——r=1 完全锁定/r≈0 退锁）

实验 1：全局耦合 —— N=20 频率展宽下 K 扫描 → 临界 K_c（湖形成条件）
实验 2：规模临界 —— N=5/10/20/50 的 K_c（规模效应——工程问 2）
实验 3：分裂/合并 —— K 时变（高→低→高）→ r 跟随（湖的拆分与合并）
实验 4：噪声容限 —— 加白噪声 → r 的鲁棒性（工程问 1）
实验 5：局部耦合 —— 环形邻接 → 多个局部同步簇（小湖群——R2 大湖群架构最小演示）

输出：控制台摘要 + fig_stage1c.png + npz
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(42)
DT = 0.01

def kuramoto(omega, K, t_max=50.0, local=False, noise=0.0, K_schedule=None):
    """欧拉积分。返回 t, r(t), theta(t)
    K: 标量（全局）或矩阵（局部）
    K_schedule: (t, K) 列表 → K 随时间变化（分裂/合并实验）
    """
    n = int(t_max / DT) + 1
    t = np.linspace(0, t_max, n)
    theta = RNG.uniform(0, 2*np.pi, len(omega))
    r = np.zeros(n)
    r[0] = abs(np.mean(np.exp(1j*theta)))
    K_now = np.full((len(omega), len(omega)), K if np.isscalar(K) else 0.0)
    if not np.isscalar(K):
        K_now = K.copy()
    sched_idx = 0
    for i in range(1, n):
        if K_schedule is not None:
            while sched_idx < len(K_schedule) and K_schedule[sched_idx][0] <= t[i]:
                K_now = np.full((len(omega), len(omega)), K_schedule[sched_idx][1])
                if local:
                    K_now = make_local(K_schedule[sched_idx][1], len(omega))
                sched_idx += 1
        # 驱动项：每单元 = Σ_j K_ij·sin(θ_j − θ_i)（均值场归一化 K_eff = K/N）
        dtheta = omega + (K_now * np.sin(theta[None, :] - theta[:, None])).sum(axis=1) / len(omega)
        if noise > 0:
            dtheta += RNG.normal(0, noise, len(omega))
        theta = theta + dtheta * DT
        r[i] = abs(np.mean(np.exp(1j*theta)))
    return t, r, theta

def make_local(K, n):
    """环形邻接耦合（每单元连左右邻居）"""
    M = np.zeros((n, n))
    for i in range(n):
        M[i, (i-1) % n] = K
        M[i, (i+1) % n] = K
    return M

def steady_r(omega, K, t_max=80.0, local=False, noise=0.0):
    t, r, _ = kuramoto(omega, K, t_max, local, noise)
    return np.mean(r[-int(len(r)*0.2):])   # 末段 20% 平均

# ---------- 实验 1：全局耦合 K 扫描 ----------
def run_exp1():
    N = 20
    omega = RNG.normal(1.0, 0.3, N)   # 频率展宽 σω=0.3
    ks = np.linspace(0, 3.0, 13)
    rs = [steady_r(omega, k, t_max=60.0) for k in ks]
    # 临界估计：r 超过 0.5 的最小 K
    kc = None
    for k, r in zip(ks, rs):
        if r > 0.5:
            kc = k; break
    print(f"[exp1] N=20 全局耦合: 临界 K_c ≈ {kc} (r>0.5)")
    print(f"       r(K): " + " ".join(f"{k:.1f}:{r:.2f}" for k, r in zip(ks, rs)))
    return ks, rs

# ---------- 实验 2：规模临界 ----------
def run_exp2():
    Ns = [5, 10, 20, 50]
    ks = np.linspace(0, 3.0, 13)
    kcs = []
    # 同 ω 样本（控制变量：N 取前 N 个频率——排除种子差异）
    omega_base = RNG.normal(1.0, 0.3, 50)
    for N in Ns:
        omega = omega_base[:N]
        rs = [steady_r(omega, k, t_max=60.0) for k in ks]
        kc = next((k for k, r in zip(ks, rs) if r > 0.5), None)
        kcs.append(kc)
        print(f"[exp2] N={N}: K_c≈{kc}")
    spread = max(x for x in kcs if x) - min(x for x in kcs if x)
    print(f"       K_c 波动范围 = {spread:.2f}（{'小——规模效应弱（均值场一致）' if spread < 0.35 else '明显——规模效应'}")
    return Ns, kcs

# ---------- 实验 3：分裂/合并 ----------
def run_exp3():
    N = 20
    omega = RNG.normal(1.0, 0.3, N)
    # K 时变：0-30 高(2.5 锁定) → 30-60 低(0.2 退锁) → 60-90 高(2.5 再锁定)
    schedule = [(0.0, 2.5), (30.0, 0.2), (60.0, 2.5)]
    t, r, _ = kuramoto(omega, 0.0, t_max=90.0, K_schedule=schedule)
    r_hi1 = np.mean(r[(t>20)&(t<28)])
    r_lo  = np.mean(r[(t>40)&(t<55)])
    r_hi2 = np.mean(r[(t>70)&(t<85)])
    print(f"[exp3] 分裂/合并: 高K r={r_hi1:.2f} → 低K r={r_lo:.2f} → 高K r={r_hi2:.2f}")
    print(f"       合并-分裂-再合并 = {'✓' if r_hi1>0.5 and r_lo<0.4 and r_hi2>0.5 else '✗'}")
    return t, r

# ---------- 实验 4：噪声容限 ----------
def run_exp4():
    N = 20
    omega = RNG.normal(1.0, 0.3, N)
    K = 2.0
    noises = [0.0, 0.1, 0.3, 0.6, 1.0]
    rs = [steady_r(omega, K, t_max=60.0, noise=no) for no in noises]
    print(f"[exp4] 噪声容限 (K=2): " + " ".join(f"σn={no:.1f}:r={r:.2f}" for no, r in zip(noises, rs)))
    # 噪声下 r 保持 > 0.5 的最大噪声
    tol = max(no for no, r in zip(noises, rs) if r > 0.5)
    print(f"       容忍噪声 ≈ σn={tol}（r 仍 > 0.5）")
    return noises, rs

# ---------- 实验 5：局部耦合 → 局部湖（小湖群） ----------
def run_exp5():
    N = 40
    omega = RNG.normal(1.0, 0.3, N)
    # 模块化耦合：4 组（组内强 K_in / 组间弱 K_out）——大湖群的最小模型（R2）
    def make_modular(K_in, K_out, groups=4):
        M = np.full((N, N), K_out)
        np.fill_diagonal(M, 0.0)
        size = N // groups
        for g in range(groups):
            idx = slice(g*size, (g+1)*size)
            M[idx, idx] = K_in
            np.fill_diagonal(M[idx, idx], 0.0)
        return M

    for K_in, K_out in ((2.0, 0.1), (2.5, 0.05), (3.0, 0.02)):
        t, r, theta = kuramoto(omega, make_modular(K_in, K_out), t_max=80.0)
        # 组内相干 vs 组间相干
        size = N // 4
        r_in = np.mean([
            abs(np.mean(np.exp(1j*theta[g*size:(g+1)*size])))
            for g in range(4)
        ])
        r_between = abs(np.mean(np.exp(1j*theta)))
        print(f"[exp5] 模块化(K_in={K_in},K_out={K_out}): 组内相干={r_in:.3f} | 全局(组间)={r_between:.3f}")
        if r_in > 0.8 and r_between < 0.5:
            print(f"       ✓ 局部湖成立（组内强相干 + 组间弱——大湖群最小演示：4 个小湖并存）")
            return t, r, theta, (K_in, r_in, r_between)
    print(f"       未找到理想参数区（记录为不完善处——组间耦合需更弱）")
    return t, r, theta, (K_in, 0.0, 0.0)

# ---------- 绘图 ----------
def plot_all(e1, e3, e4, e5):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ks, rs = e1
    axes[0, 0].plot(ks, rs, "o-")
    axes[0, 0].axhline(0.5, color="k", ls="--", lw=0.8)
    axes[0, 0].set_title("Exp1: order param r vs coupling K (N=20)")
    axes[0, 0].set_xlabel("K"); axes[0, 0].set_ylabel("r (sync)")

    t, r = e3
    axes[0, 1].plot(t, r)
    axes[0, 1].set_title("Exp3: K high->low->high (merge/split/merge)")
    axes[0, 1].set_xlabel("t"); axes[0, 1].set_ylabel("r")

    noises, rs = e4
    axes[1, 0].plot(noises, rs, "o-")
    axes[1, 0].set_title("Exp4: noise tolerance (K=2)")
    axes[1, 0].set_xlabel("noise sigma"); axes[1, 0].set_ylabel("r")

    t, r, theta, _ = e5
    axes[1, 1].plot(t, r)
    axes[1, 1].set_title("Exp5: local coupling -> local lakes (r)")
    axes[1, 1].set_xlabel("t"); axes[1, 1].set_ylabel("global r")
    fig.tight_layout()
    fig.savefig("fig_stage1c.png", dpi=110)
    print("[plot] saved fig_stage1c.png")

if __name__ == "__main__":
    e1 = run_exp1()
    e2 = run_exp2()
    e3 = run_exp3()
    e4 = run_exp4()
    e5 = run_exp5()
    plot_all(e1, e3, e4, e5)
    np.savez("stage1c_data.npz", ks=e1[0], rs=e1[1], t3=e3[0], r3=e3[1],
             noises=e4[0], rs4=e4[1], t5=e5[0], r5=e5[1])
    print("[done] stage1c complete")
