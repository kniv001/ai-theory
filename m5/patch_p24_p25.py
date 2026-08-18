# -*- coding: utf-8 -*-
"""
M5 A 类快赢：P24（间隔优势 ∝ 间隔/半衰期比）+ P25（弱 K 区噪声-同步竞争）
基于阶段 1b（缓存-写回）与 1c（Kuramoto）的扩展参数扫描。

P24：间隔优势函数——扫描 间隔时长 × 软区半衰期（λ_s）——验证优势比随 间隔/τ 单调增
P25：噪声-同步竞争——K 近临界 × 噪声强度——验证噪声降低有效 K（弱 K 区敏感）
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(17)

# ---------- P24：间隔优势函数（缓存-写回模型） ----------
def simulate_learn(lam_s, lam_h, k, events, t_max=40.0, dt=0.01):
    n = int(t_max / dt) + 1
    t = np.linspace(0, t_max, n)
    S = np.zeros(n); H = np.zeros(n)
    ev = 0
    while ev < len(events) and events[ev][0] <= t[0]:
        S[0] += events[ev][1]; ev += 1
    for i in range(1, n):
        while ev < len(events) and events[ev][0] <= t[i]:
            S[i] += events[ev][1]; ev += 1
        S[i] += S[i-1] + (-lam_s * S[i-1]) * dt
        H[i] += H[i-1] + (k * S[i-1] - lam_h * H[i-1]) * dt
    return t, S, H

def run_p24():
    lam_h, k = 0.01, 0.3
    # 间隔时长扫描（相对软区半衰期 τ = ln2/λ_s）
    gaps = [0.5, 1.0, 2.0, 4.0, 8.0]
    for lam_s in (0.3, 0.5, 1.0):
        tau = np.log(2) / lam_s   # 软区半衰期（天）
        ratios = []
        for gap in gaps:
            # 集中：gap=0.25 天内连续 3 次复习
            massed = [(0.0, 1.0), (0.25, 1.0), (0.5, 1.0), (0.75, 1.0)]
            spaced = [(0.0, 1.0), (gap, 1.0), (2*gap, 1.0), (3*gap, 1.0)]
            _, _, Hm = simulate_learn(lam_s, lam_h, k, massed)
            _, _, Hs = simulate_learn(lam_s, lam_h, k, spaced)
            ratios.append(Hs[-1] / Hm[-1])
        print(f"[P24] λ_s={lam_s} (τ={tau:.2f}d): 间隔{gaps} 优势比 = " +
              " ".join(f"{r:.2f}" for r in ratios))
        # 验证：优势比随 gap/τ 单调增
        mono = all(ratios[i+1] >= ratios[i] for i in range(len(ratios)-1))
        print(f"      优势比随间隔/半衰期比单调增 = {'✓ P24 兑现' if mono else '✗'}")
    return gaps

# ---------- P25：噪声-同步竞争（Kuramoto） ----------
def kuramoto_r(omega, K, noise, t_max=60.0, dt=0.01, reps=5):
    N = len(omega)
    n = int(t_max / dt) + 1
    rs = []
    for _ in range(reps):   # 多次重复（不同初始相位）——均值降波动
        theta = RNG.uniform(0, 2*np.pi, N)
        r_hist = np.zeros(n)
        for i in range(n):
            dtheta = omega + (K * np.sin(theta[None, :] - theta[:, None])).sum(axis=1) / N
            if noise > 0:
                dtheta += RNG.normal(0, noise, N)
            theta = theta + dtheta * dt
            r_hist[i] = abs(np.mean(np.exp(1j*theta)))
        rs.append(np.mean(r_hist[-int(n*0.2):]))
    return np.mean(rs)

def run_p25():
    N = 20
    omega = RNG.normal(1.0, 0.3, N)
    # 临界区密扫（K_c≈0.5）+ 强区对照
    Ks = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 1.50]
    noises = [0.0, 0.05, 0.1, 0.2, 0.4]
    grid = np.zeros((len(Ks), len(noises)))
    for i, K in enumerate(Ks):
        for j, no in enumerate(noises):
            grid[i, j] = kuramoto_r(omega, K, no)
    print("[P25] r(K, noise) 网格:")
    header = "      K\\noise " + " ".join(f"{n:5.2f}" for n in noises)
    print(header)
    for i, K in enumerate(Ks):
        print(f"      {K:5.2f}    " + " ".join(f"{grid[i,j]:5.2f}" for j in range(len(noises))))
    # 验证：临界区（K=0.55——刚过临界）噪声破坏同步；强区（K=1.5）鲁棒
    weak_decay = grid[Ks.index(0.55), 0] - grid[Ks.index(0.55), -1]
    strong_stable = grid[Ks.index(1.5), 0] - grid[Ks.index(1.5), -1]
    # 全网格：噪声效应在临界区最大（计算各 K 的噪声衰减）
    decays = [grid[i, 0] - grid[i, -1] for i in range(len(Ks))]
    peak_at_critical = decays[Ks.index(0.55)] >= max(decays)
    print(f"      K=0.55（刚过临界）: 无噪声 r={grid[Ks.index(0.55),0]:.2f} → 噪声0.4 r={grid[Ks.index(0.55),-1]:.2f}（衰减 {weak_decay:.2f}）")
    print(f"      K=1.5（强）: 无噪声 r={grid[Ks.index(1.5),0]:.2f} → 噪声0.4 r={grid[Ks.index(1.5),-1]:.2f}（衰减 {strong_stable:.2f}）")
    print(f"      临界区最脆弱/强区鲁棒 = {'✓ P25 兑现（噪声 = 降低有效 K——近临界最脆弱）' if peak_at_critical and strong_stable < 0.1 else '✗'}")
    return Ks, noises, grid

if __name__ == "__main__":
    g = run_p24()
    Ks, noises, grid = run_p25()
    # 图
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    lam_s_list = [0.3, 0.5, 1.0]
    for i, lam_s in enumerate(lam_s_list):
        tau = np.log(2)/lam_s
        gaps = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
        ratios = []
        for gap in gaps:
            massed = [(0.0,1.0),(0.25,1.0),(0.5,1.0),(0.75,1.0)]
            spaced = [(0.0,1.0),(gap,1.0),(2*gap,1.0),(3*gap,1.0)]
            _,_,Hm = simulate_learn(lam_s, 0.01, 0.3, massed)
            _,_,Hs = simulate_learn(lam_s, 0.01, 0.3, spaced)
            ratios.append(Hs[-1]/Hm[-1])
        axes[0].plot(gaps/tau, ratios, "o-", label=f"tau={tau:.2f}")
    axes[0].set_title("P24: spacing advantage vs gap/tau")
    axes[0].set_xlabel("gap / half-life"); axes[0].set_ylabel("H spaced / H massed")
    axes[0].legend()

    im = axes[1].imshow(grid, aspect="auto", origin="lower",
                        extent=[noises[0], noises[-1], Ks[0], Ks[-1]])
    axes[1].set_title("P25: r(K, noise) — weak-K fragile")
    axes[1].set_xlabel("noise"); axes[1].set_ylabel("K")
    fig.colorbar(im, ax=axes[1])
    fig.tight_layout()
    fig.savefig("fig_patch_p24_p25.png", dpi=110)
    print("[plot] saved fig_patch_p24_p25.png")
    print("[done] patch p24/p25 complete")
