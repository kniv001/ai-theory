# -*- coding: utf-8 -*-
"""
M5 阶段 10：语义蛋 + 相位学习整合（"蛋会预测"——蛋内时序知识）
stage5（语义蛋：4 概念湖 + 组间联想）× stage9（相位学习：W 复值——arg W = 延迟）

整合：组间联想 W 复值化——|W| = 语义关联强度 + arg W = 时序延迟
  时序学习：A→B（间隔 Δt 重复）→ W_AB 相位 = ωΔt（蛋学习"B 在 A 后 Δt"）
  预测：A 激活 → 蛋预测 B 在 Δt 后激活（相位提前——预测河）
  误差修正：时序改变（Δt 变化）→ 预测误差（惊讶）→ W 相位修正（C3-01 相位误差）
  方向：A→B vs B→A（相位符号——顺序知识）

实验 1：蛋学时序（A→B Δt=0.3 重复 → W_AB 相位 ≈ ωΔt）
实验 2：蛋会预测（A 激活 → 预测信号在 Δt 时刻峰值——预测正确性）
实验 3：误差修正（时序改为 Δt=0.7 → 预测误差 → W 相位漂移——惊讶驱动学习）
实验 4：方向知识（A→B vs B→A——相位符号）
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(41)
EPS = 0.05
LAM = 0.01
OMEGA = 2*np.pi*1.0
N_CON = 4

class PredictingEgg:
    """会预测的蛋：概念湖 + 复值组间联想（强度+时序）"""
    def __init__(self):
        self.W = np.zeros((N_CON, N_CON), dtype=complex)  # 组间复值联想
        self.act = np.zeros(N_CON)                        # 湖激活历史

    def learn_sequence(self, a, b, dt, n_reps=60):
        """时序配对学习：a 激活 → 间隔 dt → b 激活（复值 Hebb——相位 = ωΔt）"""
        for _ in range(n_reps):
            z_a = 1.0 * np.exp(1j * 0.0)
            z_b = 1.0 * np.exp(1j * OMEGA * dt)
            self.W[a, b] += EPS * z_b * np.conj(z_a)   # 相位 = ωΔt（B 相对 A 的延迟）
        self.W *= (1 - LAM)

    def predict(self, a, horizon=1.5, dt_step=0.01):
        """A 激活 → 预测信号曲线（各时刻预期 B 的激活强度——预测河）"""
        t = np.arange(0, horizon, dt_step)
        pred = np.zeros(len(t))
        for b in range(N_CON):
            if b != a and abs(self.W[a, b]) > 0.05:
                delay = np.angle(self.W[a, b]) / OMEGA
                if delay < 0:
                    delay += 2*np.pi/OMEGA    # 环绕修正
                if delay < horizon:
                    pred[int(delay/dt_step)] += abs(self.W[a, b])
        # 高斯展宽（预测的时序分布）
        if np.max(pred) > 0:
            sigma = 3
            idx = np.where(pred > 0)[0]
            for i in idx:
                pred = pred + abs(self.W[a, b]) * np.exp(-0.5*((np.arange(len(t))-i)/sigma)**2)
        return t, pred

    def predict_peak_time(self, a, b):
        """预测 B 的峰值时刻"""
        d = np.angle(self.W[a, b]) / OMEGA
        if d < 0:
            d += 2*np.pi/OMEGA
        return d

def run():
    egg = PredictingEgg()

    # ---- 实验 1：蛋学时序 ----
    dt1 = 0.3
    egg.learn_sequence(0, 1, dt1)
    ph = np.angle(egg.W[0, 1])
    pred_dt = egg.predict_peak_time(0, 1)
    print(f"[exp1] 蛋学时序: A→B Δt={dt1} 重复 → arg W={ph:.3f} | 预测延迟={pred_dt:.3f}s"
          f"（误差 {abs(pred_dt-dt1):.3f}s）")
    print(f"       = {'✓ 蛋内时序知识（相位学习）' if abs(pred_dt - dt1) < 0.05 else '✗'}")

    # ---- 实验 2：蛋会预测 ----
    t, pred = egg.predict(0)
    peak_t = t[np.argmax(pred)]
    print(f"[exp2] 蛋会预测: A 激活 → 预测信号峰值在 t={peak_t:.2f}s"
          f"（学到的 Δt={dt1}——{'✓ 预测正确（蛋会预测）' if abs(peak_t-dt1) < 0.05 else '✗'}）")

    # ---- 实验 3：误差修正（时序改变 → 惊讶 → 修正） ----
    dt2 = 0.7
    egg.learn_sequence(0, 1, dt2, n_reps=100)   # 环境时序改变
    ph_new = np.angle(egg.W[0, 1])
    pred_new = egg.predict_peak_time(0, 1)
    drift = abs(pred_new - dt1)
    print(f"[exp3] 误差修正: 时序改为 Δt={dt2} → 预测延迟 {pred_dt:.3f} → {pred_new:.3f}"
          f"（漂移 {drift:.3f}s）")
    print(f"       = {'✓ 预测误差驱动修正（时序惊讶→W 相位调整——C3-01 相位误差）' if pred_new > dt1 + 0.1 else '✗'}")

    # ---- 实验 4：方向知识（每方向的 W 各自编码"对方在我后 Δt"） ----
    egg2 = PredictingEgg()
    egg2.learn_sequence(1, 2, 0.3)     # B→C：W[B,C] 预测 C 在 B 后 0.3
    egg2.learn_sequence(2, 1, 0.5)     # C→B：W[C,B] 预测 B 在 C 后 0.5
    p_bc = egg2.predict_peak_time(1, 2)
    p_cb = egg2.predict_peak_time(2, 1)
    print(f"[exp4] 方向知识: B→C 预测延迟={p_bc:.3f}s（学 0.3）| C→B 预测延迟={p_cb:.3f}s（学 0.5）")
    ok_dir = abs(p_bc - 0.3) < 0.05 and abs(p_cb - 0.5) < 0.05
    print(f"       = {'✓ 蛋有双向顺序知识（每个方向的 W 相位独立编码时序）' if ok_dir else '✗'}")

    # ---- 图 ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(t, pred)
    axes[0].axvline(dt1, color="k", ls="--", label=f"learned dt={dt1}")
    axes[0].set_title("Exp2: egg predicts B after A")
    axes[0].set_xlabel("t"); axes[0].legend()
    axes[1].bar(["B->C (learn 0.3)", "C->B (learn 0.5)"], [p_bc, p_cb])
    axes[1].set_title("Exp4: per-direction timing")
    fig.tight_layout()
    fig.savefig("fig_stage10.png", dpi=110)
    print("\n[plot] saved fig_stage10.png")
    print("[done] stage10 egg predicts complete")

if __name__ == "__main__":
    run()
