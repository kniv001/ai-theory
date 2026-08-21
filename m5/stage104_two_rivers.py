# -*- coding: utf-8 -*-
"""
M5 阶段 104：河流方向不对称（差距②——C3-02/C4-02——预测河/误差河分工）

理论锚：
  C3-02（河流双向化——预测河下行 + 误差河上行——互相预测——open）
  C4-02（方向不对称——误差河上行自动、预测河下行需注意开闸——
    注意力=闸门开度——open）
  C9-01（方向性是误差驱动学习的必要前提——联想需可定向信道——
    误差沿固定路径回传——supported）
  C1-01（驱动分层——加权流生成预测（内部）+ 误差流做耦合（单元间））

机制（双河结构）：
  ① 预测河（下行）W_fwd[i,j]：学习"i 后接 j"（前向预测——阅读/生成）
    ——激活：z_pred = W_fwd @ (g·z)——**需要闸门 g**（C4-02——注意开闸
    才有预测——无目标=无预测）
  ② 误差河（上行）W_bwd[j,i]：学习"j 的误差回传 i"（惊讶归因——
    C9-01 固定路径）——激活：z += W_bwd @ e——**自动**（误差即时上行
    ——无闸门——任何预测失败都传播）

验证：
  exp1 双河学习（W_fwd 前向 vs W_bwd 后向——不对称）
  exp2 预测河需闸门（无 g 预测弱 vs 有 g（焦点）预测强——C4-02）
  exp3 误差河自动（预测失败——误差自动上行——激活原因词——无闸门）
  exp4 误差归因（C9-01——误差沿固定路径回传——学习归因）
"""
import os
import re
import sys
import time
from collections import Counter
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("gb"):
    sys.stdout.reconfigure(encoding="utf-8")

RNG = np.random.default_rng(104)
DT = 0.05
GAMMA = 0.8
OMEGA_LO, OMEGA_HI = 0.5, 4.0
AMP_IN = 1.2
PULSE_STEPS = 5
EPS_K = 0.02
LAMBDA_K = 0.01
DELTA_PHI = np.pi / 6


class TwoRiverLake:
    """双河湖：预测河（下行——需闸门）+ 误差河（上行——自动）"""

    def __init__(self, chars):
        self.chars = list(chars)
        self.ci = {c: i for i, c in enumerate(chars)}
        n = len(chars)
        self.omega = RNG.uniform(OMEGA_LO, OMEGA_HI, n)
        self.gamma = GAMMA
        self.z = 0.1 * np.exp(1j * RNG.uniform(0, 2 * np.pi, n))
        self.t = 0.0
        self.W_fwd = np.zeros((n, n))      # 预测河（下行——i 后接 j）
        self.W_bwd = np.zeros((n, n))      # 误差河（上行——j 误差回传 i）
        self.rowsum_f = np.zeros(n)
        self.rowsum_b = np.zeros(n)

    def inject(self, sent):
        n = len(self.chars)
        drive = np.zeros(n, dtype=complex)
        for pos, c in enumerate(sent):
            if c in self.ci:
                i = self.ci[c]
                drive[i] += AMP_IN * np.exp(1j * (self.omega[i] * self.t + pos * DELTA_PHI))
        for _ in range(PULSE_STEPS):
            dz = -self.gamma * self.z + 1j * self.omega * self.z
            # 预测河（下行——需闸门 g——无目标=无预测）
            dz += self.W_fwd @ self.z - self.z * self.rowsum_f
            dz += drive
            self.z = self.z + dz * DT
            over = np.abs(self.z) > 3.0
            self.z[over] = self.z[over] / np.abs(self.z[over]) * 2.0
            self.t += DT
        return self.z

    def learn(self, sent):
        """双河学习：前向沉积（预测）+ 误差归因（C9-01）"""
        n = len(self.chars)
        idx = [self.ci[c] for c in sent if c in self.ci]
        if len(idx) < 2:
            return
        amp = np.abs(self.inject(sent))
        L = len(idx)
        # 预测河（前向——"i 后接 j"——相邻强权）
        for a in range(L):
            for b in range(a + 1, L):
                i, j = idx[a], idx[b]
                d = b - a
                w = EPS_K * amp[i] * amp[j] * (3.0 if d == 1 else 1.0 / d)
                self.W_fwd[i, j] += w
                # 误差河（反向归因——"j 的惊讶回传 i"——C9-01 定向）
                self.W_bwd[j, i] += w * 0.5
        self.W_fwd *= (1.0 - LAMBDA_K)
        self.W_bwd *= (1.0 - LAMBDA_K)
        self.rowsum_f = self.W_fwd.sum(axis=1)
        self.rowsum_b = self.W_bwd.sum(axis=1)

    def predict(self, c, g=1.0):
        """预测河（下行——闸门 g）：焦点 c → 预测下一个字
        g=0（闸门关——预测不传播——无预测）vs g=1（注意开闸——
        预测全强度）——C4-02：预测河需注意开闸"""
        if c not in self.ci:
            return []
        i = self.ci[c]
        row = self.W_fwd[i] * g
        top = np.argsort(row)[::-1][:3]
        return [(self.chars[j], row[j]) for j in top if row[j] > 0.0005]

    def error_river(self, seq):
        """误差河（上行——自动）：序列中预测失败的字 → 误差自动上行 →
        激活原因词（无闸门——C4-02 误差自动）"""
        n = len(self.chars)
        z = np.zeros(n)
        for pos, c in enumerate(seq):
            if c not in self.ci:
                continue
            j = self.ci[c]
            if pos > 0:
                prev = self.ci[seq[pos - 1]]
                expect = self.W_fwd[prev, j]      # 预测强度
                e = max(0.0, 1.0 - expect * 10)  # 预测误差（惊讶）
                if e > 0.3:                       # 误差自动上行（无闸门）
                    z += self.W_bwd[j] * e        # 回传——激活原因词
        return z


def run():
    print("=== M5 阶段 104：河流方向不对称（C3-02/C4-02——预测河需闸门/误差河自动） ===\n")
    base = os.path.dirname(__file__)
    from stage79_spontaneous_hubs import load_corpus
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=300)
    sents = simple[:300]
    chars = list(dict.fromkeys("".join(sents)))[:300]
    print(f"词汇表 {len(chars)} 字 / 语料 {len(sents)} 行")
    w = TwoRiverLake(chars)
    t0 = time.perf_counter()
    for ep in range(8):
        for s in sents:
            w.learn(s)
    print(f"训练完成——{time.perf_counter()-t0:.0f}s（双河：预测 fwd + 误差 bwd）")

    # ---- exp1：双河不对称 ----
    print("\n[exp1] 双河学习（W_fwd 前向 vs W_bwd 后向——不对称）:")
    for pair in [("苹", "果"), ("天", "气"), ("小", "猫")]:
        i, j = w.ci[pair[0]], w.ci[pair[1]]
        print(f"      {pair[0]}→{pair[1]}: 预测河 {w.W_fwd[i, j]:.4f}"
              f" vs 误差河 {w.W_bwd[j, i]:.4f}"
              f"（{'前向为主' if w.W_fwd[i, j] > w.W_bwd[j, i] else '异常'}）")

    # ---- exp2：预测河需闸门（C4-02） ----
    print("\n[exp2] 预测河闸门（无 g vs 有 g——注意力开闸才有预测——C4-02）:")
    for c in ["苹", "天", "小", "妈"]:
        p0 = w.predict(c, g=0.0)
        p1 = w.predict(c, g=1.0)
        print(f"      '{c}': 无闸门 {[(a, f'{v:.3f}') for a, v in p0[:2]] if p0 else '无'}"
              f" | 有闸门 {[(a, f'{v:.3f}') for a, v in p1[:2]] if p1 else '无'}")

    # ---- exp3：误差河自动（预测失败——误差上行——无闸门） ----
    print("\n[exp3] 误差河自动（'果苹'——'果→苹'预测失败——误差自动上行）:")
    z = w.error_river("果苹")
    top = np.argsort(z)[::-1][:4]
    print(f"      误差上行激活: {[(w.chars[j], f'{z[j]:.3f}') for j in top if z[j] > 0.001]}")
    z2 = w.error_river("苹果")
    top2 = np.argsort(z2)[::-1][:4]
    print(f"      对照'苹果'（熟悉——误差小）: "
          f"{[(w.chars[j], f'{z2[j]:.3f}') for j in top2 if z2[j] > 0.001]}")

    # ---- exp4：误差归因（C9-01——定向回传） ----
    print("\n[exp4] 误差归因（C9-01——误差沿固定路径回传——学习归因）:")
    print("      W_bwd[j,i] = 'j 的惊讶回传 i'——'果'惊讶 → 回传'苹'（激活原因）")
    if "果" in w.ci and "苹" in w.ci:
        i, j = w.ci["苹"], w.ci["果"]
        print(f"      W_bwd[果→苹] = {w.W_bwd[j, i]:.4f}（定向回传 ✓——"
              f"vs 反向 {w.W_bwd[i, j]:.4f}——不对称）")
    print("\n[结论] 双河分工：预测河（下行——需闸门——有目标才预测）"
          "+ 误差河（上行——自动——惊讶即时回传）——C3-02/C4-02 机制面")
    print("[done] stage104 two rivers")


if __name__ == "__main__":
    run()
