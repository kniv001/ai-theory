# -*- coding: utf-8 -*-
"""
M5 阶段 103：相位通道（C13-02 组合性=时序吸引子复合——最大理论-工程差距
——工程至今用实数幅度共现——相位语义一个字没用上）

理论锚：
  C13-02（组合性 = 时序吸引子复合——原语吸引子按相位次序编排成复合轨迹——
    sh+an→shan——字→词→句同机制放大——open——本 stage M5 验证）
  C3-01（相位携带语义——时序惊讶=相位误差——open）
  C1-03（复值单元——相位=时序/震荡——supported）
  C5-01（复值误差-Hebb——学习强度+延迟（相位）——海兔时序配对——
    supported）

机制（时序配对相位学习）：
  ① 位置相位（stage50——字 i 驱动相位 = ω_i·t + pos·Δφ）
  ② 复值河道 W（n×n 复数）——学习 = 时序配对（C5-01 海兔式）：
    W[i,j] += ε·exp(i·(pos_j-pos_i)·Δφ)——直接沉积期望延迟（相位差）
    ——"苹果" → W[苹,果] 相位 = +Δφ（果在苹后一位）
  ③ 组合轨迹：注入序列 → 相位序列（0,Δφ,2Δφ…——复合吸引子）
  ④ 顺序区分：W[苹,果] vs W[果,苹]——相位差反号（C13-02 判据：次序编排）

验证：
  exp1 时序配对（W[苹,果] 相位 = +Δφ——海兔式延迟学习）
  exp2 组合轨迹（注入"苹果"——相位序列——复合吸引子）
  exp3 顺序区分（"苹果" vs "果苹"——W 相位反号——组合可区分）
  exp4 词→句（"苹果很甜"——4 字相位序列——字级→句级）
  exp5 相位误差（C3-01 时序惊讶——"果苹"预测相位 vs 实际——误差）
"""
import os
import re
import sys
import time
from collections import Counter
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("gb"):
    sys.stdout.reconfigure(encoding="utf-8")

RNG = np.random.default_rng(103)
DT = 0.05
GAMMA = 0.8
OMEGA_LO, OMEGA_HI = 0.5, 4.0
AMP_IN = 1.2
PULSE_STEPS = 5
EPS_K = 0.02
LAMBDA_K = 0.01
DELTA_PHI = np.pi / 6      # 位置相位间隔（每位置 Δφ——顺序编码）


class PhaseLake:
    """复值相位湖：位置相位驱动 + 时序配对学习（C5-01 海兔式——相位=延迟）"""

    def __init__(self, chars):
        self.chars = list(chars)
        self.ci = {c: i for i, c in enumerate(chars)}
        n = len(chars)
        self.omega = RNG.uniform(OMEGA_LO, OMEGA_HI, n)
        self.gamma = GAMMA
        self.z = 0.1 * np.exp(1j * RNG.uniform(0, 2 * np.pi, n))
        self.t = 0.0
        self.W = np.zeros((n, n), dtype=complex)   # 复值河道（相位=延迟）
        self.rowsum = np.zeros(n)

    def inject(self, sent):
        """位置相位注入：字 i 驱动相位 = ω_i·t + pos·Δφ——演化"""
        n = len(self.chars)
        drive = np.zeros(n, dtype=complex)
        for pos, c in enumerate(sent):
            if c in self.ci:
                i = self.ci[c]
                drive[i] += AMP_IN * np.exp(1j * (self.omega[i] * self.t + pos * DELTA_PHI))
        for _ in range(PULSE_STEPS):
            dz = -self.gamma * self.z + 1j * self.omega * self.z
            dz += self.W @ self.z - self.z * self.rowsum
            dz += drive
            self.z = self.z + dz * DT
            over = np.abs(self.z) > 3.0
            self.z[over] = self.z[over] / np.abs(self.z[over]) * 2.0
            self.t += DT
        return self.z

    def learn(self, sent):
        """时序配对学习（C5-01 海兔式）：W[i,j] 沉积期望延迟（位置相位差）
        ——"苹果" → W[苹,果] 相位 = +Δφ（果在苹后一位——相位=延迟）"""
        n = len(self.chars)
        idx = [self.ci[c] for c in sent if c in self.ci]
        if len(idx) < 2:
            return
        amp = np.abs(self.inject(sent))
        L = len(idx)
        for a in range(L):
            for b in range(a + 1, L):
                i, j = idx[a], idx[b]
                phase = (b - a) * DELTA_PHI          # 期望延迟（位置差）
                self.W[i, j] += EPS_K * amp[i] * amp[j] * np.exp(1j * phase)
                self.W[j, i] += EPS_K * amp[i] * amp[j] * np.exp(-1j * phase)
        self.W *= (1.0 - LAMBDA_K)
        self.rowsum = np.abs(self.W).sum(axis=1)

    def phase_of(self, a, b):
        """W[a,b] 的相位（学习到的延迟——a→b 的时序）"""
        if a in self.ci and b in self.ci:
            w = self.W[self.ci[a], self.ci[b]]
            if abs(w) > 1e-6:
                return np.angle(w)
        return None

    def phase_error(self, seq):
        """相位误差（C3-01 时序惊讶）：注入序列——预测相位（位置差×Δφ）
        vs 实际相位（学习到的 W）——误差 = 惊讶"""
        errs = []
        for a in range(len(seq) - 1):
            p = self.phase_of(seq[a], seq[a + 1])
            if p is not None:
                expect = DELTA_PHI            # 相邻（位置差 1）
                err = abs((p - expect + np.pi) % (2 * np.pi) - np.pi)
                errs.append(err)
        return np.mean(errs) if errs else None


def run():
    print("=== M5 阶段 103：相位通道（C13-02 组合性=时序吸引子复合） ===\n")
    base = os.path.dirname(__file__)
    from stage79_spontaneous_hubs import load_corpus
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=300)
    sents = simple[:300]
    chars = list(dict.fromkeys("".join(sents)))[:300]
    print(f"词汇表 {len(chars)} 字 / 语料 {len(sents)} 行")
    w = PhaseLake(chars)
    t0 = time.perf_counter()
    for ep in range(3):
        for s in sents:
            w.learn(s)
    print(f"训练完成——{time.perf_counter()-t0:.0f}s（复值河道——相位=延迟）")

    # ---- exp1：时序配对（海兔式延迟学习） ----
    print("\n[exp1] 时序配对（W[i,j] 相位 = 期望延迟——C5-01 海兔式）:")
    for pair in [("苹", "果"), ("天", "气"), ("小", "猫"), ("妈", "妈"), ("吃", "鱼")]:
        p = w.phase_of(*pair)
        if p is not None:
            print(f"      W['{pair[0]}','{pair[1]}'] 相位 = {p:.2f}"
                  f"（期望 +{DELTA_PHI:.2f}——{'对齐 ✓' if abs(p - DELTA_PHI) < 0.5 else '偏差'}）")
        else:
            print(f"      W['{pair[0]}','{pair[1]}'] 无连接")

    # ---- exp2：组合轨迹（相位序列——复合吸引子） ----
    print("\n[exp2] 组合轨迹（注入'苹果'——相位序列——字→词复合吸引子）:")
    z = w.inject("苹果")
    for c in "苹果":
        if c in w.ci:
            i = w.ci[c]
            print(f"      '{c}': 幅度 {abs(z[i]):.2f} / 相位 {np.angle(z[i]):.2f}")

    # ---- exp3：顺序区分（C13-02 判据——次序编排） ----
    print("\n[exp3] 顺序区分（'苹果' vs '果苹'——W 相位反号——组合可区分）:")
    p_ab = w.phase_of("苹", "果")
    p_ba = w.phase_of("果", "苹")
    if p_ab is not None and p_ba is not None:
        print(f"      W[苹→果] = {p_ab:.2f} vs W[果→苹] = {p_ba:.2f}"
              f"（{'方向区分 ✓' if abs(p_ab - p_ba) > 1.0 else '区分不足'}——"
              f"相位差 {abs(p_ab - p_ba):.2f} ≈ 2Δφ={2 * DELTA_PHI:.2f}）")

    # ---- exp4：词→句（4 字相位序列——尺度递归） ----
    print("\n[exp4] 词→句（'苹果很甜'——4 字相位序列——C16-01 尺度递归）:")
    seq = "苹果很甜"
    z = w.inject(seq)
    for pos, c in enumerate(seq):
        if c in w.ci:
            i = w.ci[c]
            print(f"      位置{pos} '{c}': 相位 {np.angle(z[i]):.2f}"
                  f"（期望 {pos * DELTA_PHI:.2f}——"
                  f"{'对齐' if abs((np.angle(z[i]) - pos * DELTA_PHI + np.pi) % (2*np.pi) - np.pi) < 0.5 else '偏差'}）")

    # ---- exp5：相位误差（C3-01 时序惊讶） ----
    print("\n[exp5] 相位误差（时序惊讶——C3-01——熟悉序列 vs 陌生序列）:")
    for seq in ["苹果", "果苹", "小猫吃鱼", "苹果很甜"]:
        err = w.phase_error(seq)
        print(f"      '{seq}': 相位误差 {err:.2f} 弧度"
              f"（{'熟悉' if err is not None and err < 0.5 else '惊讶/陌生' if err is not None else '无数据'}）")
    print("\n[结论] 相位通道：时序配对（海兔式）+ 组合轨迹 + 顺序区分 + 相位误差"
          "——C13-02 组合性的动力学形态（字→词→句相位层级）")
    print("[done] stage103 phase composition")


if __name__ == "__main__":
    run()
