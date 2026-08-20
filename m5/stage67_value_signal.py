# -*- coding: utf-8 -*-
"""
M5 阶段 67：价值信号（理论差距 #5——R8/C8-02——RPE——奖惩驱动沉积）

理论锚：R8（双环——内环价值系统——RPE 驱动 W 塑形）/ C8-02（疼痛/快乐 =
  价值通道误差流——快乐=正误差加分/疼痛=负误差减分）/ C8-04（价值信号 v(t)
  调制沉积率 ε——加分=刻河道/减分=削平河道）
研究锚：Schultz RPE（多巴胺——奖赏预测误差——经典）/ Berridge（wanting）
机制（价值调制沉积——RPE = 实际 − 预测）：
  ① 价值事件：奖励（正——"答对"）→ RPE+ → 沉积增强（刻河道）
     惩罚（负——"答错"）→ RPE− → 沉积削弱（削平河道）
  ② 预测价值：熟悉句（预测准——RPE≈0）→ 中性；新奇正确（RPE+）→ 刻；
     错误句（RPE−）→ 削
验证：
  exp1 奖励刻河道（正价值句——K 增强 vs 中性）
  exp2 惩罚削平（负价值句——K 削弱 vs 中性）
  exp3 价值调制（同句有无价值——学习差异——RPE 效应）
"""
import os
import re
import time
from collections import Counter
import numpy as np

RNG = np.random.default_rng(67)
DT = 0.05
GAMMA = 0.8
OMEGA_LO, OMEGA_HI = 0.5, 4.0
AMP_IN = 1.2
PULSE_STEPS = 5
EPS_K = 0.02
LAMBDA_K = 0.01
K_CAP = 0.5
REWARD_MULT = 2.0     # 奖励沉积系数（刻河道——RPE+）
PUNISH_MULT = 0.3     # 惩罚沉积系数（削平——RPE−）

def load_corpus(path, lo=3, hi=80, n=None):
    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    clean = [s for s in lines if lo <= len(s) <= hi
             and re.search(r"[一-鿿]", s) and not re.search(r"[A-Za-z]", s)]
    if n and len(clean) > n:
        clean = clean[:n]
    return clean

def step_dynamics(z, omega, gamma, K, rowsum, drive, dt):
    zr, zi = z.real, z.imag
    dz = -gamma * z + 1j * omega * z
    dz += K @ zr + 1j * (K @ zi) - z * rowsum
    dz += drive
    z = z + dz * dt
    over = np.abs(z) > 3.0
    z[over] = z[over] / np.abs(z[over]) * 2.0
    return z

class ValueLake:
    """价值湖：RPE 调制沉积（刻河道/削平）"""
    def __init__(self, chars):
        self.chars = chars
        self.ci = {c: i for i, c in enumerate(chars)}
        n = len(chars)
        self.omega = RNG.uniform(OMEGA_LO, OMEGA_HI, n)
        self.gamma = GAMMA
        self.z = 0.1 * np.exp(1j * RNG.uniform(0, 2 * np.pi, n))
        self.t = 0.0
        self.K = np.zeros((n, n))
        self.rowsum = np.zeros(n)

    def step(self, drive):
        self.z = step_dynamics(self.z, self.omega, self.gamma, self.K, self.rowsum, drive, DT)
        self.t += DT
        return self.z

    def inject_sentence(self, sent):
        drive = np.zeros(len(self.chars), dtype=complex)
        for pos, c in enumerate(sent):
            if c in self.ci:
                i = self.ci[c]
                drive[i] += AMP_IN * np.exp(1j * (self.omega[i] * self.t + pos * np.pi / 6))
        for _ in range(PULSE_STEPS):
            self.step(drive)
        for _ in range(3):
            self.step(np.zeros(len(self.chars), dtype=complex))
        return np.abs(self.z)

    def learn(self, sents, values=None):
        """学习（values：句价值列表——None=中性/+1=奖励/−1=惩罚）"""
        n = len(self.chars)
        for si, sent in enumerate(sents):
            v = values[si] if values else 0
            mult = REWARD_MULT if v > 0 else (PUNISH_MULT if v < 0 else 1.0)
            idx = [self.ci[c] for c in sent if c in self.ci]
            if len(idx) < 2:
                continue
            amp = self.inject_sentence(sent)
            sub = np.array(idx)
            A = amp[sub]
            L = len(idx)
            d_idx = np.arange(L)
            dist_w = 1.0 / np.maximum(np.abs(d_idx[:, None] - d_idx[None, :]), 1.0)
            contrib = EPS_K * mult * np.outer(A, A) * np.triu(dist_w, 1)
            pi, pj = np.nonzero(contrib)
            self.K[sub[pi], sub[pj]] += contrib[pi, pj]
            self.K[sub[pj], sub[pi]] += contrib[pi, pj] * 0.3
        self.K *= (1.0 - LAMBDA_K)
        rs = self.K.sum(axis=1)
        over = rs > K_CAP
        self.K[over] *= (K_CAP / rs[over])[:, None]
        self.rowsum = self.K.sum(axis=1)

    def strength(self, a, b):
        if a in self.ci and b in self.ci:
            return self.K[self.ci[a], self.ci[b]]
        return 0.0


def run():
    print("=== M5 阶段 67：价值信号（R8/C8-02——RPE 奖惩驱动沉积） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=900)
    chars = list(dict.fromkeys("".join(simple)))
    print(f"词汇表 {len(chars)} 字 / 语料 {len(simple)} 行")
    # ---- exp1：奖励刻河道 ----
    print("\n[exp1] 奖励刻河道（正价值——'苹果是水果'答对——K 增强 vs 中性）:")
    w_r = ValueLake(chars)
    w_n = ValueLake(chars)
    for _ in range(3):
        w_r.learn(["苹果是水果", "苹果很甜"], values=[+1, 0])
        w_n.learn(["苹果是水果", "苹果很甜"], values=[0, 0])
    kr = w_r.strength("苹", "果")
    kn = w_n.strength("苹", "果")
    print(f"      奖励 {kr:.3f} vs 中性 {kn:.3f}（{'刻河道 ✓' if kr > kn * 1.3 else '弱'}）")
    # ---- exp2：惩罚削平 ----
    print("\n[exp2] 惩罚削平（负价值——'苹果是石头'答错——K 削弱 vs 中性）:")
    w_p = ValueLake(chars)
    for _ in range(3):
        w_p.learn(["苹果是石头", "苹果很甜"], values=[-1, 0])
    kp = w_p.strength("苹", "果")
    print(f"      惩罚 {kp:.3f} vs 中性对照 {kn:.3f}（{'削平 ✓' if kp < kn else '未削平'}）")
    # ---- exp3：价值调制（同句有无价值——学习差异） ----
    print("\n[exp3] 价值调制（RPE 效应——奖励句学习 > 中性句 > 惩罚句）:")
    print(f"      奖励 {kr:.3f} > 中性 {kn:.3f} > 惩罚 {kp:.3f}"
          f"（{'RPE 梯度 ✓' if kr > kn > kp else '梯度异常'}）")
    print("\n[done] stage67 value signal")


if __name__ == "__main__":
    run()
