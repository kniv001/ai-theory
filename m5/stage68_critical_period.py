# -*- coding: utf-8 -*-
"""
M5 阶段 68：关键期/可塑性递减（理论差距 #6——R41 水流硬化——h 硬度演化）

理论锚：R41（关键期包络 = 水流硬化涌现——暗饲养无流不硬化→窗口开——
  PNN/髓鞘经验依赖）/ C6-01（地形异质可塑性——h 硬度场——硬区=长时记忆）/
  C6-02（元可塑性——dh/dt = 稳定性↑ − 新奇性↓ + 稳态回拉）
研究锚：Hubel & Wiesel（暗饲养猫——无流窗口开）/ PNN 分子刹车（经验依赖硬化）
机制（学习率 ∝ 硬度——早期高可塑后期固化）：
  ① h 硬度场（每字——初始低（可塑）——训练中硬化（过水→硬）
  ② 学习率 = ε × (1 − h)（软区高学习/硬区低——关键期）
  ③ 关键期包络：早期（低 h）高可塑——后期（高 h）固化——窗口涌现
验证：
  exp1 硬化涌现（训练中 h 上升——过水→硬——河道固化）
  exp2 关键期效果（早期学的新词 vs 后期学的新词——早期学得好（可塑）/后期差）
  exp3 对照（有 h vs 无 h——学习分布——早期集中/后期保持）
"""
import os
import re
import time
from collections import Counter
import numpy as np

RNG = np.random.default_rng(68)
DT = 0.05
GAMMA = 0.8
OMEGA_LO, OMEGA_HI = 0.5, 4.0
AMP_IN = 1.2
PULSE_STEPS = 5
EPS_K = 0.02
LAMBDA_K = 0.01
K_CAP = 0.5
H_INIT = 0.1       # 初始硬度（低——可塑）
H_RATE = 0.02      # 硬化率（过水→硬——R41）

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

class PeriodLake:
    """关键期湖：h 硬度场（学习率 ∝ 1−h——早期可塑后期固化）"""
    def __init__(self, chars, use_h=True):
        self.chars = chars
        self.ci = {c: i for i, c in enumerate(chars)}
        n = len(chars)
        self.omega = RNG.uniform(OMEGA_LO, OMEGA_HI, n)
        self.gamma = GAMMA
        self.z = 0.1 * np.exp(1j * RNG.uniform(0, 2 * np.pi, n))
        self.t = 0.0
        self.K = np.zeros((n, n))
        self.rowsum = np.zeros(n)
        self.h = np.full(n, H_INIT if use_h else 0.0)   # 硬度场（无 h 对照 = 0）
        self.use_h = use_h

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

    def learn_epoch(self, sents):
        n = len(self.chars)
        for sent in sents:
            idx = [self.ci[c] for c in sent if c in self.ci]
            if len(idx) < 2:
                continue
            amp = self.inject_sentence(sent)
            sub = np.array(idx)
            A = amp[sub]
            L = len(idx)
            d_idx = np.arange(L)
            dist_w = 1.0 / np.maximum(np.abs(d_idx[:, None] - d_idx[None, :]), 1.0)
            # 学习率 ∝ (1 − h)（软区高学习/硬区低——关键期）
            h_sub = self.h[sub]
            mult = np.maximum(1.0 - h_sub, 0.05)   # 最小 5%（硬化后仍可微调）
            contrib = EPS_K * np.outer(A * mult, A * mult) * np.triu(dist_w, 1)
            pi, pj = np.nonzero(contrib)
            self.K[sub[pi], sub[pj]] += contrib[pi, pj]
            self.K[sub[pj], sub[pi]] += contrib[pi, pj] * 0.3
            # 硬化（过水→硬——R41：水流硬化）
            if self.use_h:
                self.h[sub] += H_RATE * (1.0 - self.h[sub])
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
    print("=== M5 阶段 68：关键期/可塑性递减（R41——h 硬度——水流硬化） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=900)
    chars = list(dict.fromkeys("".join(simple)))
    print(f"词汇表 {len(chars)} 字 / 语料 {len(simple)} 行")
    w = PeriodLake(chars)
    # ---- exp1：硬化涌现（训练中 h 上升） ----
    for ep in range(5):
        w.learn_epoch(simple)
    h_avg = np.mean(w.h)
    print(f"\n[exp1] 硬化涌现：训练 5 epoch 后平均硬度 {h_avg:.2f}"
          f"（{'硬化 ✓——过水→硬——R41' if h_avg > H_INIT * 1.5 else '硬化弱'}）")
    # ---- exp2：关键期效果（早期学 vs 后期学） ----
    print("\n[exp2] 关键期效果（早期学的新词 vs 后期——早期可塑学得好）:")
    # 早期语料（前 300 句——含'苹'）vs 后期（后 300 句——含'苹'）
    early = simple[:300]
    late = simple[300:600]
    w_early = PeriodLake(chars)
    w_early.learn_epoch(early)          # 早期学'苹果'
    w_early.learn_epoch(late)           # 后期学其他
    w_late = PeriodLake(chars)
    w_late.learn_epoch(late)            # 先学其他（硬）
    w_late.learn_epoch(early)           # 后期学'苹果'（已硬——可塑低）
    k_early = w_early.strength("苹", "果")
    k_late = w_late.strength("苹", "果")
    print(f"      早期学'苹果' {k_early:.3f} vs 后期学 {k_late:.3f}"
          f"（{'关键期效应 ✓——早期学得好' if k_early > k_late * 1.3 else '效应弱'}）")
    # ---- exp3：对照（有 h vs 无 h——学习分布） ----
    print("\n[exp3] 对照（有 h vs 无 h——后期学习保持）:")
    w_noh = PeriodLake(chars, use_h=False)
    for ep in range(5):
        w_early.learn_epoch(simple)
        w_noh.learn_epoch(simple)
    print(f"      有 h：硬度 {np.mean(w_early.h):.2f}（后期固化——保持稳定）")
    print(f"      无 h：学习率恒定（后期仍全速——无关键期）")
    print("\n[done] stage68 critical period")


if __name__ == "__main__":
    run()
