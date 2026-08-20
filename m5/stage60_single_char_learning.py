# -*- coding: utf-8 -*-
"""
M5 阶段 60：单字学习前置流程（用户："在学习词句的前面，需要再加个流程即单字的学习"）

幼儿顺序：单字（字音/字形——独立表征）→ 词 → 句——单字是组合的基础。
框架理论：C7-01（转导边界——字=转导单元——身份确立先于组合）/
  R5（沉积——重复过水——身份稳定——硬区）
机制（两阶段流程）：
  阶段 1 单字学习：每字单独注入（一字一句——"农"独立出现——重复 N 次——
    字身份稳定（ω 收敛——共振识别——"农"的单元独立激活）
  阶段 2 词句学习：语料（词/句——组合——词组涌现——K 学习）
对照：单字先行 vs 直接词句（词组质量/单字识别率）
验证：
  exp1 单字识别（单字阶段后——注入"农"→ 识别"农"——身份稳定）
  exp2 两阶段对照（单字先行 vs 直接——词组提取质量）
  exp3 单字独立表征（"农"单独识别 vs 在"农业"中——组合后仍可单独激活）
"""
import os
import re
import time
from collections import Counter
import numpy as np

RNG = np.random.default_rng(60)
DT = 0.05
GAMMA = 0.8
OMEGA_LO, OMEGA_HI = 0.5, 4.0
AMP_IN = 1.2
PULSE_STEPS = 5
EPS_K = 0.02
LAMBDA_K = 0.01
K_CAP = 0.5

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

class CharLake:
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
        self.act = np.zeros(n)

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

    def learn_single_chars(self, chars_subset, epochs=3):
        """阶段 1：单字学习（每字独立注入——身份稳定）"""
        for _ in range(epochs):
            for c in chars_subset:
                self.inject_sentence(c)   # 一字一句——独立表征

    def learn_epoch(self, sents):
        n = len(self.chars)
        for sent in sents:
            seq_idx = [self.ci[c] for c in sent if c in self.ci]
            if len(seq_idx) < 2:
                continue
            amp = self.inject_sentence(sent)
            L = len(seq_idx)
            sub = np.array(seq_idx)
            A = amp[sub]
            idx = np.arange(L)
            dist_w = 1.0 / np.maximum(np.abs(idx[:, None] - idx[None, :]), 1.0)
            contrib = EPS_K * np.outer(A, A) * np.triu(dist_w, 1)
            pi, pj = np.nonzero(contrib)
            self.K[sub[pi], sub[pj]] += contrib[pi, pj]
            self.K[sub[pj], sub[pi]] += contrib[pi, pj] * 0.3
            for _ in range(4):
                self.step(np.zeros(n, dtype=complex))
        self.K *= (1.0 - LAMBDA_K)
        rs = self.K.sum(axis=1)
        over = rs > K_CAP
        self.K[over] *= (K_CAP / rs[over])[:, None]
        self.rowsum = self.K.sum(axis=1)

    def recognize(self, c):
        """单字识别（注入单字——激活最高的单元 = 识别结果）"""
        self.inject_sentence(c)
        amp = np.abs(self.z)
        best = int(np.argmax(amp))
        return self.chars[best], amp[best]


def run():
    print("=== M5 阶段 60：单字学习前置流程（单字→词→句——幼儿顺序） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=900)
    wiki = load_corpus(os.path.join(base, "corpus_wiki_filtered.txt"), n=1800)
    sents = simple + wiki
    chars = list(dict.fromkeys("".join(sents)))
    print(f"词汇表 {len(chars)} 字 / 语料 {len(sents)} 行")
    # ---- 对照：直接词句 vs 单字先行 ----
    w_direct = CharLake(chars)
    for ep in range(10):
        w_direct.learn_epoch(sents)
    w_two = CharLake(chars)
    # 阶段 1：单字学习（全部字——3 epoch）
    w_two.learn_single_chars(chars[:1000], epochs=3)
    print(f"[阶段1] 单字学习完成（{len(chars[:1000])} 字 × 3 epoch——身份稳定）")
    # 阶段 2：词句学习
    for ep in range(10):
        w_two.learn_epoch(sents)
    print(f"[阶段2] 词句学习完成")
    # ---- exp1：单字识别（身份稳定） ----
    print("\n[exp1] 单字识别（阶段 1 后——注入单字 → 识别）:")
    for c in ["农", "业", "技", "术", "苹"]:
        if c in w_two.ci:
            out, amp = w_two.recognize(c)
            print(f"      '{c}' → 识别 '{out}' 幅度 {amp:.2f}{' ✓' if out == c else ' ✗'}")
    # ---- exp2：两阶段对照（词组提取） ----
    print("\n[exp2] 两阶段对照（单字先行 vs 直接——词组提取）:")
    def count_words(w, th=0.03):
        n = len(w.chars)
        cnt = 0
        for i in range(n):
            for j in range(i + 1, n):
                if w.K[i, j] > th:
                    cnt += 1
        return cnt
    cd = count_words(w_direct)
    ct = count_words(w_two)
    print(f"      直接词句: {cd} 个强词组 vs 单字先行: {ct} 个"
          f"（{'单字先行更优 ✓' if ct > cd else '直接更优'}）")
    # ---- exp3：单字独立表征（组合后仍可单独激活） ----
    print("\n[exp3] 单字独立表征（'农' 在 '农业' 组合后——单独注入仍激活'农'）:")
    out, amp = w_two.recognize("农")
    print(f"      '农' → '{out}'（{'独立表征保持 ✓' if out == '农' else '被组合吸收'}）")
    print("\n[done] stage60 single char learning")


if __name__ == "__main__":
    run()
