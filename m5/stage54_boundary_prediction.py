# -*- coding: utf-8 -*-
"""
M5 阶段 54：预测误差词切分（现实研究增强——Goriely & Buttery 2025：
"预测误差在词首峰值——无监督词切分"——框架 A0 误差词汇）

工程增强：词切分从"K 阈值提取"→ 预测误差切分（更真实——误差驱动——研究支撑）。
机制：
  ① 时序湖（K 方向化——前→后预测）
  ② 逐位预测：注入时每步预测下一字（K 最强前向）——预测强度 = 前向 K
  ③ 词边界 = 预测强度骤降处（"农→业"预测强（词内）/ "业→属"预测弱（跨词——边界））
验证：
  exp1 预测强度曲线（"农业属于第一级产业"——逐位——词内高/边界低）
  exp2 边界检测（强度骤降 = 词边界——无监督切分）
  exp3 与 K 词组对照（预测切分 vs 词组提取——一致率）
"""
import os
import re
import time
from collections import Counter
import numpy as np

RNG = np.random.default_rng(54)
DT = 0.05
GAMMA = 0.8
OMEGA_LO, OMEGA_HI = 0.5, 4.0
AMP_IN = 1.2
PULSE_STEPS = 5
EPS_K = 0.02
LAMBDA_K = 0.01
K_CAP = 0.5
DELTA_PHI = np.pi / 6

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

class PredLake:
    """预测湖：时序 + 前向 K——预测误差词切分"""
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
                drive[i] += AMP_IN * np.exp(1j * (self.omega[i] * self.t + pos * DELTA_PHI))
        for _ in range(PULSE_STEPS):
            self.step(drive)
        for _ in range(3):
            self.step(np.zeros(len(self.chars), dtype=complex))
        return np.abs(self.z)

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

    def pred_strength(self, a, b):
        """前向预测强度（a→b——词内强/跨词弱）"""
        if a in self.ci and b in self.ci:
            return self.K[self.ci[a], self.ci[b]]
        return 0.0

    def segment(self, sent, drop_ratio=0.5):
        """预测误差切分：逐位前向预测强度——骤降处 = 词边界
        （研究：预测误差词首峰——Goriely 2025）"""
        cs = [c for c in sent if c in self.ci]
        if len(cs) < 2:
            return [sent]
        strengths = [self.pred_strength(cs[i], cs[i + 1]) for i in range(len(cs) - 1)]
        # 边界 = 强度 < 邻域中位 × drop_ratio（骤降）
        med = np.median([s for s in strengths if s > 0]) if any(s > 0 for s in strengths) else 0
        words = []
        cur = cs[0]
        for i in range(len(strengths)):
            if strengths[i] < med * drop_ratio and med > 0:
                words.append(cur)
                cur = cs[i + 1]
            else:
                cur += cs[i + 1]
        words.append(cur)
        return words, strengths


def run():
    print("=== M5 阶段 54：预测误差词切分（Goriely 2025 研究——词首误差峰） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=900)
    wiki = load_corpus(os.path.join(base, "corpus_wiki_filtered.txt"), n=1800)
    sents = simple + wiki
    freq = Counter("".join(sents))
    chars = [c for c, _ in freq.most_common(300)]
    w = PredLake(chars)
    t0 = time.perf_counter()
    for ep in range(12):
        w.learn_epoch(sents)
    print(f"训练完成——{time.perf_counter()-t0:.0f}s")
    # ---- exp1：预测强度曲线 ----
    print("\n[exp1] 前向预测强度（逐位——词内高/边界低）:")
    test = "农业属于第一级产业"
    words, strengths = w.segment(test)
    cs = [c for c in test if c in w.ci]
    for i in range(len(strengths)):
        bar = "#" * int(strengths[i] * 200)
        print(f"      {cs[i]}→{cs[i+1]}: {strengths[i]:.3f} {bar}")
    # ---- exp2：边界检测 ----
    print(f"\n[exp2] 预测误差切分（'农业属于第一级产业'）:")
    print(f"      → {words}")
    test2 = "因为下雨所以带伞"
    w2, s2 = w.segment(test2)
    print(f"      '因为下雨所以带伞' → {w2}")
    # ---- exp3：与 K 词组对照 ----
    print("\n[exp3] 对照（预测切分 vs K 词组提取）:")
    adj = Counter()
    for s in sents:
        for k in range(len(s) - 1):
            adj[s[k:k + 2]] += 1
    kwords = set()
    n = len(chars)
    for i in range(n):
        for j in range(i + 1, n):
            if w.K[i, j] > 0.02:
                ca, cb = chars[i], chars[j]
                kwords.add(ca + cb if adj[ca + cb] >= adj[cb + ca] else cb + ca)
    # 预测切分的词 vs K 词组（"农业"等）
    for wd in ["农业", "技术", "环境", "教育"]:
        in_pred = wd in words or any(wd in x for x in w.segment(wd + "属于产业")[0])
        in_k = wd in kwords
        print(f"      '{wd}': 预测切分 {'✓' if in_pred else '—'} / K 词组 {'✓' if in_k else '—'}")
    print("\n[done] stage54 boundary prediction")


if __name__ == "__main__":
    run()
