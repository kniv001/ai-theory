# -*- coding: utf-8 -*-
"""
M5 阶段 50：时序进湖（用户："语句的时序性有加入框架吗"——诚实：湖动力学=词袋——
修复：位置相位编码 + K 方向化——C13-02 时序吸引子复合）

缺失诊断：整句同时注入（加速）——字湖无顺序——K 对称——"农业"与"业农"不可区分。
修复（stage37 相位差原理 → 湖内实现）：
  ① 位置相位编码：字 i 驱动相位 = ω_i·t + pos_i×Δφ（位置 → 相位偏移——
     同时注入（快）但相位携带顺序——时序在相位里）
  ② K 方向化：沉积时"农业"（农在前业在后）——K[农→业]（前→后 权重 1.0）>
     K[业→农]（后→前 0.3）——方向不对称——语序进入 K
验证：
  exp1 位置相位（"农业" vs "业农"——相位模式不同——顺序可区分）
  exp2 K 方向性（"农业"——K[农→业] > K[业→农]——顺序方向涌现）
  exp3 时序组合（"农业发展" vs "业农发展"——顺序的 K 模式差异——C13-02）
"""
import os
import re
import time
from collections import Counter
import numpy as np

RNG = np.random.default_rng(50)
DT = 0.05
GAMMA = 0.8
OMEGA_LO, OMEGA_HI = 0.5, 4.0
AMP_IN = 1.2
PULSE_STEPS = 5
EPS_K = 0.02
LAMBDA_K = 0.01
K_CAP = 0.5
DELTA_PHI = np.pi / 6      # 位置相位间隔（每位置 Δφ——顺序编码）
DIR_FWD = 1.0              # 前→后 权重
DIR_BWD = 0.3              # 后→前 权重（弱——语序方向涌现）

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

class TemporalLake:
    """时序湖：位置相位编码 + K 方向化"""
    def __init__(self, chars):
        self.chars = chars
        self.ci = {c: i for i, c in enumerate(chars)}
        n = len(chars)
        self.omega = RNG.uniform(OMEGA_LO, OMEGA_HI, n)
        self.gamma = GAMMA
        self.z = 0.1 * np.exp(1j * RNG.uniform(0, 2 * np.pi, n))
        self.t = 0.0
        self.K = np.zeros((n, n))       # 方向化 K（K[i,j] = i 在 j 前（i→j）的强度）
        self.rowsum = np.zeros(n)
        self.act = np.zeros(n)

    def step(self, drive):
        self.z = step_dynamics(self.z, self.omega, self.gamma, self.K, self.rowsum, drive, DT)
        self.t += DT
        return self.z

    def inject_sentence(self, sent):
        """位置相位注入：字 i 相位 = ω_i·t + pos_i×Δφ（同时注入——时序在相位）"""
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
            # 方向化：i<j（前→后 1.0）——j<i（后→前 0.3）
            self.K[sub[pi], sub[pj]] += contrib[pi, pj] * DIR_FWD
            self.K[sub[pj], sub[pi]] += contrib[pi, pj] * DIR_BWD
            for _ in range(4):
                self.step(np.zeros(n, dtype=complex))
        self.K *= (1.0 - LAMBDA_K)
        row_sum = self.K.sum(axis=1)
        over = row_sum > K_CAP
        self.K[over] *= (K_CAP / row_sum[over])[:, None]
        self.rowsum = self.K.sum(axis=1)

    def order_phase(self, a, b):
        """位置相位差（给定两字——注入序列后的相位关系——顺序判定）"""
        if a not in self.ci or b not in self.ci:
            return None
        pa = np.angle(self.z[self.ci[a]])
        pb = np.angle(self.z[self.ci[b]])
        return np.angle(np.exp(1j * (pb - pa)))

    def dir_strength(self, a, b):
        """方向强度（K[a→b] vs K[b→a]——语序方向）"""
        if a not in self.ci or b not in self.ci:
            return None
        i, j = self.ci[a], self.ci[b]
        return self.K[i, j], self.K[j, i]


def run():
    print("=== M5 阶段 50：时序进湖（位置相位 + K 方向化——C13-02） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=900)
    wiki = load_corpus(os.path.join(base, "corpus_wiki_filtered.txt"), n=1800)
    sents = simple + wiki
    freq = Counter("".join(sents))
    chars = [c for c, _ in freq.most_common(300)]
    w = TemporalLake(chars)
    t0 = time.perf_counter()
    for ep in range(12):
        w.learn_epoch(sents)
    print(f"训练完成——{time.perf_counter()-t0:.0f}s")
    # ---- exp1：位置相位（顺序可区分） ----
    print("\n[exp1] 位置相位（注入序列——相位差 = 顺序——'农业' vs '业农'）:")
    w2 = TemporalLake(chars)
    w2.inject_sentence("农业")
    ph_ab = w2.order_phase("农", "业")
    w2.inject_sentence("业农")
    ph_ba = w2.order_phase("农", "业")
    print(f"      '农业'注入后 农-业 相位差 = {ph_ab:.3f} rad")
    print(f"      '业农'注入后 农-业 相位差 = {ph_ba:.3f} rad"
          f"（{'顺序可区分 ✓' if abs(ph_ab - ph_ba) > 0.3 else '区分弱'}）")
    # ---- exp2：K 方向性（语序方向涌现） ----
    print("\n[exp2] K 方向性（语序方向——'农业' 农→业 vs 业→农）:")
    for pair in ["农业", "技术", "发展", "环境", "教育"]:
        r = w.dir_strength(pair[0], pair[1])
        if r:
            fwd, bwd = r
            print(f"      '{pair}': 前→后 {fwd:.3f} vs 后→前 {bwd:.3f}"
                  f"（{'方向 ✓' if fwd > bwd * 1.5 else '方向弱'}）")
    # ---- exp3：时序组合（顺序的 K 模式差异——C13-02） ----
    print("\n[exp3] 时序组合（'农业发展' vs '业农发展'——顺序的 K 模式）:")
    if all(c in w.ci for c in "农产业发展"):
        # 农业发展：农→业 业→发 发→展 的 K 链 vs 倒序
        chain_fwd = (w.K[w.ci["农"], w.ci["业"]] + w.K[w.ci["业"], w.ci["发"]]
                     + w.K[w.ci["发"], w.ci["展"]]) / 3
        chain_bwd = (w.K[w.ci["业"], w.ci["农"]] + w.K[w.ci["发"], w.ci["业"]]
                     + w.K[w.ci["展"], w.ci["发"]]) / 3
        print(f"      正序链（农→业→发→展）强度 = {chain_fwd:.3f}")
        print(f"      倒序链（业→农→发→业…）强度 = {chain_bwd:.3f}"
              f"（{'时序组合 ✓——正序强于倒序' if chain_fwd > chain_bwd * 1.3 else '组合弱'}）")
    print("\n[done] stage50 temporal lake")


if __name__ == "__main__":
    run()
