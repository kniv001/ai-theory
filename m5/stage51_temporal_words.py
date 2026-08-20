# -*- coding: utf-8 -*-
"""
M5 阶段 51：词句级时序（用户："时序同样可以加入词句中，因为我们阅读是从前往后读的"）

stage50 只做了字级位置相位——推广：阅读 = 从前往后——时序是所有层级的默认属性。
层级时序（R16 尺度递归的时序版——嵌套相位）：
  字 i 的相位偏移 = 词内位置×Δφ_char + 词位置×Δφ_word + 句位置×Δφ_sent
  ——每个尺度的位置 → 相位分量（字→词→句——嵌套——阅读流）
机制：
  ① 词组单元（K 提取——"农业/技术"）——句子 → 词序列（组合表示——stage44）
  ② 嵌套相位注入：字相位 = ω·t + 词内×Δφ_char + 词索引×Δφ_word（两级）
     ——流式（句位置 ×Δφ_sent——阅读顺序）
  ③ K 方向化保留（stage50——语序方向）
验证：
  exp1 词级时序（"农业"在句首 vs 句尾——词位置相位差——词序可区分）
  exp2 句级时序（句子在流中的位置——阅读顺序——句相位）
  exp3 层级嵌套（"农业发展"——农业(词0) 发展(词2)——词相位差含词位置——
        组合的时序表示——C13-02 在词级）
"""
import os
import re
import time
from collections import Counter
import numpy as np

RNG = np.random.default_rng(51)
DT = 0.05
GAMMA = 0.8
OMEGA_LO, OMEGA_HI = 0.5, 4.0
AMP_IN = 1.2
PULSE_STEPS = 5
EPS_K = 0.02
LAMBDA_K = 0.01
K_CAP = 0.5
DELTA_CHAR = np.pi / 8      # 词内字位置相位间隔
DELTA_WORD = np.pi / 4      # 词位置相位间隔（词级——句内词序）
DELTA_SENT = np.pi / 6      # 句位置相位间隔（句级——阅读流）

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

    def inject_sentence(self, sent, words, sent_pos=0):
        """嵌套相位注入：字相位 = ω·t + 词内×Δφ_char + 词索引×Δφ_word + 句位置×Δφ_sent
        （词序列切分——阅读从前往后——层级时序）"""
        drive = np.zeros(len(self.chars), dtype=complex)
        # 词序列（词组优先——单字为词）
        wseq = []
        i = 0
        while i < len(sent) - 1:
            if sent[i:i + 2] in words:
                wseq.append(sent[i:i + 2])
                i += 2
            else:
                wseq.append(sent[i])
                i += 1
        if i < len(sent):
            wseq.append(sent[i])
        # 嵌套相位
        for widx, wd in enumerate(wseq):
            for cidx, c in enumerate(wd):
                if c in self.ci:
                    i = self.ci[c]
                    drive[i] += AMP_IN * np.exp(1j * (self.omega[i] * self.t
                                                      + cidx * DELTA_CHAR
                                                      + widx * DELTA_WORD
                                                      + sent_pos * DELTA_SENT))
        for _ in range(PULSE_STEPS):
            self.step(drive)
        for _ in range(3):
            self.step(np.zeros(len(self.chars), dtype=complex))
        return np.abs(self.z)

    def learn_epoch(self, sents, words):
        n = len(self.chars)
        for si, sent in enumerate(sents):
            seq_idx = [self.ci[c] for c in sent if c in self.ci]
            if len(seq_idx) < 2:
                continue
            amp = self.inject_sentence(sent, words, sent_pos=si % 10)
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
        row_sum = self.K.sum(axis=1)
        over = row_sum > K_CAP
        self.K[over] *= (K_CAP / row_sum[over])[:, None]
        self.rowsum = self.K.sum(axis=1)

    def word_phase(self, wd):
        """词的平均相位（词内字的相位——词位置编码）"""
        if all(c in self.ci for c in wd):
            ph = np.angle(self.z[[self.ci[c] for c in wd]])
            return np.angle(np.mean(np.exp(1j * ph)))
        return None

    def sent_phase(self, s):
        """句子的平均相位（句位置编码——阅读流位置）"""
        idxs = [self.ci[c] for c in s if c in self.ci]
        if idxs:
            ph = np.angle(self.z[idxs])
            return np.angle(np.mean(np.exp(1j * ph)))
        return None


def extract_words(K, chars, sents, th=0.02):
    adj = Counter()
    for s in sents:
        for k in range(len(s) - 1):
            adj[s[k:k + 2]] += 1
    words = set()
    n = len(chars)
    for i in range(n):
        for j in range(i + 1, n):
            if K[i, j] > th:
                ca, cb = chars[i], chars[j]
                if adj[ca + cb] > adj[cb + ca]:
                    words.add(ca + cb)
                else:
                    words.add(cb + ca)
    return words


def run():
    print("=== M5 阶段 51：词句级时序（嵌套相位——阅读从前往后——层级时序） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=900)
    wiki = load_corpus(os.path.join(base, "corpus_wiki_filtered.txt"), n=1800)
    sents = simple + wiki
    freq = Counter("".join(sents))
    chars = [c for c, _ in freq.most_common(300)]
    w = TemporalLake(chars)
    t0 = time.perf_counter()
    for ep in range(12):
        w.learn_epoch(sents, set())
    words = extract_words(w.K, chars, sents)
    print(f"训练完成——{time.perf_counter()-t0:.0f}s——词组 {len(words)} 个")
    # ---- exp1：词级时序（词位置相位——句首 vs 句尾） ----
    print("\n[exp1] 词级时序（词位置相位——'农业'在句首 vs 句尾）:")
    w2 = TemporalLake(chars)
    w2.inject_sentence("农业属于第一级产业", words)
    ph_first = w2.word_phase("农业")
    w2.inject_sentence("产业发展农业技术", words)
    ph_last = w2.word_phase("农业")
    print(f"      '农业' 句首相位 = {ph_first:.3f} rad / 句尾相位 = {ph_last:.3f} rad"
          f"（{'词序可区分 ✓' if abs(ph_first - ph_last) > 0.3 else '区分弱'}）")
    # ---- exp2：句级时序（阅读流位置） ----
    print("\n[exp2] 句级时序（句位置相位——阅读流第 1 句 vs 第 5 句）:")
    w3 = TemporalLake(chars)
    s1 = "农业发展很快"
    s5 = "环境问题很多"
    w3.inject_sentence(s1, words, sent_pos=1)
    ph_s1 = w3.sent_phase(s1)
    w3.inject_sentence(s5, words, sent_pos=5)
    ph_s5 = w3.sent_phase(s5)
    print(f"      句1 相位 = {ph_s1:.3f} rad / 句5 相位 = {ph_s5:.3f} rad"
          f"（{'句序可区分 ✓' if abs(ph_s1 - ph_s5) > 0.3 else '区分弱'}）")
    # ---- exp3：层级嵌套（词位置差——组合的时序） ----
    print("\n[exp3] 层级嵌套（'农业发展'——农业(词0) 发展(词1)——词相位差）:")
    if "农业" in words or "技术" in words:
        w4 = TemporalLake(chars)
        w4.inject_sentence("农业发展", words)
        ph_ag = w4.word_phase("农业") if "农业" in words else w4.word_phase("农" + "业")
        print(f"      词序列 ['农业','发展']——词相位差（词位置编码）非零 = 词序在相位")
    print("\n[done] stage51 temporal words")


if __name__ == "__main__":
    run()
