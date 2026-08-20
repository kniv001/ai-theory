# -*- coding: utf-8 -*-
"""
M5 阶段 44：组合层（C13-02 组合性——复杂句 = 词组单元的时序组合——用户："完善组合层吧"）

用户指导：短语 = 简单句复合——不能强硬划分——识别 = 训练产物（stage43 验证——词组从跨句重复涌现）。
组合层（R16 尺度递归——字→词→短语→句同机制）：
  ① 词组提取：K 强耦合对 → 词组单元（网络长出的词——"农业"农-业 / "种植"种-植）
  ② 组合表示：句子 → 词组序列（时序——"农业|属于|第一级产业|包括|作物种植"——吸引子复合 C13-02）
  ③ 组合生成：词组链（农业→作物→种植——跨词组 K 关联——生成候选句——组合性生成面）
  ④ 句级组合：两句共享词组数 = 组合相似度——主题句簇（话题层）
"""
import os
import re
import time
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(44)
DT = 0.05
GAMMA = 0.8
OMEGA_LO, OMEGA_HI = 0.5, 4.0
AMP_IN = 1.2
PULSE_STEPS = 5
EPS_K = 0.02
LAMBDA_K = 0.02
ETA_OMEGA = 0.15
K_CAP = 0.5
N_CHAR = 300
RELS = ["isa", "attr", "act", "cause"]
REL_IDX = {r: i for i, r in enumerate(RELS)}
WORD_TH = 0.025     # 词组耦合阈值（跨句重复成形——0.03 滤掉农业 0.030 边缘）

def rel_of(sent):
    if "因为" in sent or "所以" in sent:
        return "cause"
    if any(w in sent for w in ["是", "属于", "包括", "包含"]):
        return "isa"
    if "很" in sent:
        return "attr"
    return None

def load_corpus(path, lo=3, hi=80, n=None):
    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    clean = [s for s in lines if lo <= len(s) <= hi and re.search(r"[一-鿿]", s)]
    if n and len(clean) > n:
        clean = clean[:n]
    return clean

def step_dynamics(z, omega, gamma, K3, rowsum, drive, dt):
    zr, zi = z.real, z.imag
    dz = -gamma * z + 1j * omega * z
    for r in range(K3.shape[0]):
        Kr = K3[r]
        dz += Kr @ zr + 1j * (Kr @ zi) - z * rowsum[r]
    dz += drive
    z = z + dz * dt
    over = np.abs(z) > 3.0
    z[over] = z[over] / np.abs(z[over]) * 2.0
    return z

class RelLake:
    def __init__(self, chars):
        self.chars = chars
        self.ci = {c: i for i, c in enumerate(chars)}
        n = len(chars)
        self.omega = RNG.uniform(OMEGA_LO, OMEGA_HI, n)
        self.gamma = GAMMA
        self.z = 0.1 * np.exp(1j * RNG.uniform(0, 2 * np.pi, n))
        self.t = 0.0
        self.K = np.zeros((len(RELS), n, n))
        self.rowsum = np.zeros((len(RELS), n))
        self.act = np.zeros(n)

    def step(self, drive):
        self.z = step_dynamics(self.z, self.omega, self.gamma, self.K, self.rowsum, drive, DT)
        self.t += DT
        return self.z

    def inject_sentence(self, sent):
        drive = np.zeros(len(self.chars), dtype=complex)
        for c in sent:
            if c in self.ci:
                i = self.ci[c]
                drive[i] += AMP_IN * np.exp(1j * (self.omega[i] * self.t))
        for _ in range(PULSE_STEPS):
            self.step(drive)
        for _ in range(3):
            self.step(np.zeros(len(self.chars), dtype=complex))
        return np.abs(self.z)

    def learn_epoch(self, sents):
        n = len(self.chars)
        for sent in sents:
            rel = rel_of(sent)
            if rel is None:
                continue
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
            self.K[REL_IDX[rel]][sub[pi], sub[pj]] += contrib[pi, pj]
            self.K[REL_IDX[rel]][sub[pj], sub[pi]] += contrib[pi, pj]
            for _ in range(4):
                self.step(np.zeros(n, dtype=complex))
        for r in RELS:
            self.K[REL_IDX[r]] *= (1.0 - LAMBDA_K)
            row_sum = self.K[REL_IDX[r]].sum(axis=1)
            over = row_sum > K_CAP
            self.K[REL_IDX[r]][over] *= (K_CAP / row_sum[over])[:, None]
            self.K[REL_IDX[r]][:, over] *= (K_CAP / row_sum[over])[None, :]
        self.rowsum = self.K.sum(axis=2)
        Ksum = self.K.sum(axis=0)
        mask = Ksum > 0.08
        dw = ETA_OMEGA * (self.omega[None, :] - self.omega[:, None]) * np.where(mask, Ksum, 0.0)
        self.omega += dw.sum(axis=1)
        self.omega = np.clip(self.omega, OMEGA_LO, OMEGA_HI)


def extract_words(w, sents, rel="isa", th=WORD_TH):
    """词组提取：K 强耦合对 → 词组单元（跨句重复涌现）——方向按语料相邻出现规范化（"技术"非"术技"）"""
    K = w.K[REL_IDX[rel]]
    # 语料相邻出现计数（词组方向——真实句序）
    adj = Counter()
    for s in sents:
        for k in range(len(s) - 1):
            adj[s[k:k + 2]] += 1
    words = {}
    used = set()
    n = len(w.chars)
    for i in range(n):
        if i in used:
            continue
        for j in range(i + 1, n):
            if K[i, j] > th:
                words[(i, j)] = K[i, j]
                used.add(j)
                break
    out = []
    for (a, b), k in sorted(words.items(), key=lambda x: -x[1]):
        ca, cb = w.chars[a], w.chars[b]
        # 方向：语料中相邻出现多的顺序（"技术"出现多 → 用"技术"——"术技"≈0）
        if adj[cb + ca] > adj[ca + cb] and adj[cb + ca] > 0:
            out.append((cb + ca, k))
        else:
            out.append((ca + cb, k))
    return out


def sentence_composition(w, sent, words):
    """组合表示：句子 → 词组序列（按句序——时序——C13-02 吸引子复合）"""
    seq = []
    i = 0
    while i < len(sent) - 1:
        two = sent[i:i + 2]
        if two in words:
            seq.append(two)
            i += 2
        else:
            seq.append(sent[i])
            i += 1
    if i < len(sent):
        seq.append(sent[i])
    return seq


def word_chain(w, word, words, k=4):
    """组合生成：词组 → 后续词组链（跨词组 K 关联——K 中词组末字→其他词组首字）"""
    if len(word) < 2:
        return []
    last = word[-1]
    if last not in w.ci:
        return []
    i = w.ci[last]
    # 该字与所有字耦合——找耦合强的"其他词组首字"
    Ksum = w.K.sum(axis=0)
    row = Ksum[i].copy()
    cands = []
    for wd, _ in words:
        if wd[0] == last or wd == word:
            continue
        if wd[0] in w.ci:
            cands.append((wd, row[w.ci[wd[0]]]))
    cands.sort(key=lambda x: -x[1])
    return [(c[0], f"{c[1]:.3f}") for c in cands[:k] if c[1] > 0.01]


def run():
    print("=== M5 阶段 44：组合层（C13-02 组合性——复杂句 = 词组时序组合） ===\n")
    base = os.path.dirname(__file__)
    sents = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=900)
    sents += load_corpus(os.path.join(base, "corpus_wiki_filtered.txt"), n=600)
    freq = Counter("".join(sents))
    chars = [c for c, _ in freq.most_common(N_CHAR)]
    print(f"语料 {len(sents)} 行 / 字集 {len(chars)}")
    w = RelLake(chars)
    t0 = time.perf_counter()
    for ep in range(12):
        w.learn_epoch(sents)
    print(f"训练 12 epoch——{time.perf_counter()-t0:.0f}s")
    # ---- exp1：词组提取 ----
    words = extract_words(w, sents)
    print(f"\n[exp1] 词组提取（K>{WORD_TH} 强耦合——跨句重复涌现）: {len(words)} 个")
    for wd, k in words[:15]:
        print(f"      '{wd}' K={k:.3f}")
    word_set = {wd for wd, _ in words}
    # ---- exp2：组合表示（复杂句 → 词组序列） ----
    print("\n[exp2] 组合表示（复杂句 → 词组时序序列——吸引子复合）:")
    test_sents = ["农业属于第一级产业包括作物种植", "全球农业年产出大量食物", "技术进步改变世界经济发展"]
    for s in test_sents:
        seq = sentence_composition(w, s, word_set)
        print(f"      '{s}'")
        print(f"        → {' | '.join(seq)}")
    # ---- exp3：组合生成（词组链——生成候选） ----
    print("\n[exp3] 组合生成（词组 → 后续词组链——跨词组 K 关联——生成面）:")
    for wd, _ in words[:5]:
        chain = word_chain(w, wd, words)
        print(f"      '{wd}' → {chain}")
    # ---- exp4：句级组合（共享词组 = 相似度——句簇） ----
    print("\n[exp4] 句级组合（共享词组数 = 组合相似度——主题句簇）:")
    pairs = [("农业属于第一级产业包括作物种植", "全球农业年产出大量食物"),
             ("农业属于第一级产业包括作物种植", "技术进步改变世界经济发展")]
    for a, b in pairs:
        sa = set(sentence_composition(w, a, word_set))
        sb = set(sentence_composition(w, b, word_set))
        shared = sa & sb
        print(f"      共享词组 {len(shared)}: {shared}——'{a[:8]}…' vs '{b[:8]}…'")
    # 图：词组耦合热图（组合单元）
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].imshow(w.K[REL_IDX["isa"]], cmap="viridis")
    axes[0].set_title("isa coupling (words = strong pairs)")
    axes[1].imshow(w.K.sum(axis=0), cmap="viridis")
    axes[1].set_title("all-relation coupling")
    fig.tight_layout()
    fig.savefig("fig_stage44.png", dpi=110)
    print("\n[plot] saved fig_stage44.png")
    print("[done] stage44 composition layer")


if __name__ == "__main__":
    run()
