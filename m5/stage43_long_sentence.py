# -*- coding: utf-8 -*-
"""
M5 阶段 43：长句处理（短语分段——网络吃下知识句——用户："怎么让网络吃下更多的知识句/长句"）

根因（stage39 对照确认）：长句 60 字 → 句内全对 1770 对——不相干字共现——
真正词组（农业/种植/畜牧）的强关联被稀释——复杂句 85% 时结构退化。
解法（语言习得真实机制 + R16 尺度递归）：
  标点（，、；：） = 天然的短语边界——"农业属于第一级产业，包括作物种植、畜牧"
    → 短语："农业属于第一级产业" | "包括作物种植" | "畜牧"
  短语内全对（短——词组强关联成形——不稀释）+ 相邻短语对（结构信息保留）
  ——幼儿先学短语再学长句——分段 = 语言习得固有步骤
关系扩展：isa 判定加"属于/包括/包含"（知识句的类别表达）
对照：整句全对（稀释基线）vs 短语分段（新）——词组湖纯度对比
"""
import os
import re
from collections import Counter
import numpy as np
from numba import njit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(43)
DT = 0.05
GAMMA = 0.8
OMEGA_LO, OMEGA_HI = 0.5, 4.0
AMP_IN = 1.2
PULSE_STEPS = 5          # γ=0.8 稳态 4τ≈5 步（幅度 98%）——8→5（1.5×）
EPS_K = 0.02
LAMBDA_K = 0.02
ETA_OMEGA = 0.15
K_CAP = 0.5
N_CHAR = 300
RELS = ["isa", "attr", "act", "cause"]
REL_IDX = {r: i for i, r in enumerate(RELS)}
ACT_VERBS = set("吃喝看听说写读走跑玩学做买卖拿放开关洗穿唱画打种浇扫拖叠铺搬递借还教帮陪带坐骑背跳踢投端尝拿看玩读听唱讲")


@njit(cache=True)
def step_dynamics(z, omega, gamma, K3, drive, dt):
    """numba 编译的单元动力学（向量化——消除 Python 调用层——10-30×）"""
    dz = -gamma * z + 1j * omega * z
    for r in range(K3.shape[0]):
        diff = z[None, :] - z[:, None]
        dz += (K3[r] * diff).sum(axis=1)
    dz += drive
    z = z + dz * dt
    for i in range(len(z)):
        if np.abs(z[i]) > 3.0:
            z[i] = z[i] / np.abs(z[i]) * 2.0
    return z

def rel_of(sent):
    if "因为" in sent or "所以" in sent:
        return "cause"
    if any(w in sent for w in ["是", "属于", "包括", "包含"]):
        return "isa"
    if "很" in sent:
        return "attr"
    for c in sent:
        if c in ACT_VERBS:
            return "act"
    return None

def split_phrases(sent):
    """标点分段（，、；：——短语边界）"""
    parts = [p.strip() for p in re.split(r"[，、；：]", sent) if p.strip()]
    return parts if len(parts) > 1 else [sent]

def load_corpus(path, lo=3, hi=80, n=None):
    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    clean = [s for s in lines if lo <= len(s) <= hi and re.search(r"[一-鿿]", s)]
    if n and len(clean) > n:
        clean = clean[:n]
    return clean

class RelLake:
    """关系分层字湖 + 短语分段开关（K 3D 数组 + numba step）"""
    def __init__(self, chars, phrase_mode=False):
        self.chars = chars
        self.ci = {c: i for i, c in enumerate(chars)}
        n = len(chars)
        self.omega = RNG.uniform(OMEGA_LO, OMEGA_HI, n)
        self.gamma = GAMMA
        self.z = 0.1 * np.exp(1j * RNG.uniform(0, 2 * np.pi, n))
        self.t = 0.0
        self.K = np.zeros((len(RELS), n, n))   # 3D（numba 支持）
        self.act = np.zeros(n)
        self.phrase = phrase_mode

    def step(self, drive):
        self.z = step_dynamics(self.z, self.omega, self.gamma, self.K, drive, DT)
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

    def deposit_phrase(self, rel, phr_idx, amp, w=1.0):
        """短语内全对沉积（距离衰减）"""
        if len(phr_idx) < 2:
            return
        L = len(phr_idx)
        sub = np.array(phr_idx)
        A = amp[sub]
        idx = np.arange(L)
        dist_w = 1.0 / np.maximum(np.abs(idx[:, None] - idx[None, :]), 1.0)
        contrib = EPS_K * w * np.outer(A, A) * np.triu(dist_w, 1)
        pi, pj = np.nonzero(contrib)
        self.K[REL_IDX[rel]][sub[pi], sub[pj]] += contrib[pi, pj]
        self.K[REL_IDX[rel]][sub[pj], sub[pi]] += contrib[pi, pj]

    def deposit_between(self, rel, a_idx, b_idx, amp, w=0.3):
        """相邻短语对（结构信息——低权重——向量化）"""
        A = amp[np.array(a_idx)][:, None]
        B = amp[np.array(b_idx)][None, :]
        contrib = EPS_K * w * (A * B)
        ia = np.array(a_idx)
        ib = np.array(b_idx)
        self.K[REL_IDX[rel]][np.ix_(ia, ib)] += contrib
        self.K[REL_IDX[rel]][np.ix_(ib, ia)] += contrib.T

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
            if self.phrase and len(sent) > 12:
                # 短语分段：短语内全对（词组成形）+ 相邻短语对（结构）
                phrases = split_phrases(sent)
                phr_idx = [[self.ci[c] for c in p if c in self.ci] for p in phrases]
                phr_idx = [p for p in phr_idx if len(p) >= 2]
                for p in phr_idx:
                    self.deposit_phrase(rel, p, amp)
                for k in range(len(phr_idx) - 1):
                    self.deposit_between(rel, phr_idx[k], phr_idx[k + 1], amp)
            else:
                # 整句全对（基线——长句稀释）
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
        Ksum = self.K.sum(axis=0)
        mask = Ksum > 0.08
        dw = ETA_OMEGA * (self.omega[None, :] - self.omega[:, None]) * np.where(mask, Ksum, 0.0)
        self.omega += dw.sum(axis=1)
        self.omega = np.clip(self.omega, OMEGA_LO, OMEGA_HI)

    def clusters(self, rel, th=0.05):
        n = len(self.chars)
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for i in range(n):
            for j in range(i + 1, n):
                if self.K[REL_IDX[rel]][i, j] + self.K[REL_IDX[rel]][j, i] > th:
                    parent[find(i)] = find(j)
        comps = {}
        for i in range(n):
            r = find(i)
            comps.setdefault(r, []).append(i)
        return list(comps.values())

    def coherence(self, members):
        ph = np.angle(self.z[members])
        return abs(np.mean(np.exp(1j * ph)))

    def purity(self, rel):
        cls = self.clusters(rel)
        big = [c for c in cls if len(c) >= 3]
        if not big:
            return 0.0
        return len(big) * np.mean([self.coherence(c) for c in big])


def run():
    print("=== M5 阶段 43：长句处理（短语分段——网络吃下知识句） ===\n")
    base = os.path.dirname(__file__)
    sents = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=900)
    sents += load_corpus(os.path.join(base, "corpus_wiki_filtered.txt"), n=600)
    lens = [len(s) for s in sents]
    print(f"语料 {len(sents)} 行（简单 + 知识句）——长度: 短(<12) {sum(1 for l in lens if l < 12)}"
          f" / 长(≥12) {sum(1 for l in lens if l >= 12)}——最长 {max(lens)} 字")
    freq = Counter("".join(sents))
    chars = [c for c, _ in freq.most_common(N_CHAR)]
    print(f"字集 {len(chars)}")
    # 对照：整句全对（基线）vs 短语分段（新）
    w_base = RelLake(chars, phrase_mode=False)
    w_phr = RelLake(chars, phrase_mode=True)
    for ep in range(20):
        w_base.learn_epoch(sents)
        w_phr.learn_epoch(sents)
    print("\n[结果] 结构纯度（湖数 × 平均相干）——整句（基线）vs 短语分段:")
    for r in RELS:
        pb = w_base.purity(r)
        pp = w_phr.purity(r)
        print(f"      {r}: 基线 {pb:.1f} vs 短语分段 {pp:.1f}（{'✓ 改善' if pp > pb * 1.3 else '—'}）")
    # 词组湖展示（isa——知识词组）
    print("\n[词组] isa 湖 top6（短语分段版——知识词组成形）:")
    cls = w_phr.clusters("isa")
    big = sorted([c for c in cls if len(c) >= 3], key=len, reverse=True)
    for ci, members in enumerate(big[:6]):
        print(f"      湖{ci}: [{''.join(chars[i] for i in members)}] 相干={w_phr.coherence(members):.2f}")
    # 知识词组检查（农业类）
    print("\n[知识词组检查] '农' 的关联（短语版——农业/种植/作物）:")
    if "农" in w_phr.ci:
        i = w_phr.ci["农"]
        row = w_phr.K[REL_IDX["isa"]][i].copy()
        top = np.argsort(row)[::-1][:8]
        print(f"      '农' → {[(chars[j], f'{row[j]:.2f}') for j in top if row[j] > 0.02]}")
    # 图
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].imshow(w_base.K[REL_IDX["isa"]], cmap="viridis"); axes[0].set_title("whole-sentence (baseline)")
    axes[1].imshow(w_phr.K[REL_IDX["isa"]], cmap="viridis"); axes[1].set_title("phrase-split")
    fig.tight_layout()
    fig.savefig("fig_stage43.png", dpi=110)
    print("\n[plot] saved fig_stage43.png")
    print("[done] stage43 long sentence")


if __name__ == "__main__":
    run()
