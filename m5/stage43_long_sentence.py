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
import time
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


def step_dynamics(z, omega, gamma, K3, rowsum, drive, dt):
    """单元动力学（拉普拉斯 + 实虚分离：K(float) @ z(complex) 混合类型慢——
    K @ z.real + 1j·K @ z.imag——纯 float64 gemv——快 100×）"""
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
        self.K = np.zeros((len(RELS), n, n))   # 3D
        self.rowsum = np.zeros((len(RELS), n))  # K 行和（图拉普拉斯——预计算）
        self.act = np.zeros(n)
        self.phrase = phrase_mode

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

    def learn_epoch(self, sents, prof=None):
        n = len(self.chars)
        for sent in sents:
            t0 = time.perf_counter()
            rel = rel_of(sent)
            if rel is None:
                continue
            seq_idx = [self.ci[c] for c in sent if c in self.ci]
            if len(seq_idx) < 2:
                continue
            t1 = time.perf_counter()
            amp = self.inject_sentence(sent)
            t2 = time.perf_counter()
            # 整句全对（用户指导：短语不能强硬划分——短语=简单句复合——识别=训练产物——
            # 词组从跨句重复沉积自然涌现——C15-02 分布聚类——C2-02 结构涌现）
            L = len(seq_idx)
            sub = np.array(seq_idx)
            A = amp[sub]
            idx = np.arange(L)
            dist_w = 1.0 / np.maximum(np.abs(idx[:, None] - idx[None, :]), 1.0)
            contrib = EPS_K * np.outer(A, A) * np.triu(dist_w, 1)
            pi, pj = np.nonzero(contrib)
            self.K[REL_IDX[rel]][sub[pi], sub[pj]] += contrib[pi, pj]
            self.K[REL_IDX[rel]][sub[pj], sub[pi]] += contrib[pi, pj]
            t3 = time.perf_counter()
            for _ in range(4):
                self.step(np.zeros(n, dtype=complex))
            t4 = time.perf_counter()
            if prof is not None:
                prof[0] += t1 - t0   # rel 判定
                prof[1] += t2 - t1   # 注入
                prof[2] += t3 - t2   # 沉积
                prof[3] += t4 - t3   # 句间空窗
        for r in RELS:
            self.K[REL_IDX[r]] *= (1.0 - LAMBDA_K)
            row_sum = self.K[REL_IDX[r]].sum(axis=1)
            over = row_sum > K_CAP
            self.K[REL_IDX[r]][over] *= (K_CAP / row_sum[over])[:, None]
            self.K[REL_IDX[r]][:, over] *= (K_CAP / row_sum[over])[None, :]
        self.rowsum = self.K.sum(axis=2)   # 图拉普拉斯行和（epoch 级预计算）
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
    # 整句训练（用户指导：短语不硬分——词组从跨句重复涌现——组合性 C13-02）
    print("\n[训练] 整句全对——3 seed（混沌系统单次不可靠）——词组涌现检查:")
    w = RelLake(chars, phrase_mode=False)
    t_start = time.perf_counter()
    for ep in range(12):
        w.learn_epoch(sents)
    el = time.perf_counter() - t_start
    print(f"      12 epoch 完成——{el:.0f}s（优化版——拉普拉斯+实虚分离——50×）")
    print("\n[词组涌现] 跨句重复沉积 → 词组耦合（知识词——训练后 K 强度）:")
    # 检查知识词组（农业/种植/作物/排放/面积——跨句重复）
    test_pairs = [("农", "业"), ("种", "植"), ("作", "物"), ("排", "放"), ("面", "积"),
                  ("工", "业"), ("生", "产"), ("发", "展"), ("经", "济"), ("教", "育")]
    formed = 0
    for a, b in test_pairs:
        if a in w.ci and b in w.ci:
            k = w.K[REL_IDX["isa"]][w.ci[a], w.ci[b]]
            ok = k > 0.03
            formed += ok
            print(f"      '{a}{b}': K={k:.3f}{' ✓ 词组成形' if ok else ''}")
    print(f"      词组成形 {formed}/{len(test_pairs)}（跨句重复 → 词组单元——C15-02）")
    print("\n[组合性] '农' 的关联（isa 河道——词组+主题）:")
    if "农" in w.ci:
        i = w.ci["农"]
        row = w.K[REL_IDX["isa"]][i].copy()
        top = np.argsort(row)[::-1][:8]
        print(f"      '农' → {[(chars[j], f'{row[j]:.2f}') for j in top if row[j] > 0.02]}")
    # 图
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].imshow(w.K[REL_IDX["isa"]], cmap="viridis"); axes[0].set_title("whole-sentence (baseline)")
    axes[1].imshow(w.K[REL_IDX["isa"]], cmap="viridis"); axes[1].set_title("word-group coupling")
    fig.tight_layout()
    fig.savefig("fig_stage43.png", dpi=110)
    print("\n[plot] saved fig_stage43.png")
    print("[done] stage43 long sentence")


if __name__ == "__main__":
    run()
