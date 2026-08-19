# -*- coding: utf-8 -*-
"""
M5 阶段 39：层级湖（文字湖桥梁 ③——R16 尺度递归——字→词→句→篇章）

TEXT_LAKE_ROADMAP 差距③（湖规模/层级）：低层湖 → 高层湖——尺度递归。
数据：corpus_wiki_sample.txt（169k 句 wiki 抽样——正常规范语句——档 1 语料）。
机制（两层尺度递归——R16 同机制递归）：
  低层：字湖（stage38——频率吸引 + 耦合沉积 + 容量预算——K 模块 = 同步域）
  句子 → 低层湖激活向量（每湖总幅度）——句子的"湖级签名"
  高层：湖间耦合 L（相邻句的湖级签名共现 → L 沉积——真实：相邻句同话题）——
    话题湖 = 湖的组合（L 强耦合连通分量——跨域合并 R22）
验证：
  exp1 字湖细化：真实语料（500 句）→ 字湖模块（比 30 句手工语料更细/更多）
  exp2 湖级签名：句子 → 激活向量（不同话题句 → 不同签名）
  exp3 话题湖：相邻句共现 → L 模块（话题湖 = 字湖组合）——语义检查
  exp4 尺度递归：字湖 → 话题湖——两层同机制（K 模块——R16）
"""
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(39)
DT = 0.05
GAMMA = 0.8
OMEGA_LO, OMEGA_HI = 0.5, 4.0
AMP_IN = 1.2
PULSE_STEPS = 8
EPS_K = 0.02
LAMBDA_K = 0.02
ETA_OMEGA = 0.15
K_CAP = 0.5
COH_TH = 0.6
N_CHAR = 200          # 字单元数（高频字）
N_TRAIN = 302         # 训练句数（阶段 A：简单语料全量 302 句——先简单后混合——用户原则）
EPS_L = 0.05          # 湖级耦合沉积
LAMBDA_L = 0.02       # 湖级侵蚀
L_CAP = 1.0           # 湖级容量预算


def load_corpus(path, n=N_TRAIN, lo=3, hi=60):   # lo=3（简单主谓宾句 3-8 字——"我吃饭"——不能被 8 过滤）
    """加载 wiki 抽样语料——过滤（长度/ASCII 噪声）——抽样 n 句"""
    with open(path, encoding="utf-8") as f:
        sents = [l.strip() for l in f if l.strip()]
    clean = [s for s in sents if lo <= len(s) <= hi and not re.search(r"[A-Za-z0-9一-鿿]", s) is None]
    clean = [s for s in clean if re.search(r"[一-鿿]", s)]
    rng = np.random.default_rng(7)
    if len(clean) > n:
        clean = rng.choice(clean, n, replace=False).tolist()
    return clean


class CharLake:
    """低层：字湖（stage38 机制——频率吸引 + 耦合沉积 + 容量预算）"""
    def __init__(self, chars):
        self.chars = chars
        self.ci = {c: i for i, c in enumerate(chars)}
        n = len(chars)
        self.omega = RNG.uniform(OMEGA_LO, OMEGA_HI, n)
        self.gamma = GAMMA
        self.z = 0.1 * np.exp(1j * RNG.uniform(0, 2 * np.pi, n))
        self.t = 0.0
        self.K = np.zeros((n, n))
        self.act = np.zeros(n)
        self.lake_members = None   # 湖模块（训练后设置）

    def step(self, drive):
        dz = -self.gamma * self.z + 1j * self.omega * self.z
        dz += (self.K * (self.z[None, :] - self.z[:, None])).sum(axis=1)
        dz += drive
        self.z = self.z + dz * DT
        self.t += DT
        over = np.abs(self.z) > 3.0
        self.z[over] = self.z[over] / np.abs(self.z[over]) * 2.0
        return self.z

    def inject(self, c):
        if c not in self.ci:
            return
        i = self.ci[c]
        drive = np.zeros(len(self.chars), dtype=complex)
        drive[i] = AMP_IN * np.exp(1j * (self.omega[i] * self.t))
        for _ in range(PULSE_STEPS):
            self.step(drive)
        for _ in range(3):
            self.step(np.zeros(len(self.chars), dtype=complex))

    def learn_epoch(self, sents):
        n = len(self.chars)
        for sent in sents:
            seq = [c for c in sent if c in self.ci]
            self.act = np.zeros(n)
            for c in seq:
                self.inject(c)
                self.act += np.abs(self.z)
                self.act *= 0.9
            for i in range(len(seq)):
                wi = self.ci[seq[i]]
                for j in range(i + 1, len(seq)):
                    wj = self.ci[seq[j]]
                    pair = EPS_K * self.act[wi] * self.act[wj] / (j - i)
                    self.K[wi, wj] += pair
                    self.K[wj, wi] += pair
            for _ in range(4):
                self.step(np.zeros(n, dtype=complex))
        self.K *= (1.0 - LAMBDA_K)
        row_sum = self.K.sum(axis=1)
        over = row_sum > K_CAP
        self.K[over] *= (K_CAP / row_sum[over])[:, None]
        self.K[:, over] *= (K_CAP / row_sum[over])[None, :]
        for i in range(n):
            for j in range(i + 1, n):
                kij = self.K[i, j]
                if kij > 0.08:
                    pull = ETA_OMEGA * (self.omega[j] - self.omega[i]) * kij
                    self.omega[i] += pull
                    self.omega[j] -= pull
        self.omega = np.clip(self.omega, OMEGA_LO, OMEGA_HI)

    def clusters(self, th=0.08):
        n = len(self.chars)
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for i in range(n):
            for j in range(i + 1, n):
                if self.K[i, j] > th:
                    parent[find(i)] = find(j)
        comps = {}
        for i in range(n):
            r = find(i)
            comps.setdefault(r, []).append(i)
        return list(comps.values())

    def sentence_sig(self, sent):
        """句子的湖级签名：注入句子 → 每湖总幅度"""
        self.act = np.zeros(len(self.chars))
        for c in sent:
            if c in self.ci:
                self.inject(c)
                self.act += np.abs(self.z)
        if self.lake_members is None:
            return np.array([self.act.sum()])
        return np.array([self.act[m].sum() for m in self.lake_members])

    def coherence(self, members):
        ph = np.angle(self.z[members])
        return abs(np.mean(np.exp(1j * ph)))


class TopicLayer:
    """高层：话题湖——湖间耦合 L（窗口共现——主题内多句共享湖对 → 强信号）"""
    def __init__(self, n_lakes):
        self.n = n_lakes
        self.L = np.zeros((n_lakes, n_lakes))
        self.win = []          # 窗口 sig（主题内 5 句共享湖对——多句共现增强）
        self.WIN = 5

    def learn(self, sig):
        """湖级签名 → 窗口内共现沉积（真实：相邻句同话题——主题 = 多句共享模式）"""
        self.win.append(sig)
        if len(self.win) > self.WIN:
            self.win.pop(0)
        for prev in self.win[:-1]:
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    pair = EPS_L * prev[i] * sig[j]
                    if pair > 0:
                        self.L[i, j] += pair
                        self.L[j, i] += pair

    def epoch_end(self):
        self.L *= (1.0 - LAMBDA_L)
        row_sum = self.L.sum(axis=1)
        over = row_sum > L_CAP
        self.L[over] *= (L_CAP / row_sum[over])[:, None]
        self.L[:, over] *= (L_CAP / row_sum[over])[None, :]
        self.win = []

    def topics(self, th=0.05):
        parent = list(range(self.n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.L[i, j] > th:
                    parent[find(i)] = find(j)
        comps = {}
        for i in range(self.n):
            r = find(i)
            comps.setdefault(r, []).append(i)
        return list(comps.values())


def run():
    print("=== M5 阶段 39：层级湖（文字湖桥梁 ③——R16 尺度递归——字→词→句→篇章） ===\n")
    path = os.path.join(os.path.dirname(__file__), "corpus_simple_natural.txt")
    sents = load_corpus(path, N_TRAIN)
    print(f"语料：{len(sents)} 句（阶段 A——自然简单主谓宾——先简单后混合——用户原则）")
    # 字集：高频字（前 N_CHAR）
    from collections import Counter
    freq = Counter("".join(sents))
    chars = [c for c, _ in freq.most_common(N_CHAR)]
    print(f"字集：{len(chars)} 高频字")
    # ---- exp1：字湖涌现（真实语料） ----
    w = CharLake(chars)
    for ep in range(25):
        w.learn_epoch(sents)
    clusters = w.clusters()
    big = [c for c in clusters if len(c) >= 2]
    print(f"[exp1] 字湖模块 = {len(big)}（≥2 字）——top5:")
    for ci, members in enumerate(big[:5]):
        print(f"      湖{ci}: [{''.join(chars[i] for i in members)}] 相干={w.coherence(members):.2f}")
    w.lake_members = [c for c in big if len(c) >= 2]
    # ---- exp2/3：话题湖（高层——湖级签名共现） ----
    topic = TopicLayer(len(w.lake_members))
    for ep in range(25):
        for s in sents:
            sig = w.sentence_sig(s)
            norm = np.maximum(sig.sum(), 1e-9)
            topic.learn(sig / norm)
        topic.epoch_end()
    topics = topic.topics()
    tbig = [t for t in topics if len(t) >= 2]
    print(f"[exp3] 话题湖（L 模块≥2）= {len(tbig)}——湖组合:")
    for ti, members in enumerate(tbig[:6]):
        names = [f"L{m}" for m in members]
        # 每个成员湖的字（前 6 字）
        words = ["".join(chars[i] for i in w.lake_members[m][:6]) for m in members]
        print(f"      话题{ti}: {names} = [{'; '.join(words)}]")
    # ---- exp4：尺度递归检查 ----
    print(f"[exp4] 两层同机制（K 模块/L 模块——R16 尺度递归）——"
          f"字湖 {len(big)} → 话题湖 {len(tbig)}——高层 = 低层湖的组合")
    # 图：K 与 L
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    im = axes[0].imshow(w.K, cmap="viridis")
    axes[0].set_title(f"Char coupling K ({len(big)} lakes)")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    im2 = axes[1].imshow(topic.L, cmap="viridis")
    axes[1].set_title(f"Lake coupling L ({len(tbig)} topics)")
    fig.colorbar(im2, ax=axes[1], fraction=0.046)
    fig.tight_layout()
    fig.savefig("fig_stage39.png", dpi=110)
    print("\n[plot] saved fig_stage39.png")
    print("[done] stage39 hierarchical lake")


if __name__ == "__main__":
    run()
