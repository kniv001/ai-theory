# -*- coding: utf-8 -*-
"""
M5 阶段 41：因果河道（"知道为什么"的语言层实现——R45 必然性/C74-01 为什么）

用户："对于语言，只要让框架知道关系就行，让他知道为什么"
机制（框架：关系=河道 R2——因果=有方向的河道——预测河下行/误差河上行）：
  K_cause：因果句（"因为A所以B"）→ 方向性沉积（A→B——原因→结果）
    ——因果方向 = 河道方向（R2——"为什么B？" = 沿河道上行检索 A）
  属性句（"很"句）→ K_attr（补 stage40 数据短板）
  关系层：isa/attr/act/cause 四河道（stage40 三河道 + 因果）
验证：
  exp1 因果河道：因为→所以 方向性（A→B 强于 B→A——因果不对称）
  exp2 因果湖：因果链涌现（雨→伞/饿→吃/冷→衣/努力→成功）
  exp3 "为什么"检索：沿 K_cause 上行（B→找 A——"为什么带伞？"→雨）
  exp4 属性补充：K_attr 细化（很句 100+——甜/大/蓝/香 属性湖）
"""
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(41)
DT = 0.05
GAMMA = 0.8
OMEGA_LO, OMEGA_HI = 0.5, 4.0
AMP_IN = 1.2
PULSE_STEPS = 8
EPS_K = 0.02
LAMBDA_K = 0.02
ETA_OMEGA = 0.15
K_CAP = 0.5
N_CHAR = 260
RELS = ["isa", "attr", "act", "cause"]
ACT_VERBS = set("吃喝看听说写读走跑玩学做买卖拿放开关洗穿唱画打种浇扫拖叠铺搬递借还教帮陪带坐骑背跳踢投端尝拿看玩读听唱讲")

def rel_of(sent):
    """关系分类：因为/所以→cause；是→isa；很→attr；动词→act"""
    if "因为" in sent or "所以" in sent:
        return "cause"
    if "是" in sent:
        return "isa"
    if "很" in sent:
        return "attr"
    for c in sent:
        if c in ACT_VERBS:
            return "act"
    return None

def load_corpus(path, lo=2, hi=80, n=None):
    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    clean = [s for s in lines if lo <= len(s) <= hi and re.search(r"[一-鿿]", s)]
    if n and len(clean) > n:
        clean = clean[:n]
    return clean

class RelLake:
    """关系分层字湖（stage40 + 因果河道——方向性）"""
    def __init__(self, chars):
        self.chars = chars
        self.ci = {c: i for i, c in enumerate(chars)}
        n = len(chars)
        self.omega = RNG.uniform(OMEGA_LO, OMEGA_HI, n)
        self.gamma = GAMMA
        self.z = 0.1 * np.exp(1j * RNG.uniform(0, 2 * np.pi, n))
        self.t = 0.0
        self.K = {r: np.zeros((n, n)) for r in RELS}
        self.act = np.zeros(n)

    def step(self, drive):
        dz = -self.gamma * self.z + 1j * self.omega * self.z
        for r in RELS:
            dz += (self.K[r] * (self.z[None, :] - self.z[:, None])).sum(axis=1)
        dz += drive
        self.z = self.z + dz * DT
        self.t += DT
        over = np.abs(self.z) > 3.0
        self.z[over] = self.z[over] / np.abs(self.z[over]) * 2.0
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
            if rel == "cause":
                # 因果方向性：因为A所以B——A（前件）→B（后件）——K_cause[A,B] 方向
                # 句内按"因为"位置分前后——前件字 → 后件字 方向沉积
                mid = sent.find("所以")
                if mid < 0:
                    mid = len(sent) // 2
                pre = [self.ci[c] for c in sent[:mid] if c in self.ci]
                post = [self.ci[c] for c in sent[mid:] if c in self.ci]
                for a in pre:
                    for b in post:
                        self.K["cause"][a, b] += EPS_K * amp[a] * amp[b]
            else:
                self.K[rel][sub[pi], sub[pj]] += contrib[pi, pj]
                self.K[rel][sub[pj], sub[pi]] += contrib[pi, pj]
            for _ in range(4):
                self.step(np.zeros(n, dtype=complex))
        for r in RELS:
            self.K[r] *= (1.0 - LAMBDA_K)
            row_sum = self.K[r].sum(axis=1)
            over = row_sum > K_CAP
            self.K[r][over] *= (K_CAP / row_sum[over])[:, None]
            self.K[r][:, over] *= (K_CAP / row_sum[over])[None, :]
        Ksum = sum(self.K[r] for r in RELS)
        mask = Ksum > 0.08
        dw = ETA_OMEGA * (self.omega[None, :] - self.omega[:, None]) * np.where(mask, Ksum, 0.0)
        self.omega += dw.sum(axis=1)
        self.omega = np.clip(self.omega, OMEGA_LO, OMEGA_HI)

    def clusters(self, rel, th=0.05, directed=False):
        n = len(self.chars)
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for i in range(n):
            for j in range(i + 1, n):
                w = self.K[rel][i, j] + self.K[rel][j, i] if not directed else max(self.K[rel][i, j], self.K[rel][j, i])
                if w > th:
                    parent[find(i)] = find(j)
        comps = {}
        for i in range(n):
            r = find(i)
            comps.setdefault(r, []).append(i)
        return list(comps.values())

    def coherence(self, members):
        ph = np.angle(self.z[members])
        return abs(np.mean(np.exp(1j * ph)))

    def why(self, c, k=3):
        """'为什么B？' → 沿 K_cause 上行检索 A（原因）"""
        if c not in self.ci:
            return []
        i = self.ci[c]
        col = self.K["cause"][:, i].copy()   # 上行（A→B 的列 = B 的原因们）
        top = np.argsort(col)[::-1][:k]
        return [(self.chars[j], col[j]) for j in top if col[j] > 0.02]

    def predict(self, rel, c, k=3):
        if c not in self.ci:
            return []
        i = self.ci[c]
        row = self.K[rel][i].copy()
        top = np.argsort(row)[::-1][:k]
        return [(self.chars[j], row[j]) for j in top if row[j] > 0.02]


def run():
    print("=== M5 阶段 41：因果河道（'知道为什么'——R45 必然性/C74-01） ===\n")
    base = os.path.dirname(__file__)
    sents = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=900)
    sents += load_corpus(os.path.join(base, "corpus_attr_cause.txt"))
    rels_count = {}
    for s in sents:
        r = rel_of(s)
        rels_count[r] = rels_count.get(r, 0) + 1
    print(f"语料 {len(sents)} 行——关系分类: {rels_count}")
    from collections import Counter
    freq = Counter("".join(sents))
    chars = [c for c, _ in freq.most_common(N_CHAR)]
    print(f"字集 {len(chars)}")
    w = RelLake(chars)
    for ep in range(30):
        w.learn_epoch(sents)
    # ---- exp1：因果方向性 ----
    print("\n[exp1] 因果方向性（A→B vs B→A——'因为A所以B'方向沉积）:")
    for a, b in [("雨", "伞"), ("饿", "饭"), ("冷", "衣"), ("勤", "优")]:
        if a in w.ci and b in w.ci:
            ab = w.K["cause"][w.ci[a], w.ci[b]]
            ba = w.K["cause"][w.ci[b], w.ci[a]]
            print(f"      '{a}'→'{b}' = {ab:.3f} vs '{b}'→'{a}' = {ba:.3f}"
                  f"（{'方向性 ✓' if ab > ba * 1.5 else '方向弱'}）")
    # ---- exp2：因果湖 ----
    cls = w.clusters("cause")
    big = [c for c in cls if len(c) >= 3]
    print(f"\n[exp2] 因果湖（K_cause）: {len(big)} 个≥3——top5:")
    for ci, members in enumerate(big[:5]):
        print(f"      因{ci}: [{''.join(chars[i] for i in members)}] 相干={w.coherence(members):.2f}")
    # ---- exp3："为什么"检索 ----
    print("\n[exp3] '为什么'检索（沿 K_cause 上行——为什么B？→原因 A）:")
    for c in ["伞", "饭", "功", "伞", "奖"]:
        if c in w.ci:
            why = w.why(c)
            print(f"      为什么'{c}'? → {[(p, f'{v:.2f}') for p, v in why]}")
    # ---- exp4：属性补充 ----
    attr = w.clusters("attr")
    a_big = [c for c in attr if len(c) >= 3]
    print(f"\n[exp4] 属性湖（K_attr——'很'河道——补充后）: {len(a_big)} 个≥3——top5:")
    for ci, members in enumerate(a_big[:5]):
        print(f"      属{ci}: [{''.join(chars[i] for i in members)}] 相干={w.coherence(members):.2f}")
    # 图：因果方向性
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(w.K["cause"], cmap="viridis"); axes[0].set_title("K_cause")
    axes[1].imshow(w.K["attr"], cmap="viridis"); axes[1].set_title("K_attr")
    axes[2].imshow(w.K["isa"], cmap="viridis"); axes[2].set_title("K_isa")
    fig.tight_layout()
    fig.savefig("fig_stage41.png", dpi=110)
    print("\n[plot] saved fig_stage41.png")
    print("[done] stage41 causal river")


if __name__ == "__main__":
    run()
