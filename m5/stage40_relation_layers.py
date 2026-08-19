# -*- coding: utf-8 -*-
"""
M5 阶段 40：关系分层（用户问题："归类是否有加入关系连接"——关系 = 河道——R16/R2）

问题：K 无类型——"苹果是水果"(is-a)/"苹果很甜"(属性)/"我吃苹果"(动作) 混合沉积——
  "是/这"成为万能枢纽——污染湖结构 + 无法区分关系类型。
方案（框架理论：关系 = 河道——经由词 = 关系类型）：
  K 分层：K_isa（经由"是"——类别/指物）/ K_attr（经由"很"——属性）/
          K_act（经由动作动词——主谓宾）
  每句按功能词分类 → 沉积到对应层——湖检测按层进行
验证：
  exp1 类别湖（K_isa）：动物/水果/物品/颜色——通过"是"连通——自发归类（类别锚）
  exp2 属性湖（K_attr）：甜/脆/香/软——属性词与对象关联
  exp3 动作湖（K_act）：吃-苹果/看-书/洗-衣服——主谓宾关系
  exp4 关系分离：同一对象词在多层（苹果 is-a 水果 / attr 甜 / act 吃）——无冲突共存
"""
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(40)
DT = 0.05
GAMMA = 0.8
OMEGA_LO, OMEGA_HI = 0.5, 4.0
AMP_IN = 1.2
PULSE_STEPS = 8
EPS_K = 0.02
LAMBDA_K = 0.02
ETA_OMEGA = 0.15
K_CAP = 0.5
N_CHAR = 220
RELS = ["isa", "attr", "act"]
ACT_VERBS = set("吃喝看听说写读走跑玩学做买卖拿放开关洗穿唱画打种浇扫拖叠铺搬递借还教帮陪带坐骑背跳踢投端尝拿看玩读听唱讲")

def rel_of(sent):
    """句子的关系类型：'是'→isa；'很'→attr；含动作动词→act；多类按优先级（是>很>动）"""
    if "是" in sent:
        return "isa"
    if "很" in sent:
        return "attr"
    for c in sent:
        if c in ACT_VERBS:
            return "act"
    return None   # 无功能词（纯名词行——跳过关系层——或归 act？——跳过）

def load_corpus(path, lo=2, hi=80, n=None):
    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    clean = [s for s in lines if lo <= len(s) <= hi and re.search(r"[一-鿿]", s)]
    if n and len(clean) > n:
        clean = clean[:n]
    return clean

class RelLake:
    """关系分层字湖：K[rel][i][j]——按经由词分层沉积"""
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
        # 耦合 = 各层之和（同步域 = 全关系共同塑造）
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
        # 频率吸引（全层之和——强耦合）
        Ksum = sum(self.K[r] for r in RELS)
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
                if self.K[rel][i, j] > th:
                    parent[find(i)] = find(j)
        comps = {}
        for i in range(n):
            r = find(i)
            comps.setdefault(r, []).append(i)
        return list(comps.values())

    def coherence(self, members):
        ph = np.angle(self.z[members])
        return abs(np.mean(np.exp(1j * ph)))


def run():
    print("=== M5 阶段 40：关系分层（用户问题——归类的关系连接——关系=河道） ===\n")
    base = os.path.dirname(__file__)
    sents = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=900)
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
    # ---- exp1：类别湖（K_isa） ----
    isa = w.clusters("isa")
    isa_big = [c for c in isa if len(c) >= 3]
    print(f"\n[exp1] 类别湖（K_isa——'是'河道）: {len(isa_big)} 个≥3")
    for ci, members in enumerate(isa_big[:8]):
        print(f"      类{ci}: [{''.join(chars[i] for i in members)}] 相干={w.coherence(members):.2f}")
    # ---- exp2：属性湖（K_attr） ----
    attr = w.clusters("attr")
    attr_big = [c for c in attr if len(c) >= 3]
    print(f"\n[exp2] 属性湖（K_attr——'很'河道）: {len(attr_big)} 个≥3")
    for ci, members in enumerate(attr_big[:8]):
        print(f"      属{ci}: [{''.join(chars[i] for i in members)}] 相干={w.coherence(members):.2f}")
    # ---- exp3：动作湖（K_act） ----
    act = w.clusters("act")
    act_big = [c for c in act if len(c) >= 3]
    print(f"\n[exp3] 动作湖（K_act——动词河道）: {len(act_big)} 个≥3")
    for ci, members in enumerate(act_big[:8]):
        print(f"      动{ci}: [{''.join(chars[i] for i in members)}] 相干={w.coherence(members):.2f}")
    # ---- exp4：关系分离检查 ----
    print("\n[exp4] 关系分离（同一对象词跨层——无冲突共存）:")
    for c in ["苹", "小", "水", "颜"]:
        if c not in w.ci: continue
        i = w.ci[c]
        layers = []
        for r in RELS:
            top = np.argsort(w.K[r][i])[::-1][:2]
            layers.append(f"{r}:[{''.join(chars[j] for j in top if w.K[r][i][j] > 0.02)}]")
        print(f"      '{c}': " + " ".join(layers))
    # 图：三层 K
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, r in zip(axes, RELS):
        im = ax.imshow(w.K[r], cmap="viridis")
        ax.set_title(f"K_{r}")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig("fig_stage40.png", dpi=110)
    print("\n[plot] saved fig_stage40.png")
    print("[done] stage40 relation layers")


if __name__ == "__main__":
    run()
