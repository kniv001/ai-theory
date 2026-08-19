# -*- coding: utf-8 -*-
"""
M5 阶段 42：幼儿行为学习（加速学习效率——沉积不均匀化——用户："幼儿行为学习加进去吧"）

幼儿 vs LLM 效率差距（5 因子分析）：沉积不均匀化是核心——幼儿"学在刀刃上"：
  ① 新奇加权（R10 误差驱动注意/R52 惊讶）：新颖句（低频字）高沉积——惊讶处学得快
  ② 重要性标记 τ（R11 双相固化）：首现字/词句 ×2 权重——之后衰减（学过的低注意力）
  ③ 重放复习（R42 固化/C98-01 测试效应——已验证）：定期"测验"——检索 > 再输入
对照：幼儿式（新+权+重放）vs 均匀沉积（基线）——同语料同 seed——
  收敛速度（湖纯度达阈值的 epoch）+ 结构清晰度（湖数/相干）对比
"""
import os
import re
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(42)
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
NOVEL_K = 2.0          # 新颖度沉积系数（新颖句高沉积）
FIRST_TAU = 2.0        # 首现句权重（τ 标记——新东西重要）
REPLAY_EVERY = 10      # 重放间隔（epoch）
REPLAY_N = 30          # 每次重放句数
REPLAY_W = 0.5         # 重放沉积系数（检索强化——低于新学但高于零）


def rel_of(sent):
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
    """关系分层字湖 + 幼儿机制开关（child_mode: 新颖度/首现/重放）"""
    def __init__(self, chars, child_mode=False):
        self.chars = chars
        self.ci = {c: i for i, c in enumerate(chars)}
        n = len(chars)
        self.omega = RNG.uniform(OMEGA_LO, OMEGA_HI, n)
        self.gamma = GAMMA
        self.z = 0.1 * np.exp(1j * RNG.uniform(0, 2 * np.pi, n))
        self.t = 0.0
        self.K = {r: np.zeros((n, n)) for r in RELS}
        self.act = np.zeros(n)
        self.child = child_mode
        self.seen = set()          # 已见字（首现标记）
        self.novel = np.ones(n)    # 每字新颖度（1 - 频率归一）
        self.replay_pool = []      # 重要句池（重放候选）

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

    def deposit(self, rel, sent, amp, w=1.0):
        """句内全对沉积（权重 w——幼儿机制调制沉积率）"""
        seq_idx = [self.ci[c] for c in sent if c in self.ci]
        if len(seq_idx) < 2:
            return
        L = len(seq_idx)
        sub = np.array(seq_idx)
        A = amp[sub]
        idx = np.arange(L)
        dist_w = 1.0 / np.maximum(np.abs(idx[:, None] - idx[None, :]), 1.0)
        contrib = EPS_K * w * np.outer(A, A) * np.triu(dist_w, 1)
        pi, pj = np.nonzero(contrib)
        if rel == "cause":
            mid = sent.find("所以")
            if mid < 0:
                mid = len(sent) // 2
            pre = [self.ci[c] for c in sent[:mid] if c in self.ci]
            post = [self.ci[c] for c in sent[mid:] if c in self.ci]
            for a in pre:
                for b in post:
                    self.K["cause"][a, b] += EPS_K * w * amp[a] * amp[b]
        else:
            self.K[rel][sub[pi], sub[pj]] += contrib[pi, pj]
            self.K[rel][sub[pj], sub[pi]] += contrib[pi, pj]

    def sentence_weight(self, sent):
        """幼儿权重：新颖度（惊讶——R52）× 首现标记（τ——R11）"""
        w = 1.0
        new_char = False
        nov = 0.0
        cnt = 0
        for c in sent:
            if c in self.ci:
                nov += self.novel[self.ci[c]]
                cnt += 1
                if c not in self.seen:
                    new_char = True
        if cnt:
            w *= 1.0 + NOVEL_K * (nov / cnt)      # 新颖句高沉积（惊讶驱动）
        if new_char:
            w *= FIRST_TAU                         # 首现字 → 该句重要（τ）
            for c in sent:
                if c in self.ci:
                    self.seen.add(c)
        return w

    def learn_epoch(self, sents, epoch=0):
        n = len(self.chars)
        for sent in sents:
            rel = rel_of(sent)
            if rel is None:
                continue
            seq_idx = [self.ci[c] for c in sent if c in self.ci]
            if len(seq_idx) < 2:
                continue
            w = self.sentence_weight(sent) if self.child else 1.0
            amp = self.inject_sentence(sent)
            self.deposit(rel, sent, amp, w)
            if self.child and w > 1.5:             # 重要句入重放池
                self.replay_pool.append(sent)
                if len(self.replay_pool) > 200:
                    self.replay_pool.pop(0)
            for _ in range(4):
                self.step(np.zeros(n, dtype=complex))
        # 幼儿：重放复习（测试效应——C98-01——检索 > 再输入）
        if self.child and epoch % REPLAY_EVERY == REPLAY_EVERY - 1 and self.replay_pool:
            pool = self.replay_pool
            if len(pool) > REPLAY_N:
                pick = RNG.choice(len(pool), REPLAY_N, replace=False)
            else:
                pick = np.arange(len(pool))
            for pi in pick:
                sent = pool[pi]
                rel = rel_of(sent)
                if rel is None:
                    continue
                amp = self.inject_sentence(sent)
                self.deposit(rel, sent, amp, REPLAY_W)   # 重放低沉积（检索强化——非新学）
        # 侵蚀 + 容量 + 吸引
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
                if self.K[rel][i, j] + self.K[rel][j, i] > th:
                    parent[find(i)] = find(j)
        comps = {}
        for i in range(n):
            r = find(i)
            comps.setdefault(r, []).append(i)
        return list(comps.values())

    def coherence(self, members):
        ph = np.angle(self.z[members])
        return abs(np.mean(np.exp(1j * ph)))


def purity(w, rel):
    """结构纯度指标：湖数 × 平均相干（结构清晰度）"""
    cls = w.clusters(rel)
    big = [c for c in cls if len(c) >= 3]
    if not big:
        return 0.0
    cohs = [w.coherence(c) for c in big]
    return len(big) * np.mean(cohs)


def run():
    print("=== M5 阶段 42：幼儿行为学习（沉积不均匀化——加速学习效率） ===\n")
    base = os.path.dirname(__file__)
    sents = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=900)
    sents += load_corpus(os.path.join(base, "corpus_attr_cause.txt"))
    freq = Counter("".join(sents))
    chars = [c for c, _ in freq.most_common(N_CHAR)]
    total = sum(freq.values())
    print(f"语料 {len(sents)} 行 / 字集 {len(chars)}——对照：均匀 vs 幼儿式（新奇+τ+重放）")
    # 新颖度预计算（低频字 = 高新颖）
    novel = np.ones(len(chars))
    for i, c in enumerate(chars):
        novel[i] = 1.0 - (freq[c] / total) * 10   # 频率归一（高频→0 新颖——低频→高新颖）
    novel = np.clip(novel, 0.0, 1.0)
    # 两版本（同 seed 对照）
    w_base = RelLake(chars, child_mode=False)
    w_child = RelLake(chars, child_mode=True)
    w_child.novel = novel.copy()
    N_EPOCHS = 25
    curves = {"base": {r: [] for r in RELS}, "child": {r: [] for r in RELS}}
    for ep in range(N_EPOCHS):
        w_base.learn_epoch(sents, ep)
        w_child.learn_epoch(sents, ep)
        if ep % 5 == 4 or ep == N_EPOCHS - 1:
            for r in RELS:
                curves["base"][r].append(purity(w_base, r))
                curves["child"][r].append(purity(w_child, r))
    # 结果
    print("\n[结果] 结构纯度（湖数 × 平均相干——越大越清晰）:")
    print(f"{'epoch':>6} | {'isa 基线/幼儿':>14} | {'attr 基线/幼儿':>14} | {'act 基线/幼儿':>14} | {'cause 基线/幼儿':>14}")
    for i, ep in enumerate(range(4, N_EPOCHS, 5)):
        ep_i = min(i, len(curves["base"]["isa"]) - 1)
        print(f"{ep+1:>6} | "
              f"{curves['base']['isa'][ep_i]:6.1f}/{curves['child']['isa'][ep_i]:6.1f} | "
              f"{curves['base']['attr'][ep_i]:6.1f}/{curves['child']['attr'][ep_i]:6.1f} | "
              f"{curves['base']['act'][ep_i]:6.1f}/{curves['child']['act'][ep_i]:6.1f} | "
              f"{curves['base']['cause'][ep_i]:6.1f}/{curves['child']['cause'][ep_i]:6.1f}")
    # 收敛速度（纯度达到 80% 最终值的 epoch）
    print("\n[收敛] 达到最终纯度 80% 所需 epoch（幼儿 < 基线 = 加速）:")
    for r in RELS:
        b_final = curves["base"][r][-1]
        c_final = curves["child"][r][-1]
        def reach(curve, target):
            for i, v in enumerate(curve):
                if v >= target:
                    return (i + 1) * 5
            return N_EPOCHS
        b_ep = reach(curves["base"][r], 0.8 * max(b_final, 1e-6))
        c_ep = reach(curves["child"][r], 0.8 * max(c_final, 1e-6))
        print(f"      {r}: 基线 {b_ep} epoch vs 幼儿 {c_ep} epoch"
              f"（{'加速 ✓' if c_ep < b_ep else '—'}）最终纯度 {b_final:.1f} vs {c_final:.1f}")
    # 图
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for r in RELS:
        axes[0].plot([5 * (i + 1) for i in range(len(curves["base"][r]))], curves["base"][r],
                     label=f"{r}-base", linestyle="--", alpha=0.6)
        axes[0].plot([5 * (i + 1) for i in range(len(curves["child"][r]))], curves["child"][r],
                     label=f"{r}-child")
    axes[0].set_title("Structure purity (base vs child)")
    axes[0].legend(fontsize=7)
    axes[1].bar(["isa", "attr", "act", "cause"],
                [curves["base"][r][-1] for r in RELS], alpha=0.6, label="base")
    axes[1].bar(["isa", "attr", "act", "cause"],
                [curves["child"][r][-1] for r in RELS], alpha=0.6, label="child", width=0.4)
    axes[1].set_title("Final purity")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("fig_stage42.png", dpi=110)
    print("\n[plot] saved fig_stage42.png")
    print("[done] stage42 child learning")


if __name__ == "__main__":
    run()
