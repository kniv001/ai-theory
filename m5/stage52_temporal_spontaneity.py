# -*- coding: utf-8 -*-
"""
M5 阶段 52：时序 × 自发性组合（用户："现在的自发性和时序有组合在一起吗"——诚实：没有——
组合 = 时序位置 → 位置角色分布 → 自发聚类——语法角色从位置涌现）

stage49（自发）与时序（stage50-51）独立——组合价值：
  时序（位置相位）→ 每词的位置分布（句首/句尾频率——"农业"常在句首=主语位/
  "甜"常在句尾=补语位）→ 自发聚类（位置分布 + 方向邻居）→ 语法角色涌现
机制：
  ① 时序湖训练（stage50——位置相位 + K 方向化——语序方向）
  ② 位置角色分布：每词出现在句首/句中/句尾的频率（位置统计——阅读顺序）
  ③ 自发聚类：方向邻居（前向/后向）+ 位置分布 → 聚类（主语类/谓语类/补语类——无预设）
验证：
  exp1 位置角色分布（农业→句首多/甜→句尾多——语法位置涌现）
  exp2 自发聚类（位置+邻居——主语类/谓语类/补语类）
  exp3 对照（涌现类 vs 预设词类——名词/动词/形容词——对应）
"""
import os
import re
import time
from collections import Counter
import numpy as np

RNG = np.random.default_rng(52)
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

class TemporalLake:
    """时序湖（stage50——位置相位 + K 方向化）"""
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
        self.pos_dist = np.zeros((n, 3))   # 位置分布（句首/中/尾——语法角色信号）

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
                # 位置分布（句首 0-20% / 中 / 句尾 80-100%）
                frac = pos / max(len(sent) - 1, 1)
                if frac < 0.2:
                    self.pos_dist[i, 0] += 1
                elif frac > 0.8:
                    self.pos_dist[i, 2] += 1
                else:
                    self.pos_dist[i, 1] += 1
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
        row_sum = self.K.sum(axis=1)
        over = row_sum > K_CAP
        self.K[over] *= (K_CAP / row_sum[over])[:, None]
        self.rowsum = self.K.sum(axis=1)

    def role_cluster(self, min_shared=0.15):
        """自发聚类：方向邻居（前向/后向）+ 位置分布——语法角色涌现"""
        n = len(self.chars)
        # 方向邻居：前向（K[i,j] 强——i 在 j 前）和后向
        fwd_neigh = [set(np.where(self.K[i] > 0.015)[0]) for i in range(n)]
        # 位置主导（句首/句尾/中间——归一化——宽松判定）
        pd = self.pos_dist / (self.pos_dist.sum(axis=1, keepdims=True) + 1e-9)
        head_dom = pd[:, 0] > pd[:, 2] * 1.5           # 句首 > 句尾 1.5×（主语倾向）
        tail_dom = pd[:, 2] > pd[:, 0] * 1.5           # 句尾 > 句首 1.5×（补语倾向）
        mid_dom = pd[:, 1] > pd[:, 0] + pd[:, 2]       # 中间主导（动词倾向——"吃"在句中）
        # 邻居 + 位置合并聚类（贪心）
        labels = np.full(n, -1, dtype=int)
        roles = {}
        for i in range(n):
            if not fwd_neigh[i] and not head_dom[i] and not tail_dom[i]:
                continue
            best_k, best_sim = -1, 0.0
            for k, members in roles.items():
                sim = len(fwd_neigh[i] & fwd_neigh[members[0]]) / max(len(fwd_neigh[i] | fwd_neigh[members[0]]), 1)
                if sim > best_sim:
                    best_k, best_sim = k, sim
            if best_k >= 0 and best_sim >= min_shared:
                roles[best_k].append(i)
                labels[i] = best_k
            else:
                labels[i] = len(roles)
                roles[labels[i]] = [i]
        # 位置标注（角色命名——主语/谓语/补语——涌现）
        named = []
        for k, members in roles.items():
            m_head = sum(1 for i in members if head_dom[i])
            m_tail = sum(1 for i in members if tail_dom[i])
            m_mid = sum(1 for i in members if mid_dom[i])
            if m_head > len(members) * 0.4:
                tag = "主语类(句首)"
            elif m_tail > len(members) * 0.4:
                tag = "补语类(句尾)"
            elif m_mid > len(members) * 0.4:
                tag = "谓语类(中间)"
            else:
                tag = "其他"
            named.append((tag, members))
        return labels, named


def run():
    print("=== M5 阶段 52：时序 × 自发性组合（位置分布 → 语法角色涌现） ===\n")
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
    print(f"训练完成——{time.perf_counter()-t0:.0f}s（时序湖——位置相位+方向 K）")
    # ---- exp1：位置角色分布 ----
    print("\n[exp1] 位置角色分布（句首/句尾频率——语法位置涌现）:")
    pd = w.pos_dist / (w.pos_dist.sum(axis=1, keepdims=True) + 1e-9)
    tests = ["农", "甜", "吃", "看", "高", "我"]
    for c in tests:
        if c in w.ci:
            i = w.ci[c]
            print(f"      '{c}': 句首 {pd[i,0]:.2f} / 中 {pd[i,1]:.2f} / 句尾 {pd[i,2]:.2f}")
    # ---- exp2：自发聚类（时序 + 位置） ----
    labels, named = w.role_cluster()
    print("\n[exp2] 自发聚类（方向邻居 + 位置分布——语法角色）:")
    for tag, members in named[:6]:
        print(f"      {tag}（{len(members)} 字）: {''.join(chars[i] for i in members[:25])}")
    # ---- exp3：对照（涌现类 vs 预设词类） ----
    print("\n[exp3] 对照（涌现语法角色 vs 预设词类）:")
    preset = {"名词": ["农", "技", "苹", "山", "水"], "动词": ["吃", "看", "学", "写"],
              "形容词": ["甜", "高", "长", "远"]}
    for name, words in preset.items():
        found = [(c, "主语类" if labels[w.ci[c]] >= 0 and
                  w.pos_dist[w.ci[c], 0] > w.pos_dist[w.ci[c], 2] else "补语类")
                 for c in words if c in w.ci and labels[w.ci[c]] >= 0]
        print(f"      预设 {name}({words}) → 涌现位置类: {found}")
    print("\n[done] stage52 temporal spontaneity")


if __name__ == "__main__":
    run()
