# -*- coding: utf-8 -*-
"""
M5 阶段 49v3：关系彻底自发（用户两次纠正：关系词预设→模板也预设→彻底自发）

零人工词表方案（C15-02 词类子湖=分布聚类——框架理论早有）：
  ① 字湖自发训练（K——跨句重复——无预设）
  ② 词角色聚类：每词的 K 行（共现分布）→ 聚类——"农业"与"技术"同类（都连
     产业/发展——名词角色）；"甜"不同（连很/苹果——属性角色）——词类涌现
  ③ 关系类型 = 词对 (a,b) 的 (角色a, 角色b) 组合——NN 对（类别关系）/
     N-属性 对（属性关系）/ N-动作 对（动作关系）——无预设
验证：
  exp1 角色聚类（K 行——农业/技术/苹果同类？甜/高同类？吃/看同类？）
  exp2 关系类型（词对按角色组合分组——NN 对 vs N-属性 对 vs N-动词 对）
  exp3 内容检查（各组词对——类别对/属性对/动作对——语义对应）
"""
import os
import re
import time
from collections import Counter
import numpy as np

RNG = np.random.default_rng(49)
DT = 0.05
GAMMA = 0.8
OMEGA_LO, OMEGA_HI = 0.5, 4.0
AMP_IN = 1.2
PULSE_STEPS = 5
EPS_K = 0.02
LAMBDA_K = 0.01
ETA_OMEGA = 0.15
K_CAP = 0.5
N_CHAR = 300
RELS = ["isa", "attr", "act", "cause"]
REL_IDX = {r: i for i, r in enumerate(RELS)}
N_ROLES = 4          # 角色聚类数（无预设标签——事后命名）

def load_corpus(path, lo=3, hi=80, n=None):
    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    clean = [s for s in lines if lo <= len(s) <= hi
             and re.search(r"[一-鿿]", s) and not re.search(r"[A-Za-z]", s)]
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
            self.K[REL_IDX["isa"]][sub[pi], sub[pj]] += contrib[pi, pj]
            self.K[REL_IDX["isa"]][sub[pj], sub[pi]] += contrib[pi, pj]
            for _ in range(4):
                self.step(np.zeros(n, dtype=complex))
        self.K[REL_IDX["isa"]] *= (1.0 - LAMBDA_K)
        row_sum = self.K[REL_IDX["isa"]].sum(axis=1)
        over = row_sum > K_CAP
        self.K[REL_IDX["isa"]][over] *= (K_CAP / row_sum[over])[:, None]
        self.K[REL_IDX["isa"]][:, over] *= (K_CAP / row_sum[over])[None, :]
        self.rowsum = self.K.sum(axis=2)


def role_cluster(K, chars, min_shared=0.15):
    """词角色聚类：共现邻居重叠（Jaccard——"农业"与"技术"共享共现词——
    欧氏距离对高维稀疏失效——邻居重叠是正确相似度——无预设）"""
    n = len(chars)
    # 共现邻居（K 行非零 > 阈值）
    neigh = []
    for i in range(n):
        nb = set(np.where(K[i] > 0.01)[0])
        neigh.append(nb)
    # 贪心聚类（邻居 Jaccard 重叠 > 阈值合并）
    labels = np.full(n, -1, dtype=int)
    roles = {}
    for i in range(n):
        if not neigh[i]:
            continue
        best_k, best_sim = -1, 0.0
        for k, members in roles.items():
            sim = len(neigh[i] & neigh[members[0]]) / max(len(neigh[i] | neigh[members[0]]), 1)
            if sim > best_sim:
                best_k, best_sim = k, sim
        if best_k >= 0 and best_sim >= min_shared:
            roles[best_k].append(i)
            labels[i] = best_k
        else:
            labels[i] = len(roles)
            roles[labels[i]] = [i]
    role_members = {k: [chars[i] for i in members] for k, members in roles.items()}
    return labels, role_members


def run():
    print("=== M5 阶段 49v3：关系彻底自发（零人工词表——K 行聚类） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=900)
    wiki = load_corpus(os.path.join(base, "corpus_wiki_filtered.txt"), n=1800)
    sents = simple + wiki
    freq = Counter("".join(sents))
    chars = [c for c, _ in freq.most_common(N_CHAR)]
    print(f"语料 {len(sents)} 行 / 字集 {len(chars)}")
    w = RelLake(chars)
    blocks = [wiki[i:i + 600] for i in range(0, 1800, 600)]
    t0 = time.perf_counter()
    for block in blocks:
        for ep in range(10):
            w.learn_epoch(simple + block)
    print(f"训练完成——{time.perf_counter()-t0:.0f}s（自发——无任何词表）")
    # ---- exp1：词角色聚类（K 行——无预设） ----
    labels, roles = role_cluster(w.K[REL_IDX["isa"]], chars)
    print("\n[exp1] 词角色聚类（K 行分布——无预设——事后命名）:")
    for k in range(N_ROLES):
        members = roles[k]
        print(f"      角色{k}（{len(members)} 字）: {''.join(members[:25])}")
    # ---- exp2/3：关系类型 = 词对角色组合 ----
    print("\n[exp2] 关系类型（词对按 (角色a, 角色b) 组合——无预设）:")
    K = w.K[REL_IDX["isa"]]
    # 找强词对（K > 0.025——跨句重复）
    pairs = []
    n = len(chars)
    for i in range(n):
        for j in range(i + 1, n):
            if K[i, j] > 0.025:
                pairs.append((i, j, K[i, j]))
    # 按角色组合分组
    groups = {}
    for i, j, k in pairs:
        key = (labels[i], labels[j])
        groups.setdefault(key, []).append((chars[i] + chars[j], k))
    for key, items in sorted(groups.items(), key=lambda x: -len(x[1]))[:6]:
        items.sort(key=lambda x: -x[1])
        top = [f"{w}({k:.2f})" for w, k in items[:6]]
        print(f"      角色{key[0]}-角色{key[1]} 对（{len(items)} 个）: {top}")
    print("\n[done] stage49v3 relation emergence (zero preset)")


if __name__ == "__main__":
    run()
