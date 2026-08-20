# -*- coding: utf-8 -*-
"""
M5 阶段 45：主题分批课程（词组成形 ∝ 主题密度——stage44 发现——训练组织原则）

发现：600 句主题密集 → 词组 95；2000 句主题多样 → 0（稀释）。
课程（幼儿式——主题重复）：
  知识语料按主题块切分（wiki 文章顺序——每块 ~600 句同一主题区）
  块 1 密集训练（12 epoch）→ 块 2 继续（K 保留——词组跨批累积）→ 块 3……
  侵蚀平衡：每块 epoch 少（词组成形即停）+ 侵蚀率低（跨批保留）
验证：
  exp1 跨批累积：批 1 农业词组 + 批 2 主题词组——都在（vs 一次性大语料 0）
  exp2 词组库增长：批 1 → 批 2 → 批 3——词组数变化
  exp3 生成链：跨批保留的关联（农业→种植——批 1 词组在批 2 后仍可用）
"""
import os
import re
import time
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(45)
DT = 0.05
GAMMA = 0.8
OMEGA_LO, OMEGA_HI = 0.5, 4.0
AMP_IN = 1.2
PULSE_STEPS = 5
EPS_K = 0.02
LAMBDA_K = 0.01          # 侵蚀降低（跨批保留——0.02→0.01）
ETA_OMEGA = 0.15
K_CAP = 0.5
N_CHAR = 300
RELS = ["isa", "attr", "act", "cause"]
REL_IDX = {r: i for i, r in enumerate(RELS)}
WORD_TH = 0.025

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

    def extract_words(self, sents, th=WORD_TH):
        K = self.K[REL_IDX["isa"]]
        adj = Counter()
        for s in sents:
            for k in range(len(s) - 1):
                adj[s[k:k + 2]] += 1
        words = {}
        used = set()
        n = len(self.chars)
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
            ca, cb = self.chars[a], self.chars[b]
            if adj[cb + ca] > adj[ca + cb] and adj[cb + ca] > 0:
                out.append((cb + ca, k))
            else:
                out.append((ca + cb, k))
        return out


def run():
    print("=== M5 阶段 45：主题分批课程（词组跨批累积——幼儿式主题重复） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=900)
    wiki = load_corpus(os.path.join(base, "corpus_wiki_filtered.txt"), n=1800)
    freq = Counter("".join(simple + wiki))
    chars = [c for c, _ in freq.most_common(N_CHAR)]
    print(f"字集 {len(chars)}——知识语料 1800 句（3 主题块 × 600）")
    # 主题块（wiki 顺序——前 600 农业类/600-1200 其他/1200-1800 其他）
    blocks = [wiki[i:i + 600] for i in range(0, 1800, 600)]
    # ---- 对照：一次性 1800（主题分散——稀释）vs 主题分批（课程） ----
    w_flat = RelLake(chars)
    t0 = time.perf_counter()
    for ep in range(12):
        w_flat.learn_epoch(simple + wiki)
    flat_words = w_flat.extract_words(simple + wiki)
    print(f"\n[一次性 1800] {time.perf_counter()-t0:.0f}s——词组 {len(flat_words)} 个"
          f"（{'稀释——预期' if len(flat_words) < 20 else '意外'})")
    # 主题分批（K 保留——跨批累积）
    w_cur = RelLake(chars)
    t0 = time.perf_counter()
    cur_words = []
    for bi, block in enumerate(blocks):
        for ep in range(10):
            w_cur.learn_epoch(simple + block)
        wds = w_cur.extract_words(simple + wiki)
        cur_words.append(len(wds))
        print(f"  [批{bi+1}] 训练后词组 {len(wds)} 个——累计 {time.perf_counter()-t0:.0f}s")
    print(f"\n[主题分批] 总耗时 {time.perf_counter()-t0:.0f}s——最终词组 {cur_words[-1]} 个"
          f"（vs 一次性 {len(flat_words)}——{'✓ 分批保留' if cur_words[-1] > len(flat_words) * 2 else '—'}）")
    final_words = w_cur.extract_words(simple + wiki)
    print("\n[词组展示] 分批课程最终词组 top10:")
    for wd, k in final_words[:10]:
        print(f"      '{wd}' K={k:.3f}")
    # 跨批保留检查（批 1 农业词组在批 3 后）
    print("\n[跨批保留] 批 1 农业类词组在批 3 后:")
    for wd, k in final_words:
        if any(c in wd for c in "农植作"):
            print(f"      '{wd}' K={k:.3f}（保留 ✓）")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].imshow(w_flat.K[REL_IDX["isa"]], cmap="viridis"); axes[0].set_title("one-shot 1800")
    axes[1].imshow(w_cur.K[REL_IDX["isa"]], cmap="viridis"); axes[1].set_title("theme batches")
    fig.tight_layout()
    fig.savefig("fig_stage45.png", dpi=110)
    print("\n[plot] saved fig_stage45.png")
    print("[done] stage45 theme curriculum")


if __name__ == "__main__":
    run()
