# -*- coding: utf-8 -*-
"""
M5 阶段 62：批量注入加速（用户："空间换时间——内存显存可以占更满些"）

瓶颈（计时器数据）：逐句注入——每句 12 步 × 每步 K@z（gemv——Python 层未摊销）
空间换时间：批量 B 句平行——z 变 (B,n) 矩阵——K@Z.T 一次 gemm
  （BLAS gemm 比 B 次 gemv 快 10-50×——批量摊销——内存 B×n×16B 可忽略）
显存路径：RTX 5060 8GB——torch 批量 gemm——B=1000+（500MB——8GB 够）
验证：
  exp1 正确性（批量 vs 逐句——K 结果一致）
  exp2 速度对比（逐句 vs 批量——加速比）
"""
import os
import re
import time
from collections import Counter
import numpy as np

RNG = np.random.default_rng(62)
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

class BatchLake:
    """批量湖：z (B,n) 矩阵——K@Z.T gemm——空间换时间"""
    def __init__(self, chars):
        self.chars = chars
        self.ci = {c: i for i, c in enumerate(chars)}
        n = len(chars)
        self.omega = RNG.uniform(OMEGA_LO, OMEGA_HI, n)
        self.gamma = GAMMA
        self.t = 0.0
        self.K = np.zeros((n, n))
        self.rowsum = np.zeros(n)

    def step_batch(self, Z, drive):
        """批量动力学：Z (B,n)——K@Z.T 一次 gemm——快 10-50×"""
        dz = -self.gamma * Z + 1j * self.omega * Z
        # K @ Z.T（gemm——(n,n)@(n,B)）——转置回 (B,n)
        dz += (self.K @ Z.T).T - Z * self.rowsum
        dz += drive
        Z = Z + dz * DT
        over = np.abs(Z) > 3.0
        Z[over] = Z[over] / np.abs(Z[over]) * 2.0
        return Z

    def learn_epoch_batch(self, sents, B=64):
        n = len(self.chars)
        # 分批（B 句平行——空间换时间）
        for start in range(0, len(sents), B):
            batch = sents[start:start + B]
            Z = np.zeros((len(batch), n), dtype=complex)
            drives = np.zeros((len(batch), n), dtype=complex)
            seqs = []
            for bi, sent in enumerate(batch):
                idx = [self.ci[c] for c in sent if c in self.ci]
                if len(idx) < 2:
                    continue
                seqs.append((bi, idx))
                for pos, i in enumerate(idx):
                    drives[bi, i] += AMP_IN * np.exp(1j * (self.omega[i] * self.t + pos * DELTA_PHI))
            # 批量注入（所有句同步步进）
            for _ in range(PULSE_STEPS + 3):
                Z = self.step_batch(Z, drives)
            # 沉积（每句——B 次——非瓶颈）
            amp = np.abs(Z)
            for bi, idx in seqs:
                sub = np.array(idx)
                A = amp[bi, sub]
                L = len(idx)
                d_idx = np.arange(L)
                dist_w = 1.0 / np.maximum(np.abs(d_idx[:, None] - d_idx[None, :]), 1.0)
                contrib = EPS_K * np.outer(A, A) * np.triu(dist_w, 1)
                pi, pj = np.nonzero(contrib)
                self.K[sub[pi], sub[pj]] += contrib[pi, pj]
                self.K[sub[pj], sub[pi]] += contrib[pi, pj] * 0.3
        self.K *= (1.0 - LAMBDA_K)
        rs = self.K.sum(axis=1)
        over = rs > K_CAP
        self.K[over] *= (K_CAP / rs[over])[:, None]
        self.rowsum = self.K.sum(axis=1)

    def learn_epoch_seq(self, sents):
        """逐句版（对照——旧方式）"""
        n = len(self.chars)
        for sent in sents:
            idx = [self.ci[c] for c in sent if c in self.ci]
            if len(idx) < 2:
                continue
            drive = np.zeros(n, dtype=complex)
            for pos, i in enumerate(idx):
                drive[i] += AMP_IN * np.exp(1j * (self.omega[i] * self.t + pos * DELTA_PHI))
            Z = np.zeros(n, dtype=complex)
            for _ in range(PULSE_STEPS + 3):
                dz = -self.gamma * Z + 1j * self.omega * Z
                dz += self.K @ Z - Z * self.rowsum + drive
                Z = Z + dz * DT
                over = np.abs(Z) > 3.0
                Z[over] = Z[over] / np.abs(Z[over]) * 2.0
            amp = np.abs(Z)
            sub = np.array(idx)
            A = amp[sub]
            L = len(idx)
            d_idx = np.arange(L)
            dist_w = 1.0 / np.maximum(np.abs(d_idx[:, None] - d_idx[None, :]), 1.0)
            contrib = EPS_K * np.outer(A, A) * np.triu(dist_w, 1)
            pi, pj = np.nonzero(contrib)
            self.K[sub[pi], sub[pj]] += contrib[pi, pj]
            self.K[sub[pj], sub[pi]] += contrib[pi, pj] * 0.3


def run():
    print("=== M5 阶段 62：批量注入加速（空间换时间——内存占满换速度） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=900)
    wiki = load_corpus(os.path.join(base, "corpus_wiki_filtered.txt"), n=1800)
    sents = simple + wiki
    chars = list(dict.fromkeys("".join(sents)))
    print(f"词汇表 {len(chars)} 字 / 语料 {len(sents)} 行")
    # ---- exp1：正确性（批量 vs 逐句——同 seed 同 epoch——K 一致性） ----
    w1 = BatchLake(chars)
    w2 = BatchLake(chars)
    for ep in range(3):
        w1.learn_epoch_seq(sents)
        w2.learn_epoch_batch(sents, B=64)
    diff = np.abs(w1.K - w2.K).max()
    print(f"\n[exp1] 正确性（逐句 vs 批量——3 epoch 后 K 最大差异）: {diff:.2e}"
          f"（{'一致 ✓' if diff < 1e-6 else '不一致——检查'}）")
    # ---- exp2：速度对比 ----
    print("\n[exp2] 速度对比（逐句 vs 批量——1 epoch）:")
    w3 = BatchLake(chars)
    t0 = time.perf_counter()
    w3.learn_epoch_seq(sents)
    t_seq = time.perf_counter() - t0
    w4 = BatchLake(chars)
    t0 = time.perf_counter()
    w4.learn_epoch_batch(sents, B=64)
    t_batch = time.perf_counter() - t0
    print(f"      逐句: {t_seq:.1f}s / 批量(B=64): {t_batch:.1f}s——加速 {t_seq/t_batch:.1f}×")
    # 批量大小扫描（内存占满——加速比）
    print("\n[exp3] 批量大小扫描（B 增大——加速比——内存开销）:")
    for B in [16, 64, 256]:
        w5 = BatchLake(chars)
        t0 = time.perf_counter()
        w5.learn_epoch_batch(sents, B=B)
        tb = time.perf_counter() - t0
        mem_mb = B * len(chars) * 16 / 1e6
        print(f"      B={B}: {tb:.1f}s（加速 {t_seq/tb:.1f}×）——内存 {mem_mb:.1f}MB（可忽略）")
    print("\n[done] stage62 batch acceleration")


if __name__ == "__main__":
    run()
