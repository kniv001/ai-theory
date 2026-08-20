# -*- coding: utf-8 -*-
"""
M5 阶段 63：焦点稀疏训练（用户："大脑低功耗——不激活所有神经/焦点加入——
保持原神经数量级——更小激活数量的训练"）

理论锚：R10（注意闸门 g——误差驱动——只处理相关）/ C10-02（注意 = 误差驱动门控）/
  A0（误差词汇——计算 ∝ 误差量——可预测不计算）
研究锚：稀疏编码（Olshausen & Field——刺激只激活少数单元）/ 预测编码（Rao & Ballard——
  只处理预测误差）/ 大脑能耗（20W——功耗 ∝ 激活量非神经总数）/
  Williams 2025（EIG 注视选择——聚焦 = 信息采样）
机制（保持全量单元——激活稀疏）：
  ① 焦点注入：句子只驱动句内字（~30 激活——其余休眠）
  ② K 邻居传播：激活沿焦点字邻居（每字强邻居 ~100——稀疏 K——非全矩阵）
  ③ 休眠单元不更新（z 冻结——省计算）
验证：
  exp1 功能保持（稀疏 vs 全量——K 学习等效——词组提取一致）
  exp2 计算量（全量 vs 稀疏——每步乘加数——功耗比）
  exp3 稀疏度（K 非零比例——语言 K 自然稀疏——"农"只连少数）
"""
import os
import re
import time
from collections import Counter
import numpy as np

RNG = np.random.default_rng(63)
DT = 0.05
GAMMA = 0.8
OMEGA_LO, OMEGA_HI = 0.5, 4.0
AMP_IN = 1.2
PULSE_STEPS = 5
EPS_K = 0.02
LAMBDA_K = 0.01
K_CAP = 0.5
DELTA_PHI = np.pi / 6
NEIGH_K = 150        # 每字强邻居上限（稀疏 K——焦点传播半径）

def load_corpus(path, lo=3, hi=80, n=None):
    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    clean = [s for s in lines if lo <= len(s) <= hi
             and re.search(r"[一-鿿]", s) and not re.search(r"[A-Za-z]", s)]
    if n and len(clean) > n:
        clean = clean[:n]
    return clean

class SparseLake:
    """稀疏湖：全量单元——焦点激活（每次只激活句内字 + 邻居传播）"""
    def __init__(self, chars, sparse=True):
        self.chars = chars
        self.ci = {c: i for i, c in enumerate(chars)}
        n = len(chars)
        self.omega = RNG.uniform(OMEGA_LO, OMEGA_HI, n)
        self.gamma = GAMMA
        self.z = 0.1 * np.exp(1j * RNG.uniform(0, 2 * np.pi, n))
        self.t = 0.0
        self.K = np.zeros((n, n))
        self.rowsum = np.zeros(n)
        self.sparse = sparse
        self.neighbors = None    # 每字的邻居索引（稀疏 K——焦点传播）

    def build_neighbors(self, th=0.0):
        """稀疏 K：每字的 top-N 强邻居（相对强度——不受 K 绝对值影响——
        语言 K 自然稀疏——每字只连少数强关联——大脑同构）"""
        n = len(self.chars)
        self.neighbors = []
        for i in range(n):
            row = self.K[i]
            nb = np.argsort(row)[::-1][:NEIGH_K]
            nb = nb[row[nb] > th]
            self.neighbors.append(nb)

    def step_sparse(self, drive, active):
        """焦点稀疏步进：只更新激活单元（句内字 + 邻居）——休眠冻结"""
        z = self.z
        dz = -self.gamma * z + 1j * self.omega * z
        for i in active:
            s = 0.0j
            Ki = self.K[i]
            for j in self.neighbors[i]:
                s += Ki[j] * (z[j] - z[i])
            dz[i] += s
        dz += drive
        z = z + dz * DT
        over = np.abs(z[active]) > 3.0
        z[active][over] = z[active][over] / np.abs(z[active][over]) * 2.0
        return z

    def step_full(self, drive):
        """全量步进（对照）"""
        z = self.z
        dz = -self.gamma * z + 1j * self.omega * z
        dz += self.K @ z - z * self.rowsum + drive
        z = z + dz * DT
        over = np.abs(z) > 3.0
        z[over] = z[over] / np.abs(z[over]) * 2.0
        return z

    def learn_epoch(self, sents):
        n = len(self.chars)
        for sent in sents:
            idx = [self.ci[c] for c in sent if c in self.ci]
            if len(idx) < 2:
                continue
            drive = np.zeros(n, dtype=complex)
            for pos, i in enumerate(idx):
                drive[i] += AMP_IN * np.exp(1j * (self.omega[i] * self.t + pos * DELTA_PHI))
            if self.sparse:
                # 焦点 = 句内字 + 邻居（激活子集）
                active = set(idx)
                for i in idx:
                    active.update(self.neighbors[i])
                active = np.array(sorted(active))
                for _ in range(PULSE_STEPS + 3):
                    self.step_sparse(drive, active)
            else:
                for _ in range(PULSE_STEPS + 3):
                    self.step_full(drive)
            amp = np.abs(self.z)
            sub = np.array(idx)
            A = amp[sub]
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


def run():
    print("=== M5 阶段 63：焦点稀疏训练（保持全量单元——激活稀疏——低功耗） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=900)
    wiki = load_corpus(os.path.join(base, "corpus_wiki_filtered.txt"), n=1800)
    sents = simple
    chars = list(dict.fromkeys("".join(sents)))
    print(f"词汇表 {len(chars)} 字（全量单元保持——小规模验证）")
    # 预训练一轮（建立初始 K——邻居表）
    w_pre = SparseLake(chars, sparse=False)
    for _ in range(3):
        w_pre.learn_epoch(sents[:400])
    w_pre.build_neighbors()
    avg_nb = np.mean([len(nb) for nb in w_pre.neighbors])
    print(f"初始邻居：每字平均 {avg_nb:.0f} 个强邻居（vs 全量 {len(chars)}——稀疏度 {avg_nb/len(chars)*100:.1f}%）")
    # ---- 对照：全量 vs 稀疏（同 seed——K 学习等效） ----
    w_full = SparseLake(chars, sparse=False)
    w_sp = SparseLake(chars, sparse=True)
    w_sp.K = w_pre.K.copy(); w_sp.rowsum = w_pre.rowsum.copy()
    w_sp.build_neighbors()
    w_full.K = w_pre.K.copy(); w_full.rowsum = w_pre.rowsum.copy()
    t0 = time.perf_counter()
    for ep in range(5):
        w_full.learn_epoch(sents)
    t_full = time.perf_counter() - t0
    t0 = time.perf_counter()
    for ep in range(5):
        w_sp.learn_epoch(sents)
    t_sp = time.perf_counter() - t0
    diff = np.abs(w_full.K - w_sp.K).max()
    print(f"\n[exp1] 功能保持（全量 vs 稀疏——5 epoch 后 K 最大差异）: {diff:.2e}"
          f"（{'等效 ✓' if diff < 1e-4 else '差异大——邻居半径不足'}）")
    print(f"[exp2] 速度：全量 {t_full:.1f}s vs 稀疏 {t_sp:.1f}s——加速 {t_full/max(t_sp,1e-9):.1f}×")
    # 计算量对比（每步乘加数）
    n = len(chars)
    full_ops = n * n
    sparse_ops = np.mean([len(nb) for nb in w_sp.neighbors]) * n
    print(f"[exp3] 计算量：全量 {full_ops/1e6:.1f}M/步 vs 稀疏 {sparse_ops/1e6:.1f}M/步"
          f"——功耗比 {full_ops/max(sparse_ops,1):.0f}×")
    # 稀疏度最终
    nnz = np.sum(w_sp.K > 0.01)
    print(f"[exp4] 语言 K 稀疏度：非零 K {nnz}/{n*n} = {nnz/(n*n)*100:.1f}%"
          f"（语言 K 自然稀疏——农只连少数——大脑同构）")
    print("\n[done] stage63 sparse focus")


if __name__ == "__main__":
    run()
