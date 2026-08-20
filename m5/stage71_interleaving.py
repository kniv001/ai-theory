# -*- coding: utf-8 -*-
"""
M5 阶段 71：交错学习（教育锚定——Rohrer & Taylor 交错 > 集中——
与 stage44"主题密度"形成对照张力——诚实实验）

研究锚：Rohrer & Taylor（交错练习 > 集中练习——分类/技能任务）/
  Elfouly（间隔递送——已有）/ Chai & Ko（输入性质——主题密度——stage44）
张力：stage44 发现"主题密度决定词组成形"（集中好）——教育研究说"交错泛化好"——
  对照实验看：集中（主题块）vs 交错（主题交替）——不同指标（词组质量 vs 泛化）
机制：
  集中：主题块（A×N A×N……——stage45）
  交错：主题交替（A B C A B C……）
验证：
  exp1 词组质量（集中 vs 交错——stage44 的密度效应）
  exp2 泛化（跨主题关联——交错 vs 集中——教育预测交错泛化好）
  exp3 综合（哪个指标胜出——张力裁决）
"""
import os
import re
import time
from collections import Counter
import numpy as np

RNG = np.random.default_rng(71)
DT = 0.05
GAMMA = 0.8
OMEGA_LO, OMEGA_HI = 0.5, 4.0
AMP_IN = 1.2
PULSE_STEPS = 5
EPS_K = 0.02
LAMBDA_K = 0.01
K_CAP = 0.5

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

class ILake:
    """交错湖（集中/交错模式）"""
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

    def step(self, drive):
        self.z = step_dynamics(self.z, self.omega, self.gamma, self.K, self.rowsum, drive, DT)
        self.t += DT
        return self.z

    def inject_sentence(self, sent):
        drive = np.zeros(len(self.chars), dtype=complex)
        for pos, c in enumerate(sent):
            if c in self.ci:
                i = self.ci[c]
                drive[i] += AMP_IN * np.exp(1j * (self.omega[i] * self.t + pos * np.pi / 6))
        for _ in range(PULSE_STEPS):
            self.step(drive)
        for _ in range(3):
            self.step(np.zeros(len(self.chars), dtype=complex))
        return np.abs(self.z)

    def learn_epoch(self, sents):
        n = len(self.chars)
        for sent in sents:
            idx = [self.ci[c] for c in sent if c in self.ci]
            if len(idx) < 2:
                continue
            amp = self.inject_sentence(sent)
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

    def word_groups(self, th=0.02):
        n = len(self.chars)
        return sum(1 for i in range(n) for j in range(i + 1, n) if self.K[i, j] > th)

    def cross_theme(self, set_a, set_b):
        """跨主题关联（泛化——A 主题词与 B 主题词的 K）"""
        s = 0.0
        cnt = 0
        for a in set_a:
            if a in self.ci:
                for b in set_b:
                    if b in self.ci:
                        s += self.K[self.ci[a], self.ci[b]]
                        cnt += 1
        return s / max(cnt, 1)


def run():
    print("=== M5 阶段 71：交错学习（Rohrer & Taylor 交错 vs 集中——张力裁决） ===\n")
    base = os.path.dirname(__file__)
    wiki = load_corpus(os.path.join(base, "corpus_wiki_filtered.txt"), n=1800)
    # 三个主题块（wiki 顺序——前 600/600-1200/1200-1800）
    blocks = [wiki[i:i + 600] for i in range(0, 1800, 600)]
    print(f"3 主题块 × {len(blocks[0])} 句")
    # 集中：A×600 B×600 C×600（块序）
    blocked = blocks[0] + blocks[1] + blocks[2]
    # 交错：A1 B1 C1 A2 B2 C2（每块轮流）
    interleaved = []
    for i in range(600):
        for b in blocks:
            interleaved.append(b[i])
    chars = list(dict.fromkeys("".join(wiki)))
    w_blk = ILake(chars)
    w_int = ILake(chars)
    for ep in range(6):
        w_blk.learn_epoch(blocked)
        w_int.learn_epoch(interleaved)
    # ---- exp1：词组质量（集中 vs 交错） ----
    g_blk = w_blk.word_groups()
    g_int = w_int.word_groups()
    print(f"\n[exp1] 词组质量: 集中 {g_blk} vs 交错 {g_int}"
          f"（{'集中更优——密度效应（stage44）' if g_blk > g_int else '交错更优'}）")
    # ---- exp2：跨主题泛化 ----
    print("\n[exp2] 跨主题泛化（A-B 主题关联——教育预测交错泛化好）:")
    # 各主题首词（粗略——主题 A 前 100 句的字 / 主题 B 前 100 句的字）
    set_a = set("".join(blocks[0][:100]))
    set_b = set("".join(blocks[1][:100]))
    ct_blk = w_blk.cross_theme(set_a, set_b)
    ct_int = w_int.cross_theme(set_a, set_b)
    print(f"      集中 {ct_blk:.4f} vs 交错 {ct_int:.4f}"
          f"（{'交错泛化 ✓——教育预测' if ct_int > ct_blk * 1.2 else '集中泛化/接近'}）")
    # ---- exp3：张力裁决 ----
    print("\n[exp3] 张力裁决（密度 vs 泛化——两指标）:")
    print(f"      词组（密度）: {'集中' if g_blk > g_int else '交错'}")
    print(f"      泛化（交错）: {'交错 ✓' if ct_int > ct_blk * 1.2 else '集中'}——"
          f"教育研究（Rohrer & Taylor）预测交错泛化——工程两者兼用（密集成形 + 交错泛化）")
    print("\n[done] stage71 interleaving")


if __name__ == "__main__":
    run()
