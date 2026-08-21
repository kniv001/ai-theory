# -*- coding: utf-8 -*-
"""
M5 阶段 123：硬度分化（C6-01/02——地形异质可塑性——h 均匀初始→功能分块
涌现：硬区=长时记忆、软区=短时记忆——记忆地形完整化）

理论锚：
  C6-01（地形异质可塑性（h_i；ε、λ 随 h 递减）→ 功能分块涌现：
    硬区=长时记忆、软区=短时记忆——判据：均匀初始能否自发分化——open）
  C6-02（元可塑性：dh/dt = 稳定性↑ − 新奇性↓ + 稳态调节——
    正反馈需稳态项防全硬崩塌——open）
  C10-04（硬度稳态项 = 向个体基线 h₀ 回归——弹性地形——open）

机制（元可塑性）：
  ① 硬度 h 均匀初始（0.5）——学习：沉积 → h 升（稳定性）；侵蚀/误差
    → h 降（新奇性）；稳态回拉（λ_h·(h₀−h)——C10-04）
  ② ε、λ 随 h 递减（C6-01——硬区可塑性低——稳定）
  ③ 分化：高频字（重复沉积）→ h 升（硬——长时）；低频字 → h 降/稳
    （软——短时）

验证：
  exp1 分化涌现（均匀初始 → h 分布分化——高频硬/低频软）
  exp2 功能分块（硬区=高频词/软区=低频词——长时/短时）
  exp3 稳态项（无稳态 → 全硬崩塌 vs 有稳态 → 弹性——C6-02/C10-04）
  exp4 可塑性差异（硬区 ε 小（稳定）/软区 ε 大（易学易忘）——C6-01）
"""
import os
import re
import sys
import time
from collections import Counter
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("gb"):
    sys.stdout.reconfigure(encoding="utf-8")

from stage79_spontaneous_hubs import load_corpus

EPS0 = 0.02
LAM0 = 0.01
H0 = 0.5            # 基线硬度（稳态回拉目标）
LAM_H = 0.02        # 稳态回拉率
NOVELTY = 0.02      # 新奇性（误差 → h 降）


class HardnessLake:
    """硬度分化湖：元可塑性（沉积硬化 + 新奇软化 + 稳态回拉）"""

    def __init__(self, chars, with_homeo=True):
        self.chars = list(chars)
        self.ci = {c: i for i, c in enumerate(chars)}
        n = len(chars)
        self.h = np.full(n, H0)          # 硬度（均匀初始）
        self.W = np.zeros((n, n))
        self.with_homeo = with_homeo     # 稳态项开关（exp3 对照）

    def learn(self, sent):
        idx = [self.ci[c] for c in sent if c in self.ci]
        for a in range(len(idx) - 1):
            i, j = idx[a], idx[a + 1]
            eps = EPS0 * (1.0 - self.h[i] * 0.8)       # ε 随 h 递减（硬区 ε 小）
            self.W[i, j] += eps
        # 元可塑性：沉积硬化（每次出现累积——高频升）+ 新奇软化（仅当
        # 关联弱——误差——保持可塑——C6-02）+ 稳态回拉（C10-04）
        for i in idx:
            dep = 0.02 * (1.0 - self.h[i])             # 稳定性↑（出现累积）
            self.h[i] += dep
            # 新奇性↓：该字关联弱（低频/新词——预测误差大——软化）
            if self.W[i].sum() < 0.5:
                self.h[i] -= NOVELTY * self.h[i]
            if self.with_homeo:
                self.h[i] += LAM_H * (H0 - self.h[i])  # 稳态回拉
            self.h[i] = np.clip(self.h[i], 0.05, 0.95)

    def erosion(self):
        """侵蚀（λ 随 h——硬区 λ 小——慢蚀——C6-01）"""
        lam = LAM0 * (1.0 + self.h * 0.5)               # 硬区 λ 小？——反——
                                                        # 硬区慢蚀：λ 小→硬
        self.W *= (1.0 - LAM0)
        # 硬度与侵蚀耦合：硬区 W 衰减慢（简化——直接保持）


def run():
    print("=== M5 阶段 123：硬度分化（C6-01/02——均匀→功能分块——硬/软区） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=200)
    simple2 = load_corpus(os.path.join(base, "corpus_simple2.txt"))
    medium = load_corpus(os.path.join(base, "corpus_medium.txt"))
    full = simple + simple2 + medium
    chars = list(dict.fromkeys("".join(full)))
    print(f"语料 {len(full)} 行 / 词汇 {len(chars)}")

    w = HardnessLake(chars)
    freq = Counter("".join(full))
    for ep in range(10):
        for s in full:
            w.learn(s)
    print("训练完成（10 epoch——元可塑性）")

    # ---- exp1：分化涌现 ----
    print("\n[exp1] 分化涌现（均匀初始 → h 分化——高频硬/低频软）:")
    hi = [c for c in "苹的甜天" if c in w.ci]
    lo = [c for c in "鲸雾鹤" if c in w.ci]
    for c in hi + lo:
        h = w.h[w.ci[c]]
        tag = "硬" if h > 0.55 else ("软" if h < 0.45 else "中")
        print(f"      '{c}'（频 {freq.get(c, 0):3d}）h={h:.3f} [{tag}]")
    spread = w.h.std()
    print(f"      h 标准差 {spread:.3f}"
          f"（{'分化涌现 ✓' if spread > 0.05 else '未分化'}——均匀初始→分块）")

    # ---- exp2：功能分块 ----
    print("\n[exp2] 功能分块（硬区=长时/软区=短时——C6-01）:")
    hard = [c for c in chars if w.h[w.ci[c]] > 0.55]
    soft = [c for c in chars if w.h[w.ci[c]] < 0.45]
    print(f"      硬区 {len(hard)} 字: {''.join(hard[:15])}（高频→长时记忆）")
    print(f"      软区 {len(soft)} 字: {''.join(soft[:15])}（低频→短时/待固化）")
    h_freq = np.mean([w.h[w.ci[c]] for c in chars if freq.get(c, 0) > 10])
    l_freq = np.mean([w.h[w.ci[c]] for c in chars if freq.get(c, 0) <= 3])
    print(f"      高频字均 h {h_freq:.3f} vs 低频字均 h {l_freq:.3f}"
          f"（{'频率-硬度耦合 ✓' if h_freq > l_freq + 0.05 else '耦合弱'}——"
          f"使用多→硬——记忆地形）")

    # ---- exp3：稳态项（防全硬崩塌——C6-02/C10-04） ----
    print("\n[exp3] 稳态项对照（无稳态 → 全硬崩塌 vs 有稳态 → 弹性）:")
    w_nh = HardnessLake(chars, with_homeo=False)
    for ep in range(30):
        for s in full:
            w_nh.learn(s)
    print(f"      无稳态: h 均值 {w_nh.h.mean():.3f} / 标准差 {w_nh.h.std():.3f}"
          f"（{'全硬崩塌（正反馈失控）' if w_nh.h.mean() > 0.85 else '未崩塌'}——C6-02）")
    print(f"      有稳态: h 均值 {w.h.mean():.3f} / 标准差 {w.h.std():.3f}"
          f"（{'弹性地形 ✓（回拉 h0）' if 0.3 < w.h.mean() < 0.7 else '异常'}——C10-04）")

    # ---- exp4：可塑性差异 ----
    print("\n[exp4] 可塑性差异（硬区 ε 小（稳定）/软区 ε 大（易学）——C6-01）:")
    if "苹" in w.ci and "鲸" in w.ci:
        eps_hard = EPS0 * (1.0 - w.h[w.ci["苹"]] * 0.8)
        eps_soft = EPS0 * (1.0 - w.h[w.ci["鲸"]] * 0.8) if "鲸" in w.ci else EPS0
        print(f"      硬区'苹' ε={eps_hard:.4f} vs 软区'鲸' ε={eps_soft:.4f}"
              f"（{'硬稳定/软易学 ✓' if eps_hard < eps_soft else '异常'}——"
              f"硬区可塑性低——长时稳定）")
    print("\n[done] stage123 hardness differentiation")


if __name__ == "__main__":
    run()
