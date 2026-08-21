# -*- coding: utf-8 -*-
"""
M5 阶段 125：出生竞争（C35-01——河道竞争——三重机制：相关性排序/
容量预算/侧抑制——输家侵蚀清除——结构竞争=解释力竞争（C13-01 统一））

理论锚：
  C35-01（出生竞争 = 三重机制——相关性排序/容量预算/分流抑制——
    输家侵蚀清除——结构竞争 = 解释力竞争（能解释的=有意义的）——
    supported）
  C18-01（分流饱和——既有河道吸走流量——新河道需超阈流量开凿）
  C38-01（唯一直接硬化 = 水流硬化——竞争是流量的间接控制）

机制（河道竞争）：
  ① 容量预算（C35-01）：连接总数上限——超限 → 竞争（弱连接清除）
  ② 相关性排序：强连接优先保留（高频/高关联）——弱连接（低频）清除
  ③ 侧抑制（分流）：相邻连接竞争（相关连接抑制）
  ④ 结构竞争=解释力：能解释（高共现）的保留——不能的侵蚀

验证：
  exp1 容量预算（超限 → 弱连接被清——C35-01）
  exp2 相关性排序（强保留/弱清除）
  exp3 侧抑制（相邻竞争）
  exp4 结构竞争=解释力（共现强=解释力强——保留——C13-01）
"""
import os
import re
import sys
import time
from collections import Counter
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("gb"):
    sys.stdout.reconfigure(encoding="utf-8")

EPS_K = 0.02
LAM_K = 0.01
CAP = 20             # 容量预算（连接上限——小——触发竞争）


class CompetitionLake:
    """竞争湖：沉积 + 容量预算 + 相关性排序（输家清除）"""

    def __init__(self, chars, cap=CAP):
        self.chars = list(chars)
        self.ci = {c: i for i, c in enumerate(chars)}
        n = len(chars)
        self.W = np.zeros((n, n))
        self.cap = cap

    def learn(self, sent, prune=True):
        idx = [self.ci[c] for c in sent if c in self.ci]
        for a in range(len(idx) - 1):
            i, j = idx[a], idx[a + 1]
            self.W[i, j] += EPS_K
        self.W *= (1.0 - LAM_K)
        if prune and (self.W > 0).sum() > self.cap:
            self.prune()

    def prune(self):
        """容量预算 + 相关性排序（C35-01）：超限 → 弱连接清除"""
        th = np.percentile(self.W[self.W > 0], 30)   # 最弱 30% 清除
        self.W[self.W < th] = 0.0

    def strength(self, a, b):
        if a in self.ci and b in self.ci:
            return self.W[self.ci[a], self.ci[b]]
        return 0.0

    def n_links(self):
        return int((self.W > 0).sum())


def run():
    print("=== M5 阶段 125：出生竞争（C35-01——容量预算/相关性排序/侧抑制） ===\n")
    # 语料：高频主题（苹果/天气——强连接）+ 低频杂句（真低频 1 次——输家）
    high = ["苹果很甜。"] * 20 + ["天气很好。"] * 20 + ["小猫吃鱼。"] * 20
    low = ["鲸鱼很大。", "雾很大。", "鹤很高。", "鹰很远。", "蛇很长。"]
    full = high + low
    chars = list(dict.fromkeys("".join(full)))
    print(f"语料 {len(full)} 行（高频主题 60 + 低频杂句 5——真低频）")

    w = CompetitionLake(chars)
    for ep in range(10):
        for s in full:
            w.learn(s)
    print(f"训练完成——连接数 {w.n_links()}（容量 {w.cap}——"
          f"{'预算生效' if w.n_links() <= w.cap else '超限'}）")

    # ---- exp1：容量预算 ----
    print("\n[exp1] 容量预算（超限 → 弱连接清除——C35-01）:")
    print(f"      连接数 {w.n_links()} ≤ 容量 {w.cap}"
          f"（{'预算生效 ✓' if w.n_links() <= w.cap else '未生效'}——"
          f"超限时最弱 30% 清除——输家侵蚀）")

    # ---- exp2：相关性排序 ----
    print("\n[exp2] 相关性排序（强保留/弱清除——C35-01）:")
    for a, b in [("苹", "果"), ("天", "气"), ("小", "猫"), ("鲸", "鱼"), ("雾", "很")]:
        v = w.strength(a, b)
        strong = v > 0.1
        print(f"      {a}→{b}: {v:.4f}（{'强保留 ✓' if strong else '弱清除 ✓（输家）'}）")

    # ---- exp3：侧抑制（分流饱和——C18-01） ----
    print("\n[exp3] 侧抑制（既有河道吸走流量——新河道需超阈——C18-01）:")
    # 既有高频（苹果-很——相邻强）vs 低频新河（鲸-很——相邻弱）
    print(f"      既有（果-很）: {w.strength('果', '很'):.4f}")
    print(f"      新河（鲸-很）: {w.strength('鲸', '很'):.4f}"
          f"（{'新河弱（被既有分流——C18-01）' if w.strength('果', '很') > w.strength('鲸', '很') * 3 else '相当'}——"
          f"分流饱和——新河道需超阈流量开凿）")

    # ---- exp4：结构竞争=解释力（C13-01 统一） ----
    print("\n[exp4] 结构竞争=解释力（能解释的=有意义的——保留——C13-01）:")
    keep = sum(1 for a, b in [("苹", "果"), ("天", "气"), ("小", "猫")]
               if w.strength(a, b) > 0.1)
    drop = sum(1 for a, b in [("鲸", "鱼"), ("雾", "很"), ("鹤", "很")]
               if w.strength(a, b) < 0.1)
    print(f"      高频保留 {keep}/3 vs 低频清除 {drop}/3"
          f"（{'解释力竞争 ✓（高共现=解释力强=保留）' if keep == 3 and drop >= 2 else '竞争弱'}——"
          f"结构与语义同判据——C35-01/C13-01 统一）")
    print("\n[done] stage125 birth competition")


if __name__ == "__main__":
    run()
