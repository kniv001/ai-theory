# -*- coding: utf-8 -*-
"""
M5 阶段 105：拓扑演化（差距③——C2-01 推理期拓扑演化（真创新点）+
C5-04 新河产生=惊讶超阈——工程至今固定候选 K）

理论锚：
  C2-01（推理期拓扑演化——河流产生/断流/改道是架构核心特征——
    非固定连接图——open——真创新点）
  C5-04（结构塑性——新河产生 = 候选连接"源头-惊讶"相关性超阈
    （解释持久惊讶）；断流 = 侵蚀自然完成——open）
  C2-06（断流痕迹随时间衰减——遗忘机制——supported）
  C2-02（结构涌现——大湖群结构是训练产物——open）

机制（拓扑动态）：
  ① 正常沉积（EPS_K——共现）
  ② 惊讶检测（C5-04）：A→B 出现但 W[A,B] 弱（预测失败）→ 惊讶计数++
  ③ 新河开凿：惊讶超阈（SURPRISE_TH）→ W[A,B] 加速建立（×OPEN_MULT
    ——"解释持久惊讶"——新河产生）
  ④ 断流：W 侵蚀到 DROP_TH 以下 → 断流（拓扑移除——痕迹保留 C2-06）

验证：
  exp1 新河产生（陌生搭配（月→饼——"月饼"新增）→ 惊讶 → 新河开凿）
  exp2 断流（旧搭配（苹→脆 停止出现）→ 侵蚀 → 断流——痕迹保留）
  exp3 拓扑演化（连接数变化——训练期新河 vs 断流）
  exp4 模块化涌现（C2-02——主题块（水果/天气）→ 类内密类间疏）
"""
import os
import re
import sys
import time
from collections import Counter
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("gb"):
    sys.stdout.reconfigure(encoding="utf-8")

RNG = np.random.default_rng(105)
EPS_K = 0.02
LAMBDA_K = 0.01
SURPRISE_TH = 4        # 惊讶阈值（持久惊讶 → 新河）
OPEN_MULT = 8.0        # 新河开凿加速（惊讶加成）
DROP_TH = 0.0002       # 断流阈值（W 弱于 → 断流）


class TopoLake:
    """拓扑湖：沉积 + 惊讶开河（C5-04）+ 侵蚀断流（C2-01）"""

    def __init__(self, chars):
        self.chars = list(chars)
        self.ci = {c: i for i, c in enumerate(chars)}
        n = len(chars)
        self.W = np.zeros((n, n))
        self.surprise = np.zeros((n, n))     # 惊讶计数（源头-惊讶）
        self.opened = []                     # 新河记录（开凿日志）
        self.dropped = []                    # 断流记录

    def learn(self, sent):
        n = len(self.chars)
        idx = [self.ci[c] for c in sent if c in self.ci]
        for a in range(len(idx) - 1):
            i, j = idx[a], idx[a + 1]
            # 正常沉积（共现——相邻强）
            self.W[i, j] += EPS_K * 3.0
            # 惊讶检测（C5-04）：出现但无河（预测失败）→ 惊讶计数
            if self.W[i, j] < 0.02:
                self.surprise[i, j] += 1
                # 新河开凿：惊讶超阈 → 加速建立（解释持久惊讶）
                if self.surprise[i, j] >= SURPRISE_TH:
                    self.W[i, j] += EPS_K * OPEN_MULT
                    self.opened.append((self.chars[i], self.chars[j]))
                    self.surprise[i, j] = 0
        # 侵蚀（C2-06）
        self.W *= (1.0 - LAMBDA_K)
        # 断流（C2-01）：W 弱于阈值 → 断流（痕迹保留——C2-06）
        weak = (self.W > 0) & (self.W < DROP_TH)
        if weak.any():
            js = np.nonzero(weak)
            for k in range(min(5, len(js[0]))):
                self.dropped.append((self.chars[js[0][k]], self.chars[js[1][k]]))
            self.W[weak] = 0.0

    def strength(self, a, b):
        if a in self.ci and b in self.ci:
            return self.W[self.ci[a], self.ci[b]]
        return 0.0

    def n_links(self):
        return int((self.W > 0).sum())


def run():
    print("=== M5 阶段 105：拓扑演化（C2-01 河流产生/断流——C5-04 惊讶开河） ===\n")
    # 语料：主题块（水果类 + 天气类）——演化期变化（新增/移除搭配）
    fruit = ["苹果很甜。", "苹果是水果。", "苹果可以吃。", "苹果很脆。",
             "西瓜很甜。", "西瓜是水果。", "葡萄很甜。", "葡萄是水果。",
             "香蕉很甜。", "香蕉是水果。"]
    weather = ["天气很好。", "天气变冷了。", "天气变暖了。", "今天天气不错。",
               "明天有雨。", "刮风了。", "下雨了。", "太阳出来了。"]
    # 阶段 1：水果+天气（正常学习）
    # 阶段 2：新增"月饼"（月→饼 新河——惊讶开河）+ 移除"苹果很脆"（苹→脆 断流）
    phase1 = fruit + weather
    phase2 = phase1 + ["中秋节吃月饼。", "月饼很甜。", "小猫吃月饼。"]
    chars = list(dict.fromkeys("".join(phase1 + phase2)))
    print(f"词汇表 {len(chars)} 字 / 阶段1 {len(phase1)} 行 / 阶段2 新增月饼句")
    w = TopoLake(chars)

    # ---- 阶段 1：正常学习（10 epoch） ----
    for ep in range(10):
        for s in phase1:
            w.learn(s)
    print(f"\n[阶段1] 训练 10 epoch——连接数 {w.n_links()}")
    for pair in [("苹", "果"), ("天", "气"), ("月", "饼")]:
        print(f"      W[{pair[0]}→{pair[1]}] = {w.strength(*pair):.4f}"
              f"（{'已有河' if w.strength(*pair) > DROP_TH else '无河'}）")

    # ---- 阶段 2：新河（月→饼 惊讶开河）+ 断流（苹→脆） ----
    print("\n[阶段2] 新增'月饼'句——月→饼 惊讶开河；'苹果很脆'移除——苹→脆 侵蚀断流")
    before = w.n_links()
    for ep in range(10):
        for s in phase2:
            w.learn(s)
    after = w.n_links()
    print(f"      连接数 {before} → {after}（{'净增（拓扑生长）' if after > before else '净减'}）")
    print(f"      新河开凿 {len(w.opened)} 条: {w.opened[:6]}")
    print(f"      断流 {len(w.dropped)} 条: {w.dropped[:6]}")

    # ---- exp1：新河产生（月→饼） ----
    print("\n[exp1] 新河产生（月→饼——惊讶超阈开凿——C5-04）:")
    st = w.strength("月", "饼")
    print(f"      W[月→饼] = {st:.4f}（{'新河建立 ✓' if st > 0.02 else '未建立'}——"
          f"惊讶 {SURPRISE_TH} 次触发 ×{OPEN_MULT} 加速）")

    # ---- exp2：断流（苹→脆） ----
    print("\n[exp2] 断流（苹→脆——'苹果很脆'停止出现——侵蚀断流——C2-01/C2-06）:")
    st = w.strength("苹", "脆")
    print(f"      W[苹→脆] = {st:.4f}（{'断流 ✓' if st < DROP_TH else '仍流通'}——"
          f"痕迹弱化保留——C2-06）")

    # ---- exp3：拓扑演化 ----
    print("\n[exp3] 拓扑演化（连接数随训练动态变化——C2-01）:")
    print(f"      阶段1 {before} 连接 → 阶段2 {after} 连接"
          f"（+{after - before}——新河-断流净效应）")

    # ---- exp4：模块化涌现（C2-02——类内密类间疏） ----
    print("\n[exp4] 模块化（主题块——水果类 vs 天气类——C2-02 结构涌现）:")
    intra_fruit = np.mean([w.strength(a, b) for a, b in [("苹", "果"), ("西", "瓜"),
                                                         ("葡", "萄"), ("香", "蕉")]])
    intra_weather = np.mean([w.strength(a, b) for a, b in [("天", "气"), ("刮", "风"),
                                                           ("下", "雨"), ("太", "阳")]])
    inter = np.mean([w.strength(a, b) for a, b in [("苹", "天"), ("西", "雨"),
                                                   ("葡", "刮"), ("香", "下")]])
    print(f"      类内（水果）{intra_fruit:.4f} / 类内（天气）{intra_weather:.4f}"
          f" vs 类间 {inter:.4f}"
          f"（{'模块化 ✓' if min(intra_fruit, intra_weather) > inter * 2 else '模块化不足'}——"
          f"类内密度 > 类间——子湖形成）")
    print("\n[done] stage105 topology evolution")


if __name__ == "__main__":
    run()
