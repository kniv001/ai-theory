# -*- coding: utf-8 -*-
"""
M5 阶段 76：生成边界优化（"苹果很甜了一口粥"延展问题——生成停止 =
预测切分的对称应用——Goriely 边界机制——强度骤降处停止）

理论锚：C3-02（预测河下行——生成 = 预测执行）/ R2（河道——强度=熟悉度）
研究锚：Goriely 2025（预测误差词首峰——边界 = 预测骤降——切分）——
  生成对称：熟悉链（预测强——"苹果很甜"）继续——漂移（预测弱——"了一口粥"）停止
机制：
  ① 逐字生成（K 前向——预测河下行）
  ② 每步检查预测强度：强度骤降（漂移——陌生链）→ 停止
     （"果→很"强（熟悉）/ "甜→了"弱（漂移——停））
  ③ 阈值：预测强度 < 峰值的 30% → 停止（熟悉链结束）
对照：无边界（旧——延展）vs 边界（新——干净句）
"""
import os
import time
import numpy as np
from stage72_integration_v3 import PipelineLake
from stage73_evaluation import load_corpus
import stage72_integration_v3 as s72


def generate_boundary(w, seed, drop_ratio=0.3, max_len=8):
    """生成 + 边界停止（预测强度骤降 = 漂移——停——Goriely 对称）"""
    out = seed
    peak = 0.0
    for _ in range(max_len):
        if out[-1] not in w.ci:
            break
        i = w.ci[out[-1]]
        row = w.K[i].copy()
        top = np.argsort(row)[::-1]
        nxt, strength = None, 0.0
        for j in top:
            if row[j] > 0.01 and w.chars[j] not in out:
                nxt, strength = w.chars[j], row[j]
                break
        if nxt is None:
            break
        peak = max(peak, strength)
        if peak > 0 and strength < peak * drop_ratio:   # 漂移（预测骤降）→ 停
            break
        out += nxt
    return out


def run():
    print("=== M5 阶段 76：生成边界优化（预测骤降停止——Goriely 对称——干净句） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=900)
    wiki = load_corpus(os.path.join(base, "corpus_wiki_filtered.txt"), n=600)
    s72.SCALE_NOISE = 0.97
    s72.H_RATE = 0.005
    chars = list(dict.fromkeys("".join(simple)))[:500]
    w = PipelineLake(chars)
    for c in chars:
        w.inject(c)
    w._decay()
    w.build_neighbors()
    important = ["苹果很甜", "天气变冷"]
    vals = [1 if s in important else 0 for s in simple + wiki]
    for day in range(3):
        w.learn_day(simple + wiki, values=vals, important=important)
        w.sleep_night()
        if day == 1:
            w.build_neighbors()
    # 对照：无边界（旧）vs 边界（新）
    print("\n[对照] 无边界（延展）vs 边界（干净）:")
    for sd in ["苹果", "天气", "学习", "喜欢", "妈妈"]:
        old = w.generate(sd)          # 旧（无边界——延展）
        new = generate_boundary(w, sd)   # 新（边界——骤降停止）
        print(f"      '{sd}': 旧 '{old}' vs 新 '{new}'")
    # 评估：边界版长度分布（不延展）
    print("\n[评估] 边界版生成（种子 8 个——长度）:")
    for sd in ["苹果", "天气", "学习", "喜欢", "妈妈", "小猫", "老师", "水"]:
        g = generate_boundary(w, sd)
        print(f"      '{sd}' → '{g}'（{len(g)} 字）")
    print("\n[done] stage76 generation boundary")


if __name__ == "__main__":
    run()
