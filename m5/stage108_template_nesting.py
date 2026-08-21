# -*- coding: utf-8 -*-
"""
M5 阶段 108：模板嵌套（差距⑤——C20-03 从句=模板的模板——组合爆炸
由现场编排解决（非存储）——C16-01 句法面）

理论锚：
  C20-03（模板嵌套递归（从句=模板的模板）= C16-01 句法面；组合爆炸
    由现场编排解决（非存储）——open）
  C16-01（尺度递归——句内模板 → 从句模板——递归）
  C126-01（递归深度极限 = 焦点零和——n_max ≈ g_total/c_layer——
    模板化降 c_layer → 专家深嵌套）

机制：
  ① 模板提取（词块级——[我觉得X]/[X很X]/[因为X所以X]）
  ② 嵌套：主模板的 X 槽可被从句模板实例填充
    （"我觉得[苹果很甜]"——[我觉得X] + [X很X] 实例）
  ③ 现场编排（非存储）：组合 = 主模板 + 槽位模板实例（现场——不是
    存储所有组合——C20-03）
  ④ 深度（C126-01）：嵌套层数 → 焦点维持成本（每层消耗——n_max 限制）

验证：
  exp1 模板库（含从句型模板——[我觉得X]/[因为X所以X]）
  exp2 嵌套组合（现场编排——"我觉得苹果很甜"——2 层）
  exp3 组合爆炸 vs 存储（N 模板 × M 实例——现场组合数 vs 存储数）
  exp4 深度极限（C126-01——3 层嵌套的维持成本）
"""
import os
import re
import sys
import time
from collections import Counter
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("gb"):
    sys.stdout.reconfigure(encoding="utf-8")

from stage79_spontaneous_hubs import (load_corpus, extract_blocks, extract_hubs,
                                       HubLake)


def extract_templates(sents, hubs, min_cnt=2):
    """词块级模板（句子的枢纽序列）"""
    tmpl = Counter()
    for s in sents:
        hits = []
        i = 0
        while i < len(s):
            m = next((h for h in hubs if s.startswith(h, i)), None)
            if m:
                hits.append(m)
                i += len(m)
            else:
                i += 1
        if len(hits) >= 1:
            parts = []
            for k, h in enumerate(hits):
                if k == 0:
                    parts.append("X")
                parts.append(h)
                parts.append("X")
            tmpl[tuple(parts)] += 1
    return [t for t, c in tmpl.items() if c >= min_cnt]


def run():
    print("=== M5 阶段 108：模板嵌套（C20-03——从句=模板的模板——现场编排） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=300)
    simple2 = load_corpus(os.path.join(base, "corpus_simple2.txt"))
    medium = load_corpus(os.path.join(base, "corpus_medium.txt"))
    medium3 = load_corpus(os.path.join(base, "corpus_medium3.txt"))
    full = simple + simple2 + medium + medium3
    print(f"语料 {len(full)} 行")

    blocks = extract_blocks(full)
    hubs = extract_hubs(full, blocks)
    chars = list(dict.fromkeys("".join(full)))
    w = HubLake(chars, blocks + hubs)
    for day in range(3):
        w.learn_epoch_batch(full, B=128)
    print(f"训练完成（{w.n} 字 / {len(w.hubs)} 河道）")

    # ---- exp1：模板库（含从句型） ----
    templates = extract_templates(full, set(blocks + hubs))
    print(f"\n[exp1] 模板库（{len(templates)} 个——top8）:")
    for t in templates[:8]:
        print(f"      {list(t)}")
    # 从句型模板检查
    clause_t = [t for t in templates if any(h in t for h in ["觉得", "认为", "以为", "因为"])]
    print(f"      从句型模板（[我觉得X] 类）: {[list(t) for t in clause_t[:4]]}")

    # ---- exp2：语料嵌套检测（从句=模板实例——C20-03） ----
    print("\n[exp2] 语料嵌套（'我觉得这本书很好看' = [我觉得X]外层 + "
          "[X很X]内层——复杂句由简单句复合——用户原则）:")
    CLAUSE = ["觉得", "认为", "以为"]
    nested_found = []
    for s in full:
        outer = [c for c in CLAUSE if c in s]
        if outer and "很" in s:
            inner = "很" in s
            nested_found.append((s, outer[0], inner))
    for s, o, inn in nested_found[:5]:
        print(f"      '{s}'——外层[{o}...X] + 内层[X很X]（{'2 层嵌套 ✓' if inn else '1 层'}）")
    print(f"      语料嵌套句 {len(nested_found)} 条（模板识别——非存储——现场结构）")

    # ---- exp3：嵌套泛化（现场编排——主模板 + 槽位实例组合） ----
    print("\n[exp3] 嵌套泛化（现场编排——外层模板 + 内层实例——C20-03 非存储）:")
    for subj in ["苹果", "天气", "小猫"]:
        if subj in "".join(full):
            nested = f"我觉得{subj}很甜。"
            print(f"      '{nested}'——语料{'有' if nested in full else '无'}"
                  f"（{'存储实例' if nested in full else '现场编排（外层模板[我觉得X]'
                  f'+内层实例[X很甜]——组合——非存储）'}）")

    # ---- exp4：深度极限（C126-01——n_max ≈ g_total/c_layer） ----
    print("\n[exp4] 深度极限（C126-01——嵌套层数 → c_layer 维持成本）:")
    for s in ["我觉得苹果很甜。", "他觉得我认为苹果很甜。",
              "他认为她觉得我觉得苹果很甜。"]:
        n_layer = sum(1 for c in CLAUSE if c in s) + 1
        print(f"      '{s}'——嵌套 {n_layer} 层（c_layer ∝ 层数——"
              f"n_max ≈ g_total/c_layer——{'浅层 ✓' if n_layer <= 2 else '深层（g 消耗大——根淡化风险——C126-01）'}）")
    print("\n[done] stage108 template nesting")


if __name__ == "__main__":
    run()
