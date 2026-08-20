# -*- coding: utf-8 -*-
"""
M5 阶段 78：模板驱动生成（用户："句子构建 = 整体结构先行——不是字推字——
自发的句来填字"——C20-01 模板实例化——整体先行）

理论锚：C20-01（句子意义 = 模板实例化——先有模板（结构）再填槽位）/
  C20-02（模板泛化 = 槽位-词类绑定）/ C15-01（语法 = 时序模板）
机制（句来填字——整体先行）：
  ① 模板库：语料高频结构（"X很Y"/"X是Y"/"X包括Y"——功能词序列——stage47）
  ② 模板选择：种子词 → 匹配模板（种子与模板槽位的 K 关联——
     "苹果"与"X很Y"（苹果很甜——K 强）→ 选择属性模板）
  ③ 实例化：X=种子——Y 槽位填充（K 预测——"甜"）——整体结构 → 字填充
  ④ 句链：模板实例化的序列（多模板——句子组合——C13-02）
对照：逐字推（旧——漂移）vs 模板生成（整体先行——干净）
"""
import os
import re
import time
from collections import Counter
import numpy as np
from stage72_integration_v3 import PipelineLake
from stage73_evaluation import load_corpus
import stage72_integration_v3 as s72

def extract_templates(sents):
    """自发模板：枢纽字（统计自选——高频+结构位——"很/是/有"从语料自然涌现——
    非人工词表——C2-02 结构涌现）——模板 = X[枢纽]Y（槽位 = 枢纽前后）"""
    freq = Counter("".join(sents))
    hubs = [c for c, n in freq.most_common(300) if n > 30]
    tmpl = Counter()
    for s in sents:
        for h in hubs:
            if h in s and s.index(h) > 0 and s.index(h) < len(s) - 1:
                tmpl[("X", h, "Y")] += 1
    return tmpl, hubs


def template_generate(w, sents, seed, templates, hubs, max_steps=3):
    """模板驱动生成：种子 → 模板选择 → 实例化（整体先行——句来填字）"""
    last = seed[-1]
    if last not in w.ci:
        return seed
    out = seed
    for _ in range(max_steps):
        last = out[-1]
        if last not in w.ci:
            break
        i = w.ci[last]
        # 模板选择：与枢纽字的 K 关联（自发——"苹果"与"很"强——选属性模板）
        best_t, best_k = None, 0.0
        for h in hubs:
            if h in w.ci:
                k = w.K[i, w.ci[h]]
                if k > best_k:
                    best_t, best_k = ("X", h, "Y"), k
        if best_t is None or best_k < 0.002:
            break
        fw = best_t[1]
        # 实例化：种子（X 位）→ 功能词 → Y 槽位（K 预测）
        if fw in w.ci:
            j = w.ci[fw]
            row = w.K[j].copy()
            top = np.argsort(row)[::-1]
            y = None
            for k in top:
                if row[k] > 0.002 and w.chars[k] not in out and w.chars[k] != last:
                    y = w.chars[k]
                    break
            if y is None:
                break
            out = out + fw + y
    return out


def run():
    print("=== M5 阶段 78：模板驱动生成（C20-01——整体结构先行——句来填字） ===\n")
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
    templates, hubs = extract_templates(simple + wiki)
    print(f"模板库: {[t for t, c in templates.most_common(6)]}")
    print("\n[生成] 模板驱动（整体先行——句来填字）:")
    for sd in ["苹果", "天气", "老师", "中秋", "小猫", "水"]:
        g = template_generate(w, simple + wiki, sd, templates, hubs)
        print(f"      '{sd}' → '{g}'")
    print("\n[done] stage78 template generation")


if __name__ == "__main__":
    run()
