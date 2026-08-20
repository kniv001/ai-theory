# -*- coding: utf-8 -*-
"""
M5 阶段 47：模板槽位（语法层——C20-01/02/03——"X属于Y"模板——知识句压缩）

用户："目前对于框架的思考，还缺少什么功能，例如已添加的因果链"——模板槽位 = 最优先缺失
理论锚定：C20-01（句子意义=模板实例化）/ C20-02（模板泛化=槽位-词类子湖绑定——
  "X属于Y"——X 槽位↔名词湖）/ C15-01（语法=时序模板——从统计自发生成——不预设）
机制：
  ① 模板提取：含功能词（是/属于/包括/很/因为…所以）的句子 → 模板（功能词序列 + 槽位）
  ② 槽位词类：X 槽位填充词 / Y 槽位填充词——词类涌现（主语类/类别类——C15-02）
  ③ 模板压缩：知识句 → 模板ID + 槽位填充（"农业属于第一级产业" = [X属于Y] X=农业 Y=第一级产业）
  ④ 模板实例化（生成）：模板 + 槽位预测（"农业属于[?]"——Y 预测 = 类别填充）
验证：
  exp1 模板提取（高频模板——X是Y/X属于Y/X包括Y/因为X所以Y）
  exp2 槽位词类（X 槽位词 vs Y 槽位词——类聚分离）
  exp3 模板压缩（知识句 → 模板表示——压缩率）
  exp4 模板实例化（生成——"农业属于[类别]"）
"""
import os
import re
import time
from collections import Counter
import numpy as np

RNG = np.random.default_rng(47)
FUNC_WORDS = ["因为", "所以", "属于", "包括", "包含", "是", "很", "有"]

def load_corpus(path, lo=3, hi=80, n=None):
    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    clean = [s for s in lines if lo <= len(s) <= hi
             and re.search(r"[一-鿿]", s) and not re.search(r"[A-Za-z]", s)]
    if n and len(clean) > n:
        clean = clean[:n]
    return clean

def match_template(sent):
    """句子 → 模板（功能词序列 + 槽位填充）——"农业属于第一级产业" → ([X,属于,Y], [农业,第一级产业])"""
    # 找功能词位置（最长匹配优先）
    positions = []
    for fw in FUNC_WORDS:
        idx = 0
        while True:
            i = sent.find(fw, idx)
            if i < 0:
                break
            positions.append((i, i + len(fw), fw))
            idx = i + 1
    if not positions:
        return None
    positions.sort()
    # 槽位切分
    slots = []
    prev_end = 0
    tmpl = []
    for start, end, fw in positions:
        if start > prev_end:
            slots.append(sent[prev_end:start])
            tmpl.append("X")
        tmpl.append(fw)
        prev_end = end
    if prev_end < len(sent):
        slots.append(sent[prev_end:])
        tmpl.append("X")
    return (tuple(tmpl), slots)

def run():
    print("=== M5 阶段 47：模板槽位（语法层——C20——X属于Y 模板——知识句压缩） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=900)
    wiki = load_corpus(os.path.join(base, "corpus_wiki_filtered.txt"), n=3000)
    sents = simple + wiki
    print(f"语料 {len(sents)} 行")
    # ---- exp1：模板提取 ----
    tmpl_count = Counter()
    tmpl_slots = {}   # 模板 → (X槽位词列表, Y槽位词列表)
    for s in sents:
        m = match_template(s)
        if m:
            tmpl, slots = m
            tmpl_count[tmpl] += 1
            if tmpl not in tmpl_slots:
                tmpl_slots[tmpl] = [[], []]
            if len(slots) >= 1:
                tmpl_slots[tmpl][0].append(slots[0])
            if len(slots) >= 2:
                tmpl_slots[tmpl][1].append(slots[1])
    print(f"\n[exp1] 模板提取（频率 ≥20）:")
    for tmpl, cnt in tmpl_count.most_common(12):
        if cnt >= 20:
            print(f"      {list(tmpl)} × {cnt}")
    # ---- exp2：槽位词类 ----
    print("\n[exp2] 槽位词类（X 槽位词 vs Y 槽位词——top 模板）:")
    for tmpl, cnt in tmpl_count.most_common(4):
        if cnt < 20 or tmpl not in tmpl_slots:
            continue
        xs, ys = tmpl_slots[tmpl]
        x_top = Counter(xs).most_common(5)
        y_top = Counter(ys).most_common(5)
        print(f"      {list(tmpl)}: X槽=[{', '.join(w for w, _ in x_top)}]"
              f" Y槽=[{', '.join(w for w, _ in y_top)}]")
    # ---- exp3：模板压缩 ----
    print("\n[exp3] 模板压缩（知识句 → 模板 + 槽位——压缩率）:")
    test = ["农业属于第一级产业包括作物种植", "全球农业年产出大量食物",
            "因为下雨所以带伞", "苹果很甜"]
    for s in test:
        m = match_template(s)
        if m:
            tmpl, slots = m
            comp = "[" + "][".join(tmpl) + "]"
            print(f"      '{s}' → {comp} 槽位={slots}")
    # ---- exp4：模板实例化（生成） ----
    print("\n[exp4] 模板实例化（X 槽位给定 → Y 槽位预测——类别填充）:")
    for tmpl, cnt in tmpl_count.most_common(3):
        if cnt < 20 or tmpl not in tmpl_slots:
            continue
        xs, ys = tmpl_slots[tmpl]
        x_freq = Counter(xs)
        y_by_x = {}
        # X 给定 → Y 的分布（同一句的 X-Y 对）
        for s in sents:
            m = match_template(s)
            if m and m[0] == tmpl and len(m[1]) >= 2:
                y_by_x.setdefault(m[1][0], Counter())[m[1][1]] += 1
        top_x = x_freq.most_common(3)
        for x, _ in top_x:
            yc = y_by_x.get(x, Counter())
            if yc:
                y_top = yc.most_common(3)
                print(f"      {list(tmpl)} X='{x}' → Y 候选=[{', '.join(f'{w}({c})' for w, c in y_top)}]")
    print("\n[done] stage47 template slots")


if __name__ == "__main__":
    run()
