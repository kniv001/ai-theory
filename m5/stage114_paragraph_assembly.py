# -*- coding: utf-8 -*-
"""
M5 阶段 114：段落拼合验证（用户："目标是可以将句子拼合在一起的完整段落——
语料可以继续增加"）

拼合 = 跨来源句子组合成**新段落**（非检索整段——组合性 C13-02——
段落 = 句子的复合——句子从不同段落/语料来——主题锚保持下拼装）
语料：corpus_paragraph.txt 扩充至 22 段（89 行——我生成——家庭/学校/水果/
蔬菜/动物/雾/四季/饺子/上学/睡觉/公园/月亮/画画/读书/下雨——新主题）

验证：
  exp1 拼合段落（种子 → 新段落——句子跨来源——主题一致）
  exp2 新段落验证（生成的段落 ≠ 语料整段——是组合（C13-02））
  exp3 主题一致性（拼合段落的句间衔接——Maimon）
  exp4 拼合多样性（同一种子——不同组合——生成自由度）
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


def paragraph_assemble(w, seed, sents, max_sent=6, coh_th=0.004):
    """段落拼合：种子 → 首句 → 主题锚扩展（跨来源句——组合成新段落）"""
    para = []
    topic = seed
    for _ in range(max_sent):
        cands = []
        for s in sents:
            if s in para or len(s) < 5 or ("。" not in s and "？" not in s):
                continue
            if topic and topic[-1] in w.ci:
                i = w.ci[topic[-1]]
                idx = [w.ci[c] for c in s if c in w.ci]
                if not idx:
                    continue
                score = float(np.mean([w.KT[i, j] for j in idx]))
                if any(t in s for t in (topic, seed)):
                    score *= 2.0
                if score > coh_th:
                    cands.append((score, s))
        if not cands:
            break
        cands.sort(key=lambda x: -x[0])
        best = cands[0][1]
        para.append(best)
        if seed and seed in best:
            topic = seed                        # 句内含种子 → 主题保持
        elif best[0] in w.ci:
            topic = best[0]                     # 否则主题锚推进
    return para


def run():
    print("=== M5 阶段 114：段落拼合（跨来源句组合——新段落——C13-02） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=900)
    simple2 = load_corpus(os.path.join(base, "corpus_simple2.txt"))
    simple4 = load_corpus(os.path.join(base, "corpus_simple4.txt"))
    medium = load_corpus(os.path.join(base, "corpus_medium.txt"))
    para = load_corpus(os.path.join(base, "corpus_paragraph.txt"))
    full = simple + simple2 + simple4 + medium + para
    print(f"语料 {len(full)} 行（段落语料 {len(para)} 行——22 段）")

    blocks = extract_blocks(full)
    hubs = extract_hubs(full, blocks)
    chars = list(dict.fromkeys("".join(full)))
    w = HubLake(chars, blocks + hubs)
    t0 = time.perf_counter()
    for day in range(3):
        w.learn_epoch_batch(full, B=128)
    print(f"训练完成——{time.perf_counter()-t0:.0f}s")

    # ---- exp1：拼合段落 ----
    print("\n[exp1] 段落拼合（种子 → 新段落——跨来源句组合）:")
    for sd in ["苹果", "天气", "小猫", "月亮", "画画", "上学"]:
        p = paragraph_assemble(w, sd, full)
        print(f"      '{sd}' →")
        for s in p:
            src = "段落" if s in para else "基础语料"
            print(f"        [{src}] {s}")
        print()

    # ---- exp2：新段落验证（≠ 语料整段——组合） ----
    print("\n[exp2] 新段落验证（生成的段落是否 = 语料整段——组合 vs 检索）:")
    para_blocks = [para[i:i + 4] for i in range(0, len(para), 4)]
    for sd in ["苹果", "天气", "小猫"]:
        p = paragraph_assemble(w, sd, full)
        is_verbatim = any(p == b for b in para_blocks)
        cross_src = len({s in para for s in p}) > 1
        print(f"      '{sd}' 段落 {len(p)} 句——{'语料整段（检索）' if is_verbatim else '新组合 ✓（拼合——C13-02）'}"
              f"——来源{'跨（段落+基础）' if cross_src else '单'}")

    # ---- exp3：主题一致性 ----
    print("\n[exp3] 主题一致性（拼合段落的句间衔接——Maimon）:")
    for sd in ["苹果", "天气", "小猫", "月亮"]:
        p = paragraph_assemble(w, sd, full)
        if len(p) >= 2:
            shared = sum(len(set(p[i]) & set(p[i + 1])) for i in range(len(p) - 1))
            print(f"      '{sd}' {len(p)} 句——句间共享词总和 {shared}"
                  f"（{'衔接 ✓' if shared >= 2 else '弱'}——Maimon）")

    # ---- exp4：拼合多样性 ----
    print("\n[exp4] 拼合多样性（同一种子——不同组合——生成自由度）:")
    for sd in ["苹果"]:
        p1 = paragraph_assemble(w, sd, full)
        p2 = paragraph_assemble(w, sd, full)
        same = p1 == p2
        print(f"      '{sd}' 两次生成 {'相同（确定性）' if same else '不同组合 ✓（拼合自由）'}"
              f"——{len(p1)} 句 vs {len(p2)} 句")
        if not same:
            print(f"        第一次: {p1[:3]}")
            print(f"        第二次: {p2[:3]}")
    print("\n[done] stage114 paragraph assembly")


if __name__ == "__main__":
    run()
