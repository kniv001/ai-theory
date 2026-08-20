# -*- coding: utf-8 -*-
"""
M5 阶段 56：篇章组合（R16 尺度递归宏观——句→段——组合性在句间尺度——
"加法"延续——框架理论 + 研究双锚）

理论锚：R16（尺度递归——句子=词组的组合→段落=句子的组合——同机制放大）
研究锚：Maimon NAACL 2025（话语连贯 = 衔接 cohesion/一致性 consistency/相关性 relevance——
  联合训练——连贯的计算定义）/ Halliday 词汇衔接（相邻句共享词）/ Cortex 2025（连贯-创造力权衡）
机制：
  ① 衔接（cohesion）：相邻句共享词/词组数（词汇衔接——"农业属于…"与"全球农业…"共享"农业"）
  ② 段落生成（句链）：起始句 → 衔接最强的下一句（共享词驱动——连贯的句序列）
  ③ 一致性（consistency）：段落内主题持续（共享词率——窗口）
验证：
  exp1 衔接检测（同主题句对 vs 随机句对——共享词强度）
  exp2 段落生成（起始句 → 句链——衔接驱动——段落雏形）
  exp3 一致性（生成段落的主题持续——共享词率）
"""
import os
import re
import time
from collections import Counter
import numpy as np

def load_corpus(path, lo=3, hi=80, n=None):
    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    clean = [s for s in lines if lo <= len(s) <= hi
             and re.search(r"[一-鿿]", s) and not re.search(r"[A-Za-z]", s)]
    if n and len(clean) > n:
        clean = clean[:n]
    return clean

def cohesion(s1, s2):
    """衔接（词汇共享——cohesion——Maimon）"""
    c1, c2 = set(s1), set(s2)
    shared = c1 & c2
    return len(shared) / max(len(c1 | c2), 1), shared

def run():
    print("=== M5 阶段 56：篇章组合（R16 句→段——Maimon 连贯三条件） ===\n")
    base = os.path.dirname(__file__)
    wiki = load_corpus(os.path.join(base, "corpus_wiki_filtered.txt"), n=3000)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=900)
    sents = wiki + simple
    print(f"语料 {len(sents)} 行")
    # ---- exp1：衔接检测（同主题 vs 随机） ----
    print("\n[exp1] 衔接检测（共享词率——同主题 vs 随机）:")
    same_theme = [("农业属于第一级产业包括作物种植", "全球农业年产出大量食物"),
                  ("技术进步改变世界经济发展", "技术发展提高生产效率"),
                  ("因为下雨所以带伞", "下雨天路很滑")]
    rng = np.random.default_rng(7)
    for a, b in same_theme:
        c, shared = cohesion(a, b)
        r1, r2 = rng.choice(sents, 2, replace=False)
        cr, _ = cohesion(r1, r2)
        print(f"      同主题（{a[:6]}…/{b[:6]}…）: 衔接 {c:.3f} 共享={sorted(shared)[:4]}"
              f" vs 随机 {cr:.3f}（{'衔接显著 ✓' if c > cr * 2 else '—'}）")
    # ---- exp2：段落生成（句链——衔接驱动） ----
    print("\n[exp2] 段落生成（起始句 → 衔接最强句链——连贯段落雏形）:")
    seeds = ["农业属于第一级产业", "技术进步改变世界", "因为下雨所以带伞"]
    for sd in seeds:
        para = [sd]
        for _ in range(3):
            cur = para[-1]
            # 与当前句衔接最强的语料句（未用过）
            best, best_c = None, 0.0
            for s in sents:
                if s in para or len(s) > 30:
                    continue
                c, _ = cohesion(cur, s)
                if c > best_c:
                    best, best_c = s, c
            if best and best_c > 0.1:
                para.append(best)
            else:
                break
        print(f"      '{sd}' →")
        for p in para[1:]:
            print(f"        ↳ '{p[:25]}…'（衔接 {cohesion(para[para.index(p)-1], p)[0]:.2f}）")
    # ---- exp3：一致性（段落主题持续——共享词率） ----
    print("\n[exp3] 一致性（生成段落内共享词率——主题持续）:")
    for sd in seeds:
        para = [sd]
        for _ in range(3):
            cur = para[-1]
            best, best_c = None, 0.0
            for s in sents:
                if s in para or len(s) > 30:
                    continue
                c, _ = cohesion(cur, s)
                if c > best_c:
                    best, best_c = s, c
            if best and best_c > 0.1:
                para.append(best)
            else:
                break
        # 段内平均衔接
        if len(para) > 1:
            avg = np.mean([cohesion(para[i], para[i + 1])[0] for i in range(len(para) - 1)])
            print(f"      '{sd[:8]}…' → 段落 {len(para)} 句——平均衔接 {avg:.2f}"
                  f"（{'主题持续 ✓' if avg > 0.15 else '衔接弱'}）")
    print("\n[done] stage56 discourse")


if __name__ == "__main__":
    run()
