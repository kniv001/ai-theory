# -*- coding: utf-8 -*-
"""
M5 阶段 77：trigram 上下文生成（多义链区分——"猫吃鱼" vs "吃月饼"——
stage76 已知边界——单强度无法区分——上下文升级）

机制：
  ① 语料 n-gram 统计（bigram/trigram——"猫吃"→[鱼:次数]——方向一致性）
  ② 生成时最后 2 字 → 候选按 trigram 频次排序（语料一致优先——"猫吃"→鱼 ✓/
     "猫吃"→月 ✗（语料无"猫吃月"——跳过））
  ③ 无 trigram 时回退单字 K（stage76 边界——骤降停止）
对照：单字 K（stage76——多义混）vs trigram（本——区分）
"""
import os
import time
from collections import Counter
import numpy as np
from stage72_integration_v3 import PipelineLake
from stage73_evaluation import load_corpus
import stage72_integration_v3 as s72
from stage76_generation_boundary import generate_boundary


def build_ngrams(sents):
    """bigram/trigram 方向统计（语料顺序）"""
    bi = Counter()
    tri = Counter()
    for s in sents:
        for k in range(len(s) - 1):
            bi[s[k:k + 2]] += 1
        for k in range(len(s) - 2):
            tri[s[k:k + 3]] += 1
    return bi, tri


def generate_trigram(w, sents, seed, bi, tri, max_len=8):
    """trigram 上下文生成（多义区分——语料一致优先）"""
    out = seed
    for _ in range(max_len):
        last2 = out[-2:]
        last1 = out[-1]
        if last1 not in w.ci:
            break
        # trigram 一致候选（最后 2 字 → 下一字——语料方向）
        cands = []
        for k in range(1 if False else 0, 1):
            pass
        for (tg, cnt) in tri.items():
            if tg[:2] == last2:
                cands.append((tg[2], cnt))
        if not cands:
            # 回退 bigram
            for (bg, cnt) in bi.items():
                if bg[0] == last1:
                    cands.append((bg[1], cnt))
        if not cands:
            break
        # 排序——选语料一致的（去重——未用过）
        best = None
        for cand, cnt in sorted(cands, key=lambda x: -x[1]):
            if cand not in out and cand in w.ci:
                best = cand
                break
        if best is None:
            break
        out += best
    return out


def run():
    print("=== M5 阶段 77：trigram 上下文生成（多义链区分——猫吃鱼 vs 吃月饼） ===\n")
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
    bi, tri = build_ngrams(simple + wiki)
    # 对照：单字 K（stage76）vs trigram（本）
    print("\n[对照] 单字 K（多义混）vs trigram（区分）:")
    for sd in ["小猫", "苹果", "天气", "老师"]:
        k1 = generate_boundary(w, sd)
        k3 = generate_trigram(w, simple + wiki, sd, bi, tri)
        print(f"      '{sd}': 单字K '{k1}' vs trigram '{k3}'")
    # 多义测试（吃→鱼 vs 月饼——上下文）
    print("\n[多义] '吃' 的上下文区分:")
    print(f"      '小猫吃' → {generate_trigram(w, simple+wiki, '小猫吃', bi, tri)}"
          f"（应'鱼'——小猫吃鱼）")
    print(f"      '中秋吃' → {generate_trigram(w, simple+wiki, '中秋吃', bi, tri)}"
          f"（应'月饼'——中秋吃月饼）")
    print("\n[done] stage77 trigram generation")


if __name__ == "__main__":
    run()
