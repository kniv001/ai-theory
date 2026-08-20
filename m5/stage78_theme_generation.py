# -*- coding: utf-8 -*-
"""
M5 阶段 78：衔接约束生成（长链漂移解决——stage77 已知边界——
候选须与已生成序列保持关联（K 衔接——序列一致性）——漂移即停——
与 stage76 边界（强度骤降）互补：衔接 = 主题保持）

理论锚：Maimon 2025（话语连贯 = 衔接 cohesion——词汇共享——序列一致性）
机制：
  ① trigram 候选（stage77——多义区分）
  ② 衔接检查：候选与已生成序列的平均 K 关联（序列一致性——主题保持）
     ——"汤"与"中秋/月饼/太硬"的 K 弱（漂移）→ 停
     ——"鱼"与"小猫吃"的 K 强（主题内）→ 继续
  ③ 衔接弱 → 停止（干净句——不漂移）
对照：无衔接（漂移）vs 衔接约束（主题内——干净）
"""
import os
import time
from collections import Counter
import numpy as np
from stage72_integration_v3 import PipelineLake
from stage73_evaluation import load_corpus
import stage72_integration_v3 as s72
from stage77_trigram_generation import build_ngrams, generate_trigram


def generate_cohesion(w, sents, seed, bi, tri, min_coh=0.005, max_len=8):
    """衔接约束生成：trigram 候选 + 序列一致性（漂移即停）"""
    out = seed
    for _ in range(max_len):
        last2 = out[-2:]
        last1 = out[-1]
        if last1 not in w.ci:
            break
        cands = []
        for (tg, cnt) in tri.items():
            if tg[:2] == last2:
                cands.append((tg[2], cnt))
        if not cands:
            for (bg, cnt) in bi.items():
                if bg[0] == last1:
                    cands.append((bg[1], cnt))
        best = None
        for cand, cnt in sorted(cands, key=lambda x: -x[1]):
            if cand not in out and cand in w.ci:
                best = cand
                break
        if best is None:
            break
        # 衔接检查：候选与已生成序列的平均 K 关联（主题保持——Maimon cohesion）
        if len(out) >= 2:
            i = w.ci[best]
            seq_idx = [w.ci[c] for c in out if c in w.ci]
            coh = np.mean([w.K[i, j] for j in seq_idx])
            if coh < min_coh:
                break   # 漂移（与序列无关联——主题外）→ 停
        out += best
    return out


def run():
    print("=== M5 阶段 78：衔接约束生成（长链漂移解决——序列一致性——主题保持） ===\n")
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
    print("\n[对照] 无衔接（漂移）vs 衔接约束（主题内——干净）:")
    for sd in ["小猫", "中秋", "苹果", "老师", "天气"]:
        old = generate_trigram(w, simple + wiki, sd, bi, tri)
        new = generate_cohesion(w, simple + wiki, sd, bi, tri)
        print(f"      '{sd}': 无衔接 '{old}' vs 衔接 '{new}'")
    print("\n[done] stage78 theme generation")


if __name__ == "__main__":
    run()
