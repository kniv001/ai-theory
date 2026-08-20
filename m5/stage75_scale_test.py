# -*- coding: utf-8 -*-
"""
M5 阶段 75：规模测试（管线在真实规模——"少时间多学习"规模化验证——
574 → 3000 句——时间/词组/生成效率保持？）
"""
import os
import time
import numpy as np
from stage72_integration_v3 import PipelineLake
from stage73_evaluation import load_corpus
import stage72_integration_v3 as s72


def run():
    print("=== M5 阶段 75：规模测试（管线 574 → 3000 句——规模化验证） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=900)
    s72.SCALE_NOISE = 0.97
    s72.H_RATE = 0.005
    important = ["苹果很甜", "天气变冷", "我喜欢学习"]
    # 规模递增（574 → 1174 → 2374 → 3574）
    sizes = [(0, "574（simple）"), (600, "1174（+wiki600）"),
             (1800, "2374（+wiki1800）"), (3000, "3574（+wiki3000）")]
    for wiki_n, label in sizes:
        wiki = load_corpus(os.path.join(base, "corpus_wiki_filtered.txt"), n=wiki_n) if wiki_n else []
        sents = simple + wiki
        chars = list(dict.fromkeys("".join(simple)))[:500]
        w = PipelineLake(chars)
        for c in chars:
            w.inject(c)
        w._decay()
        w.build_neighbors()
        vals = [1 if s in important else 0 for s in sents]
        t0 = time.perf_counter()
        for day in range(3):
            w.learn_day(sents, values=vals, important=important)
            w.sleep_night()
            if day == 1:
                w.build_neighbors()
        dt = time.perf_counter() - t0
        groups = sum(1 for i in range(w.n) for j in range(i + 1, w.n) if w.K[i, j] > 0.02)
        gen = w.generate("苹果")
        good = gen.startswith("苹果")
        print(f"  {label}: {len(sents)} 句×3天——{dt:.0f}s——词组 {groups}——生成'{gen}'"
              f"{' ✓' if good else ''}（n={w.n}）")
    print("\n[done] stage75 scale test")


if __name__ == "__main__":
    run()
