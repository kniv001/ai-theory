# -*- coding: utf-8 -*-
"""
M5 阶段 74：管线参数平衡（量质双优——"少时间多学习"最优化——扫描）
"""
import os
import time
import numpy as np
from stage72_integration_v3 import PipelineLake
from stage73_evaluation import load_corpus
import stage72_integration_v3 as s72


def run():
    print("=== M5 阶段 74：管线参数平衡（量质双优——少时间多学习最优化） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=900)
    chars = list(dict.fromkeys("".join(simple)))
    print(f"词汇表 {len(chars)} 字 / 语料 {len(simple)} 行")
    important = ["苹果很甜", "天气变冷"]
    vals = [1 if s in important else 0 for s in simple]
    print("\n[扫描] 清噪(SCALE_NOISE) × 硬化(H_RATE)——词组量 + 生成质量:")
    results = []
    for sn in [0.85, 0.92, 0.97]:
        for hr in [0.02, 0.01, 0.005]:
            s72.SCALE_NOISE = sn
            s72.H_RATE = hr
            w = PipelineLake(chars)
            for c in chars:
                w.inject(c)
            w._decay()
            w.build_neighbors()
            t0 = time.perf_counter()
            for day in range(5):
                w.learn_day(simple, values=vals, important=important)
                w.sleep_night()
                if day == 2:
                    w.build_neighbors()
            dt = time.perf_counter() - t0
            groups = sum(1 for i in range(len(w.chars)) for j in range(i + 1, len(w.chars))
                         if w.K[i, j] > 0.02)
            gen = w.generate("苹果")
            good = gen.startswith("苹果很") and len(gen) >= 4
            results.append((groups, good, sn, hr))
            print(f"  清噪{sn} × 硬化{hr}: 词组 {groups}——生成'{gen}'"
                  f"{' ✓' if good else ''}（{dt:.0f}s）")
    # 平衡点
    best = max(results, key=lambda r: (r[1], r[0]))
    print(f"\n[平衡点] 量质双优: 清噪{best[2]} × 硬化{best[3]}——词组 {best[0]} + 生成 ✓")
    print("[done] stage74 pipeline tuning")


if __name__ == "__main__":
    run()
