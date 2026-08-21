# -*- coding: utf-8 -*-
"""
M5 阶段 98：长句阶段（wiki 30-80 字——stage39 加难原则 33%→50%→85% 混合渐进）

语料阶梯：短句 ≤8 → 中等 8-18 → 长句 30-60（wiki——mean 35 字）
机制（渐进混合——stage39 验证过：复杂句 33%→50%→85% 非一步到位）：
  ① day1-2：基础（simple+medium+why+social+neg+attr+isa——无长句）
  ② day3：+长句 33%
  ③ day4：+长句 50%
  ④ day5：+长句 85%
  对照：全混合（每天全部）
验证：
  exp1 长句可学（K[属于][农业]→产业/作物——长句关系）
  exp2 渐进 vs 一次性（长句理解对比——基础保持）
  exp3 长句生成（记忆检索/焦点构建——种子→长句）
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


def run():
    print("=== M5 阶段 98：长句阶段（wiki 30-80 字——渐进混合——stage39） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=900)
    simple2 = load_corpus(os.path.join(base, "corpus_simple2.txt"))
    simple3 = load_corpus(os.path.join(base, "corpus_simple3.txt"))
    simple4 = load_corpus(os.path.join(base, "corpus_simple4.txt"))
    simple5 = load_corpus(os.path.join(base, "corpus_simple5.txt"))
    medium = load_corpus(os.path.join(base, "corpus_medium.txt"))
    medium2 = load_corpus(os.path.join(base, "corpus_medium2.txt"))
    medium3 = load_corpus(os.path.join(base, "corpus_medium3.txt"))
    why = load_corpus(os.path.join(base, "corpus_why.txt"))
    wiki = load_corpus(os.path.join(base, "corpus_wiki_filtered.txt"), n=600)
    attr = load_corpus(os.path.join(base, "corpus_attr_cause.txt"))
    neg = load_corpus(os.path.join(base, "corpus_negation.txt"))
    social = load_corpus(os.path.join(base, "corpus_social.txt"))
    isa_sents = ["苹果是水果。", "香蕉是水果。", "西瓜是水果。", "葡萄是水果。",
                 "猫是动物。", "狗是动物。", "鸟是动物。", "鱼是动物。",
                 "水是液体。", "冰是固体。", "雪是白色的。", "天空是蓝色的。",
                 "老虎是动物。", "树是植物。", "花是植物。", "石头是固体。",
                 "苹果可以吃。", "水可以喝。", "雨是从云落下来的。",
                 "小猫吃鱼。", "猫吃老鼠。", "我吃苹果。", "小猫吃月饼。"]
    base_corpus = simple + simple2 + simple3 + simple4 + simple5 + medium + medium2 + medium3 + why + attr + neg + social + isa_sents
    long_sents = wiki
    print(f"基础语料 {len(base_corpus)} 行（短+中）+ 长句 {len(long_sents)} 行"
          f"（wiki mean {int(np.mean([len(s) for s in long_sents]))} 字）")

    blocks = extract_blocks(base_corpus + long_sents)
    hubs = extract_hubs(base_corpus + long_sents, blocks)
    chars = list(dict.fromkeys("".join(base_corpus + long_sents)))
    print(f"词汇表 {len(chars)} 字 / 枢纽 {len(blocks) + len(hubs)} 个")

    # ---- 渐进训练 ----
    w_prog = HubLake(chars, blocks + hubs)
    t0 = time.perf_counter()
    for day in range(5):
        if day < 2:
            w_prog.learn_epoch_batch(base_corpus, B=128)
        elif day == 2:
            w_prog.learn_epoch_batch(base_corpus + long_sents[:200], B=128)   # 33%
        elif day == 3:
            w_prog.learn_epoch_batch(base_corpus + long_sents[:300], B=128)   # 50%
        else:
            w_prog.learn_epoch_batch(base_corpus + long_sents[:510], B=128)   # 85%
    print(f"渐进训练完成——{time.perf_counter()-t0:.0f}s（33%→50%→85%——stage39）")

    # ---- 对照：一次性全混合 ----
    w_mix = HubLake(chars, blocks + hubs)
    mixed = base_corpus + long_sents
    t1 = time.perf_counter()
    for day in range(5):
        w_mix.learn_epoch_batch(mixed, B=128)
    print(f"一次性训练完成——{time.perf_counter()-t1:.0f}s（对照——全混合）")

    # ---- exp1：长句关系可学 ----
    print("\n[exp1] 长句关系（wiki 概念句——检索）:")
    for hub, obj in [("属于", "农业"), ("包括", "食物"), ("是", "农业"), ("是", "水")]:
        for name, w in [("渐进", w_prog), ("一次性", w_mix)]:
            ans = w.answer(hub, obj)
            print(f"      [{name}] K['{hub}']['{obj}'] → "
                  f"{[(a, f'{v:.2f}') for a, v in ans[:3]] if ans else '无'}")
        print()

    # ---- exp2：基础保持（长句加入后） ----
    print("\n[exp2] 基础保持（长句 85% 后——短中句知识）:")
    for hub, obj, name in [("很", "苹果", "苹果甜"), ("比", "火车", "火车快慢"),
                           ("吃", "猫", "猫吃鱼"), ("很", "妈妈", "妈妈做")]:
        ans = w_prog.answer(hub, obj)
        print(f"      {name}: K['{hub}']['{obj}'] → "
              f"{[(a, f'{v:.2f}') for a, v in ans[:3]] if ans else '无'}")

    # ---- exp3：长句生成（记忆检索——种子→长句候选） ----
    print("\n[exp3] 长句生成（种子 → 长句记忆检索——渐进湖）:")
    for sd in ["农业", "食物", "软件", "天气"]:
        if sd[-1] not in w_prog.ci:
            continue
        i = w_prog.ci[sd[-1]]
        cands = []
        for s in long_sents:
            if sd in s:
                idx = [w_prog.ci[c] for c in s if c in w_prog.ci]
                sc = float(np.mean([w_prog.KT[i, j] for j in idx]))
                cands.append((sc, s))
        if cands:
            cands.sort(key=lambda x: -x[0])
            print(f"      '{sd}' → '{cands[0][1][:40]}…'  [{len(cands)} 候选]")
        else:
            print(f"      '{sd}' → （长句无候选）")
    print("\n[done] stage98 long sentence")


if __name__ == "__main__":
    run()
