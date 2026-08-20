# -*- coding: utf-8 -*-
"""
M5 阶段 88：基础语料拓展 4（持续学习追加——stage87 机制直接使用）
corpus_simple4.txt（我生成 ~370 行）：
  动作高频化（吃/喝/看/写/听/说——日常动作过 30 次阈值——单字枢纽补全）/
  状态词（熟/香/叫/响）/ 心理动词（害怕/喜欢）/ 问句（怎么样/为什么——
  词块涌现）/ 量词数量 / 植物果实 / 食物做法 / 季节 / 大小比较

验证：
  exp1 词汇/河道持续增长（追加后）
  exp2 动作词进单字枢纽（吃/喝/看/写——K 河道）
  exp3 问句词块（怎么样/为什么——成块？）
  exp4 问答（"苹果怎么样？"→甜——之前零河道）
  exp5 旧知识保持（苹果甜——C2-06）
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
    print("=== M5 阶段 88：基础语料拓展 4（持续学习追加——动作词/问句补全） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=900)
    simple2 = load_corpus(os.path.join(base, "corpus_simple2.txt"))
    simple3 = load_corpus(os.path.join(base, "corpus_simple3.txt"))
    simple4 = load_corpus(os.path.join(base, "corpus_simple4.txt"))
    wiki = load_corpus(os.path.join(base, "corpus_wiki_filtered.txt"), n=600)
    attr = load_corpus(os.path.join(base, "corpus_attr_cause.txt"))
    neg = load_corpus(os.path.join(base, "corpus_negation.txt"))
    social = load_corpus(os.path.join(base, "corpus_social.txt"))
    isa_sents = ["苹果是水果", "香蕉是水果", "西瓜是水果", "葡萄是水果",
                 "猫是动物", "狗是动物", "鸟是动物", "鱼是动物",
                 "水是液体", "冰是固体", "雪是白色的", "天空是蓝色的",
                 "老虎是动物", "树是植物", "花是植物", "石头是固体",
                 "苹果可以吃", "水可以喝", "雨是从云落下来的",
                 "小猫吃鱼", "猫吃老鼠", "我吃苹果", "小猫吃月饼"]
    s1 = simple + isa_sents
    s2 = s1 + simple2
    s3 = s2 + simple3 + social + attr + neg

    # ---- 持续学习：A→B→C→D（追加 simple4） ----
    print("[持续] 阶段 A：simple+isa")
    chars0 = list(dict.fromkeys("".join(s1)))
    b0 = extract_blocks(s1)
    h0 = extract_hubs(s1, b0)
    w = HubLake(chars0, b0 + h0)
    for _ in range(3):
        w.learn_epoch_batch(s1, B=128)
    print(f"      A: {w.n} 字 / {len(w.hubs)} 河道")

    def stage(sents):
        blocks = extract_blocks(sents)
        hubs = extract_hubs(sents, blocks)
        for h in blocks + hubs:
            w.add_hub(h)
        for _ in range(3):
            w.learn_epoch_batch(sents, B=128)

    print("[持续] 阶段 B：+simple2")
    stage(simple2)
    print(f"      B: {w.n} 字 / {len(w.hubs)} 河道")
    print("[持续] 阶段 C：+simple3+social+attr+neg")
    stage(simple3 + social + attr + neg)
    print(f"      C: {w.n} 字 / {len(w.hubs)} 河道")
    t0 = time.perf_counter()
    print("[持续] 阶段 D：+simple4（本阶段——持续追加）")
    stage(simple4)
    print(f"      D: {w.n} 字 / {len(w.hubs)} 河道——{time.perf_counter()-t0:.0f}s")

    # ---- exp2：动作词进单字枢纽 ----
    print("\n[exp2] 动作词枢纽（simple4 高频化后——吃/喝/看/写）:")
    for hub, obj in [("吃", "猫"), ("喝", "水"), ("看", "电"), ("写", "作"),
                     ("吃", "鱼"), ("听", "音")]:
        ans = w.answer(hub, obj)
        print(f"      K['{hub}']['{obj}'] → {[(a, f'{v:.2f}') for a, v in ans[:3]] if ans else '（无）'}")

    # ---- exp3：问句词块 ----
    print("\n[exp3] 问句词块（simple4 问句补足后）:")
    blocks_all = extract_blocks(s3 + simple4)
    for qw in ["怎么样", "为什么", "什么", "哪里", "谁"]:
        print(f"      '{qw}' {'[词块✓]' if qw in blocks_all else '[未成块]'}")

    # ---- exp4：问答 ----
    print("\n[exp4] 问答（怎么样/为什么——之前零河道）:")
    for q in ["苹果怎么样？", "天气怎么样？", "为什么会下雨？", "为什么要洗手？"]:
        hub, obj, ans = w.ask(q)
        print(f"      Q: '{q}' → 枢纽'{hub}' '{obj}' → "
              f"{[(a, f'{v:.2f}') for a, v in ans[:3]] if ans else '（无命中）'}")

    # ---- exp5：旧知识保持 ----
    print("\n[exp5] 旧知识保持（C2-06——四阶段后）:")
    for hub, obj, name in [("很", "苹果", "苹果甜"), ("是", "月亮", "月亮圆"),
                           ("很", "柠檬", "柠檬酸"), ("是", "我", "身份")]:
        ans = w.answer(hub, obj)
        print(f"      {name}: K['{hub}']['{obj}'] → "
              f"{[(a, f'{v:.2f}') for a, v in ans[:3]] if ans else '（无）'}")
    print("\n[done] stage88 simple4 continual")


if __name__ == "__main__":
    run()
