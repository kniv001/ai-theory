# -*- coding: utf-8 -*-
"""
M5 阶段 89：基础语料拓展 5 验证（simple5——动作词/问句补足）

corpus_simple5.txt（我生成 ~400 行）：
  动作高频化（喝/看/写/说/听/读/走/跑/拿/放——单字枢纽补全）/
  "怎么样"问句 30 句（问答功能）/ 动物习性 / 植物生长 / 身体功能 /
  季节天气 / 空间时间

验证：
  exp1 动作词进单字枢纽（K 河道——喝/看/写/说/听）
  exp2 问句词块（"怎么"+"么样"——3 字拆两块——逐对合并）
  exp3 问答（"苹果怎么样？"——hub 选择？）
  exp4 旧知识保持
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
    print("=== M5 阶段 89：基础语料拓展 5（动作词/问句补足验证） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=900)
    simple2 = load_corpus(os.path.join(base, "corpus_simple2.txt"))
    simple3 = load_corpus(os.path.join(base, "corpus_simple3.txt"))
    simple4 = load_corpus(os.path.join(base, "corpus_simple4.txt"))
    simple5 = load_corpus(os.path.join(base, "corpus_simple5.txt"))
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
    bc = simple + simple2 + simple3 + simple4 + simple5 + wiki + attr + neg + social + isa_sents
    print(f"总语料 {len(bc)} 行")

    blocks = extract_blocks(bc)
    hubs = extract_hubs(bc, blocks)
    all_hubs = blocks + hubs
    print(f"\n[exp1] 词块/枢纽（56 块 + {len(hubs)} 单字）——动作词:")
    for c in ["喝", "看", "写", "说", "听", "吃", "放", "开", "关", "叫", "香"]:
        print(f"      '{c}' {'[单字枢纽✓]' if c in hubs else '[未进]'}")
    print("      问句块:", [b for b in blocks if any(q in b for q in "怎么么样为什")][:6])

    chars = list(dict.fromkeys("".join(bc)))
    w = HubLake(chars, all_hubs)
    t0 = time.perf_counter()
    for day in range(5):
        w.learn_epoch_batch(bc, B=128)
    print(f"训练完成——{time.perf_counter()-t0:.0f}s（{w.n} 字 / {len(w.hubs)} 河道）")

    print("\n[exp2] 动作河道（动作词进枢纽后——K 检索）:")
    for hub, obj in [("喝", "水"), ("看", "书"), ("写", "作"), ("说", "话"),
                     ("听", "音"), ("吃", "鱼"), ("放", "书")]:
        ans = w.answer(hub, obj)
        print(f"      K['{hub}']['{obj}'] → {[(a, f'{v:.2f}') for a, v in ans[:3]] if ans else '（无）'}")

    print("\n[exp3] 问答（怎么样问句——hub 选择观察）:")
    for q in ["苹果怎么样？", "今天的天气怎么样？", "为什么会下雨？"]:
        hub, obj, ans = w.ask(q)
        print(f"      Q: '{q}' → 枢纽'{hub}' '{obj}' → "
              f"{[(a, f'{v:.2f}') for a, v in ans[:3]] if ans else '（无命中）'}")

    print("\n[exp4] 旧知识保持:")
    for hub, obj, name in [("很", "苹果", "苹果甜"), ("是", "月亮", "月亮圆"),
                           ("很", "柠檬", "柠檬酸"), ("吃", "猫", "猫吃鱼")]:
        ans = w.answer(hub, obj)
        print(f"      {name}: K['{hub}']['{obj}'] → "
              f"{[(a, f'{v:.2f}') for a, v in ans[:3]] if ans else '（无）'}")
    print("\n[done] stage89 simple5 validation")


if __name__ == "__main__":
    run()
