# -*- coding: utf-8 -*-
"""
M5 阶段 86：基础语料拓展 3 验证（用户："继续补充基础语料——框架输入只有字符——
无法多模态了解关系——语料=唯一关系来源"）

corpus_simple3.txt（我生成 ~450 行）：
  情绪感受（高兴/害怕/惊喜）/ 状态变化（花开/水开/饭熟）/ 工具使用（用笔写/
  用筷吃）/ 运动（爬/飞/游）/ 味道气味（甜酸辣咸香）/ 声音（叫/响/鸣）/
  心理动词（喜欢/讨厌/想要）/ 数量变化（多/少）/ 简单因果（因为所以——
  更多）/ 问句对（是什么/怎么样/为什么——问答语料——"怎么样"零河道解决）

验证：
  exp1 问句词块涌现（"什么/怎么样"——问句语料后成块——零河道解决）
  exp2 新关系检索（味道 K[酸][柠檬]？/声音 K[叫][狗]？/状态 K[熟][饭]？）
  exp3 问答（语料问答对——"苹果是什么？"→水果——"水怎么样？"→干净）
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
    print("=== M5 阶段 86：基础语料拓展 3（字符输入——语料=唯一关系来源） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=900)
    simple2 = load_corpus(os.path.join(base, "corpus_simple2.txt"))
    simple3 = load_corpus(os.path.join(base, "corpus_simple3.txt"))
    wiki = load_corpus(os.path.join(base, "corpus_wiki_filtered.txt"), n=600)
    attr = load_corpus(os.path.join(base, "corpus_attr_cause.txt"))
    neg = load_corpus(os.path.join(base, "corpus_negation.txt"))
    social = load_corpus(os.path.join(base, "corpus_social.txt"))
    proper = load_corpus(os.path.join(base, "corpus_proper.txt"))
    isa_sents = ["苹果是水果", "香蕉是水果", "西瓜是水果", "葡萄是水果",
                 "猫是动物", "狗是动物", "鸟是动物", "鱼是动物",
                 "水是液体", "冰是固体", "雪是白色的", "天空是蓝色的",
                 "老虎是动物", "树是植物", "花是植物", "石头是固体",
                 "苹果可以吃", "水可以喝", "雨是从云落下来的",
                 "小猫吃鱼", "猫吃老鼠", "我吃苹果", "小猫吃月饼"]
    base_corpus = simple + simple2 + simple3 + wiki + attr + neg + social + isa_sents
    print(f"基础语料 {len(base_corpus)} 行（simple3 加入 {len(simple3)}——总行数 "
          f"{len(simple)}+{len(simple2)}+{len(simple3)}+{len(wiki)}+{len(attr)}+{len(neg)}+{len(social)}）")

    # ---- exp1：问句词块涌现 ----
    blocks = extract_blocks(base_corpus)
    print(f"\n[exp1] 问句词块（simple3 问句语料后——'什么/怎么样'零河道解决）:")
    for w in ["什么", "怎么样", "为什么", "是什么", "哪里", "时候", "谁", "吗"]:
        print(f"      '{w}' {'[词块✓]' if w in blocks else '[未成块]'}")
    print(f"      词块总数 {len(blocks)}")

    # ---- exp2/3：训练 + 检索 ----
    hubs = extract_hubs(base_corpus, blocks)
    all_hubs = blocks + hubs
    chars = list(dict.fromkeys("".join(base_corpus + proper)))
    print(f"\n词汇表 {len(chars)} 字 / 枢纽 {len(all_hubs)} 个")
    w = HubLake(chars, all_hubs)
    t0 = time.perf_counter()
    for day in range(5):
        w.learn_epoch_batch(base_corpus, B=128)
        if day == 4:
            w.learn_epoch_batch(proper, B=128)
    print(f"训练完成——{time.perf_counter()-t0:.0f}s")
    print("\n[exp2] 新关系检索（simple3 主题块）:")
    for hub, obj in [("很", "柠檬"), ("很", "辣椒"), ("叫", "狗"), ("熟", "饭"),
                     ("香", "花"), ("害怕", "黑"), ("用", "笔"), ("多", "树")]:
        ans = w.answer(hub, obj)
        if ans:
            print(f"      K['{hub}']['{obj}'] → {[(a, f'{v:.2f}') for a, v in ans[:3]]}")
        else:
            print(f"      K['{hub}']['{obj}'] → （无关联）")

    # ---- exp3：问答（问句语料——问答对） ----
    print("\n[exp3] 问答（simple3 问答对语料——之前零河道现在可答）:")
    for q in ["苹果是什么？", "水怎么样？", "小猫吃什么？", "天空怎么样？",
              "鱼生活在哪里？", "花开了吗？", "谁来了？"]:
        hub, obj, ans = w.ask(q)
        if ans:
            print(f"      Q: '{q}' → 枢纽'{hub}' '{obj}' → {[(a, f'{v:.2f}') for a, v in ans[:3]]}")
        else:
            print(f"      Q: '{q}' → （无命中）")
    print("\n[done] stage86 simple3 validation")


if __name__ == "__main__":
    run()
