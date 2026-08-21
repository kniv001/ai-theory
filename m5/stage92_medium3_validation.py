# -*- coding: utf-8 -*-
"""
M5 阶段 92：中等句 3 验证（2-3 分句复合——15-25 字目标）
corpus_medium3.txt（我生成 110 行——因果/时间/条件/场景/转折递进/心理行动/
多动作序列——更长复合）
验证：长复合句的关系可学（3 分句关联）/ 问答 / 旧知识保持
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
    print("=== M5 阶段 92：中等句 3（2-3 分句复合——C13-02） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=900)
    simple2 = load_corpus(os.path.join(base, "corpus_simple2.txt"))
    simple3 = load_corpus(os.path.join(base, "corpus_simple3.txt"))
    simple4 = load_corpus(os.path.join(base, "corpus_simple4.txt"))
    simple5 = load_corpus(os.path.join(base, "corpus_simple5.txt"))
    medium = load_corpus(os.path.join(base, "corpus_medium.txt"))
    medium2 = load_corpus(os.path.join(base, "corpus_medium2.txt"))
    medium3 = load_corpus(os.path.join(base, "corpus_medium3.txt"))
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
    base_corpus = simple + simple2 + simple3 + simple4 + simple5 + wiki + attr + neg + social + isa_sents
    full = base_corpus + medium + medium2 + medium3
    print(f"总语料 {len(full)} 行（基础 {len(base_corpus)} + 中等 {len(medium) + len(medium2) + len(medium3)}）")

    blocks = extract_blocks(full)
    hubs = extract_hubs(full, blocks)
    chars = list(dict.fromkeys("".join(full)))
    w = HubLake(chars, blocks + hubs)
    t0 = time.perf_counter()
    for day in range(5):
        w.learn_epoch_batch(full, B=128)
    print(f"训练完成——{time.perf_counter()-t0:.0f}s（{w.n} 字 / {len(w.hubs)} 河道）")

    print("\n[exp1] 长复合关系（3 分句——因为所以/如果就/虽然但是）:")
    for hub, obj in [("因为", "迟到"), ("因为", "感冒"), ("所以", "晚"), ("如果", "下雨"),
                     ("虽然", "冷"), ("先", "洗"), ("一边", "听")]:
        ans = w.answer(hub, obj)
        print(f"      K['{hub}']['{obj}'] → {[(a, f'{v:.2f}') for a, v in ans[:3]] if ans else '（无）'}")

    print("\n[exp2] 问答（长复合语料后）:")
    for q in ["为什么今天不去公园？", "为什么妹妹喜欢小狗？", "你什么时候起床？", "你喜欢什么动物？"]:
        hub, obj, ans = w.ask(q)
        print(f"      Q: '{q}' → 枢纽'{hub}' '{obj}' → "
              f"{[(a, f'{v:.2f}') for a, v in ans[:3]] if ans else '（无命中）'}")

    print("\n[exp3] 旧知识保持（语料扩展到 3200+ 行后）:")
    for hub, obj, name in [("很", "苹果", "苹果甜"), ("比", "火车", "火车快慢"),
                           ("吃", "猫", "猫吃鱼"), ("很", "妈妈", "妈妈做")]:
        ans = w.answer(hub, obj)
        print(f"      {name}: K['{hub}']['{obj}'] → "
              f"{[(a, f'{v:.2f}') for a, v in ans[:3]] if ans else '（无）'}")
    print("\n[done] stage92 medium3 validation")


if __name__ == "__main__":
    run()
