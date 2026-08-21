# -*- coding: utf-8 -*-
"""
M5 阶段 91：中等句语料验证（用户："基础语料基本补完——开始补中等长度的句子"）

corpus_medium.txt（我生成 ~280 行——10-25 字）：
  修饰扩展（红红的苹果很甜）/ 复合句（因为所以/但是/然后/要是/不但而且）/
  场景句（妈妈在厨房做饭）/ 因果+动作 / 比较+转折 / 心理动词+原因 /
  状态+动作序列——短句复合（C13-02 组合性——基础构件→复合）

验证：
  exp1 中等句可学（新词块/关系涌现）
  exp2 复合理解（比较+转折——K[比][苹果]；因果+动作）
  exp3 问答（中句语料后——"苹果比什么大？"）
  exp4 对照（基础 vs 基础+中等——理解深度）
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
    print("=== M5 阶段 91：中等句语料（短句复合——C13-02 组合性进阶） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=900)
    simple2 = load_corpus(os.path.join(base, "corpus_simple2.txt"))
    simple3 = load_corpus(os.path.join(base, "corpus_simple3.txt"))
    simple4 = load_corpus(os.path.join(base, "corpus_simple4.txt"))
    simple5 = load_corpus(os.path.join(base, "corpus_simple5.txt"))
    medium = load_corpus(os.path.join(base, "corpus_medium.txt"))
    medium2 = load_corpus(os.path.join(base, "corpus_medium2.txt"))
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
    medium_all = medium + medium2
    full = base_corpus + medium_all
    print(f"基础语料 {len(base_corpus)} 行 + 中等句 {len(medium_all)} 行"
          f"（medium 175 + medium2 106——8-16 字复合）——总 {len(full)} 行")
    lens = [len(s) for s in medium_all]
    print(f"中等句长度: min {min(lens)} / mean {np.mean(lens):.0f} / max {max(lens)}")

    # ---- 对照：基础 vs 基础+中等 ----
    blocks = extract_blocks(full)
    hubs = extract_hubs(full, blocks)
    chars = list(dict.fromkeys("".join(full)))
    w = HubLake(chars, blocks + hubs)
    t0 = time.perf_counter()
    for day in range(5):
        w.learn_epoch_batch(full, B=128)
    print(f"训练完成——{time.perf_counter()-t0:.0f}s（{w.n} 字 / {len(w.hubs)} 河道）")

    # ---- exp1：新关系涌现 ----
    print("\n[exp1] 中等句的新关系（修饰/复合/心理）:")
    for hub, obj in [("很", "红红"), ("很", "妈妈"), ("很", "奶奶"), ("喜欢", "红"),
                     ("比", "苹果"), ("觉得", "书"), ("要是", "下雨")]:
        ans = w.answer(hub, obj)
        print(f"      K['{hub}']['{obj}'] → {[(a, f'{v:.2f}') for a, v in ans[:3]] if ans else '（无）'}")

    # ---- exp2：复合理解 ----
    print("\n[exp2] 复合句理解（比较+转折/因果+动作——多层关系）:")
    for hub, obj in [("比", "大象"), ("比", "火车"), ("比", "苹果"), ("所以", "饿")]:
        ans = w.answer(hub, obj)
        print(f"      K['{hub}']['{obj}'] → {[(a, f'{v:.2f}') for a, v in ans[:3]] if ans else '（无）'}")

    # ---- exp3：问答 ----
    print("\n[exp3] 问答（中等句语料后）:")
    for q in ["苹果比什么大？", "大象比什么大？", "为什么要穿棉衣？", "你喜欢什么颜色？"]:
        hub, obj, ans = w.ask(q)
        print(f"      Q: '{q}' → 枢纽'{hub}' '{obj}' → "
              f"{[(a, f'{v:.2f}') for a, v in ans[:3]] if ans else '（无命中）'}")

    # ---- exp4：模板生成（中等句结构） ----
    print("\n[exp4] 模板（中等句的枢纽序列——复合模板涌现）:")
    from collections import Counter as C2
    tmpl = C2()
    for s in medium_all:
        hits = sorted([h for h in set(blocks + hubs) if h in s], key=s.index)
        if len(hits) >= 2:
            parts = []
            for i, h in enumerate(hits):
                if i == 0 and s.index(h) > 0:
                    parts.append("X")
                elif i > 0:
                    parts.append("X")
                parts.append(h)
            parts.append("X")
            tmpl[tuple(parts)] += 1
    for t, c in tmpl.most_common(6):
        print(f"      {list(t)} × {c}")
    print("\n[done] stage91 medium validation")


if __name__ == "__main__":
    run()
