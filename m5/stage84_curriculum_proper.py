# -*- coding: utf-8 -*-
"""
M5 阶段 84：专名后置课程（用户："专有名词要最后训练——过于提前的拓展名词
会直接影响判断——例如小孩会产生为什么苹果有手机这个概念——过早见识专名的问题"）

理论锚：
  C29-01（先入为主原则——联想目标优先性 = 习得时间优先性——奠基→沉积→
    新奇-已知——基础语义先奠基——专名后扩展——先入为主的正确应用）
  C26-01（文化转导 = 第二层——专名 = 文化符号——学习序列中文化层后置）
  C87-01（世界框架早期定型——早期经验塑造骨架——污染骨架比补充知识更糟）
  C22-01（基础语义河道先建成——专名后学挂载——多义消歧有基础）

编排（三层课程）：
  ① 基础先行（day1-2）：simple + social + attr/neg + isa（经验锚定 + 社会基础）
  ② 概念主题（day3-4）：wiki 主体（农业/软件/系统——概念词——稳定语义——
     无污染型专名——corpus_proper 已分离）
  ③ 专名最后（day5）：corpus_proper（苹果电脑/乔布斯/北京/北京大学——
     命名/机构——文化层）

对照：混合（现状——全部混在一起每天）——"苹果"判断对比
验证：
  exp1 编排 vs 混合——"苹果"河道（水果优先 vs 公司污染）
  exp2 专名后置后——"苹果"稳定（甜/脆/吃——不受电/脑干扰）
  exp3 专名本身（最后学习——"苹果电脑"成块——指称不污染水果）
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
    print("=== M5 阶段 84：专名后置课程（用户：专名最后训练——过早=判断污染） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=900)
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
    # wiki 主体 = 去专名句（corpus_proper 已分离——标注元数据）
    wiki_proper_text = "".join(proper)
    wiki_base = [s for s in wiki if not any(p in s for p in
                  ["苹果电脑", "乔布斯", "帕罗奥多", "全录", "赵紫阳", "屠呦呦",
                   "诺贝尔", "北京大学", "北京市", "上海市", "微软"])]
    base_corpus = simple + social + attr + neg + isa_sents + wiki_base
    print(f"基础语料 {len(base_corpus)} 行 / 专名语料 {len(proper)} 条（最后训练）")

    blocks = extract_blocks(base_corpus)
    hubs = extract_hubs(base_corpus, blocks)
    all_hubs = blocks + hubs
    chars = list(dict.fromkeys("".join(base_corpus + proper)))
    print(f"词汇表 {len(chars)} 字 / 枢纽 {len(all_hubs)} 个")

    # ---- 编排：基础先行 → 概念 → 专名最后 ----
    w_cur = HubLake(chars, all_hubs)
    t0 = time.perf_counter()
    for day in range(5):
        if day < 3:
            w_cur.learn_epoch_batch(base_corpus, B=128)          # day1-3 基础
        elif day == 3:
            w_cur.learn_epoch_batch(wiki_base, B=128)            # day4 概念
        else:
            w_cur.learn_epoch_batch(base_corpus + proper, B=128) # day5 专名后置
    print(f"编排训练完成——{time.perf_counter()-t0:.0f}s（基础→概念→专名最后——"
          f"C29-01 先入为主）")

    # ---- 对照：混合（现状——全部混在一起） ----
    w_mix = HubLake(chars, all_hubs)
    mixed = base_corpus + proper
    t1 = time.perf_counter()
    for day in range(5):
        w_mix.learn_epoch_batch(mixed, B=128)
    print(f"混合训练完成——{time.perf_counter()-t1:.0f}s（对照——现状）")

    # ---- exp1：苹果判断（水果 vs 公司污染） ----
    print("\n[exp1] '苹果'河道——编排 vs 混合（C29-01 先入为主——水果奠基）:")
    for name, w in [("编排", w_cur), ("混合", w_mix)]:
        ans = w.answer("很", "苹果")
        print(f"      [{name}] K[很]['苹果'] → {[(a, f'{v:.2f}') for a, v in ans[:3]]}"
              if ans else f"      [{name}] K[很]['苹果'] → 无")
    print("      期望：编排 → 甜/脆（水果语义——奠基）——混合 → 电/脑（公司污染）")

    # ---- exp2：稳定语义保持（编排后苹果仍水果） ----
    print("\n[exp2] 专名后置后——'苹果'基础语义稳定:")
    ans = w_cur.answer("很", "苹果")
    clean = [(a, v) for a, v in ans[:4]]
    print(f"      K[很]['苹果'] → {[(a, f'{v:.2f}') for a, v in clean]}")
    ans2 = w_cur.answer("吃", "苹果")
    if ans2:
        print(f"      K[吃]['苹果'] → {[(a, f'{v:.2f}') for a, v in ans2[:3]]}")
    # 专名块是否单独成立（"苹果电脑"成块——不抢占"苹果"）
    in_blocks = "苹果电脑" in blocks
    print(f"      '苹果电脑' 词块: {'[成块——独立指称]' if in_blocks else '[未成块]'}")

    # ---- exp3：专名本身（最后学习——指称可检索） ----
    print("\n[exp3] 专名河道（最后训练后——指称关系）:")
    for hub, obj in [("是", "苹果电脑"), ("是", "北京"), ("是", "北京大学")]:
        ans = w_cur.answer(hub, obj)
        if ans:
            print(f"      K['{hub}']['{obj}'] → {[(a, f'{v:.2f}') for a, v in ans[:3]]}")
        else:
            print(f"      K['{hub}']['{obj}'] → （无关联）")
    print("\n[done] stage84 curriculum proper nouns last")


if __name__ == "__main__":
    run()
