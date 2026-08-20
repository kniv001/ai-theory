# -*- coding: utf-8 -*-
"""
M5 阶段 85：基础语料拓展验证（用户："继续拓展基础语料——社会文化的庞大体系
目前的基础语料还是太少了"——基础厚 → 社会湖有依托）

corpus_simple2.txt（我生成 ~450 行——互补主题）：
  天气自然（下雨/月亮/彩虹）/ 身体感觉（头疼/饿了）/ 时间（明天/星期一）/
  方位（上面/旁边）/ 颜色形状（红色/圆形）/ 数量比较（三个/比）/ 家庭扩展
  （外婆/舅舅）/ 交通（火车/飞机）/ 衣物房间（衣服/客厅）/ 所属（我的书）/
  礼貌（谢谢/对不起）/ 健康（跑步/早睡）

验证：
  exp1 新词块涌现（天气/月亮/火车/红色/圆形/明天——相邻共现）
  exp2 新关系检索（K[是][月亮]→圆；K[比][苹果]→小/大；K[很][火车]→快）
  exp3 新词汇覆盖（新字进词汇表——基础词汇量扩展）
  exp4 训练后问答/生成（"月亮"→圆——基础句检索）
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
    print("=== M5 阶段 85：基础语料拓展验证（corpus_simple2——基础厚→社会湖依托） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=900)
    simple2 = load_corpus(os.path.join(base, "corpus_simple2.txt"))
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
    base_corpus = simple + simple2 + wiki + attr + neg + social + isa_sents
    print(f"基础语料 {len(base_corpus)} 行（simple {len(simple)} + simple2 {len(simple2)}"
          f" + wiki {len(wiki)} + attr/neg/social/isa）")

    # ---- exp1：新词块涌现 ----
    blocks = extract_blocks(base_corpus)
    print(f"\n[exp1] 新词块涌现（corpus_simple2 的主题词——相邻共现——非词表）:")
    for w in ["天气", "月亮", "火车", "红色", "圆形", "明天", "衣服", "客厅",
              "谢谢", "跑步", "星期", "比"]:
        print(f"      '{w}' {'[词块✓]' if w in blocks else '[未成块]'}")
    print(f"      词块总数 {len(blocks)}（simple1 时代 116——拓展后 {len(blocks)}）")

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
            w.learn_epoch_batch(proper, B=128)   # 专名最后（stage84 纪律）
    print(f"训练完成——{time.perf_counter()-t0:.0f}s（5 天——专名最后）")
    print("\n[exp2] 新关系检索（拓展语料的关系河道）:")
    for hub, obj in [("是", "月亮"), ("是", "火车"), ("很", "火车"), ("很", "红色"),
                     ("是", "圆"), ("很", "明天"), ("比", "苹果"), ("很", "苹果")]:
        ans = w.answer(hub, obj)
        if ans:
            print(f"      K['{hub}']['{obj}'] → {[(a, f'{v:.2f}') for a, v in ans[:3]]}")
        else:
            print(f"      K['{hub}']['{obj}'] → （无关联）")

    # ---- exp4：问答/生成（基础句） ----
    print("\n[exp3] 问答（基础语料覆盖后的问句驱动）:")
    for q in ["月亮是什么？", "火车怎么样？", "苹果比什么大？", "红色怎么样？"]:
        hub, obj, ans = w.ask(q)
        if ans:
            print(f"      Q: '{q}' → 枢纽'{hub}' '{obj}' → {[(a, f'{v:.2f}') for a, v in ans[:3]]}")
        else:
            print(f"      Q: '{q}' → （无命中）")
    print("\n[done] stage85 simple2 validation")


if __name__ == "__main__":
    run()
