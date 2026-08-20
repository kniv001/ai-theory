# -*- coding: utf-8 -*-
"""
M5 阶段 83：社会湖（用户："需要有社会湖——网络需要理解后来人造词汇的关系"——
纠正：网络流行语禁止入语库（散乱无关联——C22-01 无关系密度——悬空符号）——
社会湖语义输入 = 个体视角社会关系语料——稳定关系——可学）

用户方向（2026-08-21）：
  ① 社会关系（家庭/学校/朋友——人际语言）
  ② 个体视角（你是谁——身份/角色——"我是小明/我是学生"）
  ③ 专有名词拓展（为什么会有——命名/指称——"名字是爸爸妈妈起的"）
  ④ 基础社会词汇（交流/说话的概念——"说话是交流的方式"）

理论锚：
  C56-01（社会湖三层——内社会湖=身份湖网络+社会价值河道+社会行为模板——
    个体视角 = 内社会湖的第一层）
  C26-01/C17-01（专有名词 = 文化转导符号——命名 = 社会约定——C62-01 同型：
    名字的意义 = 符号预测关系）
  C13-01/C122-01（社会词汇意义 = 社会语料关系统计——稳定共现——可学）
  C22-01（社会语料关系密度高——子湖可形成——vs 网络流行语散乱）

验证：
  exp1 身份河道（"我是小明"——K[我] 与 小/学/家 关联）
  exp2 关系河道（"妈妈爱我"——K[爱] 与 妈/家 关联；"朋友帮我"）
  exp3 专名河道（"小明"→同学/名字——专名 = 指称关系集）
  exp4 社会词汇（"说"→话/礼貌；"交流"→说话）
  exp5 问答（"你是谁？"→我；"小明是谁？"→同学）
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
    print("=== M5 阶段 83：社会湖（个体视角社会关系——身份/专名/社会词汇——C56-01） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=900)
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
    sents = simple + wiki + attr + neg + isa_sents + social
    print(f"语料 {len(sents)} 行（含社会语料 {len(social)} 条——corpus_social.txt）")

    # 社会词汇抽查（应成块/进枢纽——统计涌现）
    blocks = extract_blocks(sents)
    hubs = extract_hubs(sents, blocks)
    all_hubs = blocks + hubs
    print(f"\n[exp0] 社会词块涌现（相邻共现——非词表）:")
    for w in ["小明", "小红", "妈妈", "爸爸", "说话", "交流", "朋友", "老师"]:
        print(f"      '{w}' {'[词块✓]' if w in blocks else '[未成块]'}")
    print(f"      单字枢纽: {''.join(hubs)}")

    chars = list(dict.fromkeys("".join(sents)))
    print(f"\n词汇表 {len(chars)} 字 / 枢纽 {len(all_hubs)} 个")
    w = HubLake(chars, all_hubs)
    t0 = time.perf_counter()
    for ep in range(6):
        w.learn_epoch_batch(sents, B=128)
    print(f"训练完成——{time.perf_counter()-t0:.0f}s")

    # ---- exp1：身份河道（个体视角——你是谁） ----
    print("\n[exp1] 身份河道（个体视角——'我是小明/我是学生'）:")
    for hub, obj in [("是", "我"), ("是", "你"), ("是", "他"), ("是", "她")]:
        ans = w.answer(hub, obj)
        if ans:
            print(f"      K['{hub}']['{obj}'] → {[(a, f'{v:.2f}') for a, v in ans[:3]]}")

    # ---- exp2：关系河道（社会关系——家庭/朋友） ----
    print("\n[exp2] 关系河道（社会关系——'妈妈爱我/朋友帮我'）:")
    for hub, obj in [("爱", "妈"), ("爱", "家"), ("帮", "同"), ("帮", "朋"),
                     ("叫", "妈"), ("爱", "我")]:
        ans = w.answer(hub, obj)
        if ans:
            print(f"      K['{hub}']['{obj}'] → {[(a, f'{v:.2f}') for a, v in ans[:3]]}")
        else:
            print(f"      K['{hub}']['{obj}'] → （无关联——字不在枢纽/语料覆盖）")

    # ---- exp3：专名河道（为什么会有——命名/指称） ----
    print("\n[exp3] 专名河道（'小明/北京'——指称关系集——C17-01/C26-01）:")
    for hub, obj in [("是", "小明"), ("是", "小红"), ("是", "北京"), ("是", "名字"),
                     ("叫", "小明")]:
        ans = w.answer(hub, obj)
        if ans:
            print(f"      K['{hub}']['{obj}'] → {[(a, f'{v:.2f}') for a, v in ans[:3]]}")

    # ---- exp4：社会词汇（交流/说话的概念） ----
    print("\n[exp4] 社会词汇（'说话/交流/礼貌'——社会功能基础词）:")
    for hub, obj in [("是", "说话"), ("是", "交流"), ("说", "话"), ("说", "谢"),
                     ("是", "礼貌"), ("谢", "说")]:
        ans = w.answer(hub, obj)
        if ans:
            print(f"      K['{hub}']['{obj}'] → {[(a, f'{v:.2f}') for a, v in ans[:3]]}")
        else:
            print(f"      K['{hub}']['{obj}'] → （无关联）")

    # ---- exp5：问答（个体视角） ----
    print("\n[exp5] 问答（个体视角——'你是谁/小明是谁'——问句驱动枢纽检索）:")
    for q in ["你是谁？", "小明是谁？", "你是谁的朋友？"]:
        hub, obj, ans = w.ask(q)
        if ans:
            print(f"      Q: '{q}' → 枢纽'{hub}' '{obj}' → {[(a, f'{v:.2f}') for a, v in ans[:3]]}")
        else:
            print(f"      Q: '{q}' → （无命中——语料覆盖不足）")
    print("\n[done] stage83 social lake")


if __name__ == "__main__":
    run()
