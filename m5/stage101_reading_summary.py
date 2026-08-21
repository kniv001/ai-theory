# -*- coding: utf-8 -*-
"""
M5 阶段 101：读后概括（C43-01 理解=重构——读完长句 → 主题提取 → 骨架重建）

阅读能力输出面（stage99 流式读 + stage100 回视 → 本 stage 概括）：
  ① 流式阅读（含误差驱动回视——stage100）→ 读后状态 z（整合）
  ② 主题提取：读后 z × 非结构字 → 主题字（句首主题——"农业"）
  ③ 骨架重建（C43-01 重构——压缩）：主题 → 焦点区（stage97）→
     读后状态中与主题强关联的字按链排列（"农业属于产业"——骨架）
  对照：长句原文 vs 概括（信息压缩率 + 主题保持）

验证：
  exp1 读后主题提取（长句 → 主题字）
  exp2 骨架重建概括（主题 → 概括句——压缩）
  exp3 多长句概括（农业/食物/软件——不同主题）
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
                                       HubLake, AMP_IN, DT)
from stage100_reading_review import read_flow

MAX_SUMMARY = 8       # 概括最大字数（压缩）


def summarize(w, sent, decay=0.85, max_len=MAX_SUMMARY):
    """读后概括（v3——词块级提取——非字链重建——C43-01 重构提取面）：
    流式读（回视）→ 主题块 → 原文中与主题关联最强的词块（按阅读顺序）
    ——"农业属于产业"（主谓宾骨架——压缩——句来填字：骨架=词块级）"""
    z, g, _ = read_flow(w, sent, decay=decay, err_driven=True)
    n = w.n
    # 1. 主题块：句首 2 字（阅读起点——"农业"/"太阳"）——排除结构字
    #   （单字枢纽——"是"类读后激活高但非内容）——选读后激活高的
    hub_single = {h for h in w.hubs if len(h) == 1}
    head = [c for c in sent[:4] if c in w.ci and c not in hub_single][:2]
    if not head:
        return sent[:max_len], "?"
    topic = max(head, key=lambda c: z[w.ci[c]])
    ti = w.ci[topic]
    # 2. 词块提取：原文中命中的词块（词块=词汇单元——"属于/产业/作物…"）
    blocks_in = []
    i = 0
    while i < len(sent):
        m = next((h for h in w.hubs if len(h) > 1 and sent.startswith(h, i)), None)
        if m:
            blocks_in.append(m)
            i += len(m)
        else:
            i += 1
    # 3. 关联分：每块与主题的 KT 关联 × 块内读后激活（读懂了 → 核心块）
    scored = []
    for h in blocks_in:
        if topic in h:
            continue
        idx = [w.ci[c] for c in h if c in w.ci]
        if not idx:
            continue
        rel = np.mean([w.KT[ti, j] + w.KT[j, ti] for j in idx])
        act = np.mean([z[j] for j in idx])
        scored.append((rel * act, h, sent.index(h)))
    scored.sort(key=lambda x: -x[0])
    # 4. 概括 = 主题 + 关联最强的 2 块（按阅读顺序排列）
    top2 = sorted(scored[:2], key=lambda x: x[2])
    out = topic + "".join(h for _, h, _ in top2)
    return out[:max_len], topic


def run():
    print("=== M5 阶段 101：读后概括（C43-01 理解=重构——主题提取+骨架重建） ===\n")
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
    full = simple + simple2 + simple3 + simple4 + simple5 + medium + medium2 + medium3 + why + wiki + attr + neg + social + isa_sents
    print(f"全语料 {len(full)} 行")

    blocks = extract_blocks(full)
    hubs = extract_hubs(full, blocks)
    chars = list(dict.fromkeys("".join(full)))
    w = HubLake(chars, blocks + hubs)
    t0 = time.perf_counter()
    for day in range(4):
        w.learn_epoch_batch(full, B=128)
    print(f"训练完成——{time.perf_counter()-t0:.0f}s")

    # ---- exp1/2/3：读后概括 ----
    print("\n[读后概括] 流式阅读（回视）→ 主题提取 → 骨架重建（C43-01 压缩）:")
    long_sents = [
        "农业属于第一级产业，包括作物种植、畜牧、渔业养殖、林业等活动，负责主副食和经济作物供应",
        "软件是计算机系统的组成部分，包括操作系统、应用程序和工具软件，用户通过软件完成各种任务",
        "水是生命之源，生物体内的各种化学反应都离不开水，人体大约由百分之七十的水构成",
        "太阳是太阳系的中心，地球围绕太阳运行，太阳为地球提供光和热，使生命得以存在",
    ]
    for s in long_sents:
        summ, topic = summarize(w, s)
        ratio = len(summ) / len(s)
        print(f"      原文({len(s)}字): {s[:44]}…")
        print(f"      概括({len(summ)}字——主题'{topic}'): {summ}  [压缩 {ratio:.0%}]")
        print()
    print("[done] stage101 reading summary")


if __name__ == "__main__":
    run()
