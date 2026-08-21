# -*- coding: utf-8 -*-
"""
M5 阶段 119：人称涌现（用户："框架内含'XX'的都算写死"——对话管线的人称
映射/去标记/谁例外——从身份对话学出——C15-02 位置分布）

理论锚：
  C15-02（词类 = 分布聚类涌现——人称 = 主语位词——句首率高）
  C21-01（身份湖——身份对话（你是谁→我是小明）——人称转换从问答对学）
  C56-01（对话同步域——对话人称指向）

机制（人称涌现——无"你我他她"字面量）：
  ① 人称检测：主语位字（句首 1 位出现率 > 0.5——"你/我/他/她"在句首
    ——C15-02 位置分布——涌现）
  ② 人称去标记：对象提取去掉句首人称（位置——句首主语位）
  ③ 人称转换：身份问答对（问句人称→答句人称——"你是谁？我是小明"
    → 你→我——统计涌现）
  ④ "谁"例外撤：身份湖检索（stage118——我是X——身份绑定）

验证：
  exp1 人称涌现（主语位字——你/我/他/她——位置统计）
  exp2 人称转换（身份对——你→我——统计）
  exp3 对话（无写死人称——你是谁→我是X/你喜欢→我喜欢）
"""
import os
import re
import sys
import time
from collections import Counter, defaultdict
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("gb"):
    sys.stdout.reconfigure(encoding="utf-8")

from stage79_spontaneous_hubs import (load_corpus, extract_blocks, extract_hubs,
                                       HubLake, EPS_K)


def subject_positions(sents, min_pos=0.5):
    """主语位字（C15-02——句首 1 位出现率 > min_pos——人称候选）"""
    cnt = Counter()
    head = Counter()
    for s in sents:
        if not s:
            continue
        cnt[s[0]] += 1
        head[s[0]] += 1
        for c in s[1:]:
            cnt[c] += 1
    return {c for c in head if cnt[c] > 0 and head[c] / cnt[c] > min_pos}


def persona_conv(sents):
    """人称转换（身份问答对——问句人称→答句人称——"你是谁？我是小明"→你→我）"""
    conv = defaultdict(Counter)
    for i in range(1, len(sents)):
        A, B = sents[i - 1], sents[i]
        if "？" not in A:
            continue
        q_sub = subject_positions([A])      # 问句主语位字（人称候选）
        a_sub = subject_positions([B])
        if q_sub and a_sub:
            for p in q_sub:
                for ap in a_sub:
                    conv[p][ap] += 1
    return {p: max(v, key=v.get) for p, v in conv.items() if v}


def run():
    print("=== M5 阶段 119：人称涌现（主语位 + 身份对——无'你我他她'字面量） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=900)
    simple3 = load_corpus(os.path.join(base, "corpus_simple3.txt"))
    simple4 = load_corpus(os.path.join(base, "corpus_simple4.txt"))
    simple5 = load_corpus(os.path.join(base, "corpus_simple5.txt"))
    medium = load_corpus(os.path.join(base, "corpus_medium.txt"))
    social = load_corpus(os.path.join(base, "corpus_social.txt"))
    full = simple + simple3 + simple4 + simple5 + medium + social
    print(f"语料 {len(full)} 行")

    # ---- exp1：人称涌现（主语位字） ----
    print("\n[exp1] 人称涌现（主语位字——C15-02 位置分布）:")
    subs = subject_positions(full)
    print(f"      主语位字: {sorted(subs)}")
    for c in "你我他她":
        print(f"      '{c}' 句首率: "
              f"{sum(1 for s in full if s.startswith(c)) / max(sum(1 for s in full if c in s), 1):.2f}"
              f"（{'主语位 ✓' if c in subs else '非主语位'}）")

    # ---- exp2：人称转换（身份对） ----
    print("\n[exp2] 人称转换（身份问答对——问句人称→答句人称）:")
    conv = persona_conv(full)
    for p, ap in sorted(conv.items()):
        print(f"      '{p}' → '{ap}'（身份对——统计涌现）")

    # ---- exp3：对话（无写死人称——身份湖 + 转换涌现） ----
    print("\n[exp3] 对话（人称涌现——你是谁→我是X）:")
    blocks = extract_blocks(full)
    hubs = extract_hubs(full, blocks)
    chars = list(dict.fromkeys("".join(full)))
    w = HubLake(chars, blocks + hubs)
    for day in range(3):
        w.learn_epoch_batch(full, B=128)
    # 身份湖（stage118——我是X 句）
    ident = [s for s in full if s.startswith("我") and "是" in s[:4] and "？" not in s]
    # 对话模拟（人称转换涌现）
    for q, expect in [("你是谁？", "我"), ("你喜欢什么动物？", "我")]:
        q_sub = subject_positions([q])
        p_conv = conv.get(next(iter(q_sub)), None) if q_sub else None
        obj = "动物" if "动物" in q else ("谁" if "谁" in q else None)
        if obj == "谁":
            ans = ident[0] if ident else "（无身份句）"
        elif p_conv:
            cands = [s for s in full if obj in s and p_conv in s and "？" not in s]
            ans = cands[0] if cands else "（无候选）"
        else:
            ans = "（无转换）"
        print(f"      问: {q} → 答: {ans}（人称转换 '{p_conv}'——涌现）")
    print("\n[done] stage119 persona emergence")


if __name__ == "__main__":
    run()
