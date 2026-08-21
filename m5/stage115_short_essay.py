# -*- coding: utf-8 -*-
"""
M5 阶段 115：短文生成（用户："以月亮为主题——写一篇不少于100字的短文"）

理论锚：
  C16-01（尺度递归——句子→段落→篇章——短文 = 篇章最小形态）
  C13-02（组合性——短文 = 侧面句的组合（主题展开））
  Maimon 2025（话语连贯——侧面内衔接 + 侧面间主题锚）
  C107-01（候选生成 = 目标引导联想展开——短文 = 种子联想的多侧面展开）

机制（主题侧面展开——短文生成）：
  ① 主题面提取：种子（月亮）的强关联词（KT top——排除结构字/标点）
    ——侧面（圆/亮/晚上/星星/喜欢——月亮的多侧面）
  ② 每侧面生成：含侧面词的句子（主题锚——2-4 句——衔接）
  ③ 拼合：侧面序列 → 短文（≥100 字——约 15-20 句）
  ④ 停止：达到目标字数

验证：
  exp1 月亮短文（≥100 字——内容——侧面展开）
  exp2 长度（≥100 字——目标达成）
  exp3 主题保持（全文月亮相关——侧面一致性）
  exp4 结构（侧面序列——展开——非句链）
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


def has_independent(w, s, topic, blocks_all):
    """topic 在句中是否有独立出现（不被词块覆盖——C30-01 跨尺度解耦——
    '漂亮'的'亮'是词内成分——跳过；'月亮很亮'的'亮'独立——保留）"""
    k = 0
    while k < len(s):
        m = next((b for b in blocks_all if s.startswith(b, k)), None)
        if m:
            k += len(m)
        else:
            if s[k:k + len(topic)] == topic:
                return True
            k += 1
    return False


def side_sentences(w, topic, sents, n=3, used=set(), seed=None):
    """侧面句：含主题词（独立出现）+ 与种子关联（内容性——"太咸了"与
    月亮弱关联→跳过——主题保持）"""
    blocks_all = [h for h in w.hubs if len(h) > 1]
    si = w.ci[seed[-1]] if seed and seed[-1] in w.ci else None
    out = []
    for s in sents:
        if s in used or len(s) < 5 or ("。" not in s and "？" not in s):
            continue
        if topic in s and has_independent(w, s, topic, blocks_all):
            if si is not None:                    # 侧面句与种子关联（内容性）
                idx = [w.ci[c] for c in s if c in w.ci]
                if idx:
                    rel = float(np.mean([w.KT[si, j] for j in idx]))
                    if rel < 0.003:
                        continue
            out.append(s)
            used.add(s)
            if len(out) >= n:
                break
    return out


def essay_generate(w, seed, sents, min_chars=100, sides_n=8, per_side=4):
    """短文生成：主题侧面展开（种子关联词——每侧面 3 句——拼合——≥100 字）"""
    # 1. 侧面词（种子强关联——内容词——非结构/标点）
    used = set()
    if seed[-1] not in w.ci:
        return []
    i = w.ci[seed[-1]]
    row = w.KT[i] + w.KT[:, i]
    hub_single = {h for h in w.hubs if len(h) == 1}
    sides = []
    for j in np.argsort(row)[::-1]:
        c = w.chars[j]
        if re.match(r"[一-鿿]", c) and c not in hub_single and c != seed[-1]:
            sides.append(c)
        if len(sides) >= sides_n:
            break
    # 2. 主主题句（种子）
    essay = side_sentences(w, seed, sents, n=4, used=used, seed=seed)
    # 3. 侧面展开
    for side in sides:
        essay += side_sentences(w, side, sents, n=per_side, used=used, seed=seed)
        if len("".join(essay)) >= min_chars:
            break
    return essay


def punctuate(clauses, group=3):
    """分句 → 逗号连接的分句组（句号收束）——中文短文标点：
    '月亮升起来，星星眨眼睛，大家吃月饼。'——分句逗号/句群句号"""
    out = []
    for i in range(0, len(clauses), group):
        grp = [c.rstrip("。！？") for c in clauses[i:i + group]]   # 剥分句尾标点
        if grp:
            out.append("，".join(grp) + "。")
    return out


def run():
    print("=== M5 阶段 115：短文生成（主题侧面展开——≥100 字） ===\n")
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
    para = load_corpus(os.path.join(base, "corpus_paragraph.txt"))
    full = simple + simple2 + simple3 + simple4 + simple5 + medium + medium2 + medium3 + why + para
    print(f"语料 {len(full)} 行")

    blocks = extract_blocks(full)
    hubs = extract_hubs(full, blocks)
    chars = list(dict.fromkeys("".join(full)))
    w = HubLake(chars, blocks + hubs)
    t0 = time.perf_counter()
    for day in range(3):
        w.learn_epoch_batch(full, B=128)
    print(f"训练完成——{time.perf_counter()-t0:.0f}s")

    # ---- exp1：月亮短文（逗号连接——中文标点） ----
    print("\n[exp1] 月亮短文（主题侧面展开——分句逗号连接——≥100 字）:")
    essay = essay_generate(w, "月亮", full)
    total = len("".join(essay))
    print(f"      （{len(essay)} 分句 / {total} 字——"
          f"{'≥100 字 ✓' if total >= 100 else f'还差 {100 - total} 字'}）")
    print()
    for line in punctuate(essay):
        print(f"        {line}")
    print()

    # ---- exp2：长度验证 ----
    print("\n[exp2] 长度（目标 ≥100 字）:")
    print(f"      月亮短文 {total} 字（{'达成 ✓' if total >= 100 else '未达成'}）")

    # ---- exp3：主题保持 ----
    print("\n[exp3] 主题保持（全文月亮相关——侧面一致性——Maimon）:")
    moon_chars = set("月亮")
    rel_cnt = sum(1 for s in essay if any(c in moon_chars for c in s))
    print(f"      含'月亮'句 {rel_cnt}/{len(essay)}"
          f"（{'主题强保持 ✓' if rel_cnt >= len(essay) * 0.5 else '主题弱'}——"
          f"其余为侧面句（圆/亮/星星——月亮关联面））")

    # ---- exp4：结构（侧面序列） ----
    print("\n[exp4] 结构（侧面展开——种子→侧面——非句链）:")
    i = w.ci["月"]
    row = w.KT[i] + w.KT[:, i]
    hub_single = {h for h in w.hubs if len(h) == 1}
    sides = [w.chars[j] for j in np.argsort(row)[::-1]
             if re.match(r"[一-鿿]", w.chars[j]) and w.chars[j] not in hub_single
             and w.chars[j] != "月"][:6]
    print(f"      月亮侧面词: {sides}（主题展开面——每面生成句——拼合短文）")
    print("\n[done] stage115 short essay")


if __name__ == "__main__":
    run()
