# -*- coding: utf-8 -*-
"""
M5 阶段 94：完整句生成检验（用户："目前能完整说一句话吗"）

机制（收敛——trigram + 衔接——stage77/78 验证过的干净生成）：
  ① trigram 统计（全语料——"苹果很甜"类完整片段）
  ② 生成：种子 → trigram 链（多义区分——stage77）
  ③ 衔接约束（stage78_theme——候选与序列的 K 关联——漂移即停——
     Maimon 话语连贯）
  ④ 完整句 = 语料句检索式生成（非字推字漂移）
质量评估：完整度（≥5 字）/ 自然度（trigram 来自语料）/ 停止（无候选即停）
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


def build_ngrams(sents):
    bi, tri = Counter(), Counter()
    for s in sents:
        for i in range(len(s) - 1):
            bi[s[i:i + 2]] += 1
        for i in range(len(s) - 2):
            tri[s[i:i + 3]] += 1
    return bi, tri


def generate(w, seed, bi, tri, min_coh=0.004, max_len=12):
    """trigram 链 + 衔接约束（漂移即停——stage78_theme）"""
    out = seed
    for _ in range(max_len):
        last2 = out[-2:]
        last1 = out[-1]
        if last1 not in w.ci:
            break
        cands = []
        for (tg, cnt) in tri.items():
            if tg[:2] == last2:
                cands.append((tg[2], cnt))
        if not cands:
            for (bg, cnt) in bi.items():
                if bg[0] == last1:
                    cands.append((bg[1], cnt))
        if not cands:
            break
        best = None
        for cand, cnt in sorted(cands, key=lambda x: -x[1]):
            if cand not in out and cand in w.ci:
                best = cand
                break
        if best is None:
            break
        if len(out) >= 2:
            i = w.ci[best]
            # 主题锚：候选与种子的关联（Maimon 全局 cohesion——跨主题即停）
            if seed[-1] in w.ci:
                coh = w.KT[i, w.ci[seed[-1]]]
                if coh < min_coh:
                    break
            # 句界：候选是句尾（语料中"。"后的字——预测骤降——A0）
            if best in "。！？" or (len(out) >= 4 and best in "了"):
                if best in "。！？":
                    out += best
                    break
        out += best
    return out


def run():
    print("=== M5 阶段 94：完整句生成检验（trigram + 衔接——语料句检索生成） ===\n")
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
    isa_sents = ["苹果是水果", "香蕉是水果", "西瓜是水果", "葡萄是水果",
                 "猫是动物", "狗是动物", "鸟是动物", "鱼是动物",
                 "水是液体", "冰是固体", "雪是白色的", "天空是蓝色的",
                 "老虎是动物", "树是植物", "花是植物", "石头是固体",
                 "苹果可以吃", "水可以喝", "雨是从云落下来的",
                 "小猫吃鱼", "猫吃老鼠", "我吃苹果", "小猫吃月饼"]
    full = simple + simple2 + simple3 + simple4 + simple5 + medium + medium2 + medium3 + why + wiki + attr + neg + social + isa_sents
    print(f"全语料 {len(full)} 行")

    blocks = extract_blocks(full)
    hubs = extract_hubs(full, blocks)
    chars = list(dict.fromkeys("".join(full)))
    w = HubLake(chars, blocks + hubs)
    t0 = time.perf_counter()
    for day in range(4):
        w.learn_epoch_batch(full, B=128)
    bi, tri = build_ngrams(full)
    print(f"训练完成——{time.perf_counter()-t0:.0f}s（{w.n} 字 / {len(w.hubs)} 河道"
          f"——trigram {len(tri)} 个）")

    # ---- 生成检验 ----
    print("\n[生成] trigram 链 + 衔接（完整句——语料句检索生成）:")
    seeds = ["苹果", "天气", "小猫", "妈妈", "水", "月亮", "老师", "我", "鱼"]
    for sd in seeds:
        if sd[-1] not in w.ci:
            print(f"      '{sd}' → （词不在湖）")
            continue
        g = generate(w, sd, bi, tri)
        tag = "完整句" if len(g) >= 5 and g != sd else ("短语" if len(g) > len(sd) else "未生成")
        print(f"      '{sd}' → '{g}'  [{tag}]")

    # ---- 语料对照（期望生成接近语料中的真实句） ----
    print("\n[对照] 语料中的真实完整句（生成的目标——检索而非编造）:")
    for sd in ["苹果", "天气", "小猫", "妈妈", "水"]:
        sents_with = [s for s in full if s.startswith(sd) and len(s) >= 5][:3]
        print(f"      '{sd}' 语料句: {sents_with}")
    print("\n[done] stage94 sentence generation")


if __name__ == "__main__":
    run()
