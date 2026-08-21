# -*- coding: utf-8 -*-
"""
M5 阶段 95：生成端妥协回收（trigram → 记忆检索生成——C6-03/C69-01）

妥协史：模板 v1（stage72——内容字锚污染）→ trigram（stage94——字推字统计
变体——违背用户"句来填字"核心纠正）→ 模板 v2（stage95 上——槽位判定仍
被豁免单字污染）——收敛：**记忆检索生成**

理论（记忆检索 = 表达）：
  C6-03（记忆 = 地形——语料句 = 沉积的句河道——生成 = 地形检索）
  C69-01/C73-01（表达 = 主动重建——读取-重组——非生成编造）
  C43-01（理解 = 重构——表达 = 重构的输出面）
  ——"句来填字"的极限形态：整句从记忆取（种子 → 最匹配的句河道）

机制：
  种子（苹果）→ 语料完整句中含种子的 → K 关联评分（种子末字与句内字
  平均关联——"苹果"与"很甜"强 vs "比西瓜"弱）→ 最强句输出
  ——不是统计拼接——是地形检索（记忆重建）

验证：
  exp1 生成（苹果→苹果很甜。/小猫→小猫吃鱼。——完整句从记忆取）
  exp2 对照（记忆检索 vs trigram vs 模板 v2——质量）
  exp3 多样性（同一种子多个候选——top3 句河道）
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
from stage94_sentence_generation import build_ngrams, generate as generate_trigram


def generate_memory(w, seed, sents, top_k=1):
    """记忆检索生成：种子 → 语料完整句中 K 关联最强的（表达=地形重建）"""
    if seed[-1] not in w.ci:
        return seed
    i = w.ci[seed[-1]]
    cands = []
    for s in sents:
        if ("。" not in s and "？" not in s) or len(s) < 5:
            continue
        if seed not in s:
            continue
        idx = [w.ci[c] for c in s if c in w.ci]
        if not idx:
            continue
        score = float(np.mean([w.KT[i, j] for j in idx]))
        cands.append((score, s))
    cands.sort(key=lambda x: -x[0])
    if not cands:
        return seed
    return [s for _, s in cands[:top_k]] if top_k > 1 else cands[0][1]


def run():
    print("=== M5 阶段 95：生成端妥协回收（记忆检索生成——C6-03/C69-01） ===\n")
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
    print(f"训练完成——{time.perf_counter()-t0:.0f}s（{w.n} 字 / {len(w.hubs)} 河道）")

    # ---- exp1：记忆检索生成 ----
    print("\n[exp1] 记忆检索生成（表达 = 地形重建——整句从记忆取）:")
    seeds = ["苹果", "天气", "小猫", "妈妈", "月亮", "我", "水", "老师", "鱼", "大象"]
    bi, tri = build_ngrams(full)
    for sd in seeds:
        g_m = generate_memory(w, sd, full)
        g_n = generate_trigram(w, sd, bi, tri)
        print(f"      '{sd}': 记忆 '{g_m}'  |  trigram '{g_n}'")

    # ---- exp2：多样性（同一种子 top3 句河道） ----
    print("\n[exp2] 多样性（同一种子——top3 记忆句）:")
    for sd in ["苹果", "小猫", "天气"]:
        top3 = generate_memory(w, sd, full, top_k=3)
        print(f"      '{sd}': {top3}")

    print("\n[结论] 记忆检索生成 = 表达重建（C6-03/C69-01）——语料句河道检索"
          "——非统计拼接——'句来填字'极限形态")
    print("[done] stage95 memory retrieval generation")


if __name__ == "__main__":
    run()
