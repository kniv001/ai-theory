# -*- coding: utf-8 -*-
"""
M5 阶段 80：动作类自发（写死动词表清零——hardcoded_audit 待办③——ACT_VERBS 替换）

写死替换：ACT_VERBS（stage40/41/42/43——吃喝看听说写读…动词表）→ 位置分布涌现
  动词的统计特征（C15-02 词类=分布聚类）：
    ① 句中位偏置：动词在谓语位（"猫吃鱼"——吃在句中）——均值≈0.5
    ② 位置集中：谓语位集中（std 小）
    ③ 上下文双向开放：前接主语类/后接宾语类（两侧多样性）
  ——全部纯统计——无动词词表

理论锚：
  C15-02（词类=分布聚类涌现——动词类从位置分布生长——非词典）
  C22-01（域内关系湖——动作字 → 动作河道——K[吃] = 动作关系）
  C16-01（尺度递归——"学习"块动词位 vs "科学"块名词位——字级混合由块解决）

验证：
  exp1 动词性分数（位置×上下文——对照 ACT_VERBS 命中率——涌现 vs 写死表）
  exp2 动作河道（K[吃][猫]→鱼；K[吃][苹果]→可吃？——动作关系自发）
  exp3 动作问答（"猫吃什么？"→'吃'河道→鱼——与 stage79"为什么"对称）
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
                                       HubLake)

# 写死的 ACT_VERBS（仅作对照——验证涌现命中率——不是运行依赖）
ACT_VERBS = set("吃喝看听说写读走跑玩学做买卖拿放开关洗穿唱画打种浇扫拖叠铺搬递借还教帮陪带坐骑背跳踢投端尝")


# ---------- exp1：动词性分数（位置分布——C15-02） ----------

def verb_score(sents, min_freq=5):
    """动词性 = 句中位偏置 × 位置集中 × 上下文熵（纯位置特征——去频率——
    词类判定与频率无关——C15-02 分布特征）
    中位偏置：均值≈0.5（谓语位——"猫吃鱼"吃在句中——vs 名词主语位 0-0.3）
    位置集中：std 小（谓语位固定）
    上下文熵：前接主语类/后接宾语类（双向开放）"""
    freq = Counter()
    pos = defaultdict(list)
    ctx = defaultdict(lambda: (set(), set()))
    for s in sents:
        L = len(s)
        for i, c in enumerate(s):
            if not re.match(r"[一-鿿]", c):
                continue
            freq[c] += 1
            pos[c].append(i / L)
            if i > 0:
                ctx[c][0].add(s[i - 1])
            if i < L - 1:
                ctx[c][1].add(s[i + 1])
    score = {}
    for c, n in freq.items():
        if n < min_freq:
            continue
        p = np.array(pos[c])
        mid_bias = max(0.0, 1.0 - abs(p.mean() - 0.5) / 0.3)   # 均值→0.5 → 1
        ctx_div = np.log2(len(ctx[c][0]) + 1) + np.log2(len(ctx[c][1]) + 1)
        score[c] = ctx_div * mid_bias / (p.std() + 0.05)
    return sorted(score.items(), key=lambda kv: -kv[1])


def run():
    print("=== M5 阶段 80：动作类自发（写死动词表清零——位置分布——C15-02） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=900)
    wiki = load_corpus(os.path.join(base, "corpus_wiki_filtered.txt"), n=600)
    attr = load_corpus(os.path.join(base, "corpus_attr_cause.txt"))
    isa_sents = ["苹果是水果", "香蕉是水果", "西瓜是水果", "葡萄是水果",
                 "猫是动物", "狗是动物", "鸟是动物", "鱼是动物",
                 "水是液体", "冰是固体", "雪是白色的", "天空是蓝色的",
                 "老虎是动物", "树是植物", "花是植物", "石头是固体",
                 "苹果可以吃", "水可以喝", "雨是从云落下来的",
                 "小猫吃鱼", "猫吃老鼠", "我吃苹果", "小猫吃月饼"]
    sents = simple + wiki + attr + isa_sents

    # ---- exp1：动词性分数（位置分布涌现——对照 ACT_VERBS） ----
    vs = verb_score(sents)
    print(f"[exp1] 动词性分数 top 25（纯位置特征——中位偏置×集中×上下文——无动词表）:")
    top25 = [c for c, _ in vs[:25]]
    print(f"      {' '.join(top25)}")
    hit = sum(1 for c in top25 if c in ACT_VERBS)
    print(f"      ACT_VERBS 对照命中：{hit}/25")
    print("      位置特征对比（动词 vs 名词——C15-02 分布区分）:")
    for c in ["吃", "喝", "开", "学", "看", "写", "做", "猫", "苹", "甜", "水", "果"]:
        r = next((s for v, s in vs if v == c), None)
        print(f"      '{c}' 动词性={r:.1f}" if r else f"      '{c}' 未上榜")

    # ---- exp2：动作河道（HubLake——扩展枢纽含动作字） ----
    blocks = extract_blocks(sents)
    hubs = extract_hubs(sents, blocks)
    # 动作字并入枢纽（位置统计涌现的动词位字——不写死：动词性 top 且非块内）
    in_block = set("".join(blocks))
    verb_hubs = [c for c, _ in vs[:12] if c not in in_block and c not in hubs]
    all_hubs = blocks + hubs + verb_hubs
    chars = list(dict.fromkeys("".join(sents)))
    print(f"\n词汇表 {len(chars)} 字 / 枢纽 {len(all_hubs)} 个（词块 {len(blocks)}"
          f" + 单字 {len(hubs)} + 动词位 {len(verb_hubs)}:{''.join(verb_hubs)}）")
    w = HubLake(chars, all_hubs)
    t0 = time.perf_counter()
    for ep in range(6):
        w.learn_epoch_batch(sents, B=128)
    print(f"训练完成——{time.perf_counter()-t0:.0f}s")
    print("\n[exp2] 动作河道（K[动作字][对象] 检索——无 ACT_VERBS——动词位涌现）:")
    for hub, obj in [("吃", "猫"), ("吃", "苹果"), ("喝", "水"), ("看", "书"),
                     ("吃", "老鼠"), ("写", "字")]:
        ans = w.answer(hub, obj)
        if ans:
            print(f"      K['{hub}']['{obj}'] → {[(a, f'{v:.2f}') for a, v in ans[:3]]}")
        else:
            print(f"      K['{hub}']['{obj}'] → （无关联）")

    # ---- exp3：动作问答（"猫吃什么？"——'吃'河道——与"为什么"对称） ----
    print("\n[exp3] 动作问答（问句驱动 → 动作字河道 → 对象检索）:")
    for q in ["猫吃什么？", "苹果可以吃吗？", "小猫吃什么？"]:
        hub, obj, ans = w.ask(q)
        if ans:
            print(f"      Q: '{q}' → 枢纽'{hub}' '{obj}' → {[(a, f'{v:.2f}') for a, v in ans[:3]]}")
        else:
            print(f"      Q: '{q}' → （无命中——语料覆盖不足）")
    print("\n[done] stage80 spontaneous actions")


if __name__ == "__main__":
    run()
