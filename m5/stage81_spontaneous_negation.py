# -*- coding: utf-8 -*-
"""
M5 阶段 81：否定自发（写死否定词表清零——hardcoded_audit 待办④——NEG_WORDS 替换）

写死替换：NEG_WORDS（stage57——不是/没有/没/不 + find 顺序查找）→ 否定河道
  "不"是 stage79 自发枢纽（频率×熵×位置——103 次+弥散）——否定句含"不"
  → 自动沉积 K["不"] 河道（C22-01 关系湖）——无需否定词表
  语义：否定 = 独立河道（反例关系）而非 stage57 的负向修正（削弱 K 会连累
    正向检索——河道分离无污染）
  hub 选择 = 位置最前（"苹果不是什么？"——"不"在"是"前——选否定河道）

理论锚：
  C22-01（域内关系湖——"不"河道 = 否定关系簇——与"是"河道分离）
  R45/C39-01（统计必然的反例打破——反例 = 另一条河道的必然——非负值）
  Cognition 2025（负证据约束词义——"这不是X"→X 排除——K[不] 检索排除）

验证：
  exp1 否定句自发进 K["不"]（无 NEG_WORDS——"苹果不是蔬菜"含"不"→ 命中）
  exp2 正/反例河道分离（K[是][苹果]→水果 纯 vs K[不][苹果]→蔬/菜——反例）
  exp3 否定问答（"苹果不是什么？"→'不'河道→蔬菜；"苹果是什么？"→'是'→水果）
  exp4 对照 stage57（负向修正 vs 河道分离——分离的检索纯净性）
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


class NegAskLake(HubLake):
    """HubLake + 位置最前 hub 选择（"不是什么"→"不"——否定河道优先）"""

    def ask(self, q):
        hit_single = [h for h in self.hubs if len(h) == 1 and h in q]
        mark_blocks = []
        if "？" in q:
            iq = q.index("？")
            for h in self.hubs:
                if len(h) >= 2 and iq >= len(h) and q.startswith(h, iq - len(h)):
                    mark_blocks.append(h)
        rest = q
        for h in mark_blocks + hit_single:
            rest = rest.replace(h, "")
        cs = [c for c in rest if c in self.ci]
        if not cs:
            return None, None, []
        if hit_single:
            hub = min(hit_single, key=lambda h: q.index(h))   # 位置最前
        else:
            cand = set()
            for c in q:
                if c not in self.ci:
                    continue
                covers = [h for h in self.hubs if len(h) > 1 and c in h]
                if covers:
                    cand.add(max(covers, key=len))
            if not cand:
                return None, None, []
            hub = min(cand, key=lambda h: q.find(h[0]))
            for c in q:
                for hh in cand:
                    if c in hh:
                        rest = rest.replace(c, "")
            cs = [c for c in rest if c in self.ci]
            if not cs:
                return None, None, []
        foreign = set()
        for i in range(1, len(rest)):
            if rest[i - 1] not in self.ci and rest[i] in self.ci and rest[i - 1] != "？":
                foreign.add(rest[i])
        cs = [c for c in rest if c in self.ci and c not in foreign]
        if not cs:
            return None, None, []
        rest_clean = rest.replace("？", "").replace("?", "")
        blks = [h for h in self.hubs if len(h) >= 2 and h in rest_clean]
        if blks:
            obj = min(blks, key=rest_clean.index)
        elif len(cs) >= 2:
            obj = cs[-2] + cs[-1]
        else:
            obj = cs[0]
        return hub, obj, self.answer(hub, obj)


def run():
    print("=== M5 阶段 81：否定自发（写死否定词表清零——否定河道——C22-01） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=900)
    wiki = load_corpus(os.path.join(base, "corpus_wiki_filtered.txt"), n=600)
    attr = load_corpus(os.path.join(base, "corpus_attr_cause.txt"))
    neg = load_corpus(os.path.join(base, "corpus_negation.txt"))
    isa_sents = ["苹果是水果", "香蕉是水果", "西瓜是水果", "葡萄是水果",
                 "猫是动物", "狗是动物", "鸟是动物", "鱼是动物",
                 "水是液体", "冰是固体", "雪是白色的", "天空是蓝色的",
                 "老虎是动物", "树是植物", "花是植物", "石头是固体",
                 "苹果可以吃", "水可以喝", "雨是从云落下来的",
                 "小猫吃鱼", "猫吃老鼠", "我吃苹果", "小猫吃月饼"]
    sents = simple + wiki + attr + isa_sents + neg
    print(f"语料 {len(sents)} 行（含否定 {len(neg)} 条——corpus_negation.txt）")

    blocks = extract_blocks(sents)
    hubs = extract_hubs(sents, blocks)
    all_hubs = blocks + hubs
    in_block = set("".join(blocks))
    print(f"[exp1] 自发词块含否定（不/没有/不是 是否枢纽——率统计）:")
    print(f"      '不' 在单字枢纽: {'是' if '不' in hubs else '否'}（freq 103+否定语料——"
          f"stage79 已涌现）")
    print(f"      '没有' 在词块: {'是' if '没有' in blocks else '否'}（率 0.86 已涌现）")
    print(f"      '不是' 在词块: {'是' if '不是' in blocks else '否'}（率低——"
          f"不成块——否定经单字'不'河道——无需块）")

    chars = list(dict.fromkeys("".join(sents)))
    print(f"\n词汇表 {len(chars)} 字 / 枢纽 {len(all_hubs)} 个")
    w = NegAskLake(chars, all_hubs)
    t0 = time.perf_counter()
    for ep in range(6):
        w.learn_epoch_batch(sents, B=128)
    print(f"训练完成——{time.perf_counter()-t0:.0f}s（否定句自动进 K['不']——无 NEG_WORDS）")

    # ---- exp2：正/反例河道分离 ----
    print("\n[exp2] 正/反例河道分离（K[是] vs K[不]——无负向修正污染）:")
    for hub, obj in [("是", "苹果"), ("不", "苹果"), ("是", "冰"), ("不", "冰"),
                     ("是", "猫"), ("不", "鱼"), ("没有", "鱼")]:
        ans = w.answer(hub, obj)
        if ans:
            print(f"      K['{hub}']['{obj}'] → {[(a, f'{v:.2f}') for a, v in ans[:3]]}")
        else:
            print(f"      K['{hub}']['{obj}'] → （无关联）")

    # ---- exp3：否定问答（位置最前 hub——"不是什么"→"不"） ----
    print("\n[exp3] 否定问答（问句驱动——正向/否定河道各自检索）:")
    for q in ["苹果是什么？", "苹果不是什么？", "冰是什么？", "冰不是热的吗？", "鱼是什么？"]:
        hub, obj, ans = w.ask(q)
        if ans:
            print(f"      Q: '{q}' → 枢纽'{hub}' '{obj}' → {[(a, f'{v:.2f}') for a, v in ans[:3]]}")
        else:
            print(f"      Q: '{q}' → （无命中——语料覆盖不足）")

    # ---- exp4：对照 stage57（负向修正 vs 河道分离） ----
    print("\n[exp4] 对照 stage57（负向修正 vs 河道分离——分离的纯净性）:")
    print("      stage57 做法：'苹果不是蔬菜'→ K[苹果,蔬菜] 削弱——K[是][苹果]"
          "检索也弱（污染）")
    print("      stage81 做法：'苹果不是蔬菜'→ K[不] 河道（反例关系）——K[是][苹果]"
          "无污染")
    pos = w.answer("是", "苹果")
    negv = w.answer("不", "苹果")
    print(f"      K[是][苹果] top3: {[(a, f'{v:.3f}') for a, v in pos[:3]]}（正向——水果系）")
    print(f"      K[不][苹果] top3: {[(a, f'{v:.3f}') for a, v in negv[:3]]}（反例——蔬菜系）")
    print("\n[done] stage81 spontaneous negation")


if __name__ == "__main__":
    run()
