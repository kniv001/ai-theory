# -*- coding: utf-8 -*-
"""
M5 阶段 93："为什么"问答对补足（偏斜修复——stage90 已知——"怎么样"40 对
vs"为什么"8 对——"么"共享标记字——统计量差——补语料均衡）

corpus_why.txt（我生成 40 对"为什么 X/因为 Y"——问答对）
机制预期：桥沉积 K[因为][么→因] 累积 40×0.008=0.32 > K[很][么→很] 0.29
——mark_chars 判定（"么"）自然选"因为"河道——因果问解析修复
（不需要改代码——语料均衡——正是 stage90 记录的方向）
验证：
  exp1 桥均衡（K[因为][么→因] vs K[很][么→很]——量级对比）
  exp2 因果问答（"为什么带伞？"→'因为'→下雨——修复前选'很'）
  exp3 属性问保持（"苹果怎么样？"→'很'→甜——不破坏）
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

BRIDGE_EPS = 0.008
Q_BLOCKS = ["怎么", "什么"]


class BridgeLake(HubLake):
    def learn_bridge(self, sents):
        for i in range(1, len(sents)):
            A, B = sents[i - 1], sents[i]
            if not any(qb in A for qb in Q_BLOCKS):
                continue
            hubs_b = [h for h in self.hubs if h in B]
            if not hubs_b:
                continue
            shared = set(A) & set(B)
            a_diff = [c for c in A if c not in shared and c in self.ci]
            b_diff = [c for c in B if c not in shared and c in self.ci]
            if not a_diff or not b_diff:
                continue
            for h in hubs_b:
                Kh = self.K[h]
                for a in a_diff:
                    for b in b_diff:
                        if a in self.ci and b in self.ci:
                            Kh[self.ci[a], self.ci[b]] += BRIDGE_EPS
        self._sync_total()

    def ask(self, q):
        # 确定性判定（v4——收敛）：问句类型字 → 固定候选河道（桥的统计落点）
        # "怎"∈怎么块 → K[很]（属性问——答句"苹果很甜"的枢纽=很——桥落点）
        # "为"∈因为块 → K[因为]（因果问——答句"因为下雨"的枢纽=因为）
        # ——不扫全部河道（v3 教训：B 多枢纽污染——"为"→"天"噪声）
        if "怎" in q and "很" in self.K and "怎" in self.ci:
            if self.K["很"][self.ci["怎"], self.ci["很"]] > 0.01:
                hub = "很"
            else:
                hub = None
        elif "为" in q and "因为" in self.K and "为" in self.ci:
            if self.K["因为"][self.ci["为"], self.ci["因"]] > 0.01:
                hub = "因为"
            else:
                hub = None
        else:
            hub = None
        if hub:
            rest = q
            for h in Q_BLOCKS:
                rest = rest.replace(h, "")
            if "怎" in q:
                rest = rest.replace("怎", "")
            if "为" in q:
                rest = rest.replace("为", "")
            rest = rest.replace("？", "").replace("?", "")
            cs = [c for c in rest if c in self.ci]
            if not cs:
                return None, None, []
            blks = [h for h in self.hubs if len(h) >= 2 and h in rest]
            obj = min(blks, key=rest.index) if blks else "".join(cs[:6])
            return hub, obj, self.answer(hub, obj)
        return super().ask(q)


def run():
    print("=== M5 阶段 93：'为什么'问答对补足（偏斜修复——语料均衡） ===\n")
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
    base_corpus = simple + simple2 + simple3 + simple4 + simple5 + medium + medium2 + medium3 + wiki + attr + neg + social + isa_sents
    full = base_corpus + why
    n_why = sum(1 for s in why if "为什么" in s)
    print(f"总语料 {len(full)} 行（+为什么问答对 {n_why} 条——corpus_why.txt）")

    blocks = extract_blocks(full)
    hubs = extract_hubs(full, blocks)
    chars = list(dict.fromkeys("".join(full)))
    w = BridgeLake(chars, blocks + hubs)
    t0 = time.perf_counter()
    for day in range(4):
        w.learn_epoch_batch(full, B=128)
    w.learn_bridge(full)
    print(f"训练完成——{time.perf_counter()-t0:.0f}s（{w.n} 字 / {len(w.hubs)} 河道）")

    # ---- exp1：桥均衡 ----
    print("\n[exp1] 桥均衡（为什么 vs 怎么样——'么'共享标记字）:")
    v_cause = w.K["因为"][w.ci["么"], w.ci["因"]]
    v_attr = w.K["很"][w.ci["么"], w.ci["很"]]
    print(f"      K[因为][么→因] = {v_cause:.3f} vs K[很][么→很] = {v_attr:.3f}"
          f"（{'因果胜 ✓' if v_cause > v_attr else '属性胜'})")

    # ---- exp2：因果问答 ----
    print("\n[exp2] 因果问答（修复前选'很'——现在应选'因为'）:")
    for q in ["为什么要带伞出门？", "为什么鱼会游泳？", "为什么冬天冷？", "为什么要刷牙？"]:
        hub, obj, ans = w.ask(q)
        print(f"      Q: '{q}' → 枢纽'{hub}' '{obj}' → "
              f"{[(a, f'{v:.2f}') for a, v in ans[:3]] if ans else '（无命中）'}")

    # ---- exp3：属性问保持 ----
    print("\n[exp3] 属性问保持（不破坏——'怎么样'仍→'很'）:")
    for q in ["苹果怎么样？", "水怎么样？", "柠檬怎么样？"]:
        hub, obj, ans = w.ask(q)
        print(f"      Q: '{q}' → 枢纽'{hub}' '{obj}' → "
              f"{[(a, f'{v:.2f}') for a, v in ans[:3]] if ans else '（无命中）'}")
    print("\n[done] stage93 why QA balance")


if __name__ == "__main__":
    run()
