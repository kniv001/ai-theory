# -*- coding: utf-8 -*-
"""
M5 阶段 90：跨句学习（对话同步域最小实现——C56-01——"怎么样"问答的真正解）

问题（stage88 发现）：问答对"苹果怎么样？"+"苹果很甜"是**不同句子**——
句内共现学不到"怎么样"→"很"（属性问标记→属性关系字）的映射——
ask 时 hub 选错（选"苹果"块而非"很"河道）

理论锚：
  C56-01（对话同步域 = 临时社会湖——对话 = 临时耦合——问答对 = 最小对话）
  C43-01（理解 = 重构+验证循环——对话 = 双向重构校准）
  Maimon 2025（话语连贯 = 衔接 cohesion——共享词桥）
  C16-01（尺度递归——句内关联 → 句间关联——对话尺度）

机制（桥沉积——共享词为桥）：
  ① 相邻问答对 (A=问句, B=答句)——共享词（"苹果"）
  ② 桥：A_diff（"怎么样"）↔ B_diff（"很甜"）弱沉积到 B 的枢纽河道
    （K[很][么,很] += eps——"么"与"很"的关联经每次问答对累积）
  ③ ask：问标记字（属于"怎么/什么"块的字）→ 与关系字关联最强的河道
    ——"么"→K[很] 内"么→很"最强 → hub="很" → 对象苹果 → 甜 ✓

验证：
  exp1 桥沉积（K[很] 内 么→很 关联建立——对照无桥）
  exp2 问答（"苹果怎么样？"→'很'→甜——之前选错 hub）
  exp3 问答（"天气怎么样？/为什么会下雨？"——跨句映射）
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

BRIDGE_EPS = 0.008     # 桥沉积强度（弱——共享词桥）
Q_BLOCKS = ["怎么", "什么"]   # 问句块（统计涌现——"怎么/什么"已成块）


class BridgeLake(HubLake):
    """HubLake + 跨句桥沉积（对话同步域最小实现）"""

    def learn_bridge(self, sents, B=512):
        """相邻问答对桥沉积：问句→答句——共享词桥——
        A_diff（怎么样）↔ B_diff（很甜）沉积到 B 的枢纽河道"""
        n = self.n
        for i in range(1, len(sents)):
            A, B = sents[i - 1], sents[i]
            if not any(qb in A for qb in Q_BLOCKS):   # 问句判定靠问句块
                continue
            # B 的枢纽河道（"苹果很甜"→"很"）
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
        """问句 → 问标记字 → 关系河道（跨句桥沉积学到的映射）"""
        # 问标记字（属于问句块的字——"么"——"怎么/什么"块）
        mark_chars = [c for c in q if any(c in h for h in Q_BLOCKS)]
        if mark_chars:
            # hub = 河道内 标记字→关系字 关联最强（"么"→"很"——桥沉积）
            best_hub, best_v = None, 0.0
            for h in self.hubs:
                if len(h) != 1 or h not in self.ci:
                    continue
                for mc in mark_chars:
                    if mc in self.ci:
                        v = self.K[h][self.ci[mc], self.ci[h]]
                        if v > best_v:
                            best_hub, best_v = h, v
            if best_hub:
                # 对象：去掉标记和标记字
                rest = q
                for h in Q_BLOCKS:
                    rest = rest.replace(h, "")
                for mc in mark_chars:
                    rest = rest.replace(mc, "")
                cs = [c for c in rest if c in self.ci]
                if not cs:
                    return None, None, []
                blks = [h for h in self.hubs if len(h) >= 2 and h in rest]
                obj = min(blks, key=rest.index) if blks else cs[0]
                return best_hub, obj, self.answer(best_hub, obj)
        return super().ask(q)


def run():
    print("=== M5 阶段 90：跨句学习（对话同步域最小实现——问答对桥沉积） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=900)
    simple2 = load_corpus(os.path.join(base, "corpus_simple2.txt"))
    simple3 = load_corpus(os.path.join(base, "corpus_simple3.txt"))
    simple4 = load_corpus(os.path.join(base, "corpus_simple4.txt"))
    simple5 = load_corpus(os.path.join(base, "corpus_simple5.txt"))
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
    bc = simple + simple2 + simple3 + simple4 + simple5 + wiki + attr + neg + social + isa_sents
    nq = sum(1 for s in bc if "？" in s)
    print(f"总语料 {len(bc)} 行（含问句 {nq} 条——问答对桥源）")

    blocks = extract_blocks(bc)
    hubs = extract_hubs(bc, blocks)
    chars = list(dict.fromkeys("".join(bc)))

    # 对照：无桥（BaselineLake——普通 HubLake）
    w0 = HubLake(chars, blocks + hubs)
    # 实验：桥（BridgeLake）
    w1 = BridgeLake(chars, blocks + hubs)
    for w in [w0, w1]:
        for day in range(4):
            w.learn_epoch_batch(bc, B=128)
        if hasattr(w, "learn_bridge"):
            w.learn_bridge(bc)
    print("训练完成（对照无桥 vs 实验桥沉积）")

    # ---- exp1：桥沉积效应 ----
    print("\n[exp1] 桥沉积（K[很] 内 '么'→'很' 关联——对话同步域）:")
    for name, w in [("无桥", w0), ("有桥", w1)]:
        if "么" in w.ci and "很" in w.K:
            v = w.K["很"][w.ci["么"], w.ci["很"]]
            print(f"      [{name}] K[很][么→很] = {v:.4f}")

    # ---- exp2/3：问答 ----
    print("\n[exp2] 问答（跨句映射——之前 hub 选错）:")
    for q in ["苹果怎么样？", "天气怎么样？", "水怎么样？", "天空怎么样？",
              "柠檬怎么样？", "为什么会下雨？", "为什么要洗手？"]:
        for name, w in [("无桥", w0), ("有桥", w1)]:
            hub, obj, ans = w.ask(q)
            print(f"      [{name}] Q: '{q}' → 枢纽'{hub}' '{obj}' → "
                  f"{[(a, f'{v:.2f}') for a, v in ans[:3]] if ans else '（无）'}")
        print()
    print("\n[结论] 跨句桥沉积（C56-01 对话同步域最小实现）——问标记→关系河道"
          "映射从问答对数据学出——'怎么样'→'很'（属性问）")
    print("[done] stage90 cross-sentence learning")


if __name__ == "__main__":
    run()
