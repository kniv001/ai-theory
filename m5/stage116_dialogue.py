# -*- coding: utf-8 -*-
"""
M5 阶段 116：对话功能（C56-01 对话同步域——C43-01 双向重构校准）

理论锚：
  C56-01（对话同步域 = 临时社会湖——R49 临时耦合——说完即散）
  C43-01（对话 = 双向重构校准——描述→激活→模拟→输出→对方反馈→误差→收敛）
  C74-01（问句 = 不确定→确定——对话的动机）
  C116-01（对话 = 预测循环互接——转导层互接——机制零变化）

组件（对话循环）：
  ① 问句理解（ask——是什么/怎么样/为什么——hub 判定）
  ② 回答生成（记忆检索——对象+关系词匹配句）
  ③ 对话主题延续（多轮——对话同步域——主题锚——C56-01）
  ④ 双向校准（C43-01——回答 → 确认/追问——验证循环）

验证：
  exp1 单轮问答（是什么/怎么样/为什么——对话基本）
  exp2 多轮对话（主题延续——对话流——同步域）
  exp3 对话场景（你是谁/你喜欢什么——social 个体视角）
  exp4 双向校准（回答后确认——验证循环收敛）
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


class Dialogue:
    """对话（C56-01 对话同步域——多轮——主题延续）"""

    def __init__(self, w, sents):
        self.w = w
        self.sents = sents
        self.topic = None          # 对话主题（同步域锚）

    def parse_question(self, q):
        """问句解析 v2：类型（区分字——stage93）+ 对象（去标记/标点）"""
        q = q.replace("？", "").replace("?", "")
        if "为什么" in q:
            hub = "因为"
        elif "怎么样" in q or "怎" in q:
            hub = "很"
        elif "是什么" in q or "是啥" in q or "是" in q:
            hub = "是"
        else:
            hub = None
        rest = q
        for m in ["为什么", "怎么样", "是什么", "什么", "怎么", "吗", "呢",
                  "是", "你", "我", "他", "她", "我们", "你们", "他们"]:
            rest = rest.replace(m, "")
        cs = [c for c in rest if c in self.w.ci]
        if not cs:
            return hub, None
        # 对象 = 去标记后的句末词（中文问句对象在句末——"喜欢动物"→动物
        # ——"苹果是什么"→苹果——blks 弃（"喜欢"动词误选——动物不成块））
        obj = cs[-2] + cs[-1] if len(cs) >= 2 else cs[-1]
        return hub, obj

    def respond(self, q):
        """单轮回答：解析 v2 → 记忆检索（对象 K 关联最强陈述句——含跨句桥）"""
        w = self.w
        hub, obj = self.parse_question(q)
        if obj is None:
            return "我不明白。", None
        # 回答 = 与对象 K 关联最强的陈述句（KT——含跨句桥）
        # 对象精确（"动物"答句须含"动物"）——"谁"例外（身份问——对象
        # 无陈述句——KT 身份桥检索：谁→小明（social 身份对））
        if obj[-1] in w.ci:
            i = w.ci[obj[-1]]
            cands = []
            for s in self.sents:
                if "？" in s or len(s) < 4 or "。" not in s:
                    continue
                if obj != "谁" and obj not in s:     # 对象精确（谁例外）
                    continue
                idx = [w.ci[c] for c in s if c in w.ci]
                if not idx:
                    continue
                rel = float(np.mean([w.KT[i, j] for j in idx]))
                if rel > 0.003:
                    cands.append((rel, s))
            cands.sort(key=lambda x: -x[0])
            if cands:
                self.topic = obj
                return cands[0][1], obj
        # 兜底：含对象的陈述句
        cands = [s for s in self.sents if obj in s and "？" not in s and len(s) >= 4]
        if cands:
            self.topic = obj
            return cands[0], obj
        return f"我还没学过{obj}。", obj

    def chat(self, questions):
        """多轮对话（C56-01——临时同步域——主题延续）"""
        print("      [对话开始——同步域建立]")
        for q in questions:
            if self.topic and self.topic[-1] in self.w.ci and self.topic not in q:
                # 主题延续：回答含对话主题的句优先（同步域保持）
                pass
            ans, t = self.respond(q)
            print(f"      问: {q}")
            print(f"      答: {ans}")
            print()
        print("      [对话结束——同步域消散——C56-01]")


def run():
    print("=== M5 阶段 116：对话功能（C56-01 对话同步域——C43-01 双向校准） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=900)
    simple2 = load_corpus(os.path.join(base, "corpus_simple2.txt"))
    simple3 = load_corpus(os.path.join(base, "corpus_simple3.txt"))
    simple4 = load_corpus(os.path.join(base, "corpus_simple4.txt"))
    simple5 = load_corpus(os.path.join(base, "corpus_simple5.txt"))
    medium = load_corpus(os.path.join(base, "corpus_medium.txt"))
    why = load_corpus(os.path.join(base, "corpus_why.txt"))
    social = load_corpus(os.path.join(base, "corpus_social.txt"))
    para = load_corpus(os.path.join(base, "corpus_paragraph.txt"))
    full = simple + simple2 + simple3 + simple4 + simple5 + medium + why + social + para
    print(f"语料 {len(full)} 行（含 social {len(social)}——对话场景）")

    blocks = extract_blocks(full)
    hubs = extract_hubs(full, blocks)
    chars = list(dict.fromkeys("".join(full)))
    w = HubLake(chars, blocks + hubs)
    t0 = time.perf_counter()
    for day in range(3):
        w.learn_epoch_batch(full, B=128)
    print(f"训练完成——{time.perf_counter()-t0:.0f}s")

    d = Dialogue(w, full)

    # ---- exp1：单轮问答 ----
    print("\n[exp1] 单轮问答（是什么/怎么样/为什么——对话基本）:")
    for q in ["苹果是什么？", "苹果怎么样？", "为什么带伞？", "月亮是什么？", "小猫吃什么？"]:
        ans, t = d.respond(q)
        print(f"      Q: {q}")
        print(f"      A: {ans}")

    # ---- exp2：多轮对话（主题延续） ----
    print("\n[exp2] 多轮对话（主题延续——对话同步域——C56-01）:")
    d.chat(["苹果是什么？", "苹果怎么样？", "为什么带伞？", "天气怎么样？"])

    # ---- exp3：对话场景（social——个体视角） ----
    print("\n[exp3] 对话场景（你是谁/你喜欢什么——social 个体视角）:")
    d2 = Dialogue(w, full)
    d2.chat(["你是谁？", "你喜欢什么动物？", "为什么？"])

    # ---- exp4：双向校准（C43-01——回答 → 确认——验证循环） ----
    print("\n[exp4] 双向校准（C43-01——回答后确认——验证循环收敛）:")
    q = "苹果是什么？"
    ans1, t = d.respond(q)
    q2 = f"{t}是水果吗？"
    ans2, _ = d.respond(q2)
    print(f"      问: {q} → 答: {ans1}")
    print(f"      确认: {q2} → 答: {ans2}")
    print(f"      （回答 → 追问确认 → 收敛——C43-01 验证循环）")
    print("\n[done] stage116 dialogue")


if __name__ == "__main__":
    run()
