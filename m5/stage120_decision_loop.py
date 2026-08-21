# -*- coding: utf-8 -*-
"""
M5 阶段 120：决策环（S1——行为层——"执行"——A0 五成分最后一块——
C115-01 五层最小实现——一切=选择——A5）

理论锚：
  A5（选择原语——一切行为/认知/表达 = 选择——行为层唯一动作）
  C115-01（决策环五层：①输入层（价值注入）②数据层（树状关系图——
    候选域——联想展开 C107）③服务层（代价/利益竞争——价值排序 C108）
    ④行为层（选择模块——C114）⑤反馈层（沉积-侵蚀——环闭合+学习））
  C104-01（候选生成 = g 引导的检索 + 停止规则）
  C105-01（截止时间 = 内部成本交叉——候选耗尽/价值收敛）

机制（五层最小实现——"选水果"场景）：
  ① 输入：目标注入（"水果"——价值/需求）
  ② 数据层：候选生成（目标联想展开——K 检索——候选集——剪枝预算）
  ③ 服务层：价值评估（候选与正价值词（甜/好/喜欢）的关联——A4——
    排序）
  ④ 行为层：选择（价值竞争——最优——C114 选择模块）
  ⑤ 反馈层：选择 → 沉积强化（K 更新——学习闭环——环闭合）

验证：
  exp1 候选生成（目标水果 → 苹果/香蕉/西瓜——联想展开）
  exp2 价值评估（候选价值排序——A4）
  exp3 选择输出（最优——行为层）
  exp4 反馈沉积（选择强化——学习闭环）
  exp5 截止（候选耗尽/价值收敛——停止）
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
                                       HubLake, EPS_K)

POS_VALUE = "甜好喜欢好吃香"      # 正价值词（候选价值的锚——A4 正端）


class DecisionLoop:
    """决策环（C115-01 五层——一切=选择——A5）"""

    def __init__(self, w, sents):
        self.w = w
        self.sents = sents
        self.history = []              # 决策史（反馈层沉积）

    def generate_cands(self, goal, budget=5):
        """② 数据层：候选生成（目标联想展开——选项词块——C107——
        决策的候选 = 词块选项（买什么→苹果/西瓜）——排除属性词/结构字）"""
        if goal[-1] not in self.w.ci:
            return []
        i = self.w.ci[goal[-1]]
        row = self.w.KT[i] + self.w.KT[:, i]
        cands = []
        # 词块选项（len>1——与目标关联——苹果/西瓜/香蕉）
        blk_scores = []
        for h in self.w.hubs:
            if len(h) <= 1 or h in POS_VALUE:
                continue
            if all(c in self.w.ci for c in h):
                v = sum(self.w.KT[i, self.w.ci[c]] for c in h)
                blk_scores.append((v, h))
        blk_scores.sort(key=lambda x: -x[0])
        cands = [h for _, h in blk_scores[:budget]]
        return cands

    def value_of(self, cand, goal):
        """③ 服务层：价值评估（选项与正价值词的关联——A4 正端——排序）"""
        v = 0.0
        for c in cand:
            if c not in self.w.ci:
                continue
            i = self.w.ci[c]
            for p in POS_VALUE:                      # 正价值词关联（甜/好）
                if p in self.w.ci:
                    v += self.w.KT[i, self.w.ci[p]] + self.w.KT[self.w.ci[p], i]
        return v

    def decide(self, goal, budget=5):
        """决策环（五层）——选择最优选项（行为层）——反馈沉积（K 河道）"""
        cands = self.generate_cands(goal, budget)
        if not cands:
            return None, []
        values = [(self.value_of(c, goal), c) for c in cands]
        values.sort(key=lambda x: -x[0])
        choice = values[0][1]
        # ⑤ 反馈：选择强化（K 河道——非 KT（会被 sync 覆盖）——学习闭环）
        for h in self.w.hubs:
            if goal[-1] in h and len(h) > 1:         # 含目标字的词块河道
                for c in choice:
                    if c in self.w.ci:
                        self.w.K[h][self.w.ci[goal[-1]], self.w.ci[c]] += EPS_K * 0.5
                break
        self.w._sync_total()
        self.history.append((goal, choice))
        return choice, values


def run():
    print("=== M5 阶段 120：决策环（S1——五层——一切=选择——A5） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=900)
    simple2 = load_corpus(os.path.join(base, "corpus_simple2.txt"))
    simple4 = load_corpus(os.path.join(base, "corpus_simple4.txt"))
    medium = load_corpus(os.path.join(base, "corpus_medium.txt"))
    para = load_corpus(os.path.join(base, "corpus_paragraph.txt"))
    full = simple + simple2 + simple4 + medium + para
    print(f"语料 {len(full)} 行")

    blocks = extract_blocks(full)
    hubs = extract_hubs(full, blocks)
    chars = list(dict.fromkeys("".join(full)))
    w = HubLake(chars, blocks + hubs)
    for day in range(3):
        w.learn_epoch_batch(full, B=128)
    print(f"训练完成（{w.n} 字 / {len(w.hubs)} 河道）")

    d = DecisionLoop(w, full)

    # ---- exp1：候选生成 ----
    print("\n[exp1] 候选生成（目标'水果'→ 联想展开——数据层②——C107）:")
    cands = d.generate_cands("水果", budget=5)
    for c in cands:
        print(f"      候选: {c}")

    # ---- exp2/3：价值评估 + 选择 ----
    print("\n[exp2/3] 价值评估 + 选择（服务层③ + 行为层④——A4）:")
    choice, values = d.decide("水果")
    print(f"      候选价值排序:")
    for v, c in values[:5]:
        mark = " ◀选择" if c == choice else ""
        print(f"        {v:.3f}  {c}{mark}")
    print(f"      选择: {choice}（行为层——最优——C114）")

    # ---- exp4：反馈沉积（学习闭环） ----
    print("\n[exp4] 反馈沉积（选择强化——K 更新——环闭合⑤——学习）:")
    if "水" in w.ci and "果" in w.ci:
        before = w.KT[w.ci["水"], w.ci["果"]]
    else:
        before = 0
    d.decide("水果")                     # 再次决策（强化）
    after = w.KT[w.ci["水"], w.ci["果"]] if "水" in w.ci and "果" in w.ci else 0
    print(f"      K[水→果]: {before:.4f} → {after:.4f}"
          f"（{'沉积强化 ✓（选择反馈）' if after > before else '未强化'}——"
          f"环闭合——选择→沉积→下次决策）")

    # ---- exp5：截止（候选耗尽/收敛） ----
    print("\n[exp5] 截止（候选耗尽 → 停止——C105 内部成本交叉）:")
    choice, values = d.decide("月亮")
    if choice:
        print(f"      '月亮' → 选择 '{choice}'（候选 {len(values)} 个——"
              f"价值收敛 → 决策完成——C105 截止涌现）")
    else:
        print(f"      '月亮' → 无候选（候选耗尽——停止——C104 停止规则）")
    print("\n[done] stage120 decision loop")


if __name__ == "__main__":
    run()
