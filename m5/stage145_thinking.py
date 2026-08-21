# -*- coding: utf-8 -*-
"""
M5 阶段 145：思考（用户："目前框架有决策环了，应该支持思考了吧"——
思考 = 决策环在回答内部运行——C109-01 元决策——候选展开/价值评估/
选择/沉积——无直接答案的问题走思考）

理论锚：
  C109-01（思考 = 决策环的输出（元决策）——思考的启动/继续/停止 =
    决策环实例——候选：继续/停止/转向）
  C115-01（决策环五层：输入/数据层候选/服务层价值/行为层选择/
    反馈层沉积）
  C104-01（候选生成 = g 引导联想链递归展开——排序×剪枝×预算）
  C105-01（截止 = 内部成本交叉——思考深度受限）
  C28-02（联想链可审计——思考痕迹）

机制（think = 决策环在回答内部）：
  ① 输入层：问题注入（对象/类型解析——同 chat）
  ② 数据层：候选生成（g 引导联想展开——obj → KT 关联词 → 含词句——
    多步递归——排序×剪枝（价值低剪））
  ③ 服务层：价值评估（候选与问题类型关联——hub 匹配——rel 排序）
  ④ 行为层：选择（最高价值候选——或组合多句推理链）
  ⑤ 反馈层：沉积（选择强化——决策环闭环）
  截止：深度上限（C105——展开成本交叉——思考停止）
  思考痕迹：展开路径记录（审计——C28-02）

验证：
  exp1 思考启动（无直接答案 → 进入思考——决策环）
  exp2 联想展开（候选生成——多步——路径）
  exp3 选择（价值评估——最佳候选/组合）
  exp4 思考痕迹（审计——路径可读）
  exp5 对比（直接检索 vs 思考——无答案问解决）
"""
import os
import re
import sys
import time
from collections import Counter
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("gb"):
    sys.stdout.reconfigure(encoding="utf-8")

from lake_app import TextLakeApp


class ThinkingLake(TextLakeApp):
    """文字湖 + 思考（决策环在回答内部——C109-01）"""

    def learn_bridge(self):
        """跨句桥（stage117——问答对——问句字→答句枢纽——稀疏沉积——
        '为什么带伞？'+'因为下雨。'→K[因为][为→因]——hub 涌现正确
        （'为什么'→'因为'——非'认为'））"""
        marks = self.q_only_words()
        for i in range(1, len(self.sents)):
            A, B = self.sents[i - 1], self.sents[i]
            if not any(m in A for m in marks):
                continue
            hubs_b = [h for h in self.w.hubs if h in B]
            if not hubs_b:
                continue
            shared = set(A) & set(B)
            a_diff = [c for c in A if c not in shared and c in self.w.ci]
            for h in hubs_b:
                rep = h[0] if len(h) > 1 else h
                if rep not in self.w.ci:
                    continue
                for a in a_diff:
                    if a in self.w.ci and a != rep:
                        v = float(self.w.K[h][self.w.ci[a], self.w.ci[rep]])
                        if v < 0.02:
                            self.w.K[h][self.w.ci[a], self.w.ci[rep]] += 0.02
        self.w.KT_sp = sum(self.w.K.values())
        if self.w.use_gpu:
            import torch
            self.w.KT = torch.tensor(self.w.KT_sp.toarray(),
                                     device="cuda", dtype=torch.float32)
        print("[think] 跨句桥学习（问答对→答句枢纽——hub 涌现正确化）")

    def parse(self, q):
        """问题解析（对象/类型——同 chat 管线）"""
        qc = q.replace("？", "").replace("?", "")
        rest = qc
        for m in sorted(self.q_only_words(), key=len, reverse=True):
            rest = rest.replace(m, "")
        for m in ["是", "你", "我", "他", "她", "我们", "你们", "他们"]:
            rest = rest.replace(m, "")
        cs = [c for c in rest if c in self.w.ci]
        obj = cs[-2] + cs[-1] if len(cs) >= 2 else (cs[-1] if cs else None)
        hub = self.hub_emerge(q, obj=obj)
        return obj, hub

    def think(self, q, max_depth=3, budget=8):
        """思考 = 决策环（C115-01 五层——C109-01 元决策）：
        无直接答案 → 联想展开 → 评估 → 选择 → 沉积
        返回 (答案, 思考痕迹)"""
        trace = []                            # 思考痕迹（审计——C28-02）
        # ① 输入层：问题注入
        obj, hub = self.parse(q)
        if not obj or obj[-1] not in self.w.ci:
            return "我不明白。", trace
        trace.append(f"①输入: 对象={obj} 类型={hub}")
        i0 = self.w.ci[obj[-1]]

        # 直接检索（先试——有答案不思考）
        direct = self.chat(q)
        if direct != "我不明白。":
            trace.append(f"直接检索: {direct}（无需思考）")
            return direct, trace
        trace.append("直接检索空 → 启动思考（决策环）")

        # ② 数据层：候选生成（g 引导联想展开——多步递归）
        #    第 1 步：对象关联词（**基础语料共现**——含 obj 的句中词——
        #    相关性强——非 wiki 全量 KT（'衣服'→服务员/装饰——泛））
        co = Counter()
        for s in self.sents:
            if "？" in s:                    # 排除问句（问句标记'为'
                continue                     # 与对象同句——人为共现污染）
            if obj in s:
                for c in s:
                    if c in self.w.ci and re.match(r"[一-鿿]", c) \
                            and c not in obj:
                        co[c] += 1
        cand_words = [c for c, _ in co.most_common(budget)]
        trace.append(f"②展开: 对象关联词 {cand_words[:5]}…")
        #    候选句池（含关联词的陈述句——剪枝：与对象基础共现——
        #    rel = 句中字与对象的共现数——排除'的'万能字污染——
        #    非 KT_sp（wiki 泛化））
        cands = []
        for wd in cand_words:
            for s in self.sents:
                if wd not in s or "？" in s or len(s) < 4:
                    continue
                rel = sum(co[c] for c in s)               # 共现次数加权
                if rel >= 3:                              # 至少 3 次共现
                    cands.append((rel, s, wd))
        # ③ 服务层：价值评估（hub 类型匹配加分——C13-01 关系类型）
        scored = []
        for rel, s, wd in cands:
            v = rel
            if hub and hub in s:
                v *= 2.0                      # 类型匹配（为什么→因为）
            scored.append((v, s, wd))
        scored.sort(key=lambda x: -x[0])
        trace.append(f"③评估: {len(scored)} 候选——top: "
                     f"{[s[:12] for _, s, _ in scored[:3]]}…")
        if not scored:
            trace.append("④选择: 无候选 → 停止（诚实——C109 停止）")
            return "我不明白。", trace

        # ④ 行为层：选择（最高价值——组合推理链——多句——去重）
        chosen = []
        for v, s, wd in scored:
            if s not in [x[1] for x in chosen]:
                chosen.append((v, s, wd))
            if len(chosen) >= 3:
                break
        ans = "。".join(s for _, s, _ in chosen) + "。"
        trace.append(f"④选择: {[s[:12] for _, s, _ in chosen]}…")
        # ⑤ 反馈层：沉积（选择强化——决策环闭环——C5-01）
        good = [s for _, s, _ in chosen]
        self.correct([], good, eps_good=0.05)
        trace.append("⑤沉积: 选择强化（闭环）")
        return ans, trace


def run():
    print("=== M5 阶段 145：思考（决策环在回答内部——C109-01 元决策）"
          " ===\n")
    app = ThinkingLake()
    base = os.path.dirname(__file__)
    app.load_corpus(
        os.path.join(base, "corpus_simple_natural.txt"),
        os.path.join(base, "corpus_simple2.txt"),
        os.path.join(base, "corpus_simple3.txt"),
        os.path.join(base, "corpus_simple4.txt"),
        os.path.join(base, "corpus_moon.txt"),
        os.path.join(base, "corpus_complex.txt"),
        os.path.join(base, "corpus_why.txt"),
    )
    app.load_corpus(os.path.join(base, "corpus_wiki_filtered.txt"),
                    group="wiki")
    app.learn_bridge()                       # 跨句桥（hub 涌现正确化）

    # ---- exp1/2/3/4：思考完整流程 ----
    print("\n[exp1-4] 思考完整流程（无直接答案 → 决策环 → 联想展开 → "
          "选择 → 痕迹）:")
    for q in ["为什么要穿衣服？", "为什么要保护环境？", "为什么要吃饭？"]:
        ans, trace = app.think(q)
        print(f"\n  Q: {q}")
        for t in trace:
            print(f"    {t}")
        print(f"  → 思考答案: {ans}")

    # ---- exp5：对比（直接检索 vs 思考） ----
    print("\n[exp5] 对比（直接检索 vs 思考——无答案问解决）:")
    q = "为什么要穿衣服？"
    direct = app.chat(q)
    ans, _ = app.think(q)
    print(f"  '{q}'")
    print(f"  直接检索: {direct}")
    print(f"  思考: {ans}"
          f"（{'思考解决 ✓（直接无答案→思考出推理链——决策环）'
              if direct == '我不明白。' and ans != '我不明白。' else '对比'}——"
          f"C109-01 思考=决策环输出）")
    print("\n[结论] 思考：决策环接入回答内部（C115-01 五层——"
          "输入/候选展开/价值评估/选择/沉积——C109-01 元决策）——"
          "无直接答案问 → 联想展开（g 引导——C104）→ 组合推理链 → "
          "选择强化（闭环）——思考痕迹可审计（C28-02）——框架支持思考")
    print("[done] stage145 thinking")


if __name__ == "__main__":
    run()
