# -*- coding: utf-8 -*-
"""
M5 阶段 117：类型词涌现（用户："类型词硬编码——转变为框架训练涌现"——
stage116 的 parse if-else 是写死映射——hardcoded_audit 新条目）

理论锚：
  C15-01（语法 = 时序模板——统计自发生成——非先天规则库）
  C122-01（规则 = 关系统计——类型词映射 = 训练产物）
  C13-01（意义 = 预测关系集——问句类型 ↔ 关系河道——从问答对学出）
  stage90（跨句桥——问答对沉积——问句标记字 ↔ 答句类型词）

机制（hub 涌现——非写死）：
  ① 跨句桥学习（stage90——问答对：问句字 ↔ 答句枢纽——"苹果怎么样？"
    +"苹果很甜" → 桥沉积 K[很][怎→很]）
  ② hub 判定：
    a. 直接命中：问句中的单字枢纽（"是"in"苹果是什么"——"是"是涌现
      枢纽）→ hub="是"
    b. 桥检索（无直接）：问句字 → 与各枢纽的跨句桥关联 argmax
      （"怎"→K[很][怎→很] 桥——"怎么样"→"很"——数据学出非写死）
  ③ 回答类型筛选（hub in 答句——"为"答句含"为"（因为）——工作）

验证：
  exp1 hub 涌现（是什么→是/怎么样→很/为什么→为/喜欢→喜欢——桥/直接）
  exp2 对话（涌现 hub 下全部答对——与 stage116 对照）
  exp3 桥来源（问答对语料——问句字↔类型词的桥强度）
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

BRIDGE_EPS = 0.008


class BridgeHubLake(HubLake):
    """HubLake + 跨句桥（stage90——问答对——问句字↔答句枢纽）"""

    def __init__(self, chars, hubs, sents=None):
        super().__init__(chars, hubs)
        self.sents = sents or []

    def q_only_words(self):
        """问句专属词（涌现——只在问句出现的词/字——什么/怎么/吗/为什么
        ——C15-01 统计——替代写死问句标记列表）"""
        cached = getattr(self, "_q_only_words", None)
        if cached is not None:
            return cached
        q_lines = [s for s in self.sents if "？" in s]
        st_lines = [s for s in self.sents if "？" not in s]
        q_text = "".join(q_lines)
        st_text = "".join(st_lines)
        q_chars = set(q_text) - set(st_text)          # 问句专属字
        words = set()
        for w in q_chars:
            words.add(w)
        for s in q_lines:                              # 问句专属词（块级）
            for i in range(len(s) - 1):
                w2 = s[i:i + 2]
                if w2 in q_text and w2 not in st_text:
                    words.add(w2)
        self._q_only_words = words
        return words

    def learn_bridge(self, sents, B=512):
        """问答对桥沉积：问句（含问句块）→ 答句——A_diff↔B_diff 沉积到
        B 的枢纽河道——"苹果怎么样？"+"苹果很甜" → K[很][怎→很]"""
        q_marks = [w for w in self.q_only_words() if len(w) >= 1]
        for i in range(1, len(sents)):
            A, B = sents[i - 1], sents[i]
            if not any(m in A for m in q_marks):
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

    def hub_emerge(self, q):
        """hub 涌现（非写死）：
        a. 直接命中：问句中的单字枢纽（"是"in"苹果是什么"）——排除问句
          专属字（怎/什/么——只在问句出现——非类型词——"怎"豁免进枢纽
          但非类型）——直接优先（真类型词明确）
        b. 桥检索：问句字 → 各枢纽（含词块）的跨句桥 argmax（怎→很/
          么→喜欢块）"""
        q_chars = [c for c in q if c in self.ci]
        if not q_chars:
            return None
        q_only = getattr(self, "_q_only", None)
        if q_only is None:
            q_lines = "".join(s for s in self.sents if "？" in s)
            st_lines = "".join(s for s in self.sents if "？" not in s)
            self._q_only = set(q_lines) - set(st_lines)   # 问句专属字
            q_only = self._q_only
        direct = [h for h in self.hubs if len(h) == 1 and h in q
                  and h not in q_only]
        if direct:                               # 直接命中（真类型词）优先
            best, best_v = None, -1.0
            for h in direct:
                v = sum(self.K[h][self.ci[c], self.ci[h]]
                        for c in q_chars if c in self.ci)
                if v > best_v:
                    best, best_v = h, v
            return best
        # b. 桥检索：问句字 → 各枢纽（含词块——"喜欢"块）的桥关联——argmax
        scores = {}
        for h in self.hubs:
            if h not in self.ci:                 # 词块用首字（"喜欢"→"喜"）
                continue
            rep = h[0] if len(h) > 1 else h
            if rep not in self.ci:
                continue
            v = sum(self.K[h][self.ci[c], self.ci[rep]]
                    for c in q_chars if c in self.ci)
            if v > 0.001:
                scores[h] = v
        if scores:
            return max(scores, key=scores.get)
        return None


class DialogueEmerge:
    """对话（涌现 hub——C56-01 同步域）"""

    def __init__(self, w, sents):
        self.w = w
        self.sents = sents

    def parse_question(self, q):
        """解析（对象提取保持——hub 涌现替代 if-else）"""
        q = q.replace("？", "").replace("?", "")
        hub = self.w.hub_emerge(q)               # 涌现（非写死）
        rest = q
        # 问句专属词（涌现——什么/怎么/吗……）——替代写死列表
        for m in sorted(self.w.q_only_words(), key=len, reverse=True):
            rest = rest.replace(m, "")
        # 人称（封闭语法类——妥协记录——将来词类绑定涌现）
        for m in ["是", "你", "我", "他", "她", "我们", "你们", "他们"]:
            rest = rest.replace(m, "")
        cs = [c for c in rest if c in self.w.ci]
        if not cs:
            return hub, None
        obj = cs[-2] + cs[-1] if len(cs) >= 2 else cs[-1]
        return hub, obj

    def respond(self, q):
        w = self.w
        hub, obj = self.parse_question(q)
        if obj is None:
            return "我不明白。", None
        # 人称转换（对话惯例——你↔我/他→他/她→她——封闭语法类——
        # 妥协记录：KT 人称桥被句内共现污染（他比你高——你-他句内 vs
        # 你-我身份桥）——词类绑定完备后从身份对学）
        p_conv = None
        for p in q:
            if p in "你我":
                p_conv = "我" if p == "你" else "你"
                break
            if p in "他她":
                p_conv = p
                break
        if obj[-1] in w.ci:
            i = w.ci[obj[-1]]
            cands = []
            for s in self.sents:
                if "？" in s or len(s) < 4 or "。" not in s:
                    continue
                if obj != "谁" and obj not in s:
                    continue                             # 对象精确（谁例外——
                                                         # 待身份机制 stage118）
                if hub and hub not in s:         # 类型匹配（涌现 hub）
                    continue
                if p_conv and p_conv not in s:   # 人称一致（你→我）
                    continue
                idx = [w.ci[c] for c in s if c in w.ci]
                if not idx:
                    continue
                rel = float(np.mean([w.KT[i, j] for j in idx]))
                if rel > 0.003:
                    cands.append((rel, s))
            cands.sort(key=lambda x: -x[0])
            if cands:
                return cands[0][1], obj
        cands = [s for s in self.sents if obj in s and "？" not in s
                 and len(s) >= 4 and (hub is None or hub in s)]
        if cands:
            return cands[0], obj
        return f"我还没学过{obj}。", obj


def run():
    print("=== M5 阶段 117：类型词涌现（hub 涌现——非写死——C15-01/C122-01） ===\n")
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
    print(f"语料 {len(full)} 行")

    blocks = extract_blocks(full)
    hubs = extract_hubs(full, blocks)
    chars = list(dict.fromkeys("".join(full)))
    w = BridgeHubLake(chars, blocks + hubs, sents=full)
    for day in range(3):
        w.learn_epoch_batch(full, B=128)
    w.learn_bridge(full)
    print(f"训练完成（{w.n} 字 / {len(w.hubs)} 河道——含跨句桥）")

    # ---- exp1：hub 涌现 ----
    print("\n[exp1] hub 涌现（直接命中 + 桥 argmax——非写死）:")
    for q in ["苹果是什么？", "苹果怎么样？", "为什么带伞？", "你喜欢什么动物？"]:
        hub = w.hub_emerge(q)
        print(f"      '{q}' → hub='{hub}'"
              f"（{'直接命中' if hub and hub in q.replace('？','') else '桥检索'}）")

    # ---- exp2：对话（涌现 hub——全部答对） ----
    print("\n[exp2] 对话（涌现 hub——类型匹配——与 stage116 对照）:")
    d = DialogueEmerge(w, full)
    for q in ["苹果是什么？", "苹果怎么样？", "为什么带伞？", "你是谁？",
              "你喜欢什么动物？", "月亮是什么？"]:
        ans, _ = d.respond(q)
        print(f"      问: {q}")
        print(f"      答: {ans}")

    # ---- exp3：桥来源（问句字↔类型词——问答对学出） ----
    print("\n[exp3] 桥来源（问答对——问句字↔类型词——跨句桥强度）:")
    for qc, tc in []:
        if qc in w.ci and tc in w.ci:
            v = w.K["很" if tc == "很" else ("因为" if tc == "因" else "喜欢")].get
            pass
    for h, qc, tc in [("很", "么", "很"), ("因为", "为", "因"), ("喜欢", "么", "喜")]:
        if h in w.K and qc in w.ci and tc in w.ci:
            v = w.K[h][w.ci[qc], w.ci[tc]]
            print(f"      K['{h}'][{qc}→{tc}] = {v:.3f}（问答对桥——涌现）")
    print("\n[done] stage117 hub emergence")


if __name__ == "__main__":
    run()
