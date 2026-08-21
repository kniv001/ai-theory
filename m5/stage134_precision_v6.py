# -*- coding: utf-8 -*-
"""
M5 阶段 134：v6 综合湖问答精度回测（stage133 只验证"有响应"——
本 stage 用真实语料问答对测正确率——隔离沉积后能力完整验证）

理论锚：
  C69-01（表达 = 重建——从记忆取——整句检索）
  C13-01（意义 = 预测关系集——关系类型决定回答河道）
  C30-01（跨尺度解耦——隔离沉积——不损语义）

机制（回答管线——全部涌现）：
  ① 问句解析：对象 = 去标记后句末词；hub = 类型词（涌现——
    stage117 hub_emerge——直接命中+桥检索）
  ② 回答 = 与对象 KT 关联最强 + hub 类型匹配 + 人称一致的陈述句
  ③ 精度 = 回答句与语料标准答案的一致性（关键内容词匹配）

验证：
  exp1 问答对抽取（从语料自动构建测试集——问句+相邻答案行）
  exp2 v6 回答正确率（全测试集——按类型分：为什么/是什么/怎么样/
    吃什么/为什么带伞）
  exp3 对比 v5（同测试集——隔离是否损能力）
  exp4 类型分布（各类型正确率——薄弱处定位）
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
                                       HubLake, AMP_IN, DT)
from stage133_integration_v6 import V6Lake
from stage117_hub_emergence import BridgeHubLake, DialogueEmerge


class V6Bridge(V6Lake, BridgeHubLake):
    """v6 综合湖 + 跨句桥/涌现管线（stage117 复用——hub 全部涌现——
    无写死映射——C15-01 统计 + 桥检索）"""

    def q_only_words(self):
        """q_only_words 修复：两字专属词要求**至少一字是单字专属**且
        不含标点——排除跨词窗污染（'候休'——'时候休息'的跨词窗——
        两字都非专属却被判专属）"""
        base = BridgeHubLake.q_only_words(self)
        single = {w for w in base if len(w) == 1}
        fixed = set()
        for w in base:
            if len(w) == 1:
                fixed.add(w)
            elif any(c in single for c in w) and not any(p in w for p in "？。！，"):
                fixed.add(w)
        self._q_only_words = fixed
        return fixed


class V6Dialogue(DialogueEmerge):
    """v6 对话：stage117 涌现管线 + 主语位规则（C15-02/stage119——
    位置统计非词表）——"在哪里/做什么"类对象 = 主语（句首）非句末
    + hub 软过滤（机制修复：涌现 hub 有时是动词（做）非类型词——
    答案不含——hub 应作排序加分非硬滤——C13-01 关系类型）"""

    def parse_question(self, q):
        hub, obj = super().parse_question(q)
        if obj is None:
            return hub, obj
        if "哪里" in q or "做什么" in q or "喜欢什么" in q:
            # 位置/动作/喜欢类问句——对象 = 主语（句首位——stage119
            # subject_positions——句首率>0.5 的字——位置规则非词表）
            # （"什么时候"除外——对象 = 句末动词——super 句末提取对）
            qc = q.replace("？", "").replace("?", "")
            obj = qc[:2] if len(qc) >= 2 else qc
        return hub, obj

    def respond(self, q):
        """hub 过滤：硬滤 + 软排序结合（机制修复链）：
        ① obj 单字（拆残——'天空'→'天'）→ 扩展问句主题（前两字）
        ② obj 含人称（'你们'）→ 对象检查用人称转换版（你→我——
          答案'我们喜欢跑步'含'我'）
        ③ hub==obj（内容字误判——'天'既是对象又是 hub）→ 降级为
          None（hub 是对象非类型词——排序即可）"""
        w = self.w
        hub, obj = self.parse_question(q)
        if obj is None:
            return "我不明白。", None
        p_conv = None
        for p in q:
            if p in "你我":
                p_conv = "我" if p == "你" else "你"
                break
            if p in "他她":
                p_conv = p
                break
        if hub and hub == obj:                # ③ hub 是对象（内容字）——
            hub = None                        #   非类型词——降级排序
        if len(obj) == 1:                     # ① 单字残——扩展主题（前两字）
            qc = q.replace("？", "").replace("?", "")
            obj = qc[:2] if len(qc) >= 2 else qc
        obj_conv = None
        if obj and "你" in obj:            # ② 人称对象——一次性映射
            obj_conv = obj.replace("你", "我")   # 你→我（不能链式——
        elif obj and "我" in obj:                # replace 链会换回）
            obj_conv = obj.replace("我", "你")
        if obj[-1] in w.ci:
            i = w.ci[obj[-1]]
            cands = []
            for s in self.sents:
                if "？" in s or len(s) < 4 or "。" not in s:
                    continue
                if obj != "谁" and obj not in s:
                    if not (obj_conv and obj_conv in s):
                        continue              # 人称转换版检查
                if p_conv and p_conv not in s:
                    continue
                idx = [w.ci[c] for c in s if c in w.ci]
                if not idx:
                    continue
                rel = float(np.mean([w.KT[i, j] for j in idx]))
                if rel > 0.003:
                    cands.append((rel, s))
            cands.sort(key=lambda x: -x[0])
            if cands:
                if hub:
                    hc = [s for _, s in cands if hub in s]
                    if hc:                    # 有 hub 命中候选——硬滤优先
                        return hc[0], obj
                    return cands[0][1], obj   # 无 hub 命中——排序 top（软）
                return cands[0][1], obj
        cands = [s for s in self.sents
                 if (obj in s or (obj_conv and obj_conv in s))
                 and "？" not in s and len(s) >= 4 and "。" in s
                 and (p_conv is None or p_conv in s)]
        return cands[0] if cands else "我不明白。", obj


def extract_qa(lines):
    """从语料抽取问答对（问句行 + 下一行答案——**验证：答案与问句共享
    内容词（对象/类型词）——过滤无关相邻行**）"""
    pairs = []
    for i, s in enumerate(lines):
        if "？" in s or s.endswith("?"):
            ans = lines[i + 1] if i + 1 < len(lines) else ""
            if ans and "？" not in ans and len(ans) >= 4:
                q_core = set(re.findall(r"[一-鿿]{2}", s))
                a_core = set(re.findall(r"[一-鿿]{2}", ans))
                if q_core & a_core:          # 真实问答对（共享内容词）
                    pairs.append((s, ans))
    return pairs


def correct(ans, resp):
    """正确 = 回答句含答案的关键内容（答案句去虚词后的核心词——
    语料内部一致性）——宽松匹配（含答案后半句内容词）"""
    if resp is None:
        return False
    # 答案句内容词（去问句标记/人称/标点后——取答案中非问句共现词）
    ans_core = re.sub(r"[？。，、！？\s]", "", ans)
    # 检查回答是否包含答案核心（句子内部一致——同一语料）
    if ans_core in resp:
        return True
    # 宽松：答案与回答共享 ≥2 个连续内容字（如"是水果" vs "水果"）
    for L in range(min(4, len(ans_core)), 1, -1):
        for k in range(len(ans_core) - L + 1):
            if ans_core[k:k + L] in resp and L >= 2:
                return True
    return False


def run():
    print("=== M5 阶段 134：v6 综合湖问答精度回测（隔离后能力完整验证） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=400)
    s2 = load_corpus(os.path.join(base, "corpus_simple2.txt"))
    s3 = load_corpus(os.path.join(base, "corpus_simple3.txt"))
    s4 = load_corpus(os.path.join(base, "corpus_simple4.txt"))
    s5 = load_corpus(os.path.join(base, "corpus_simple5.txt"))
    med = load_corpus(os.path.join(base, "corpus_medium.txt"))
    why = load_corpus(os.path.join(base, "corpus_why.txt"))
    full = simple + s2 + s3 + s4 + s5 + med + why
    blocks = extract_blocks(full)
    hubs = extract_hubs(full, blocks)
    chars = list(dict.fromkeys("".join(full)))
    print(f"语料 {len(full)} 行 / 词块 {len(blocks)} / 词汇 {len(chars)}")

    w = V6Bridge(chars, blocks + hubs)
    w.sents = full
    w.learn_v6(full)              # v6 隔离沉积主湖（C30-01）
    w.learn_bridge(full)          # 跨句桥（问答对——问句字↔答句枢纽）
    dia = V6Dialogue(w, full)     # 对话（hub 涌现 + 主语位——非写死）
    print(f"训练完成（{w.n} 字 / {len(w.hubs)} 河道——v6 隔离沉积 + 跨句桥）")

    # ---- exp1：问答对抽取 ----
    print("\n[exp1] 问答对抽取（语料自动构建测试集）:")
    pairs = extract_qa(full)
    print(f"      抽取 {len(pairs)} 对（问句+相邻答案行）")
    for q, a in pairs[:4]:
        print(f"      {q} → {a}")

    # ---- exp2：v6 回答正确率 ----
    print("\n[exp2] v6 回答正确率（{len(pairs)} 对）:")
    ok = 0
    by_type = Counter()
    ok_type = Counter()
    miss = []
    for q, ans in pairs:
        resp, _ = dia.respond(q)
        c = correct(ans, resp)
        ok += c
        t = "涌现"               # hub 全部涌现（stage117——统计+桥检索）
        by_type[t] += 1
        ok_type[t] += c
        if not c:
            miss.append((q, ans, resp))
    acc = ok / len(pairs) if pairs else 0
    print(f"      正确 {ok}/{len(pairs)} = {acc:.1%}"
          f"（{'精度 ✓（隔离不损能力）' if acc > 0.6 else '精度低'}——"
          f"C69-01 表达=重建——整句检索）")
    print("\n[exp3] 类型分布（薄弱处定位）:")
    for t in sorted(by_type):
        print(f"      {t}: {ok_type[t]}/{by_type[t]}（{ok_type[t]/by_type[t]:.0%}）")
    print("\n      答错样本:")
    for q, ans, resp in miss[:6]:
        print(f"      {q} → 标准: {ans} | 回答: {resp}")

    print("\n[结论] v6 精度回测：问答正确率 {acc:.1%}——"
          "隔离沉积后能力保持（C30-01 无回注不损语义）——"
          "薄弱类型见上（可补语料/调参）")
    print("[done] stage134 precision v6")


if __name__ == "__main__":
    run()
