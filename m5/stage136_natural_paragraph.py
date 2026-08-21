# -*- coding: utf-8 -*-
"""
M5 阶段 136：自然段落生成（用户："我需要的是自然语言，不是断句"——
stage115 侧面拼合 = 列表感——本 stage 承接式级联——句间衔接）

理论锚：
  C43-01（表达 = 重建——一次连续流的重建——非列表拼合）
  C23-01（联想 = 河道激活级联——句→句 = 级联延续）
  C13-02（组合性 = 时序吸引子复合——句间 = 相位续接）
  Maimon 2025（话语连贯——句间共享成分（衔接）——主题锚）

机制（承接式级联——句间衔接）：
  ① 起始：主题句（含主题——月亮）
  ② 每步：上一句**句末成分**（末 2 字——词块优先）→ 作为桥词——
    候选句 = 含桥词的句（句间共享成分——Maimon 衔接）∧ 与主题
    KT 关联（内容性——不漂移）
  ③ 主题锚：窗口内主题词出现（每 3-4 句回主题一次——篇章锚——
    非每句强制）
  ④ 链式步进至 ≥100 字——自然延续（"月亮升起来"→"月亮很圆"→
    "晚上看月亮"——承接流非列表）

验证：
  exp1 自然段落（承接式——月亮主题——≥100 字）
  exp2 句间衔接（相邻句共享成分率——Maimon）
  exp3 主题保持（窗口内主题出现）
  exp4 对比 stage115（句间衔接率提升——列表感消除）
"""
import os
import re
import sys
import time
from collections import Counter
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("gb"):
    sys.stdout.reconfigure(encoding="utf-8")

from stage79_spontaneous_hubs import (load_corpus, extract_blocks, extract_hubs)
from stage133_integration_v6 import V6Lake
from stage115_short_essay import punctuate, has_independent


def bridge_of(w, s, k=2):
    """句末成分（桥词——词块优先——末字兜底）"""
    core = re.sub(r"[。！？，、\s]", "", s)
    if not core:
        return None
    blocks_all = sorted([h for h in w.hubs if len(h) > 1], key=len, reverse=True)
    for b in blocks_all:
        if core.endswith(b):
            return b
    return core[-k:]


def chain_paragraph(w, topic, sents, min_chars=100, max_steps=25):
    """承接式级联段落：句末成分 → 下一句桥——句间衔接（Maimon）——
    主题锚（窗口内回主题）"""
    used = set()
    blocks_all = [h for h in w.hubs if len(h) > 1]
    ti = w.ci[topic[-1]] if topic[-1] in w.ci else None
    # 起始主题句
    starters = [s for s in sents if topic in s and has_independent(w, s, topic, blocks_all)
                and s not in used and len(s) >= 5 and "。" in s]
    if not starters:
        return []
    out = [starters[0]]
    used.add(starters[0])
    cur = starters[0]
    for step in range(max_steps):
        if len("".join(out)) >= min_chars:
            break
        bridge = bridge_of(w, cur)
        if not bridge:
            break
        # 候选：含桥词（承接——共享成分）∪ 含主题词（主题持续——不断链）
        # ∧ 与主题关联（内容性）∧ 未用 ∧ 非近重复
        cands = []
        for s in sents:
            if s in used or len(s) < 5 or "。" not in s:
                continue
            if not (bridge in s or bridge[-1] in s or topic in s):
                continue
            idx = [w.ci[c] for c in s if c in w.ci]
            if not idx or ti is None:
                continue
            rel = float(np.mean([w.KT[ti, j] for j in idx]))
            if rel < 0.004:                # 收紧（滤对比句——太阳/走廊）
                continue
            # 近重复过滤（与已用句共享 ≥70% 字——去'月亮升起来'重复）
            sc = set(re.sub(r"[。！？，、\s]", "", s))
            if any(len(sc & set(re.sub(r"[。！？，、\s]", "", u))) >=
                   min(len(sc), len(u)) * 0.7 for u in used):
                continue
            # 桥强度：含完整桥词优先（强承接）——含主题词次之（持续）
            if bridge in s:
                bonus = 1.5
            elif topic in s:
                bonus = 1.0
            else:
                bonus = 1.2
            cands.append((rel * bonus, s))
        cands.sort(key=lambda x: -x[0])
        if not cands:
            break                          # 链断（级联尽头——停止）
        nxt = cands[0][1]
        out.append(nxt)
        used.add(nxt)
        cur = nxt
    return out


def cohesion(sents):
    """句间衔接率：相邻句共享成分（任意 2 字词/末词）"""
    if len(sents) < 2:
        return 0.0
    n_link = 0
    for a, b in zip(sents, sents[1:]):
        wa = set(re.findall(r"[一-鿿]{2}", a))
        wb = set(re.findall(r"[一-鿿]{2}", b))
        if wa & wb:
            n_link += 1
    return n_link / (len(sents) - 1)


def run():
    print("=== M5 阶段 136：自然段落生成（承接式级联——句间衔接——"
          "Maimon——用户：自然语言非断句） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=400)
    s2 = load_corpus(os.path.join(base, "corpus_simple2.txt"))
    s3 = load_corpus(os.path.join(base, "corpus_simple3.txt"))
    s4 = load_corpus(os.path.join(base, "corpus_simple4.txt"))
    s5 = load_corpus(os.path.join(base, "corpus_simple5.txt"))
    med = load_corpus(os.path.join(base, "corpus_medium.txt"))
    para = load_corpus(os.path.join(base, "corpus_paragraph.txt"))
    why = load_corpus(os.path.join(base, "corpus_why.txt"))
    full = simple + s2 + s3 + s4 + s5 + med + para + why
    blocks = extract_blocks(full)
    hubs = extract_hubs(full, blocks)
    chars = list(dict.fromkeys("".join(full)))
    print(f"语料 {len(full)} 行 / 词块 {len(blocks)} / 词汇 {len(chars)}")

    w = V6Lake(chars, blocks + hubs)
    w.learn_v6(full)
    print(f"训练完成（{w.n} 字 / {len(w.hubs)} 河道——v6 隔离沉积）")

    # ---- exp1/2：自然段落（承接式——月亮——≥100 字——句间衔接） ----
    print("\n[exp1] 月亮自然段落（承接式级联——句间衔接——≥100 字）:")
    if "月" in w.ci:
        clauses = chain_paragraph(w, "月", full, min_chars=100)
        # 自然段落：句号句序列（c 剥尾标点——统一单句号——中文段落
        # 本为句号句流——非机械逗号拼合）
        essay = [c.rstrip("。！？") + "。" for c in clauses]
        n_chars = sum(len(c) for c in essay)
        print(f"      {' '.join(essay)}")
        print(f"      （{len(clauses)} 句 / {n_chars} 字——"
              f"{'≥100 ✓' if n_chars >= 100 else '不足'}）")
        # exp2 句间衔接
        coh = cohesion(clauses)
        print(f"      [exp2] 句间衔接率 {coh:.2f}"
              f"（{'承接 ✓（相邻句共享成分——Maimon）' if coh > 0.5 else '断裂'}——"
              f"桥词级联——非列表）")

    # ---- exp3：主题保持（窗口内主题出现） ----
    print("\n[exp3] 主题保持（窗口内主题出现——篇章锚）:")
    if "月" in w.ci and clauses:
        windows = [clauses[i:i + 3] for i in range(0, len(clauses), 3)]
        anchored = sum(1 for win in windows if any("月" in c or "亮" in c
                                                   or "星" in c for c in win))
        print(f"      {len(windows)} 窗口——{anchored} 含主题"
              f"（{'主题锚 ✓（每 3 步回主题——篇章锚）'
                  if anchored >= len(windows) * 0.6 else '漂移'}——"
              f"非每句强制——自然）")

    # ---- exp4：对比 stage115（衔接率——列表感消除） ----
    print("\n[exp4] 对比 stage115（衔接率——列表感消除）:")
    if "月" in w.ci:
        from stage115_short_essay import essay_generate
        old = essay_generate(w, "月", full, min_chars=100)
        coh_old = cohesion(old)
        print(f"      stage115 侧面拼合衔接率 {coh_old:.2f} vs "
              f"stage136 承接级联 {coh:.2f}"
              f"（{'自然化 ✓（承接替代拼合——句间流动）'
                  if coh > coh_old else '未提升'}——"
              f"C43-01 表达=重建连续流）")
    print("\n[结论] 自然段落：承接式级联（句末成分→下一句桥——Maimon "
          "句间衔接）——衔接率 {coh:.2f}——主题锚（窗口回主题）——"
          "自然语言非断句（用户反馈修复）")
    print("[done] stage136 natural paragraph")


if __name__ == "__main__":
    run()
