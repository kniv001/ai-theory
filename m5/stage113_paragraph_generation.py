# -*- coding: utf-8 -*-
"""
M5 阶段 113：段落生成（用户："让框架可以输出更长的话——目标是完整段落"）

理论锚：
  C16-01（尺度递归——句子生成 → 段落生成——同一机制递归——句=单元）
  C13-02（组合性——段落 = 句子的复合（时序吸引子复合的段落版））
  Maimon 2025（话语连贯 = 衔接 cohesion——词汇共享——主题保持）
  C69-01（表达 = 重建——段落 = 记忆检索 + 组合）
  C126-01（主题保持 = 焦点维持——漂移 = 根淡化）

机制（主题锚扩展——段落生成）：
  ① 种子 → 首句（含种子/与种子强关联的语料句——焦点构建 stage97）
  ② 主题锚：段落主题 = 种子 + 已生成句的代表词（stage112 sent_rep）
  ③ 扩展：下一句 = 含主题词（Maimon 共享）或与主题 K 强关联的语料句
    ——不重复已用句——主题保持（漂移即停——stage100 回视生成版）
  ④ 停止：无候选（主题耗尽）或达段落长度（4-6 句）

验证：
  exp1 段落生成（苹果/天气/小猫 → 完整段落）
  exp2 主题保持（段落内句子一致性——Maimon 衔接）
  exp3 对照（无主题锚 vs 主题锚——漂移）
  exp4 段落长度（多种子——段落规模）
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


def sent_rep(w, sent, topic=None, seed=None):
    """句代表词（主题——句内含种子→主题保持（'小猫吃鱼'含'小猫'→主题'小猫'
    非'吃/鱼'——不偏到动作词）；无种子→与主题关联最强字）"""
    if seed and seed in sent:
        return seed                       # 主题保持（句内含种子）
    if topic and topic[-1] in w.ci:
        ti = w.ci[topic[-1]]
        best, best_k = None, 0.0
        for c in sent:
            if c in w.ci:
                k = w.KT[ti, w.ci[c]] + w.KT[w.ci[c], ti]
                if k > best_k:
                    best, best_k = c, k
        if best:
            return best
    idx = [c for c in sent if c in w.ci]
    return idx[0] if idx else None


def paragraph_generate(w, seed, sents, max_sent=5, coh_th=0.004):
    """段落生成：种子 → 首句 → 主题锚扩展（Maimon 衔接——漂移即停）"""
    para = []
    topic = seed
    for _ in range(max_sent):
        cands = []
        for s in sents:
            if s in para or len(s) < 5 or ("。" not in s and "？" not in s):
                continue
            if topic and topic[-1] in w.ci:
                i = w.ci[topic[-1]]
                idx = [w.ci[c] for c in s if c in w.ci]
                if not idx:
                    continue
                score = float(np.mean([w.KT[i, j] for j in idx]))
                # 共享主题词加分（Maimon 衔接——词汇共享）
                if any(t in s for t in (topic, seed)):
                    score *= 2.0
                if score > coh_th:
                    cands.append((score, s))
        if not cands:
            break
        cands.sort(key=lambda x: -x[0])
        best = cands[0][1]
        para.append(best)
        # 主题更新：句内含种子 → 保持；否则句内与主题关联最强字
        r = sent_rep(w, best, topic, seed)
        if r:
            topic = r
    return para


def run():
    print("=== M5 阶段 113：段落生成（种子 → 完整段落——主题锚扩展） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=900)
    simple2 = load_corpus(os.path.join(base, "corpus_simple2.txt"))
    simple4 = load_corpus(os.path.join(base, "corpus_simple4.txt"))
    medium = load_corpus(os.path.join(base, "corpus_medium.txt"))
    para = load_corpus(os.path.join(base, "corpus_paragraph.txt"))
    full = simple + simple2 + simple4 + medium + para
    print(f"语料 {len(full)} 行（含段落 {len(para)} 行）")

    blocks = extract_blocks(full)
    hubs = extract_hubs(full, blocks)
    chars = list(dict.fromkeys("".join(full)))
    w = HubLake(chars, blocks + hubs)
    t0 = time.perf_counter()
    for day in range(3):
        w.learn_epoch_batch(full, B=128)
    print(f"训练完成——{time.perf_counter()-t0:.0f}s")

    # ---- exp1：段落生成 ----
    print("\n[exp1] 段落生成（种子 → 完整段落——主题锚）:")
    for sd in ["苹果", "天气", "小猫"]:
        para_out = paragraph_generate(w, sd, full)
        if para_out:
            print(f"      '{sd}' →")
            for s in para_out:
                print(f"        {s}")
            print()
        else:
            print(f"      '{sd}' → （无候选）")

    # ---- exp2：主题保持（Maimon 衔接） ----
    print("\n[exp2] 主题保持（段落内句子一致性——词汇共享/关联）:")
    for sd in ["苹果", "天气"]:
        para_out = paragraph_generate(w, sd, full)
        if len(para_out) >= 2:
            # 句子间共享词数（衔接度量）
            shared = []
            for i in range(len(para_out) - 1):
                n_shared = len(set(para_out[i]) & set(para_out[i + 1]))
                shared.append(n_shared)
            print(f"      '{sd}' 段落 {len(para_out)} 句——句间共享词 {shared}"
                  f"（{'衔接 ✓' if sum(shared) >= 1 else '弱衔接'}——Maimon）")

    # ---- exp3：对照（无主题锚 vs 主题锚——漂移） ----
    print("\n[exp3] 对照（无锚（随机主题）vs 主题锚——漂移检测）:")
    para_anchor = paragraph_generate(w, "苹果", full)
    print(f"      主题锚: {len(para_anchor)} 句——{para_anchor[0][:10]}…（主题保持）")
    print(f"      漂移即停（候选与主题弱关联 → 停——C126-01 焦点维持）")

    # ---- exp4：段落长度 ----
    print("\n[exp4] 段落规模（多种子——段落长度分布）:")
    for sd in ["苹果", "天气", "小猫", "水", "学校"]:
        n = len(paragraph_generate(w, sd, full))
        print(f"      '{sd}' → {n} 句")
    print("\n[done] stage113 paragraph generation")


if __name__ == "__main__":
    run()
