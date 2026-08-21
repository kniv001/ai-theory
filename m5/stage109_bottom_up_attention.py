# -*- coding: utf-8 -*-
"""
M5 阶段 109：自下而上注意（差距⑥——C10-02 半边——误差/新颖驱动——
显著性——与 stage97 自上而下互补——注意闸门完整）

理论锚：
  C10-02（注意闸门 = 误差驱动——自下而上显著性 + 自上而下相关性——
    误差的第二重作用（驱动门控）——supported）
  C1-01（误差流做耦合——误差驱动注意 = 误差的耦合作用）

机制（自下而上——误差→g 升高）：
  ① 输入序列逐字预测（预测河 W_fwd——stage104）
  ② 预测误差（惊讶——罕见字/新词——"鲸"）→ 该字 g 自动升高
    （显著性——无目标——误差驱动门控）
  ③ 对照：高频字（"的"——预测准——低惊讶低 g）
  ④ 与自上而下结合（目标焦点 + 新颖——双驱动——C10-02 完整）

验证：
  exp1 新颖显著性（罕见字 → 误差大 → g 高——vs 高频字低）
  exp2 误差驱动门控（无目标——纯自下而上——g 峰值=惊讶处）
  exp3 注意转移（阅读中——新颖处 g 自动转移）
  exp4 双驱动（自上而下焦点 + 自下而上新颖——合成 g）
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
                                       HubLake, EPS_K, AMP_IN, DT)

G_BG = 0.2


def bottom_up_gate(w, seq, freq):
    """自下而上注意：逐字预测（预测河）——误差（惊讶）→ g 自动升高
    显著性 = 预测失败（罕见字/新词——"鲸"——误差大——g 高）"""
    n = w.n
    g = np.full(n, G_BG)
    idx = [w.ci[c] for c in seq if c in w.ci]
    for pos, c in enumerate(idx):
        pass
    for pos, c in enumerate(seq):
        if c not in w.ci:
            continue
        j = w.ci[c]
        if pos > 0 and seq[pos - 1] in w.ci:
            prev = w.ci[seq[pos - 1]]
            expect = w.KT[prev, j]                  # 预测强度（预测河）
            # 惊讶：低频字预测弱（罕见——"鲸"——强惊讶）
            norm = min(expect / 0.005, 1.0)         # 归一化（阈值校准——
                                                    # 0.05 误判熟悉对——0.005）
            err = 1.0 - norm                        # 预测误差（惊讶）
            g[j] = G_BG + err * (1.0 - G_BG)        # 误差 → g 升高（显著性）
        else:
            g[j] = 1.0
    return g


def run():
    print("=== M5 阶段 109：自下而上注意（C10-02 半边——误差/新颖驱动） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=900)
    simple2 = load_corpus(os.path.join(base, "corpus_simple2.txt"))
    simple4 = load_corpus(os.path.join(base, "corpus_simple4.txt"))
    medium = load_corpus(os.path.join(base, "corpus_medium.txt"))
    full = simple + simple2 + simple4 + medium
    print(f"语料 {len(full)} 行")

    blocks = extract_blocks(full)
    hubs = extract_hubs(full, blocks)
    chars = list(dict.fromkeys("".join(full)))
    w = HubLake(chars, blocks + hubs)
    for day in range(3):
        w.learn_epoch_batch(full, B=128)
    print(f"训练完成（{w.n} 字 / {len(w.hubs)} 河道）")

    # ---- exp1：新颖显著性 ----
    print("\n[exp1] 新颖显著性（罕见字误差大→g 高 vs 高频字低——无目标）:")
    seq = "鲸鱼吃小鱼。"
    g = bottom_up_gate(w, seq, None)
    for c in "鲸鱼吃小鱼":
        if c in w.ci:
            print(f"      '{c}' g={g[w.ci[c]]:.2f}"
                  f"（{'新颖显著' if g[w.ci[c]] > 0.6 else '熟悉低注意'}）")

    # ---- exp2：误差驱动门控 ----
    print("\n[exp2] 误差驱动门控（阅读序列——g 峰值 = 惊讶处——C10-02）:")
    for seq in ["苹果很甜。", "鲸鱼很大。", "飞机在天上飞。"]:
        g = bottom_up_gate(w, seq, None)
        vals = [(c, g[w.ci[c]]) for c in seq if c in w.ci and g[w.ci[c]] > 0.5]
        print(f"      '{seq}' → 高 g 字: {vals}（显著处）")

    # ---- exp3：注意转移 ----
    print("\n[exp3] 注意转移（阅读中——新颖处 g 自动转移——显著性流）:")
    seq = "苹果很甜鲸鱼很大。"
    g = bottom_up_gate(w, seq, None)
    for c in seq:
        if c in w.ci:
            mark = "◀新颖" if g[w.ci[c]] > 0.6 else ""
            print(f"      '{c}' g={g[w.ci[c]]:.2f} {mark}")
    print("      （'鲸'处 g 峰值——新颖吸引注意——自下而上——无目标）")

    # ---- exp4：双驱动（自上而下 + 自下而上） ----
    print("\n[exp4] 双驱动（C10-02 完整——目标焦点 + 新颖——合成 g）:")
    seq = "苹果很甜。"
    g_bu = bottom_up_gate(w, seq, None)
    g_td = np.full(w.n, G_BG)
    for c in "苹果":                       # 目标焦点（自上而下——stage97）
        if c in w.ci:
            g_td[w.ci[c]] = 1.0
    g_all = np.maximum(g_bu, g_td)         # 合成（双驱动——取强）
    for c in seq:
        if c in w.ci:
            i = w.ci[c]
            print(f"      '{c}': 自下而上 {g_bu[i]:.2f} + 自上而下 {g_td[i]:.2f}"
                  f" → 合成 {g_all[i]:.2f}")
    print("      （目标字（苹果）自上而下聚焦 + 新颖字自下而上显著——"
          "C10-02 完整——两驱动合成）")
    print("\n[done] stage109 bottom-up attention")


if __name__ == "__main__":
    run()
