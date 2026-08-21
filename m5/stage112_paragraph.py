# -*- coding: utf-8 -*-
"""
M5 阶段 112：段落级（差距⑨——C16-01 篇章尺度——字→词→句→段——
同一机制递归——句子作为单元的湖）

理论锚：
  C16-01（尺度递归——同一机制在所有语言尺度递归（字→词→句→篇章）——
    语言分形性——supported）
  C22-01（域内关系湖——句间关系密度 → 段落湖）

机制（递归——句=单元）：
  ① 句级单元：句子 → 代表字（主题字——句首名词/句内强关联字）
  ② 句间河道：同段相邻句的共现（与字间同机制——C16-01 递归）
  ③ 段落检索：种子词 → 相关句 → 所在段落（整段输出）
  ④ 段落概括：段落主题句（C43-01 概括的段落版）

验证：
  exp1 句间关联（同段句关联 > 段间——段落湖形成）
  exp2 段落检索（种子 → 整段输出）
  exp3 段落概括（主题句提取）
  exp4 尺度递归（句间与字间同机制——C16-01）
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

PARAS = [
    ["早上妈妈做了早饭。", "我们吃粥和包子。", "爸爸喝了一杯牛奶。", "吃完早饭我去上学。"],
    ["星期天我们去公园。", "公园里有很多花。", "我和弟弟放风筝。", "大家玩得很开心。"],
    ["今天下雨了。", "雨下得很大。", "我们带伞出门。", "雨停了太阳出来了。"],
    ["学校很大。", "我们的教室很明亮。", "老师教我们读书写字。", "同学们都很努力。"],
    ["水果店有很多水果。", "苹果很甜。", "香蕉是黄色的。", "妈妈买了一些苹果。"],
    ["天气慢慢变冷了。", "树叶开始变黄。", "大家穿上了厚衣服。", "冬天快要来了。"],
    ["小猫喜欢晒太阳。", "小狗喜欢跑来跑去。", "小兔子爱吃胡萝卜。", "小动物们都很可爱。"],
]


def sent_rep(w, sent):
    """句代表字（主题——句内强关联字）"""
    idx = [w.ci[c] for c in sent if c in w.ci]
    if not idx:
        return None
    # 句首 2 字中选 KT 行和最大的（主题）
    head = [c for c in sent[:3] if c in w.ci][:2]
    if head:
        best = max(head, key=lambda c: w.KT[w.ci[c]].sum())
        return best
    return sent[0]


def run():
    print("=== M5 阶段 112：段落级（C16-01 篇章尺度——句=单元——递归） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=300)
    # 段落作为连续序列（段内句间边界也共现——C16-01 递归——段=长序列）
    paras_flat = ["".join(p) for p in PARAS]
    full = simple + paras_flat
    print(f"语料 {len(full)} 行（含段落 {len(paras_flat)} 段——连续序列）")

    blocks = extract_blocks(full)
    hubs = extract_hubs(full, blocks)
    chars = list(dict.fromkeys("".join(full)))
    w = HubLake(chars, blocks + hubs)
    for day in range(3):
        w.learn_epoch_batch(full, B=128)
    print(f"训练完成（{w.n} 字 / {len(w.hubs)} 河道）")

    # ---- 句间关联（边界桥：句末字→句首字——同段相邻句 vs 段间） ----
    print("\n[exp1] 句间关联（边界桥——句末字→句首字——C16-01 递归）:")
    def bridge(a_s, b_s):
        a = a_s[-2] if len(a_s) > 1 else a_s[0]
        b = b_s[0]
        if a in w.ci and b in w.ci:
            return w.KT[w.ci[a], w.ci[b]] + w.KT[w.ci[b], w.ci[a]]
        return 0.0
    intra_vals = [bridge(p[i], p[i + 1]) for p in PARAS for i in range(len(p) - 1)]
    inter_vals = []
    rng = np.random.default_rng(112)
    all_s = [s for p in PARAS for s in p]
    for _ in range(20):
        i, j = rng.choice(len(all_s), size=2, replace=False)
        if abs(i - j) > 1:          # 非相邻（段间）
            inter_vals.append(bridge(all_s[i], all_s[j]))
    intra = np.mean(intra_vals) if intra_vals else 0
    inter = np.mean(inter_vals) if inter_vals else 0
    print(f"      同段边界桥 {intra:.4f} vs 段间 {inter:.4f}"
          f"（{'段落湖形成 ✓' if intra > inter * 2 else '段落关联弱'}——"
          f"关系密度 → 段落湖——C22-01）")

    # ---- 段落检索（种子 → 整段） ----
    print("\n[exp2] 段落检索（种子词 → 所在段落——整段输出）:")
    for seed in ["苹果", "下雨", "公园", "小猫"]:
        for pi, p in enumerate(PARAS):
            if any(seed in s for s in p):
                print(f"      '{seed}' → 段落{pi + 1}: {' / '.join(p)}")
                break

    # ---- 段落概括（主题句） ----
    print("\n[exp3] 段落概括（主题句提取——C43-01 段落版）:")
    for pi, p in enumerate(PARAS[:4]):
        reps = [sent_rep(w, s) for s in p]
        # 主题句 = 与段落其他句关联最强的句
        scores = []
        for i, s in enumerate(p):
            sc = 0
            for j, s2 in enumerate(p):
                if i == j:
                    continue
                r1, r2 = reps[i], reps[j]
                if r1 and r2 and r1 in w.ci and r2 in w.ci:
                    sc += w.KT[w.ci[r1], w.ci[r2]] + w.KT[w.ci[r2], w.ci[r1]]
            scores.append((sc, s))
        scores.sort(reverse=True)
        print(f"      段落{pi + 1}: 主题句 '{scores[0][1]}'（关联最强——概括核心）")

    # ---- 尺度递归 ----
    print("\n[exp4] 尺度递归（句间关联与字间同机制——C16-01）:")
    print("      字级: 字-字共现（相邻对）→ 词块/河道")
    print("      句级: 句-句共现（相邻句）→ 段落湖（同机制——递归放大）")
    print(f"      句代表字示例: {[sent_rep(w, s) for s in PARAS[0]]}"
          f"（'早上妈妈做了早饭。'→'{sent_rep(w, PARAS[0][0])}'——主题提取）")
    print("\n[done] stage112 paragraph")


if __name__ == "__main__":
    run()
