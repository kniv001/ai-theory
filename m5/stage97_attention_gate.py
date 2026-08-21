# -*- coding: utf-8 -*-
"""
M5 阶段 97：注意闸门 v2（C10-02——焦点聚集构建——联想自动）
用户纠正："焦点存在并不代表着全部的注意力——我们依然可以自动地去联想——
只是想要构建的时候可以焦点区聚集"

理论（纠正后）：
  C10-01（双通道——广播通道全局快=自动联想——河道通道定向=构建）
  C4-02（预测河下行需注意开闸——构建（生成）受闸门控制——联想自动）
  R10/C45-01（g 与 e 互补——焦点控制构建——联想（e）自动）
  C72-01（直觉=级联部分先行——自动联想——不需要焦点）

v1 错误：把闸门用于问答（联想/检索）——抑制了语义关联（水果激活不了）
v2 正确：联想自动（检索排序——不抑制）——焦点聚集构建（生成时候选
  约束在焦点区——背景词不参与构建）

机制：
  ① 联想自动（问答）：问句注入 → 级联激活（自由）→ 回答 = 激活集按
    与焦点的关联排序（排序而非抑制）
  ② 焦点聚集（构建）：种子 = 焦点 → 焦点区（种子强关联词集）→
    生成候选 ∈ 焦点区（背景词不参与构建——C4-02 预测河开闸）

验证：
  exp1 联想自动（问句 → 激活 → 排序回答——语义关联显现——无抑制）
  exp2 焦点聚集构建（种子 → 焦点区 → 完整句——构建受焦点控制）
  exp3 对照（无焦点构建 vs 焦点构建——候选空间差异）
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

EVO_STEPS = 12
FOCUS_K = 120         # 焦点区大小（种子强关联 top-N——"动物"类关联词）


def dynamic_assoc(w, q, k=4):
    """联想自动（问答——无闸门）：问句注入 → 级联激活（自由）→
    回答 = 激活集按与焦点（内容字）的关联排序——排序非抑制"""
    idx = [w.ci[c] for c in q if c in w.ci]
    if not idx:
        return []
    n = w.n
    z = np.zeros(n, dtype=complex)
    drive = np.zeros(n, dtype=complex)
    for pos, i in enumerate(idx):
        drive[i] += AMP_IN * np.exp(1j * (w.omega[i] * w.t + pos * np.pi / 6))
    for _ in range(EVO_STEPS):
        dz = -w.gamma * z + 1j * w.omega * z
        dz += (w.KT @ z.real + 1j * (w.KT @ z.imag)) - z * w.rsT
        dz += drive
        z = z + dz * DT
        over = np.abs(z) > 3.0
        z[over] = z[over] / np.abs(z[over]) * 2.0
    amp = np.abs(z)
    amp[idx] = 0                       # 问句字排除
    for j in range(n):
        if not re.match(r"[一-鿿]", w.chars[j]):
            amp[j] = 0                 # 标点排除
    for h in w.hubs:
        if len(h) == 1 and h in w.ci:
            amp[w.ci[h]] = 0           # 结构字排除（自发角色）
    # 焦点 = 问句内容字（对象——排除问标记块字——"是什么"的"什/么"不参与）
    mark = set("什么怎么为什么吗呢")
    focus = [w.ci[c] for c in q if c in w.ci
             and re.match(r"[一-鿿]", c) and c not in mark]
    if focus:
        rel = np.zeros(n)
        for f in focus:
            rel += w.KT[f] + w.KT[:, f]
        score = amp * rel              # 激活 × 相关性（排序——非抑制）
    else:
        score = amp
    top = np.argsort(score)[::-1][:k]
    return [(w.chars[j], amp[j]) for j in top if amp[j] > 0.01]


def build_focused(w, seed, sents, max_len=8):
    """焦点聚集构建（生成——C4-02 预测河开闸）：种子 = 焦点——
    焦点区 = 种子强关联词集——构建候选 ∈ 焦点区（背景不参与构建）"""
    if seed[-1] not in w.ci:
        return seed
    i = w.ci[seed[-1]]
    # 焦点区：种子强关联词（KT 行/列 top-N）
    row = w.KT[i] + w.KT[:, i]
    fz = np.argsort(row)[::-1][:FOCUS_K]
    focus_set = {w.chars[j] for j in fz if re.match(r"[一-鿿]", w.chars[j])}
    focus_set.add(seed[-1])
    # 语料完整句检索：种子出现 且 句内字都在焦点区（或少量例外）→ 候选
    cands = []
    for s in sents:
        if ("。" not in s and "？" not in s) or len(s) < 5:
            continue
        if seed not in s:
            continue
        outside = [c for c in s if c in w.ci and c not in focus_set]
        if len(outside) > 1:           # 焦点区约束（最多 1 个区外字——构建聚集）
            continue
        idx = [w.ci[c] for c in s if c in w.ci]
        score = float(np.mean([w.KT[i, j] for j in idx]))
        cands.append((score, s))
    cands.sort(key=lambda x: -x[0])
    if not cands:
        return seed
    return cands[0][1]


def run():
    print("=== M5 阶段 97 v2：注意闸门（焦点聚集构建——联想自动——用户纠正） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=900)
    simple2 = load_corpus(os.path.join(base, "corpus_simple2.txt"))
    simple3 = load_corpus(os.path.join(base, "corpus_simple3.txt"))
    simple4 = load_corpus(os.path.join(base, "corpus_simple4.txt"))
    simple5 = load_corpus(os.path.join(base, "corpus_simple5.txt"))
    medium = load_corpus(os.path.join(base, "corpus_medium.txt"))
    medium2 = load_corpus(os.path.join(base, "corpus_medium2.txt"))
    medium3 = load_corpus(os.path.join(base, "corpus_medium3.txt"))
    why = load_corpus(os.path.join(base, "corpus_why.txt"))
    wiki = load_corpus(os.path.join(base, "corpus_wiki_filtered.txt"), n=600)
    attr = load_corpus(os.path.join(base, "corpus_attr_cause.txt"))
    neg = load_corpus(os.path.join(base, "corpus_negation.txt"))
    social = load_corpus(os.path.join(base, "corpus_social.txt"))
    isa_sents = ["苹果是水果。", "香蕉是水果。", "西瓜是水果。", "葡萄是水果。",
                 "猫是动物。", "狗是动物。", "鸟是动物。", "鱼是动物。",
                 "水是液体。", "冰是固体。", "雪是白色的。", "天空是蓝色的。",
                 "老虎是动物。", "树是植物。", "花是植物。", "石头是固体。",
                 "苹果可以吃。", "水可以喝。", "雨是从云落下来的。",
                 "小猫吃鱼。", "猫吃老鼠。", "我吃苹果。", "小猫吃月饼。"]
    full = simple + simple2 + simple3 + simple4 + simple5 + medium + medium2 + medium3 + why + wiki + attr + neg + social + isa_sents
    print(f"全语料 {len(full)} 行")

    blocks = extract_blocks(full)
    hubs = extract_hubs(full, blocks)
    chars = list(dict.fromkeys("".join(full)))
    w = HubLake(chars, blocks + hubs)
    t0 = time.perf_counter()
    for day in range(4):
        w.learn_epoch_batch(full, B=128)
    print(f"训练完成——{time.perf_counter()-t0:.0f}s（{w.n} 字 / {len(w.hubs)} 河道）")

    # ---- exp1：联想自动（问答——排序而非抑制） ----
    print("\n[exp1] 联想自动（问句 → 自由级联 → 按焦点相关性排序——无抑制）:")
    for q in ["苹果是什么？", "为什么带伞？", "小猫吃什么？", "苹果怎么样？",
              "月亮是什么？", "大象是什么？"]:
        ans = dynamic_assoc(w, q)
        print(f"      Q: '{q}' → {[(a, f'{v:.2f}') for a, v in ans[:4]] if ans else '（无）'}")

    # ---- exp2：焦点聚集构建（生成——候选约束在焦点区） ----
    print("\n[exp2] 焦点聚集构建（种子=焦点——候选∈焦点区——背景不参与构建）:")
    seeds = ["苹果", "天气", "小猫", "妈妈", "月亮", "大象", "水", "鱼"]
    for sd in seeds:
        g = build_focused(w, sd, full)
        tag = "完整句" if len(g) >= 5 and g != sd else ("短语" if len(g) > len(sd) else "未生成")
        print(f"      '{sd}' → '{g}'  [{tag}]")

    # ---- exp3：对照（无焦点 vs 焦点——候选空间） ----
    print("\n[exp3] 对照（无焦点构建 vs 焦点构建——'苹果'的候选句）:")
    all_c = [s for s in full if "苹果" in s and len(s) >= 5]
    fz_c = build_focused(w, "苹果", full)
    print(f"      语料含'苹果'句 {len(all_c)} 个——焦点构建选出: '{fz_c}'")
    print("      （焦点区约束：句内字须与'苹果'强关联——背景句不参与）")
    print("\n[done] stage97 v2 attention gate")


if __name__ == "__main__":
    run()
