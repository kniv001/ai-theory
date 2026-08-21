# -*- coding: utf-8 -*-
"""
M5 阶段 96：问句动力学化（妥协回收——ask if-else → 湖级联激活）

妥协：stage93 v4 的 ask 是规则式（"怎"→K[很]/"为"→K[因为]——if-else——
近似写死——虽然数据驱动但仍是解析）
理论目标：问句驱动 = 湖动力学（A1 原语——注入问句 → 级联激活 → 回答）
  C13-01（意义 = 预测关系集——问句 = 关系集激活请求）
  C43-01（理解 = 重构——问句注入 → 地形重构 → 回答）

机制（动力学问句）：
  ① 注入问句（相位驱动——与学习同机制）
  ② 演化 N 步（K 级联——关联传播——"苹果是什么" → 苹果+是+什么 驱动
     → 级联激活 水果/甜——关系集）
  ③ 回答 = |z| top（排除问句字本身——激活的语义关联词）

验证：
  exp1 动力学问句（苹果是什么→水果？/为什么带伞→雨？——级联激活）
  exp2 对照（规则 ask vs 动力学 ask——回答质量）
  exp3 同种子多问法（"苹果是什么/苹果怎么样"——同一对象不同问法——
    级联差异——语义面）
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
                                       HubLake, AMP_IN, PULSE_STEPS, DT)

EVO_STEPS = 12          # 问句级联演化步数（多步——关联传播）


def dynamic_ask(w, q, k=4):
    """动力学问句：注入问句 → 级联激活 → 回答（排除问句字——A1 原语）"""
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
    amp[idx] = 0                     # 排除问句字（回答 = 激活的语义关联）
    for h in w.hubs:                 # 排除单字枢纽（结构字——自发涌现的角色——
        if len(h) == 1 and h in w.ci:      # 背景激活淹没语义——非词表）
            amp[w.ci[h]] = 0
    top = np.argsort(amp)[::-1][:k]
    return [(w.chars[j], amp[j]) for j in top if amp[j] > 0.02]


def run():
    print("=== M5 阶段 96：问句动力学化（妥协回收——ask → 湖级联激活） ===\n")
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

    # ---- exp1：动力学问句 ----
    print("\n[exp1] 动力学问句（注入问句 → 级联激活 → 回答——无解析）:")
    for q in ["苹果是什么？", "为什么带伞？", "小猫吃什么？", "苹果怎么样？",
              "水是什么？", "月亮是什么？"]:
        ans = dynamic_ask(w, q)
        print(f"      Q: '{q}' → {[(a, f'{v:.2f}') for a, v in ans[:4]] if ans else '（无激活）'}")

    # ---- exp2：对照（规则 ask vs 动力学） ----
    print("\n[exp2] 对照（规则 ask vs 动力学——回答质量）:")
    for q in ["苹果是什么？", "为什么带伞？", "小猫吃什么？"]:
        hub, obj, r_ans = w.ask(q)
        d_ans = dynamic_ask(w, q)
        print(f"      Q: '{q}'")
        print(f"        规则: 枢纽'{hub}' '{obj}' → {[(a, f'{v:.2f}') for a, v in r_ans[:3]] if r_ans else '无'}")
        print(f"        动力学: {[(a, f'{v:.2f}') for a, v in d_ans[:4]] if d_ans else '无'}")

    # ---- exp3：同一对象不同问法（语义面） ----
    print("\n[exp3] 同一对象不同问法（级联差异 = 语义面）:")
    for q in ["苹果是什么？", "苹果怎么样？", "苹果可以吃吗？"]:
        ans = dynamic_ask(w, q)
        print(f"      Q: '{q}' → {[(a, f'{v:.2f}') for a, v in ans[:4]] if ans else '（无）'}")
    print("\n[done] stage96 dynamic QA")


if __name__ == "__main__":
    run()
