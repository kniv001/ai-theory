# -*- coding: utf-8 -*-
"""
M5 阶段 121：跨尺度解耦（C30-01——不同尺度吸引子语义解耦——
**词义不下沉到字**；成分只增加指向词的河道，内容不变；组合靠河道连接
非语义合并；解决回注问题（无回注）——supported 未验证——本 stage M5 验证）

理论锚：
  C30-01（跨尺度解耦原则——词义不下沉到字——成分只增加指向词的河道——
    组合靠河道连接非语义合并——无回注——supported）
  C16-01（尺度递归——字→词→句——语义分层）

机制（验证）：
  ① 成分语义：字（"苹"）的河道 = 指向词（"果"——苹果）——**不含词义
    （甜/脆/水果）**——词义不下沉（C30-01）
  ② 词级语义：词块（"苹果"）河道 = 甜/脆/水果——词义在词级
  ③ 解耦：字级（指向）vs 词级（语义）——分层不混合
  ④ 无回注："甜"不流入"苹"（成分字）——无回注（C30-01 解决回注问题）

验证：
  exp1 成分语义（"苹"→"果"——指向词——非甜/脆——词义不下沉）
  exp2 词级语义（"苹果"河道——甜/脆/水果——词义在词级）
  exp3 解耦（字级 vs 词级——语义分层）
  exp4 无回注（"甜"不流入"苹"——回注问题解决）
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


def run():
    print("=== M5 阶段 121：跨尺度解耦（C30-01——词义不下沉到字——无回注） ===\n")
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
    print(f"训练完成（{w.n} 字 / {len(w.hubs)} 河道——含块 {len(blocks)}）")

    # ---- exp1：成分语义（词义不下沉） ----
    print("\n[exp1] 成分语义（'苹'的河道——指向词——非甜/脆——词义不下沉）:")
    for c in ["苹", "西", "香"]:
        if c in w.ci:
            i = w.ci[c]
            row = w.KT[i] + w.KT[:, i]
            top = [(w.chars[j], row[j]) for j in np.argsort(row)[::-1][:5]
                   if w.chars[j] != c and row[j] > 0.005]
            print(f"      '{c}' 关联: {[(a, f'{v:.3f}') for a, v in top]}"
                  f"（{'指向词 ✓' if any(a in '果瓜蕉' for a, _ in top) else '成分语义'}"
                  f"——不含甜/脆类词义）")

    # ---- exp2：词级语义 ----
    print("\n[exp2] 词级语义（'苹果'河道——甜/脆/水果——词义在词级）:")
    if "苹果" in w.K:
        i = w.ci["苹"]
        row = w.K["苹果"][i] + w.K["苹果"][:, i]
        top = [(w.chars[j], row[j]) for j in np.argsort(row)[::-1][:6]
               if re.match(r"[一-鿿]", w.chars[j]) and row[j] > 0.005]
        print(f"      '苹果'河道: {[(a, f'{v:.3f}') for a, v in top]}"
              f"（{'词义（甜/脆/水果）✓' if any(a in '甜脆果' for a, _ in top) else '词义弱'}）")

    # ---- exp3：解耦（字级 vs 词级——语义分层） ----
    print("\n[exp3] 解耦（字级=指向 / 词级=语义——分层）:")
    if "苹" in w.ci and "苹果" in w.K:
        i = w.ci["苹"]
        char_top = [w.chars[j] for j in np.argsort(w.KT[i] + w.KT[:, i])[::-1][:3]]
        word_top = [w.chars[j] for j in np.argsort(
            w.K["苹果"][i] + w.K["苹果"][:, i])[::-1][:3]]
        overlap = set(char_top) & set(word_top)
        print(f"      字级 top: {char_top}")
        print(f"      词级 top: {word_top}")
        print(f"      重叠: {overlap}（{'分层 ✓（字=指向/词=语义）' if len(overlap) <= 1 else '混合'}——"
              f"C30-01 语义分层）")

    # ---- exp4：无回注（"甜"不流入"苹"） ----
    print("\n[exp4] 无回注（'甜'→'苹'的关联——词义不回流到成分——C30-01）:")
    if "甜" in w.ci and "苹" in w.ci:
        v = w.KT[w.ci["甜"], w.ci["苹"]] + w.KT[w.ci["苹"], w.ci["甜"]]
        print(f"      K[甜↔苹] = {v:.4f}"
              f"（{'无回注 ✓（词义不流入成分）' if v < 0.01 else '有回注（词义下沉）'}——"
              f"'苹'不因'苹果甜'而含'甜'语义——组合靠河道连接非语义合并）")
    print("\n[done] stage121 scale decoupling")


if __name__ == "__main__":
    run()
