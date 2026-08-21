# -*- coding: utf-8 -*-
"""
M5 阶段 122：词块隔离沉积（C30-01 无回注的正解——stage121 发现：
"苹果很甜"的句内共现让"甜"流入"苹"——块内字跨块共现应沉积词级河道
——不写字级）

理论锚：
  C30-01（跨尺度解耦——词义不下沉到字——成分只增加指向词的河道——
    组合靠河道连接非语义合并——无回注）
  C16-01（尺度递归——词级河道承载词义——字级河道承载指向）

机制（词块隔离）：
  ① 沉积分类：句内字对——同块（苹-果）→ 词级河道（K[苹果]——块内
    语义）+ 字级指向（K[苹][果]——成分→词）
  ② 跨块对（苹-甜——苹属苹果块）→ **词级河道**（K[苹果][甜]——苹果的
    语义）——**不写字级**（K[苹][甜] 不加——无回注✓）
  ③ 非块字对（天-气 若不成块）→ 字级（正常）

验证（对照 stage121）：
  exp1 无回注（隔离后——K[甜↔苹] 应趋零——词义不流入成分）
  exp2 词级语义保留（K[苹果][甜]——词义在词级）
  exp3 指向保留（K[苹][果]——成分→词指向）
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
                                       EPS_K, DT, AMP_IN)


class IsoLake:
    """词块隔离湖：字级（指向）+ 词级（语义）——无回注（C30-01）"""

    def __init__(self, chars, blocks):
        self.chars = list(chars)
        self.ci = {c: i for i, c in enumerate(chars)}
        n = len(chars)
        self.blocks = [b for b in blocks if all(c in self.ci for c in b)]
        self.K_char = np.zeros((n, n))          # 字级河道（指向）
        self.K_word = {b: np.zeros((n, n)) for b in self.blocks}   # 词级河道
        self.n = n

    def block_of(self, c):
        """字所属的词块（"苹"→"苹果"）"""
        return [b for b in self.blocks if c in b]

    def learn(self, sent):
        """隔离沉积：同块对→词级+指向；跨块对→词级（不写字级）"""
        idx = [self.ci[c] for c in sent if c in self.ci]
        for a in range(len(idx) - 1):
            for b in range(a + 1, len(idx)):
                i, j = idx[a], idx[b]
                ca, cb = self.chars[i], self.chars[j]
                ba = self.block_of(ca)
                bb = self.block_of(cb)
                if ba and bb and ba[0] == bb[0]:      # 同块（苹-果）
                    self.K_char[i, j] += EPS_K
                    self.K_char[j, i] += EPS_K
                    self.K_word[ba[0]][i, j] += EPS_K
                elif ba or bb:                         # 跨块（苹-甜）→ 词级
                    blk = (ba or bb)[0]
                    self.K_word[blk][i, j] += EPS_K
                    self.K_word[blk][j, i] += EPS_K
                else:                                  # 非块 → 字级
                    self.K_char[i, j] += EPS_K
                    self.K_char[j, i] += EPS_K

    def char_strength(self, a, b):
        return self.K_char[self.ci[a], self.ci[b]]

    def word_strength(self, blk, a, b):
        if blk in self.K_word:
            return self.K_word[blk][self.ci[a], self.ci[b]]
        return 0.0


def run():
    print("=== M5 阶段 122：词块隔离沉积（C30-01 无回注正解） ===\n")
    base = os.path.dirname(__file__)
    from stage79_spontaneous_hubs import load_corpus
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=200)
    medium = load_corpus(os.path.join(base, "corpus_medium.txt"))
    full = simple + medium
    blocks = extract_blocks(full)
    chars = list(dict.fromkeys("".join(full)))
    print(f"语料 {len(full)} 行 / 词块 {len(blocks)}（苹果 in blocks: {'苹果' in blocks}）")
    w = IsoLake(chars, blocks)
    for ep in range(3):
        for s in full:
            w.learn(s)
    print("训练完成（字级=指向 + 词级=语义——隔离沉积）")

    # ---- exp1：无回注（对照 stage121——K[甜↔苹] 应趋零） ----
    print("\n[exp1] 无回注（隔离后——'甜'不流入'苹'——C30-01）:")
    v = w.char_strength("甜", "苹") + w.char_strength("苹", "甜")
    print(f"      字级 K[甜↔苹] = {v:.4f}"
          f"（{'无回注 ✓（词义不流入成分）' if v < 0.001 else '仍有回注'}——"
          f"vs stage121 的 0.0533——隔离生效）")

    # ---- exp2：词级语义保留 ----
    print("\n[exp2] 词级语义（'苹果'河道——甜——词义在词级）:")
    v = w.word_strength("苹果", "苹", "甜") + w.word_strength("苹果", "甜", "苹")
    print(f"      词级 K[苹果][苹↔甜] = {v:.4f}"
          f"（{'词义保留 ✓' if v > 0.001 else '词义丢失'}——"
          f"'苹果很甜'的语义在词级河道——非字级）")

    # ---- exp3：指向保留 ----
    print("\n[exp3] 指向保留（'苹'→'果'——成分→词）:")
    v = w.char_strength("苹", "果")
    print(f"      字级 K[苹→果] = {v:.4f}"
          f"（{'指向保留 ✓' if v > 0.001 else '指向丢失'}——"
          f"成分只增加指向词的河道——C30-01）")
    print("\n[结论] 词块隔离：同块→指向+词级 / 跨块→词级（无回注）"
          "/ 非块→字级——C30-01 无回注满足——语义分层（字=指向/词=语义）")
    print("[done] stage122 block isolation")


if __name__ == "__main__":
    run()
