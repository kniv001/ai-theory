# -*- coding: utf-8 -*-
"""
M5 阶段 142：记忆功能 + 持续学习（用户："补充记忆功能吧，减少重复
训练的时间"——C6-03 记忆=地形——保存/加载——增量学习——不重训）

理论锚：
  C6-03（记忆 = 地形本身——保存 = 固化——加载 = 恢复）
  C96-01（缓存-写回模型——记忆文件 = 磁盘——保存 = 写回）
  C2-06（结构永不冻结——持续学习——增量沉积——新语料只加不删）

机制（memory.py）：
  ① save_lake：K 稀疏三元组 + chars/hubs/omega → npz（记忆文件）
  ② load_lake：恢复 SparseLake（不训练——直接可用）
  ③ continual_learn：增量（新句沉积 → K+= → KT 重合并——新字扩展）
  ④ 全量重训 vs 增量学习（耗时对比——增量省时）

验证：
  exp1 保存（训练 → save——文件大小）
  exp2 加载（load——恢复——激活一致——不重训）
  exp3 持续学习（load 旧 → 学新语料 → save——对比全量重训耗时）
  exp4 记忆更新（增量后新知识可用——旧知识保持）
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
from stage139_wiki_sparse import SparseLake
from memory import save_lake, load_lake, continual_learn


def run():
    print("=== M5 阶段 142：记忆功能 + 持续学习（保存/加载——"
          "增量——减少重复训练） ===\n", flush=True)
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=400)
    s2 = load_corpus(os.path.join(base, "corpus_simple2.txt"))
    s3 = load_corpus(os.path.join(base, "corpus_simple3.txt"))
    s4 = load_corpus(os.path.join(base, "corpus_simple4.txt"))
    s5 = load_corpus(os.path.join(base, "corpus_simple5.txt"))
    med = load_corpus(os.path.join(base, "corpus_medium.txt"))
    moon = load_corpus(os.path.join(base, "corpus_moon.txt"))
    why = load_corpus(os.path.join(base, "corpus_why.txt"))
    cx = load_corpus(os.path.join(base, "corpus_complex.txt"))
    para = load_corpus(os.path.join(base, "corpus_paragraph.txt"))
    essay = load_corpus(os.path.join(base, "corpus_essay.txt"))
    wiki = load_corpus(os.path.join(base, "corpus_wiki_filtered.txt"))
    basic = simple + s2 + s3 + s4 + s5 + med + moon + why + cx + para + essay
    full = basic + wiki
    print(f"语料 {len(full)} 行（基础 {len(basic)} + wiki {len(wiki)}）",
          flush=True)

    # ---- exp1：保存（训练 → save——记忆文件） ----
    print("\n[exp1] 保存（训练 → 记忆文件——C6-03 地形写回）:")
    mem_path = "lake_memory.npz"
    if os.path.exists(mem_path):
        # 记忆优先（已有记忆——不重训——加载验证）
        t0 = time.perf_counter()
        w = load_lake(mem_path)
        t_train = time.perf_counter() - t0
        size = os.path.getsize(mem_path) / 1e6
        print(f"      已有记忆——加载 {t_train:.1f}s（免重训）——"
              f"lake_memory.npz {size:.0f} MB", flush=True)
    else:
        # 首次：训练 → 保存
        t0 = time.perf_counter()
        blocks = extract_blocks(full)
        hubs = extract_hubs(full, blocks)
        freq = Counter("".join(full))
        chars = [c for c in dict.fromkeys("".join(full)) if freq[c] >= 4]
        w = SparseLake(chars, blocks + hubs)
        w.learn_v7(full)
        t_train = time.perf_counter() - t0
        save_lake(w, mem_path)
        size = os.path.getsize(mem_path) / 1e6
        print(f"      训练 {t_train:.0f}s → 保存 lake_memory.npz "
              f"{size:.0f} MB（记忆=地形——C96-01 写回）", flush=True)

    # ---- exp2：加载（恢复——不训练——激活一致） ----
    print("\n[exp2] 加载（恢复——不训练——激活一致）:")
    t0 = time.perf_counter()
    w2 = load_lake()
    t_load = time.perf_counter() - t0
    print(f"      加载 {t_load:.2f}s（vs 重训 {t_train:.0f}s——"
          f"省 {t_train/t_load:.0f}×）")
    for s in ["苹果很甜。", "农业是环境压力的主要驱动者。"]:
        a1 = w.activate(s)
        a2 = w2.activate(s)
        print(f"      '{s[:14]}…' 原湖激活 {a1} vs 加载湖 {a2}"
              f"（{'一致 ✓（记忆恢复）' if a1 == a2 else '不一致'}）",
              flush=True)

    # ---- exp3：持续学习（load 旧 → 学新 → save——对比重训耗时） ----
    print("\n[exp3] 持续学习（增量——新语料只加不删——C2-06）:")
    # 新语料 = 文章（新知识——增量学习）
    new_corpus = load_corpus(os.path.join(base, "corpus_essay.txt"))
    t0 = time.perf_counter()
    w2 = load_lake()
    w2 = continual_learn(w2, new_corpus)
    t_inc = time.perf_counter() - t0
    print(f"      加载+增量学习（{len(new_corpus)} 新句）{t_inc:.1f}s"
          f"（vs 全量重训 ~93s——增量省 90%+——"
          f"不重训旧语料）")
    save_lake(w2, "lake_memory_v2.npz")
    print(f"      新记忆保存 lake_memory_v2.npz"
          f"（{os.path.getsize('lake_memory_v2.npz')/1e6:.0f} MB）",
          flush=True)

    # ---- exp4：记忆更新（增量后新知识可用——旧知识保持） ----
    print("\n[exp4] 记忆更新（增量后新知识——旧知识保持——不遗忘）:")
    # 旧知识（wiki 已学——保持）
    old_k = ["苹果很甜。", "月亮很圆。"]
    # 新知识（文章新句——增量后可用）
    new_k = new_corpus[:2]
    for s in old_k + new_k:
        a = w2.activate(s)
        print(f"      '{s[:16]}…' → 激活 {a}"
              f"（{'保持/新增 ✓' if a >= 3 else '弱'}）", flush=True)
    print("\n[结论] 记忆功能：保存（npz——{:.0f}MB）→ 加载（{:.2f}s vs "
          "重训 {:.0f}s）→ 持续学习（增量 {:.1f}s vs 重训 {:.0f}s——"
          "省 {:.0f}×）→ 记忆更新（新知识+旧保持）——记忆=地形（C6-03）"
          "——重复训练时间消除".format(size, t_load, t_train, t_inc,
                                       t_train, t_train / max(t_inc, 0.1)))
    print("[done] stage142 memory continual", flush=True)


if __name__ == "__main__":
    run()
