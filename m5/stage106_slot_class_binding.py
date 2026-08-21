# -*- coding: utf-8 -*-
"""
M5 阶段 106：槽位-词类绑定（差距④——C20-02 模板泛化=槽位↔词类子湖——
名词槽↔名词湖——模板可无限实例化）

理论锚：
  C20-02（模板泛化 = 槽位-词类子湖绑定河道（名词槽↔名词湖）→
    模板可无限实例化——open——判据：M5 仿真（槽位绑定涌现））
  C15-02（词类子湖 = 分布聚类涌现——open）
  C20-01（句子意义 = 模板实例化——supported 方向）

机制：
  ① 词类聚类（C15-02——stage49 角色聚类：K 行邻居重叠——名词类/属性类）
  ② 模板提取（[X很X]——槽位词收集——"苹果很甜" X=苹果 Y=甜）
  ③ 槽位-词类绑定（C20-02）：X 槽位填充词的聚类 = 名词类（主语位）
    Y 槽位填充词 = 属性类（补语位）——绑定从语料统计涌现
  ④ 泛化：新词（聚类=名词类）→ 填 X 槽（"香蕉很甜"——语料无但模板
    允许——无限实例化——C20-02）

验证：
  exp1 词类聚类（名词类/属性类——C15-02）
  exp2 槽位词类（X 槽 = 名词类？Y 槽 = 属性类？——绑定涌现）
  exp3 模板泛化（新名词 → X 槽——"香蕉很甜"——语料无但实例化）
  exp4 泛化边界（非名词（"很"类）→ X 槽被拒——绑定约束）
"""
import os
import re
import sys
import time
from collections import Counter, defaultdict
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("gb"):
    sys.stdout.reconfigure(encoding="utf-8")

from stage79_spontaneous_hubs import (load_corpus, extract_blocks, extract_hubs,
                                       HubLake)


def role_cluster(w, words, k=3):
    """词类聚类（C15-02——stage49：K 行邻居重叠——同类共享共现邻居）"""
    n = w.n
    neigh = []
    for c in words:
        if c[0] in w.ci:
            i = w.ci[c[0]]
            row = w.KT[i] + w.KT[:, i]
            top = set(np.argsort(row)[::-1][:15])
            neigh.append((c, top))
    # 聚类：邻居重叠（Jaccard）
    clusters = []
    used = set()
    for i, (c1, n1) in enumerate(neigh):
        if i in used:
            continue
        cl = [c1]
        used.add(i)
        for j, (c2, n2) in enumerate(neigh):
            if j in used:
                continue
            inter = len(n1 & n2)
            union = len(n1 | n2)
            if union > 0 and inter / union > 0.25:
                cl.append(c2)
                used.add(j)
        clusters.append(cl)
    return clusters


def run():
    print("=== M5 阶段 106：槽位-词类绑定（C20-02——模板泛化=槽位↔词类湖） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=300)
    simple2 = load_corpus(os.path.join(base, "corpus_simple2.txt"))
    medium = load_corpus(os.path.join(base, "corpus_medium.txt"))
    full = simple + simple2 + medium
    print(f"语料 {len(full)} 行")

    blocks = extract_blocks(full)
    hubs = extract_hubs(full, blocks)
    chars = list(dict.fromkeys("".join(full)))
    w = HubLake(chars, blocks + hubs)
    for day in range(3):
        w.learn_epoch_batch(full, B=128)
    print(f"训练完成（{w.n} 字 / {len(w.hubs)} 河道）")

    # ---- 模板提取（[X很X]——槽位词） ----
    tmpl_slots = defaultdict(list)      # 模板 → 槽位词
    for s in full:
        if "很" in s:
            i = s.index("很")
            if i > 0 and i < len(s) - 1:
                tmpl_slots[("X", "很", "Y")].append((s[:i], s[i + 1:]))
    xs = [x for x, _ in tmpl_slots.get(("X", "很", "Y"), []) if x]
    ys = [y for _, y in tmpl_slots.get(("X", "很", "Y"), []) if y]
    print(f"\n[X很X] 模板: X 槽 {len(xs)} 个 / Y 槽 {len(ys)} 个")
    print(f"      X 槽词: {xs[:12]}")
    print(f"      Y 槽词: {ys[:12]}")

    # ---- exp1：词类聚类（C15-02） ----
    print("\n[exp1] 词类聚类（K 行邻居重叠——C15-02——名词/属性分湖）:")
    test_words = ["苹果", "西瓜", "香蕉", "葡萄", "天空", "天气", "水", "柠檬",
                  "甜", "脆", "酸", "冷", "热", "蓝", "香"]
    cl = role_cluster(w, test_words)
    for i, c in enumerate(cl):
        print(f"      类{i}: {c}")

    # ---- exp2：槽位词类（绑定涌现） ----
    print("\n[exp2] 槽位-词类绑定（X 槽=名词类？Y 槽=属性类？——C20-02）:")
    x_cl = role_cluster(w, [x for x in xs[:12] if x in w.ci])
    y_cl = role_cluster(w, [y for y in ys[:12] if y in w.ci])
    x_merged = [c for cl_ in x_cl for c in cl_]
    y_merged = [c for cl_ in y_cl for c in cl_]
    print(f"      X 槽聚类: {x_merged[:12]}（{len(x_cl)} 类）")
    print(f"      Y 槽聚类: {y_merged[:12]}（{len(y_cl)} 类）")

    # ---- exp3：模板泛化（C20-02——新名词填 X 槽——无限实例化） ----
    print("\n[exp3] 模板泛化（新名词 → X 槽——语料无但模板允许）:")
    for new_word in ["香蕉", "桃子", "樱桃", "橘子", "草莓"]:
        if new_word[0] in w.ci:
            i = w.ci[new_word[0]]
            row = w.KT[i] + w.KT[:, i]
            # 与名词类（苹果）的聚类一致性（邻居重叠）
            if "苹" in w.ci:
                j = w.ci["苹"]
                overlap = len(set(np.argsort(row)[::-1][:15]) & set(np.argsort(w.KT[j] + w.KT[:, j])[::-1][:15]))
                ok = overlap >= 3
            else:
                ok = True
            # Y 槽预测（"很"后的属性——K 预测）
            y_pred = ""
            if "很" in w.ci:
                hi = w.ci["很"]
                hrow = w.KT[hi].copy()
                top = np.argsort(hrow)[::-1]
                for k in top:
                    if hrow[k] > 0.005 and w.chars[k] not in new_word:
                        y_pred = w.chars[k]
                        break
            gen = f"{new_word}很{y_pred}" if y_pred else new_word
            print(f"      '{new_word}' → 实例化 '{gen}'"
                  f"（{'名词类 ✓' if ok else '类外'}——语料{'有' if gen in full else '无'}——"
                  f"模板泛化{' ✓' if ok and y_pred else ''}）")

    # ---- exp4：泛化边界（非名词 → X 槽被拒） ----
    print("\n[exp4] 泛化边界（结构字 → X 槽被拒——绑定约束）:")
    for bad in ["很", "是", "了", "在"]:
        if bad in w.ci:
            i = w.ci[bad]
            row = w.KT[i] + w.KT[:, i]
            if "苹" in w.ci:
                j = w.ci["苹"]
                overlap = len(set(np.argsort(row)[::-1][:15]) & set(np.argsort(w.KT[j] + w.KT[:, j])[::-1][:15]))
                print(f"      '{bad}' 与名词类重叠 {overlap}（{'拒绝 ✓（绑定约束）' if overlap < 3 else '误入'}）")
    print("\n[done] stage106 slot-class binding")


if __name__ == "__main__":
    run()
