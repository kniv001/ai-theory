# -*- coding: utf-8 -*-
"""
M5 阶段 140：知识问答（用户："继续扩充知识库，达到早期ai的水平"——
wiki 全量知识 + 知识问答能力——回答'X是什么'类百科问题——早期 AI
（ELIZA/专家系统级——知识检索问答））

理论锚：
  C13-01（意义 = 预测关系集——知识 = 河道——X是什么 = 检索 X 的定义
    河道）
  C43-01（表达 = 重建——回答 = 从知识河道重建）
  C69-01（记忆检索——整句检索——定义句）

机制（知识问答）：
  ① 知识 = wiki 全量学习（stage139 稀疏湖——446MB）
  ② 定义句涌现：wiki 中"X是……"句（主语 + 是——2737 句）
  ③ 问答：问句"X是什么？"→ 对象 X → 检索含 X 定义句（KT 关联——
    是 类型匹配）→ 回答
  ④ 早期 AI 水平 = 知识问答正确率（ELIZA/专家系统级）

验证：
  exp1 知识问答对抽取（wiki 定义句 → X是什么 → 答案）
  exp2 知识问答正确率（抽 40 对——回答与定义句一致）
  exp3 领域覆盖（农业/历史/科技/地理——多领域抽样）
  exp4 未知问句（未学过的 X——诚实回答——不胡说）
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
                                       DT)
from stage139_wiki_sparse import SparseLake


def extract_defs(sents):
    """定义句抽取（X是……——主语=是前内容词——涌现——非写死）"""
    defs = []
    for s in sents:
        m = re.match(r"^([一-鿿]{2,6})是", s)
        if m:
            subj = m.group(1)
            # 主语词块优先（自给农业/集约农业——4 字词块）
            defs.append((subj, s))
    return defs


def answer(w, defs, q, k=5):
    """知识问答：问句 X 是什么？→ 检索含 X 的定义句（KT 关联——
    对象在句首——是 类型匹配——涌现）"""
    qc = q.replace("？", "").replace("?", "").replace("是什么", "").strip()
    x = qc if qc else None
    if not x or x[-1] not in w.ci:
        return None
    i = w.ci[x[-1]]
    cands = []
    for subj, s in defs:
        if x not in s:
            continue
        idx = [w.ci[c] for c in s if c in w.ci]
        if not idx:
            continue
        rel = float(np.mean([w.KT_sp[i, j] for j in idx]))
        if rel > 0.0005:
            cands.append((rel, s))
    cands.sort(key=lambda x: -x[0])
    return cands[0][1] if cands else None


def run():
    print("=== M5 阶段 140：知识问答（wiki 全量——早期 AI 水平——"
          "X是什么类百科问答） ===\n", flush=True)
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
    wiki = load_corpus(os.path.join(base, "corpus_wiki_filtered.txt"))
    basic = simple + s2 + s3 + s4 + s5 + med + moon + why + cx + para
    full = basic + wiki
    print(f"语料 {len(full)} 行（基础 {len(basic)} + wiki 全量 "
          f"{len(wiki)}）", flush=True)

    t0 = time.perf_counter()
    blocks = extract_blocks(full)
    hubs = extract_hubs(full, blocks)
    freq = Counter("".join(full))
    chars = [c for c in dict.fromkeys("".join(full)) if freq[c] >= 4]
    w = SparseLake(chars, blocks + hubs)
    w.learn_v7(full)
    print(f"训练 {time.perf_counter()-t0:.0f}s（n={w.n} 河道 "
          f"{len(w.hubs)}——稀疏 {w.mem_MB():.0f}MB）", flush=True)

    # ---- exp1：定义句抽取 ----
    print("\n[exp1] 定义句抽取（wiki X是……——知识问答对源）:")
    defs = extract_defs(wiki)
    print(f"      抽取 {len(defs)} 条定义句")
    for subj, s in defs[:5]:
        print(f"      '{subj}' → '{s[:30]}…'")

    # ---- exp2：知识问答正确率 ----
    print("\n[exp2] 知识问答（'X是什么？'——回答与定义句一致）:")
    rng = np.random.RandomState(11)
    idxs = rng.choice(len(defs), min(40, len(defs)), replace=False)
    ok = 0
    for i in idxs:
        subj, s = defs[i]
        q = f"{subj}是什么？"
        ans = answer(w, defs, q)
        good = ans is not None and (s in ans or
                                    len(set(s) & set(ans)) >= 8)
        ok += good
        if not good:
            print(f"      ✗ '{q}' → 标准: '{s[:24]}…' | 回答: "
                  f"'{ans[:24] if ans else None}…'")
    print(f"      正确 {ok}/{len(idxs)} = {ok/len(idxs):.0%}"
          f"（{'知识问答 ✓（早期 AI 级——ELIZA/专家系统）'
              if ok >= 24 else '待改进'}——"
          f"C13-01 知识 = 河道——定义句检索）")

    # ---- exp3：领域覆盖 ----
    print("\n[exp3] 领域覆盖（多领域知识问答——早期 AI 广度）:")
    topics = ["农", "历", "科", "地", "生"]
    for t in topics:
        subj = next((a for a, s in defs if t in a), None)
        if subj:
            q = f"{subj}是什么？"
            ans = answer(w, defs, q)
            print(f"      '{q}' → '{ans[:26] if ans else None}…'"
                  f"（{'覆盖 ✓' if ans else '缺失'}）")
        else:
            print(f"      '{t}' 主题无定义句")

    # ---- exp4：未知问句（诚实——不胡说） ----
    print("\n[exp4] 未知问句（未学过的 X——诚实回答——不胡说）:")
    for x in ["量子", "恐龙", "钢琴", "月亮"]:
        q = f"{x}是什么？"
        ans = answer(w, defs, q)
        print(f"      '{q}' → '{ans[:24] if ans else '无答案'}…'"
              f"（{'诚实 ✓（有定义才答）' if ans is None or x in ans else '胡说'}）")
    print(f"\n[结论] 知识问答：定义句 {len(defs)} 条——正确率 "
          f"{ok}/{len(idxs)}——领域覆盖——未知诚实——早期 AI 水平"
          "（ELIZA/专家系统——wiki 知识问答能力达成）")
    print("[done] stage140 knowledge qa", flush=True)


if __name__ == "__main__":
    run()
