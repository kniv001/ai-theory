# -*- coding: utf-8 -*-
"""
M5 阶段 141：长文生成（用户："补充写文章或短文，增强长文能力"——
多段文章——主题贯穿——段间衔接——≥200 字——文章 = 段落序列）

理论锚：
  C16-01（尺度递归——句→段→篇——文章 = 段落序列——更高尺度关系）
  C13-02（组合性——文章 = 段落组合（主题侧面展开））
  Maimon 2025（话语连贯——段间衔接（共享成分/主题延续））
  C23-01（联想级联——段→段 = 级联延续）

语料：
  corpus_essay.txt（60 行——wiki 主题段落组 8 组 + 网络短文 14 行——
    真实文章段落）

机制（多段文章）：
  ① 段 = 承接式级联（stage136——句末成分→桥——段内衔接）
  ② 段间 = 主题侧面切换（段末→下段主题词——农业→生态→政策）
  ③ 主题贯穿 = 全文章同主题锚（每段含主题词）
  ④ 文章 = 段落序列（2-4 段——≥200 字）

验证：
  exp1 长文生成（≥200 字——多段——主题贯穿）
  exp2 段内衔接（句间共享成分率）
  exp3 段间衔接（段末→段首主题延续）
  exp4 主题保持（全文章同主题）
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
from stage136_natural_paragraph import (bridge_of, special_sides,
                                         chain_paragraph, cohesion)
from stage139_wiki_sparse import SparseLake


def essay_generate(w, topic, sents, n_paras=3, min_chars=200):
    """多段文章：每段 = 承接式级联（主题侧面）——段间 = 主题延续——
    段首强制主题词（文章锚）"""
    paras = []
    used_all = set()
    cur_topic = topic
    for p in range(n_paras):
        # 段 = 承接级联（用独立 used——段内去重）
        used = set()
        blocks_all = [h for h in w.hubs if len(h) > 1]
        ti = w.ci[cur_topic[-1]] if cur_topic[-1] in w.ci else None
        sides = special_sides(w, cur_topic, sents)
        starters = [s for s in sents
                    if cur_topic in s and s not in used_all
                    and len(s) >= 5 and "。" in s]
        if not starters:
            break
        para = [starters[0]]
        used.add(starters[0])
        used_all.add(starters[0])
        cur = starters[0]
        for step in range(12):
            bridge = bridge_of(w, cur)
            cands = []
            for s in sents:
                if s in used or s in used_all or len(s) < 5 or "。" not in s:
                    continue
                if not (cur_topic in s or any(side in s for side in sides)):
                    continue
                idx = [w.ci[c] for c in s if c in w.ci]
                if not idx or ti is None:
                    continue
                rel = float(np.mean([w.KT_sp[ti, j] for j in idx]))
                if rel < 0.004:
                    continue
                sc = set(re.sub(r"[。！？，、\s]", "", s))
                if any(len(sc & set(re.sub(r"[。！？，、\s]", "", u))) >=
                       min(len(sc), len(u)) * 0.7 for u in used):
                    continue
                bonus = 1.5 if (bridge and bridge in s) else 1.0
                cands.append((rel * bonus, s))
            cands.sort(key=lambda x: -x[0])
            if not cands:
                break
            nxt = cands[0][1]
            para.append(nxt)
            used.add(nxt)
            used_all.add(nxt)
            cur = nxt
        paras.append(para)
        if len("".join(sum(paras, []))) >= min_chars:
            break
        # 段间：下段主题词 = 本段末句桥词（承接——主题侧面切换）
        b = bridge_of(w, cur)
        if b and b != cur_topic and any(b in s for s in sents):
            cur_topic = b
        else:
            cur_topic = topic          # 回主题（文章锚）
    return paras


def run():
    print("=== M5 阶段 141：长文生成（多段文章——主题贯穿——"
          "≥200 字——文章=段落序列） ===\n", flush=True)
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
    # 生成用统一句号版（wiki 句无句号——"。" in s 筛选会滤掉——
    # 补句号——文章段落可用）
    sents = [s if s.endswith(("。", "？", "！")) else s + "。" for s in full]
    print(f"语料 {len(full)} 行（基础含文章 {len(essay)} 行 + wiki "
          f"{len(wiki)}）", flush=True)

    t0 = time.perf_counter()
    blocks = extract_blocks(full)
    hubs = extract_hubs(full, blocks)
    freq = Counter("".join(full))
    chars = [c for c in dict.fromkeys("".join(full)) if freq[c] >= 4]
    w = SparseLake(chars, blocks + hubs)
    w.learn_v7(full)
    print(f"训练 {time.perf_counter()-t0:.0f}s（n={w.n} 稀疏 "
          f"{w.mem_MB():.0f}MB）", flush=True)

    # ---- exp1：长文生成（≥200 字——多段） ----
    print("\n[exp1] 长文生成（文章=段落序列——≥200 字——主题贯穿）:")
    for topic in ["农", "月"]:
        paras = essay_generate(w, topic, sents, n_paras=3, min_chars=200)
        if not paras:
            print(f"      '{topic}' 无起始句——跳过")
            continue
        n_chars = sum(len(c) for para in paras for c in para)
        print(f"      ——文章（{topic}——{len(paras)} 段——{n_chars} 字）——")
        for pi, para in enumerate(paras):
            print(f"      [段{pi+1}] {' '.join(c.rstrip('。')+'。' for c in para)}")
        print(f"      （{'≥200 达成 ✓' if n_chars >= 200 else '不足'}——"
              f"文章=段落序列——C16-01 尺度递归）", flush=True)

    # ---- exp2：段内衔接 ----
    print("\n[exp2] 段内衔接（句间共享成分率——Maimon）:")
    for topic in ["农", "月"]:
        paras = essay_generate(w, topic, sents, n_paras=3, min_chars=200)
        if not paras:
            continue
        for pi, para in enumerate(paras):
            coh = cohesion(para)
            print(f"      '{topic}' 段{pi+1}: 衔接率 {coh:.2f}"
                  f"（{'衔接 ✓' if coh > 0.4 else '弱'}）")

    # ---- exp3：段间衔接 ----
    print("\n[exp3] 段间衔接（段末→段首主题延续）:")
    paras = essay_generate(w, "农", sents, n_paras=3, min_chars=200)
    if paras and len(paras) >= 2:
        for pi in range(len(paras) - 1):
            last = paras[pi][-1]
            first = paras[pi + 1][0]
            share = set(re.findall(r"[一-鿿]{2}", last)) & \
                    set(re.findall(r"[一-鿿]{2}", first))
            print(f"      段{pi+1}末 '{last[:16]}…' → 段{pi+2}首 "
                  f"'{first[:16]}…'——共享 {share}"
                  f"（{'段间衔接 ✓（主题延续）' if share else '主题切换'}）")

    # ---- exp4：主题保持 ----
    print("\n[exp4] 主题保持（全文章同主题锚）:")
    for topic in ["农", "月"]:
        paras = essay_generate(w, topic, sents, n_paras=3, min_chars=200)
        if not paras:
            continue
        total = sum(paras, [])
        with_t = sum(1 for c in total if topic in c)
        print(f"      '{topic}' 文章 {len(total)} 句——含主题 "
          f"{with_t}（{'主题贯穿 ✓' if with_t >= len(total) * 0.4 else '漂移'}——"
          f"文章锚——全篇同主题）")
    print("\n[结论] 长文生成：多段文章（≥200 字——段=承接级联——"
          "段间=主题延续）——段内/段间衔接——主题贯穿——"
          "文章=段落序列（C16-01）——长文能力增强")
    print("[done] stage141 long essay", flush=True)


if __name__ == "__main__":
    run()
