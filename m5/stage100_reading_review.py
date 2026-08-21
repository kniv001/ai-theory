# -*- coding: utf-8 -*-
"""
M5 阶段 100：阅读回视（R126 根淡化的修复——"忘了前面"=根淡化≠根删除——
重读重建快——R70；误差驱动回视——C1-01——当前字与已读内容断开→焦点回移）

stage99 验证：流式阅读句首根淡化（g 衰减——句首激活 0.005 vs 句尾 0.100）
本 stage：回视机制——焦点回移（重读前部——g 恢复——激活重建）
  ① 固定回视：读长句中段回视句首（重读——激活恢复——R70 重读重建快）
  ② 误差驱动回视：当前字与已读内容关联弱（断开——逗号后新主题/
     陌生词）→ 焦点回移（重建上下文——C1-01 误差驱动）
  ③ 读后回答：回视 vs 不回视（句首信息检索——理解质量）

验证：
  exp1 回视 vs 不回视（句首激活恢复——R70）
  exp2 误差驱动回视（断开处触发——焦点回移迹）
  exp3 读后回答（回视后——句首主题检索——理解）
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

READ_STEPS = 3
G_BG = 0.2
REVIEW_N = 3          # 回视重读的字数（前部）
ERR_TH = 0.02         # 断开阈值（当前字与已读的关联低于此 → 回视）


def read_flow(w, sent, decay=0.85, review=False, err_driven=False):
    """流式阅读（stage99）+ 回视：焦点回移（重读前部——g 恢复——R70）
    err_driven：当前字与已读焦点的关联弱 → 回视（C1-01 误差驱动）"""
    n = w.n
    z = np.zeros(n, dtype=complex)
    g = np.full(n, G_BG)
    read_idx = []
    reviews = []
    for pos, c in enumerate(sent):
        if c not in w.ci:
            continue
        i = w.ci[c]
        # 断开检测（误差驱动回视）：当前字与已读字的平均关联
        do_review = False
        if err_driven and read_idx and len(read_idx) >= 4:
            coh = np.mean([w.KT[i, j] for j in read_idx])
            if coh < ERR_TH:
                do_review = True
        elif review and pos > 10 and pos < 20 and len(read_idx) >= REVIEW_N:
            do_review = True
        drive = np.zeros(n, dtype=complex)
        if do_review:
            # 回视 = 重读前部（重新注入——激活重建——R70 重读重建快）
            for ri, j in enumerate(read_idx[:REVIEW_N]):
                drive[j] += AMP_IN * np.exp(1j * (w.omega[j] * w.t + (pos + ri) * np.pi / 6))
                g[j] = 1.0
            reviews.append(pos)
        g = np.where(g > G_BG + 0.01, g * decay, g)
        g[i] = 1.0
        read_idx.append(i)
        drive[i] += AMP_IN * np.exp(1j * (w.omega[i] * w.t + pos * np.pi / 6))
        for _ in range(READ_STEPS):
            gz = g * z
            dz = -w.gamma * z + 1j * w.omega * z
            dz += (w.KT @ gz.real + 1j * (w.KT @ gz.imag)) - z * w.rsT
            dz += drive
            z = z + dz * DT
            over = np.abs(z) > 3.0
            z[over] = z[over] / np.abs(z[over]) * 2.0
    return np.abs(z), g, reviews


def run():
    print("=== M5 阶段 100：阅读回视（R70 重读重建快——R126 修复——误差驱动） ===\n")
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
    print(f"训练完成——{time.perf_counter()-t0:.0f}s")

    long_sent = "农业属于第一级产业，包括作物种植、畜牧、渔业养殖、林业等活动，负责主副食和经济作物供应"

    # ---- exp1：回视 vs 不回视（句首激活恢复——R70） ----
    print("\n[exp1] 回视 vs 不回视（句首'农业'读后激活——R70 重读重建快）:")
    z_nr, g_nr, _ = read_flow(w, long_sent, review=False)
    z_rv, g_rv, _ = read_flow(w, long_sent, review=True)
    for c in ["农", "业"]:
        if c in w.ci:
            i = w.ci[c]
            print(f"      '{c}': 不回视 {z_nr[i]:.3f} (g={g_nr[i]:.2f})"
                  f" vs 回视 {z_rv[i]:.3f} (g={g_rv[i]:.2f})"
                  f"（{'恢复 ✓' if z_rv[i] > z_nr[i] * 1.3 else '恢复不足'}）")

    # ---- exp2：误差驱动回视（断开处——焦点回移迹） ----
    print("\n[exp2] 误差驱动回视（当前字与已读关联弱 → 回视——C1-01）:")
    z_ed, g_ed, reviews = read_flow(w, long_sent, err_driven=True)
    print(f"      回视触发位置: {reviews}")
    # 断开分析（每字的已读关联）
    print("      每字与已读的平均关联（断开处=低）:")
    read_idx = []
    for pos, c in enumerate(long_sent):
        if c not in w.ci:
            continue
        i = w.ci[c]
        if read_idx:
            coh = np.mean([w.KT[i, j] for j in read_idx])
            mark = " ←断开" if coh < ERR_TH and len(read_idx) >= 4 else ""
            if pos % 6 == 0 or mark:
                print(f"        '{c}' 关联 {coh:.4f}{mark}")
        read_idx.append(i)

    # ---- exp3：读后回答（回视 vs 不回视——句首主题检索） ----
    print("\n[exp3] 读后回答（'农业'主题——回视 vs 不回视——读后整合）:")
    for name, zz in [("不回视", z_nr), ("回视", z_rv), ("误差驱动", z_ed)]:
        if "农" in w.ci:
            i = w.ci["农"]
            rel = w.KT[i] + w.KT[:, i]
            score = zz * rel
            for j in range(w.n):
                if not re.match(r"[一-鿿]", w.chars[j]):
                    score[j] = 0
            for h in w.hubs:
                if len(h) == 1 and h in w.ci:
                    score[w.ci[h]] = 0
            top = np.argsort(score)[::-1][:4]
            print(f"      [{name}] 激活×农业关联: "
                  f"{[(w.chars[j], f'{zz[j]:.3f}') for j in top if zz[j] > 0.005]}")
    print("\n[done] stage100 reading review")


if __name__ == "__main__":
    run()
