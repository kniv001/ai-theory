# -*- coding: utf-8 -*-
"""
M5 阶段 99：阅读注意力（用户："完善焦点注意力后——优化阅读能力（注意力相关）"）

理论锚：
  C4-02（预测河下行需注意开闸——阅读 = 焦点随序列移动的预测流）
  C16-01（尺度递归——阅读 = 时序——stage51 位置相位）
  R126/C126-01（焦点零和分配——g 分层维持——已读内容 g 衰减——
    "忘了前面" = 根淡化——长句阅读的机制）
  C45-01（焦点与信息——阅读速率）
  C43-01（理解 = 重构——读完 = 地形重构——阅读后检索）

机制（流式阅读——焦点移动）：
  ① 逐字注入长句——当前字 = 焦点（g=1）——已读字 g 衰减（×decay——
     R126 根淡化）——未读 g 低
  ② 每步演化（预测整合——当前字与已读内容的关联——阅读理解）
  ③ 读后状态 = 全文整合（句首字激活保持 vs 衰减——遗忘曲线）

验证：
  exp1 流式阅读 vs 一次性（长句——句首字读后激活——R126 根淡化）
  exp2 阅读后回答（"农业属于什么？"→产业——读后状态检索）
  exp3 焦点衰减扫描（decay——衰减快=忘得快——阅读速率）
  exp4 阅读中的预测（当前字 → 下一字——焦点区预测）
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

READ_STEPS = 3        # 每字演化步数（预测整合）
G_BG = 0.2


def read_flow(w, sent, decay=0.85):
    """流式阅读：逐字注入——焦点=当前字——已读衰减（R126 根淡化）——
    每步演化（当前字与已读内容整合——预测流）——返回读后状态 + 焦点迹"""
    n = w.n
    z = np.zeros(n, dtype=complex)
    g = np.full(n, G_BG)
    focus_trace = []
    for pos, c in enumerate(sent):
        if c not in w.ci:
            continue
        i = w.ci[c]
        g = np.where(g > G_BG + 0.01, g * decay, g)   # 已读衰减（根淡化）
        g[i] = 1.0                                    # 当前焦点
        focus_trace.append(i)
        drive = np.zeros(n, dtype=complex)
        drive[i] += AMP_IN * np.exp(1j * (w.omega[i] * w.t + pos * np.pi / 6))
        for _ in range(READ_STEPS):
            gz = g * z
            dz = -w.gamma * z + 1j * w.omega * z
            dz += (w.KT @ gz.real + 1j * (w.KT @ gz.imag)) - z * w.rsT
            dz += drive
            z = z + dz * DT
            over = np.abs(z) > 3.0
            z[over] = z[over] / np.abs(z[over]) * 2.0
    return np.abs(z), g, focus_trace


def read_once(w, sent):
    """一次性注入（对照——无焦点移动）"""
    n = w.n
    z = np.zeros(n, dtype=complex)
    drive = np.zeros(n, dtype=complex)
    for pos, c in enumerate(sent):
        if c in w.ci:
            i = w.ci[c]
            drive[i] += AMP_IN * np.exp(1j * (w.omega[i] * w.t + pos * np.pi / 6))
    for _ in range(READ_STEPS * len(sent)):
        dz = -w.gamma * z + 1j * w.omega * z
        dz += (w.KT @ z.real + 1j * (w.KT @ z.imag)) - z * w.rsT
        dz += drive
        z = z + dz * DT
        over = np.abs(z) > 3.0
        z[over] = z[over] / np.abs(z[over]) * 2.0
    return np.abs(z)


def run():
    print("=== M5 阶段 99：阅读注意力（焦点随序列移动——C4-02/R126） ===\n")
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

    long_sent = "农业属于第一级产业，包括作物种植、畜牧、渔业养殖、林业等活动，负责主副食和经济作物供应"
    print(f"\n测试长句: {long_sent}")

    # ---- exp1：流式 vs 一次性（句首字读后激活——R126 根淡化） ----
    print("\n[exp1] 流式阅读 vs 一次性（句首'农业'读后激活——R126）:")
    z_flow, g_flow, trace = read_flow(w, long_sent)
    z_once = read_once(w, long_sent)
    for c in ["农", "业", "属", "产", "业", "食", "供", "应"]:
        if c in w.ci:
            i = w.ci[c]
            print(f"      '{c}': 流式 {z_flow[i]:.3f} (g={g_flow[i]:.2f})"
                  f" vs 一次性 {z_once[i]:.3f}")
    # 句首 vs 句尾（根淡化——R126）
    first = [w.ci[c] for c in "农业" if c in w.ci]
    last = [w.ci[c] for c in "供应" if c in w.ci]
    print(f"      句首均值 {np.mean([z_flow[i] for i in first]):.3f}"
          f" vs 句尾均值 {np.mean([z_flow[i] for i in last]):.3f}"
          f"（{'根淡化（句首弱）' if np.mean([z_flow[i] for i in first]) < np.mean([z_flow[i] for i in last]) else '句首保持'}——R126）")

    # ---- exp2：阅读后回答（焦点=主题词 → 读后状态检索） ----
    print("\n[exp2] 阅读后回答（焦点'农业' → 读后整合状态）:")
    if "农" in w.ci:
        i = w.ci["农"]
        rel = w.KT[i] + w.KT[:, i]
        score = z_flow * rel
        for h in w.hubs:
            if len(h) == 1 and h in w.ci:
                score[w.ci[h]] = 0
        top = np.argsort(score)[::-1][:5]
        print(f"      读后激活 × 农业关联: {[(w.chars[j], f'{z_flow[j]:.3f}') for j in top if z_flow[j] > 0.01]}")

    # ---- exp3：焦点衰减扫描（decay——遗忘速率） ----
    print("\n[exp3] 焦点衰减扫描（decay——阅读速率 vs 保持）:")
    for decay in [0.95, 0.85, 0.7]:
        zf, gf, _ = read_flow(w, long_sent, decay=decay)
        keep = np.mean([zf[w.ci[c]] for c in "农业" if c in w.ci])
        print(f"      decay={decay}: 句首'农业'读后激活 {keep:.3f}"
              f"（{'保持' if keep > 0.01 else '遗忘'})")

    # ---- exp4：阅读中的预测（当前字 → 下一字——焦点区） ----
    print("\n[exp4] 阅读预测（'农业属于第一级' 后 → 下一字候选）:")
    if "级" in w.ci:
        i = w.ci["级"]
        row = w.KT[i].copy()
        row[i] = 0
        for h in w.hubs:
            if len(h) == 1 and h in w.ci:
                row[w.ci[h]] = 0
        top = np.argsort(row)[::-1][:4]
        print(f"      '级' 后预测: {[(w.chars[j], f'{row[j]:.3f}') for j in top if row[j] > 0.01]}")
    print("\n[done] stage99 reading attention")


if __name__ == "__main__":
    run()
