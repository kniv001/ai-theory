# -*- coding: utf-8 -*-
"""
M5 阶段 82：短句重复训练（用户："短句可以重复训练——日常以短句交流为主"）
+ 句界自发收编（hardcoded_audit 待办⑤——STOP_CHARS→预测骤降——A0）

理论锚：
  C97-01（缓存-写回——间隔优于集中——"集中复习的问题 = S 满时写回饱和"）
    ——短句重复 = 多天分布（日常每天都交流短句——间隔形态）非单天连续
  C20-01（模板实例化——短句 = 模板实例——短句硬化 = 构件硬化——
    长句由短句复合——C13-02 组合性）
  A0/C1-01（句界 = 预测骤降——误差词汇——"。"后预测弱 vs 句内强——
    无 STOP_CHARS 列表——Goriely 预测误差分割）

机制：
  ① 语料构成：短句（≤8 字）占比高（日常主体——用户原则）
  ② 训练流：daily = 长句单次 + 短句 ×REP（多天分布——间隔重复）
  ③ 句界自发：字 t 后预测强度——句内强（"甜"→"很"）vs 句界弱
    （"。"→任意句首字——分布广——预测骤降）
验证：
  exp1 短句构成（语料统计——短句占比）
  exp2 短句重复 vs 单次（K 预测强度——"苹果很甜"硬化——模板强度对照）
  exp3 句界自发（预测骤降——句内 vs 句界——无 STOP_CHARS）
  exp4 生成（短句模板激活——"苹果"→"苹果很甜"——构件硬化）
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

REP = 5          # 短句每天重复次数（日常高频——间隔分布）
SHORT_LEN = 8    # 短句阈值（字）


def predict_strength(w, seq):
    """给定序列末字 → 下一个字的预测强度（K 合并河道行 top1）——
    句内强（"甜"→"很"）/句界弱（"。"→任意——骤降）"""
    if seq[-1] not in w.ci:
        return 0.0
    i = w.ci[seq[-1]]
    row = w.KT[i].copy()
    row[i] = 0
    return float(row.max())


def run():
    print("=== M5 阶段 82：短句重复训练（日常主体——构件硬化——C97-01/C20-01）"
          "+ 句界自发（预测骤降——待办⑤） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=900)
    wiki = load_corpus(os.path.join(base, "corpus_wiki_filtered.txt"), n=600)
    attr = load_corpus(os.path.join(base, "corpus_attr_cause.txt"))
    neg = load_corpus(os.path.join(base, "corpus_negation.txt"))
    isa_sents = ["苹果是水果", "香蕉是水果", "西瓜是水果", "葡萄是水果",
                 "猫是动物", "狗是动物", "鸟是动物", "鱼是动物",
                 "水是液体", "冰是固体", "雪是白色的", "天空是蓝色的",
                 "老虎是动物", "树是植物", "花是植物", "石头是固体",
                 "苹果可以吃", "水可以喝", "雨是从云落下来的",
                 "小猫吃鱼", "猫吃老鼠", "我吃苹果", "小猫吃月饼"]
    sents = simple + wiki + attr + neg + isa_sents
    shorts = [s for s in sents if len(s) <= SHORT_LEN]
    longs = [s for s in sents if len(s) > SHORT_LEN]
    print(f"语料 {len(sents)} 行——短句(≤{SHORT_LEN}字) {len(shorts)} ({len(shorts)/len(sents):.0%})"
          f" / 长句 {len(longs)}")

    # ---- exp1：短句构成（日常主体——用户原则） ----
    print(f"\n[exp1] 短句构成（日常以短句交流为主——语料短句占比 {len(shorts)/len(sents):.0%}）:")
    print(f"      短句示例: {shorts[:8]}")

    # ---- 训练流：daily = 长句单次 + 短句×REP（多天分布——间隔形态） ----
    blocks = extract_blocks(sents)
    hubs = extract_hubs(sents, blocks)
    all_hubs = blocks + hubs
    chars = list(dict.fromkeys("".join(sents)))
    print(f"\n词汇表 {len(chars)} 字 / 枢纽 {len(all_hubs)} 个")
    daily = longs + shorts * REP
    print(f"每日流 {len(daily)} 行（长句 {len(longs)} 单次 + 短句 {len(shorts)}×{REP}"
          f" 重复——日常高频——间隔分布）")

    # 对照：w_single（短句单次）vs w_rep（短句重复）
    t0 = time.perf_counter()
    w_rep = HubLake(chars, all_hubs)
    for day in range(5):
        w_rep.learn_epoch_batch(daily, B=128)
    t1 = time.perf_counter()
    print(f"重复训练完成——{t1-t0:.0f}s（5 天×每日流——短句每天 5 轮）")
    w_single = HubLake(chars, all_hubs)
    for day in range(5):
        w_single.learn_epoch_batch(longs + shorts, B=128)
    print(f"单次训练完成——{time.perf_counter()-t1:.0f}s（5 天×短句单次）")

    # ---- exp2：短句重复 vs 单次（模板硬化对照） ----
    print("\n[exp2] 短句重复 vs 单次（K 预测强度——构件硬化）:")
    for seq in ["苹果很甜", "天气变冷", "小猫吃鱼", "水可以喝"]:
        ps = predict_strength(w_single, seq)
        pr = predict_strength(w_rep, seq)
        print(f"      '{seq}' 末字预测: 单次 {ps:.4f} vs 重复 {pr:.4f}"
              f"（{'硬化 ✓' if pr > ps * 1.3 else '增强不足'}）")

    # ---- exp3：句界自发（预测骤降——无 STOP_CHARS） ----
    print("\n[exp3] 句界自发（预测骤降——A0——无 STOP_CHARS 列表）:")
    print("      句内预测（字→下一字——强）vs 句界预测（。→任意句首——弱）:")
    tests = [("苹果很甜", "甜", "句内"), ("天气变冷", "冷", "句内"),
             ("。", "。", "句界"), ("！", "！", "句界"), ("？", "？", "句界")]
    for seq, mark, kind in tests:
        st = predict_strength(w_rep, seq)
        print(f"      '{seq}' 后预测强度 {st:.4f}（{kind}——"
              f"{'强预测' if st > 0.01 else '骤降=句界 ✓'}）")

    # ---- exp4：生成（短句模板激活——构件硬化） ----
    print("\n[exp4] 生成（短句硬化 → 构件模板激活）:")
    for sd in ["苹果", "天气", "小猫"]:
        if sd[-1] in w_rep.ci:
            i = w_rep.ci[sd[-1]]
            row = w_rep.KT[i].copy()
            row[i] = 0
            top = np.argsort(row)[::-1][:3]
            print(f"      '{sd}' 后接候选: {[(w_rep.chars[j], f'{row[j]:.3f}') for j in top]}")
    print("\n[done] stage82 daily short repetition + sentence boundary")


if __name__ == "__main__":
    run()
