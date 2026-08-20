# -*- coding: utf-8 -*-
"""
M5 阶段 57：否定/反义（语言逻辑基础——R45 统计必然的反例打破——
框架理论 + 研究双锚）

理论锚：R45（必然性 = 统计必然极限——反例打破 → 河道重构）/ C39-01（统计必然可被反例打破）
研究锚：Cognition 2025（20 个月婴儿用负证据约束词义——"这不是 danu"→ 排除——
  否定 = 词义空间的约束）/ McDermott-Hinman 2026（否定习得 = 纯语言瓶颈）/
  Zhang & Zhou 2026（中文"不/不是"——否定词）
机制：
  ① 否定词检测（不是/没有/没/不——中文否定）
  ② 反例打破（R45）："苹果不是蔬菜" → K[苹果,蔬菜] 负向（削弱——
     与"苹果是水果"（正向——必然）对比——反例打破统计必然）
  ③ 负证据约束（Cognition）："这不是X" → X 被排除（词义边界——
     "苹果"与"蔬菜"的 isa 关联降——"苹果"的类别空间被约束）
验证：
  exp1 否定检测（不是/没有/不——否定标记）
  exp2 反例打破（否定句训练后——K[苹果,蔬菜] 显著低于正向对 K[苹果,水果]）
  exp3 负证据约束（类别空间——"苹果"的 isa 关联分布——蔬菜类被排除）
"""
import os
import re
import time
from collections import Counter
import numpy as np

RNG = np.random.default_rng(57)
DT = 0.05
GAMMA = 0.8
OMEGA_LO, OMEGA_HI = 0.5, 4.0
AMP_IN = 1.2
PULSE_STEPS = 5
EPS_K = 0.02
LAMBDA_K = 0.01
K_CAP = 0.5
NEG_WORDS = ["不是", "没有", "没", "不"]   # 否定词（中文——Zhang & Zhou）
NEG_PENALTY = 0.5      # 反例惩罚（否定对的 K 沉积系数——削弱）

def load_corpus(path, lo=3, hi=80, n=None):
    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    clean = [s for s in lines if lo <= len(s) <= hi
             and re.search(r"[一-鿿]", s) and not re.search(r"[A-Za-z]", s)]
    if n and len(clean) > n:
        clean = clean[:n]
    return clean

def neg_sentences():
    """否定句（反例——自然——"苹果不是蔬菜"类）"""
    return [
        "苹果不是蔬菜", "水不是食物", "猫不是鸟", "狗不是猫", "石头不是动物",
        "桌子不是食物", "汽车不是房子", "火不是冷的", "冰不是热的", "太阳不是月亮",
        "天空不是绿色", "大米不是水果", "牛奶不是蔬菜", "飞机不是汽车", "鱼不是鸟",
        "山不是水", "树不是花", "老虎不是狮子", "马不是牛", "鸟不是鱼",
        "苹果没有翅膀", "鱼没有腿", "石头不会跑", "树不会走路", "水没有味道",
        "太阳不冷", "冰不热", "雪不是黑色", "草不是红色", "海不是绿色",
    ]

def step_dynamics(z, omega, gamma, K, rowsum, drive, dt):
    zr, zi = z.real, z.imag
    dz = -gamma * z + 1j * omega * z
    dz += K @ zr + 1j * (K @ zi) - z * rowsum
    dz += drive
    z = z + dz * dt
    over = np.abs(z) > 3.0
    z[over] = z[over] / np.abs(z[over]) * 2.0
    return z

class NegLake:
    """否定湖：时序 + 反例打破（R45）"""
    def __init__(self, chars):
        self.chars = chars
        self.ci = {c: i for i, c in enumerate(chars)}
        n = len(chars)
        self.omega = RNG.uniform(OMEGA_LO, OMEGA_HI, n)
        self.gamma = GAMMA
        self.z = 0.1 * np.exp(1j * RNG.uniform(0, 2 * np.pi, n))
        self.t = 0.0
        self.K = np.zeros((n, n))
        self.rowsum = np.zeros(n)
        self.act = np.zeros(n)

    def step(self, drive):
        self.z = step_dynamics(self.z, self.omega, self.gamma, self.K, self.rowsum, drive, DT)
        self.t += DT
        return self.z

    def inject_sentence(self, sent):
        drive = np.zeros(len(self.chars), dtype=complex)
        for pos, c in enumerate(sent):
            if c in self.ci:
                i = self.ci[c]
                drive[i] += AMP_IN * np.exp(1j * (self.omega[i] * self.t + pos * np.pi / 6))
        for _ in range(PULSE_STEPS):
            self.step(drive)
        for _ in range(3):
            self.step(np.zeros(len(self.chars), dtype=complex))
        return np.abs(self.z)

    def learn_epoch(self, sents, negs):
        n = len(self.chars)
        for sent in sents:
            self._deposit(sent, 1.0)
        for sent in negs:
            self._deposit(sent, NEG_PENALTY)   # 反例（削弱——R45）
        self.K *= (1.0 - LAMBDA_K)
        rs = self.K.sum(axis=1)
        over = rs > K_CAP
        self.K[over] *= (K_CAP / rs[over])[:, None]
        self.rowsum = self.K.sum(axis=1)

    def _deposit(self, sent, w):
        n = len(self.chars)
        seq_idx = [self.ci[c] for c in sent if c in self.ci]
        if len(seq_idx) < 2:
            return
        amp = self.inject_sentence(sent)
        L = len(seq_idx)
        sub = np.array(seq_idx)
        A = amp[sub]
        idx = np.arange(L)
        dist_w = 1.0 / np.maximum(np.abs(idx[:, None] - idx[None, :]), 1.0)
        contrib = EPS_K * np.outer(A, A) * np.triu(dist_w, 1)
        pi, pj = np.nonzero(contrib)
        # 否定句：跨否定词的词对 → 负向修正（反例打破 R45——"苹果不是蔬菜"→
        #   "苹-蔬" K 减——Cognition 2025 负证据约束）；同侧词对 → 正常（词保持）
        if w < 1.0:
            neg_pos = sent.find("不是")
            if neg_pos < 0:
                neg_pos = sent.find("没有")
            if neg_pos < 0:
                neg_pos = sent.find("不")
            pre = [self.ci[c] for c in sent[:neg_pos] if c in self.ci]
            post = [self.ci[c] for c in sent[neg_pos + 2:] if c in self.ci]
            for a in pre:
                for b in post:
                    if a in sub and b in sub:
                        self.K[a, b] -= EPS_K * NEG_PENALTY * amp[a] * amp[b]
                        self.K[b, a] -= EPS_K * NEG_PENALTY * amp[a] * amp[b] * 0.3
            # 同侧正常沉积（"苹果"词内保持）
        self.K[sub[pi], sub[pj]] += contrib[pi, pj]
        self.K[sub[pj], sub[pi]] += contrib[pi, pj] * 0.3
        for _ in range(4):
            self.step(np.zeros(n, dtype=complex))

    def strength(self, a, b):
        if a in self.ci and b in self.ci:
            return self.K[self.ci[a], self.ci[b]]
        return 0.0


def run():
    print("=== M5 阶段 57：否定/反义（R45 反例打破——负证据约束词义） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=900)
    wiki = load_corpus(os.path.join(base, "corpus_wiki_filtered.txt"), n=1800)
    sents = simple + wiki
    negs = neg_sentences()
    print(f"语料 {len(sents)} 行 + 否定句 {len(negs)} 条")
    freq = Counter("".join(sents + negs))
    # 扩大词汇表（用户：300 字集太小——测试字被挤掉反复出现——
    # 改为语料全收录——去重全量——"苹果/蔬菜/技术"全部在表）
    chars = list(dict.fromkeys("".join(negs) + "".join(sents)))
    print(f"词汇表扩大：{len(chars)} 字（语料全收录——vs 之前 300）")
    # ---- 对照：无否定训练 vs 有否定训练 ----
    w_pos = NegLake(chars)
    for ep in range(10):
        w_pos.learn_epoch(sents, [])
    w_neg = NegLake(chars)
    for ep in range(10):
        w_neg.learn_epoch(sents, negs)
    # ---- exp1：否定检测 ----
    print("\n[exp1] 否定词检测（不是/没有/不——中文否定——Zhang & Zhou）:")
    for s in ["苹果不是蔬菜", "水没有味道", "鱼没有腿", "苹果是水果"]:
        found = [w for w in NEG_WORDS if w in s]
        print(f"      '{s}' → 否定词 {found if found else '无（正向句）'}")
    # ---- exp2：反例打破（R45） ----
    print("\n[exp2] 反例打破（否定句训练后——K 对比）:")
    pos_pairs = [("苹", "果"), ("水", "果"), ("猫", "鸟"), ("雪", "黑")]
    for a, b in pos_pairs:
        kp = w_pos.strength(a, b)
        kn = w_neg.strength(a, b)
        print(f"      '{a}{b}'（反例对——{a}不是{b}）: 无否定 {kp:.3f} vs 有否定 {kn:.3f}"
              f"（{'反例削弱 ✓' if kn < kp * 0.8 else '削弱不足'}）")
    # ---- exp3：负证据约束（词义边界——Cognition 2025） ----
    print("\n[exp3] 负证据约束（'苹果'的类别空间——蔬菜被排除）:")
    if "苹" in w_neg.ci:
        i = w_neg.ci["苹"]
        # 苹 与 果/蔬/菜 的关联
        for t in ["果", "蔬", "菜", "食"]:
            if t in w_neg.ci:
                j = w_neg.ci[t]
                kp = w_pos.K[i, j]
                kn = w_neg.K[i, j]
                print(f"      '苹'-'{t}': 无否定 {kp:.3f} vs 有否定 {kn:.3f}"
                      f"（{'排除 ✓' if kn < kp * 0.6 else '保持'}）")
    print("\n[done] stage57 negation")


if __name__ == "__main__":
    run()
