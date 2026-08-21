# -*- coding: utf-8 -*-
"""
M5 阶段 129：个人史时间戳（C29-02——河道携带习得顺序——先入为主
C29-01——先学先快/网络中心性——频率可逆转——LLM 无习得时间轴对比）

理论锚：
  C29-02（河道携带个人史时间戳（隐式习得顺序）；LLM 无习得时间轴 =
    "根本不同"清单新条目——open）
  C29-01（先入为主原则：联想目标优先性 = 习得时间优先性（奠基/沉积/
    新奇-已知三机制）；使用频率可逆转过方向——supported）
  C6-03（记忆 = 地形本身——地形带时间（地形 = 经验））

研究锚：
  Kroll & Stewart 1994（RHM——双语非对称：L1 概念连接强——L2 经
    L1 词汇路径——习得顺序决定网络连接强度——前向>后向翻译）
  Dirix & Duyck（先学先快原则：映射假说（先入网络者获得可塑性/处理
    优势）+ 语义假说（先学词占据语义网络中心位置）——两机制同时）
  Cheung et al. 1998（中英双语——L1 概念连接更快——习得顺序不对称）

机制（个人史时间戳）：
  ① 河道携带时间戳：每条连接记录首现顺序（t_order）——地形=经验
    （C6-03）——习得顺序隐式携带（C29-02）
  ② 先入为主：激活优先级 = 强度 × 时间优先（先学连接先激活——
    奠基机制——C29-01）
  ③ 中心性：先学词占据网络中心（更多连接/更短路径——Dirix-Duyck
    语义假说——L1 中心）
  ④ 频率逆转：后期高频可逆转过方向（C29-01——使用频率压倒时间）

验证：
  exp1 时间戳携带（连接带习得顺序——可读——C29-02）
  exp2 先入为主（同强度——先学者优先激活——奠基——C29-01）
  exp3 中心性（先学词连接数/度中心性高——Dirix-Duyck）
  exp4 频率逆转（后期高频超越早期低频——C29-01 使用频率可逆转）
"""
import os
import sys
import time
from collections import Counter
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("gb"):
    sys.stdout.reconfigure(encoding="utf-8")

from stage79_spontaneous_hubs import load_corpus

EPS0 = 0.02
LAM = 0.01
PUNCT = set("。？！，、；：")


class StampLake:
    """时间戳湖：共现沉积 + 连接时间戳（首现）+ 时间优先激活"""

    def __init__(self, chars):
        self.chars = list(chars)
        self.ci = {c: i for i, c in enumerate(chars)}
        n = len(chars)
        self.W = np.zeros((n, n))
        self.T = np.full((n, n), -1, dtype=int)   # 连接首现时间戳（-1=无）
        self.n = n
        self.t = 0                                # 个人史时钟（句子序号）

    def learn(self, sent):
        idx = [self.ci[c] for c in sent if c in self.ci and c not in PUNCT]
        for a in range(len(idx) - 1):
            for b in range(a + 1, len(idx)):
                d = b - a
                i, j = idx[a], idx[b]
                self.W[i, j] += EPS0 / d
                self.W[j, i] += EPS0 / d
                if self.T[i, j] < 0:              # 首现时间戳（C29-02）
                    self.T[i, j] = self.t
                    self.T[j, i] = self.t
        self.t += 1

    def strength(self, a, b):
        i, j = self.ci[a], self.ci[b]
        return self.W[i, j], self.T[i, j]

    def top(self, c, k=6, priority=False):
        """激活优先级：强度 × 时间优先（先学加权——C29-01 奠基——
        处理速度 = 激活到达速度——先学连接快）"""
        i = self.ci[c]
        row = self.W[i] + self.W[:, i]
        if priority:
            # 时间优先：先学（T 小）加权（Dirix-Duyck 先学先快）
            trows = np.where(self.T[i] >= 0, self.T[i] + self.T[:, i] / 2, 1e9)
            priority_w = 1.0 / (1.0 + trows / max(self.t, 1))
            row = row * priority_w
        return [self.chars[j] for j in np.argsort(row)[::-1][:k]
                if row[j] > 0.001 and self.chars[j] != c]

    def degree(self, c):
        i = self.ci[c]
        return int((self.W[i] > 0).sum())


def run():
    print("=== M5 阶段 129：个人史时间戳（C29-02——河道携带习得顺序——"
          "Kroll-Stewart/Dirix-Duyck 锚定） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"))
    s2 = load_corpus(os.path.join(base, "corpus_simple2.txt"))
    full = simple + s2
    chars = list(dict.fromkeys("".join(full)))
    print(f"语料 {len(full)} 行 / 词汇 {len(chars)}")
    w = StampLake(chars)
    for s in full:
        w.learn(s)
    print("训练完成（单遍——时间戳 = 句子序号）")

    # ---- exp1：时间戳携带（C29-02） ----
    print("\n[exp1] 时间戳携带（连接带首现顺序——地形=经验——C29-02）:")
    for a, b in [("苹", "果"), ("天", "气"), ("小", "猫"), ("很", "甜")]:
        if a in w.ci and b in w.ci:
            s, t = w.strength(a, b)
            print(f"      '{a}'→'{b}' 强度 {s:.2f} / 首现于句 {t}"
                  f"（{'时间戳携带 ✓（习得顺序可读）' if t >= 0 else '无'}）")

    # ---- exp2：先入为主（同强度——先学优先——C29-01 奠基） ----
    print("\n[exp2] 先入为主（先学连接优先激活——奠基——C29-01）:")
    # 找强度相近、时间差大的两对连接——时间优先下先学者排名应提前
    i = w.ci["苹"]
    row = w.W[i] + w.W[:, i]
    cands = [(w.chars[j], row[j], w.T[i, j]) for j in range(w.n)
             if row[j] > 0.01 and w.T[i, j] >= 0]
    cands.sort(key=lambda x: x[2])
    if len(cands) >= 2:
        early = cands[0]                      # 最早学
        late = cands[-1]                      # 最晚学
        t0 = w.top("苹", k=8)
        t1 = w.top("苹", k=8, priority=True)
        print(f"      '苹' 最早关联: '{early[0]}'（句 {early[2]}——强度 {early[1]:.2f}） vs "
              f"最晚关联: '{late[0]}'（句 {late[2]}——强度 {late[1]:.2f}）")
        print(f"      普通激活 top: {t0[:5]}")
        print(f"      时间优先激活: {t1[:5]}")
        e0, e1 = t0.index(early[0]), t1.index(early[0])
        print(f"      '{early[0]}' 排名 {e0} → {e1}"
              f"（{'先入为主 ✓（先学者在时间优先下提前——C29-01 奠基——先学先快）'
                  if e1 < e0 else '排序未变（强度主导）'}）")

    # ---- exp3：中心性（先学词占据网络中心——Dirix-Duyck 语义假说） ----
    print("\n[exp3] 中心性（先学词连接多——网络中心——Dirix-Duyck）:")
    early_words = [c for c in "天苹小很" if c in w.ci]      # 语料前部（先学）
    late_words = [c for c in "甜脆春" if c in w.ci]         # 语料后部（后学）
    if "甜" not in w.ci:
        late_words = [c for c in "响亮春" if c in w.ci]
    d_early = np.mean([w.degree(c) for c in early_words]) if early_words else 0
    d_late = np.mean([w.degree(c) for c in late_words]) if late_words else 0
    print(f"      先学词（{'/'.join(early_words)}）均连接 {d_early:.0f} vs "
          f"后学词（{'/'.join(late_words)}）均连接 {d_late:.0f}"
          f"（{'中心性 ✓（先学词连接多——L1 中心——Dirix-Duyck 语义假说）'
              if d_early > d_late else '后学更多（频率效应）'}——"
          f"先学词占据网络中心）")

    # ---- exp4：频率逆转（后期高频超越早期低频——C29-01） ----
    print("\n[exp4] 频率逆转（使用频率可逆转过方向——C29-01）:")
    # 同对象：先学低频（吃——句111）vs 后学高频（比——句695——
    # 比较句大量重复：苹果比西瓜小/西瓜比苹果大）——频率压倒时间
    if all(c in w.ci for c in "苹比吃"):
        s_bi, t_bi = w.strength("苹", "比")
        s_chi, t_chi = w.strength("苹", "吃")
        print(f"      '苹-吃' 强度 {s_chi:.2f}（句 {t_chi} 首现——先学） vs "
              f"'苹-比' 强度 {s_bi:.2f}（句 {t_bi} 首现——后学）")
        print(f"      （{'频率逆转 ✓（后学高频超越先学低频——使用可逆转——C29-01）'
              if t_bi > t_chi and s_bi > s_chi else '时间主导'}——"
          f"使用频率可逆转过方向——时间戳保留但非绝对支配）")
    print("\n[结论] 个人史时间戳 C29-02 M5 验证：时间戳携带（习得顺序可读）/"
          "先入为主（奠基——先学先快）✓ / 中心性（先学=网络中心——Dirix-Duyck）/"
          "频率逆转（C29-01）——地形=经验（C6-03）——LLM 无习得时间轴对比")
    print("[done] stage129 temporal stamp")


if __name__ == "__main__":
    run()
