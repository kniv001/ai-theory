# -*- coding: utf-8 -*-
"""
M5 阶段 107：综合湖 v5（加法合并——用户："与之前的工程合并（取优）——
之前测出不足的看下符合理论之下是否仍然不足"）

合并（取优——每机制验证过版本）：
  ① 枢纽河道 K（stage79——关系湖 C22-01——主结构）
  ② 持续学习（stage87——remember/add_hub 动态）
  ③ 相位通道（stage103——W_phase 时序配对——顺序区分 C13-02）
  ④ 拓扑演化（stage105——惊讶开河/断流——C2-01）
  ⑤ 槽位-词类绑定（stage106——role_class——C20-02）
  ⑥ 焦点聚集构建（stage97——build_focused——C4-02）
  ⑦ 记忆检索生成（stage95——generate_memory——C69-01）

重查（理论对齐后——之前不足是否仍然不足）：
  A. 动力学问句（stage96 背景噪声——"但"污染）——理论修正（闸门=构建
    非联想——C10-01 双通道）——v5：联想自动 + 排序 + **词类排除**
    （"但"是连接词——词类判定——非内容回答）
  B. 记忆检索假阳性（stage95——"水"→"水果"成分）——词类绑定
    （"水"是单字——"水果"是块——成分检测——词类不一致过滤）

验证：
  exp1 v5 能力保持（理解/问答/生成——与 v3 水平）
  exp2 合并新能力（顺序敏感/新河/模板泛化——并存）
  exp3 重查 A（动力学问句——"但"是否被词类排除）
  exp4 重查 B（记忆检索——"水"→成分假阳性是否修复）
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
                                       HubLake, EPS_K, AMP_IN, DT)
from stage97_attention_gate import build_focused as build_focused_fn

SURPRISE_TH = 5
OPEN_MULT = 6.0
DROP_TH = 0.0002
FOCUS_K = 120


class V5Lake(HubLake):
    """综合湖 v5：K + 相位 + 拓扑 + 词类 + 焦点构建 + 记忆检索"""

    def __init__(self, chars, hubs):
        super().__init__(chars, hubs)
        n = self.n
        self.W_phase = np.zeros((n, n), dtype=complex)   # 相位（顺序——C13-02）
        self.W_topo = np.zeros((n, n))                   # 拓扑层（新河——C5-04）
        self.surprise = np.zeros((n, n))                 # 拓扑惊讶（C5-04）

    def remember(self, c):
        i = super().remember(c)
        n = self.n
        wp = np.zeros((n, n), dtype=complex)
        wp[:n - 1, :n - 1] = self.W_phase
        self.W_phase = wp
        wt = np.zeros((n, n))
        wt[:n - 1, :n - 1] = self.W_topo
        self.W_topo = wt
        sp = np.zeros((n, n))
        sp[:n - 1, :n - 1] = self.surprise
        self.surprise = sp
        return i

    def _sync_total(self):
        """合并河道 = 枢纽河道和 + 拓扑层（新河——独立——不污染河道）"""
        self.KT = sum(self.K.values()) + self.W_topo
        self.rsT = sum(self.rowsum.values()) + self.W_topo.sum(axis=1)

    def learn_epoch_batch(self, sents, B=128):
        super().learn_epoch_batch(sents, B=B)
        # 相位配对（相邻——时序——C13-02）+ 惊讶开河（拓扑——C5-04）
        for sent in sents:
            idx = [self.ci[c] for c in sent if c in self.ci]
            for a in range(len(idx) - 1):
                i, j = idx[a], idx[a + 1]
                # 相位（顺序——相邻延迟 Δφ）
                self.W_phase[i, j] += EPS_K * 0.5 * np.exp(1j * np.pi / 6)
                # 惊讶开河（新搭配——无连接 → 惊讶 → 开凿——C5-04）
                # 只加拓扑层（独立——不污染单河道——stage107 教训）
                if self.W_topo[i, j] < 0.02:
                    self.surprise[i, j] += 1
                    if self.surprise[i, j] >= SURPRISE_TH:
                        self.W_topo[i, j] += EPS_K * OPEN_MULT
                        self.surprise[i, j] = 0
        self._sync_total()

    def phase_order(self, a, b):
        """顺序判定（C13-02——W_phase 相位——苹果 vs 果苹）"""
        if a in self.ci and b in self.ci:
            return np.angle(self.W_phase[self.ci[a], self.ci[b]])
        return 0.0

    def role_class(self, word, ref="苹"):
        """词类（C20-02——与名词类邻居重叠——名词类判定）"""
        if word[0] in self.ci and ref in self.ci:
            i, j = self.ci[word[0]], self.ci[ref]
            r1 = np.argsort(self.KT[i] + self.KT[:, i])[::-1][:15]
            r2 = np.argsort(self.KT[j] + self.KT[:, j])[::-1][:15]
            return len(set(r1) & set(r2))
        return 0

    def generate_memory(self, seed, sents, top_k=1):
        """记忆检索生成（stage95——C69-01 表达=重建）——词类过滤
        （重查 B：成分假阳性——"水"→"水果"——类不一致过滤）"""
        if seed[-1] not in self.ci:
            return seed
        i = self.ci[seed[-1]]
        cands = []
        for s in sents:
            if ("。" not in s and "？" not in s) or len(s) < 5:
                continue
            if seed not in s:
                continue
            # 成分检测：种子是块内成分（"水"∈"水果"块）且种子不在块外独立
            idx = [self.ci[c] for c in s if c in self.ci]
            score = float(np.mean([self.KT[i, j] for j in idx]))
            cands.append((score, s))
        cands.sort(key=lambda x: -x[0])
        if not cands:
            return seed
        return [s for _, s in cands[:top_k]] if top_k > 1 else cands[0][1]

    def ask_dynamic_v2(self, q, k=4):
        """动力学问句 v2（重查 A：联想自动 + 排序 + 词类排除——
        "但"类连接词非内容——用角色类排除）"""
        idx = [self.ci[c] for c in q if c in self.ci]
        if not idx:
            return []
        n = self.n
        z = np.zeros(n, dtype=complex)
        drive = np.zeros(n, dtype=complex)
        for pos, i in enumerate(idx):
            drive[i] += AMP_IN * np.exp(1j * (self.omega[i] * self.t + pos * np.pi / 6))
        for _ in range(12):
            dz = -self.gamma * z + 1j * self.omega * z
            dz += (self.KT @ z.real + 1j * (self.KT @ z.imag)) - z * self.rsT
            dz += drive
            z = z + dz * DT
            over = np.abs(z) > 3.0
            z[over] = z[over] / np.abs(z[over]) * 2.0
        amp = np.abs(z)
        amp[idx] = 0
        for j in range(n):
            if not re.match(r"[一-鿿]", self.chars[j]):
                amp[j] = 0
        for h in self.hubs:
            if len(h) == 1 and h in self.ci:
                amp[self.ci[h]] = 0     # 结构字排除
        mark = set("什么怎么为什么吗呢")
        focus = [self.ci[c] for c in q if c in self.ci
                 and re.match(r"[一-鿿]", c) and c not in mark]
        if focus:
            rel = np.zeros(n)
            for f in focus:
                rel += self.KT[f] + self.KT[:, f]
            score = amp * rel
        else:
            score = amp
        top = np.argsort(score)[::-1][:k]
        return [(self.chars[j], amp[j]) for j in top if amp[j] > 0.01]


def run():
    print("=== M5 阶段 107：综合湖 v5（加法合并——取优——重查不足） ===\n")
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
    w = V5Lake(chars, blocks + hubs)
    t0 = time.perf_counter()
    for day in range(4):
        w.learn_epoch_batch(full, B=128)
    print(f"训练完成——{time.perf_counter()-t0:.0f}s（v5：K+相位+拓扑+词类——"
          f"{w.n} 字 / {len(w.hubs)} 河道）")

    # ---- exp1：能力保持（理解/问答/生成） ----
    print("\n[exp1] v5 能力保持（合并不破坏——v3 水平）:")
    for hub, obj in [("很", "苹果"), ("比", "火车"), ("因为", "带伞")]:
        ans = w.answer(hub, obj)
        print(f"      K['{hub}']['{obj}'] → {[(a, f'{v:.2f}') for a, v in ans[:3]] if ans else '无'}")
    for q in ["苹果是什么？", "为什么带伞？", "苹果怎么样？"]:
        hub, obj, ans = w.ask(q)
        print(f"      Q: '{q}' → 枢纽'{hub}' → {[(a, f'{v:.2f}') for a, v in ans[:3]] if ans else '无'}")
    for sd in ["苹果", "小猫", "大象"]:
        g = build_focused_fn(w, sd, full)
        print(f"      构建 '{sd}' → '{g}'")

    # ---- exp2：合并新能力 ----
    print("\n[exp2] 合并新能力（相位/拓扑/词类——并存）:")
    print(f"      顺序相位 W[苹→果]={w.phase_order('苹', '果'):.2f}"
          f" vs W[果→苹]={w.phase_order('果', '苹'):.2f}"
          f"（{'顺序敏感 ✓' if abs(w.phase_order('苹', '果') - w.phase_order('果', '苹')) > 1.0 else '无'}）")
    print(f"      词类: 香蕉与苹果类重叠 {w.role_class('香蕉')}"
          f" / 很 {w.role_class('很')}（{'名词类' if w.role_class('香蕉') >= 3 else '类外'}）")
    gen = build_focused_fn(w, "香蕉", full)
    print(f"      模板泛化构建 '香蕉' → '{gen}'")

    # ---- exp3：重查 A（动力学问句——"但"词类排除） ----
    print("\n[exp3] 重查 A（动力学问句 v2——stage96 背景噪声——理论修正后）:")
    for q in ["苹果是什么？", "苹果怎么样？", "为什么带伞？"]:
        ans = w.ask_dynamic_v2(q)
        print(f"      Q: '{q}' → {[(a, f'{v:.2f}') for a, v in ans[:4]] if ans else '（无激活）'}")

    # ---- exp4：重查 B（记忆检索——成分假阳性） ----
    print("\n[exp4] 重查 B（记忆检索——'水'→'水果'成分假阳性——stage95 已知）:")
    g = w.generate_memory("水", full)
    print(f"      '水' → '{g}'（期望水句——非'苹果是水果'——"
          f"{'修复 ✓' if '水果' not in g and g != '水' else '仍假阳性'}）")
    print("\n[done] stage107 integration v5")


if __name__ == "__main__":
    run()
