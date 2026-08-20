# -*- coding: utf-8 -*-
"""
M5 阶段 61：问答（"是什么"检索——与"为什么"对称——理解功能完整）

理论锚：C74-01（"为什么" = 不确定性→确定性的转化请求——"是什么"对称——
  isa 检索——类别确定）/ C13-01（意义 = 预测关系集——"是什么" = 关系集检索）
研究锚：儿童问句习得（what 问句先于 why——问句的认知基础——不确定性→确定性）
机制（检索式回答——不是生成式编造——GOAL 可解释）：
  ① "是什么"：isa 河道检索（"苹果是什么？" → K_isa 的强关联——类别——"水果"）
  ② "为什么"：cause 河道上行检索（"为什么带伞？" → 原因——"下雨"——stage41 已有）
  ③ "怎么样"：attr 河道检索（"苹果怎么样？" → 属性——"甜"）
  ④ 组合："是什么" + "怎么样"（苹果：是水果——甜）——完整理解
验证：
  exp1 "是什么"（isa 检索——类别）
  exp2 "为什么"（cause 上行——原因——与 stage41 一致）
  exp3 "怎么样"（attr 检索——属性）
  exp4 完整理解（多维度检索——苹果：类别+属性+因果）
"""
import os
import re
import time
from collections import Counter
import numpy as np

RNG = np.random.default_rng(61)
DT = 0.05
GAMMA = 0.8
OMEGA_LO, OMEGA_HI = 0.5, 4.0
AMP_IN = 1.2
PULSE_STEPS = 5
EPS_K = 0.02
LAMBDA_K = 0.01
K_CAP = 0.5
DELTA_PHI = np.pi / 6
RELS = ["isa", "attr", "act", "cause"]
REL_IDX = {r: i for i, r in enumerate(RELS)}

def load_corpus(path, lo=3, hi=80, n=None):
    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    clean = [s for s in lines if lo <= len(s) <= hi
             and re.search(r"[一-鿿]", s) and not re.search(r"[A-Za-z]", s)]
    if n and len(clean) > n:
        clean = clean[:n]
    return clean

def step_dynamics(z, omega, gamma, K3, rowsum, drive, dt):
    zr, zi = z.real, z.imag
    dz = -gamma * z + 1j * omega * z
    for r in range(K3.shape[0]):
        Kr = K3[r]
        dz += Kr @ zr + 1j * (Kr @ zi) - z * rowsum[r]
    dz += drive
    z = z + dz * dt
    over = np.abs(z) > 3.0
    z[over] = z[over] / np.abs(z[over]) * 2.0
    return z

class QALake:
    """问答湖：四河道（isa/attr/act/cause）——检索式回答"""
    def __init__(self, chars):
        self.chars = chars
        self.ci = {c: i for i, c in enumerate(chars)}
        n = len(chars)
        self.omega = RNG.uniform(OMEGA_LO, OMEGA_HI, n)
        self.gamma = GAMMA
        self.z = 0.1 * np.exp(1j * RNG.uniform(0, 2 * np.pi, n))
        self.t = 0.0
        self.K = np.zeros((len(RELS), n, n))
        self.rowsum = np.zeros((len(RELS), n))

    def step(self, drive):
        self.z = step_dynamics(self.z, self.omega, self.gamma, self.K, self.rowsum, drive, DT)
        self.t += DT
        return self.z

    def inject_sentence(self, sent):
        drive = np.zeros(len(self.chars), dtype=complex)
        for pos, c in enumerate(sent):
            if c in self.ci:
                i = self.ci[c]
                drive[i] += AMP_IN * np.exp(1j * (self.omega[i] * self.t + pos * DELTA_PHI))
        for _ in range(PULSE_STEPS):
            self.step(drive)
        for _ in range(3):
            self.step(np.zeros(len(self.chars), dtype=complex))
        return np.abs(self.z)

    def learn_epoch(self, sents):
        n = len(self.chars)
        for sent in sents:
            rel = "isa"
            if "因为" in sent or "所以" in sent:
                rel = "cause"
            elif "很" in sent:
                rel = "attr"
            seq_idx = [self.ci[c] for c in sent if c in self.ci]
            if len(seq_idx) < 2:
                continue
            amp = self.inject_sentence(sent)
            L = len(seq_idx)
            sub = np.array(seq_idx)
            A = amp[sub]
            idx = np.arange(L)
            dist_w = 1.0 / np.maximum(np.abs(idx[:, None] - idx[None, :]), 1.0)
            contrib = EPS_K * np.outer(A, A) * np.triu(dist_w, 1)
            pi, pj = np.nonzero(contrib)
            self.K[REL_IDX[rel]][sub[pi], sub[pj]] += contrib[pi, pj]
            self.K[REL_IDX[rel]][sub[pj], sub[pi]] += contrib[pi, pj] * 0.3
            for _ in range(4):
                self.step(np.zeros(n, dtype=complex))
        for r in RELS:
            self.K[REL_IDX[r]] *= (1.0 - LAMBDA_K)
            rs = self.K[REL_IDX[r]].sum(axis=1)
            over = rs > K_CAP
            self.K[REL_IDX[r]][over] *= (K_CAP / rs[over])[:, None]
            self.K[REL_IDX[r]][:, over] *= (K_CAP / rs[over])[None, :]
        self.rowsum = self.K.sum(axis=2)


    def learn_epoch_batch(self, sents, B=64):
        """批量注入（stage62——27-49x 加速——空间换时间）"""
        n = len(self.chars)
        for start in range(0, len(sents), B):
            batch = sents[start:start + B]
            Z = np.zeros((len(batch), n), dtype=complex)
            drives = np.zeros((len(batch), n), dtype=complex)
            seqs = []
            for bi, sent in enumerate(batch):
                rel = "isa"
                if "因为" in sent or "所以" in sent:
                    rel = "cause"
                elif "很" in sent:
                    rel = "attr"
                idx = [self.ci[c] for c in sent if c in self.ci]
                if len(idx) < 2:
                    continue
                seqs.append((bi, idx, rel))
                for pos, i in enumerate(idx):
                    drives[bi, i] += AMP_IN * np.exp(1j * (self.omega[i] * self.t + pos * DELTA_PHI))
            for _ in range(PULSE_STEPS + 3):
                dz = -self.gamma * Z + 1j * self.omega * Z
                for r in range(len(RELS)):
                    dz += (self.K[REL_IDX[RELS[r]]] @ Z.T).T - Z * self.rowsum[REL_IDX[RELS[r]]]
                dz += drives
                Z = Z + dz * DT
                over = np.abs(Z) > 3.0
                Z[over] = Z[over] / np.abs(Z[over]) * 2.0
            amp = np.abs(Z)
            for bi, idx, rel in seqs:
                sub = np.array(idx)
                A = amp[bi, sub]
                L = len(idx)
                d_idx = np.arange(L)
                dist_w = 1.0 / np.maximum(np.abs(d_idx[:, None] - d_idx[None, :]), 1.0)
                contrib = EPS_K * np.outer(A, A) * np.triu(dist_w, 1)
                pi, pj = np.nonzero(contrib)
                self.K[REL_IDX[rel]][sub[pi], sub[pj]] += contrib[pi, pj]
                self.K[REL_IDX[rel]][sub[pj], sub[pi]] += contrib[pi, pj] * 0.3
        for r in RELS:
            self.K[REL_IDX[r]] *= (1.0 - LAMBDA_K)
            rs = self.K[REL_IDX[r]].sum(axis=1)
            over = rs > K_CAP
            self.K[REL_IDX[r]][over] *= (K_CAP / rs[over])[:, None]
            self.K[REL_IDX[r]][:, over] *= (K_CAP / rs[over])[None, :]
        self.rowsum = self.K.sum(axis=2)

    def answer(self, rel, c, k=3):
        """检索式回答（沿河道——isa:类别/cause:原因上行/attr:属性/act:动作）"""
        if c not in self.ci:
            return []
        i = self.ci[c]
        if rel == "cause":
            col = self.K[REL_IDX["cause"]][:, i].copy()   # 上行（原因）
            top = np.argsort(col)[::-1][:k]
            return [(self.chars[j], col[j]) for j in top if col[j] > 0.003]
        row = self.K[REL_IDX[rel]][i].copy()
        top = np.argsort(row)[::-1][:k]
        return [(self.chars[j], row[j]) for j in top if row[j] > 0.004]

    def ask(self, q):
        """问句 → 检索（"苹果是什么？" → 提取对象词 → 对应河道检索）"""
        # 简单问句解析（是什么/为什么/怎么样）
        if "为什么" in q:
            obj = self._obj(q, ["为什么"])
            return "为什么", obj, self.answer("cause", obj)
        if "是什么" in q or "是啥" in q:
            obj = self._obj(q, ["是什么", "是啥"])
            return "是什么", obj, self.answer("isa", obj)
        if "怎么样" in q or "如何" in q:
            obj = self._obj(q, ["怎么样", "如何"])
            return "怎么样", obj, self.answer("attr", obj)
        return None, None, []

    def _obj(self, q, markers):
        for m in markers:
            q = q.replace(m, "")
        # 取剩余部分的长词（>1 字优先——"苹果"）
        cs = [c for c in q if c in self.ci]
        if not cs:
            return None
        # 简单：取最后 2 字（词优先）——"苹果" 或 最后 1 字
        if len(cs) >= 2:
            two = cs[-2] + cs[-1]
            if all(c in self.ci for c in two):
                return two
        return cs[-1]


def run():
    print("=== M5 阶段 61：问答（是什么检索——与为什么对称——理解完整） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=900)
    wiki = load_corpus(os.path.join(base, "corpus_wiki_filtered.txt"), n=1800)
    attr = load_corpus(os.path.join(base, "corpus_attr_cause.txt"))
    sents = simple + wiki + attr
    isa_sents = ["苹果是水果", "香蕉是水果", "西瓜是水果", "葡萄是水果",
                 "猫是动物", "狗是动物", "鸟是动物", "鱼是动物",
                 "水是液体", "冰是固体", "雪是白色的", "天空是蓝色的",
                 "老虎是动物", "树是植物", "花是植物", "石头是固体",
                 "苹果可以吃", "水可以喝", "雨是从云落下来的"]
    sents = sents + isa_sents
    chars = list(dict.fromkeys("".join(sents)))
    print(f"词汇表 {len(chars)} 字 / 语料 {len(sents)} 行")
    w = QALake(chars)
    t0 = time.perf_counter()
    blocks = [wiki[i:i + 600] for i in range(0, 1800, 600)]
    for block in blocks:
        for ep in range(8):
            w.learn_epoch_batch(simple + attr + block + isa_sents, B=128)
    print(f"训练完成——{time.perf_counter()-t0:.0f}s")
    # ---- 问答演示 ----
    print("\n[问答] 检索式回答（不是生成式编造——可审计）:")
    questions = ["苹果是什么？", "为什么带伞？", "苹果怎么样？", "技术是什么？",
                 "为什么下雨带伞？", "雪怎么样？"]
    for q in questions:
        kind, obj, ans = w.ask(q)
        if ans:
            print(f"      Q: '{q}'")
            print(f"      A: '{obj}'{kind} → {[(a, f'{v:.2f}') for a, v in ans[:3]]}")
        else:
            print(f"      Q: '{q}' → （检索无结果——语料覆盖不足）")
    print("\n[done] stage61 qa")


if __name__ == "__main__":
    run()
