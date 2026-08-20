# -*- coding: utf-8 -*-
"""
M5 阶段 53：综合链路（用户："框架需要的是加法，而不是分类讨论"——组装集成——
把已有机制合成一个完整系统——端到端：读 → 理解 → 检索 → 生成）

组装（加法——不是新分类）：
  ① 时序湖（stage50-51：位置相位 + K 方向化——阅读顺序）
  ② 关系分层（stage40：isa/attr/act/cause——四河道——作为组件保留）
  ③ 因果检索（stage41："为什么"——沿 K_cause 上行）
  ④ 词组单元（stage44：K 强耦合——网络长出的词）
  ⑤ 模板槽位（stage47：X是Y/X包括X——结构）
  ⑥ 生成（stage46：词组链——组合性生成面）
端到端演示：输入句子 → 注入（时序）→ 湖激活 → 关系提取 → 词组识别 →
  "为什么"检索 → 模板匹配 → 生成回应（词组链）
"""
import os
import re
import time
from collections import Counter
import numpy as np

RNG = np.random.default_rng(53)
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

class IntegratedLake:
    """综合湖：时序 + 关系分层 + 因果 + 词组 + 模板——组装为一个系统"""
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
        self.act = np.zeros(n)
        self.pos_dist = np.zeros((n, 3))

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

    # ---- 组装功能 ----
    def words(self, th=0.02):
        """词组单元（跨句重复——网络长出的词）"""
        K = self.K[REL_IDX["isa"]]
        adj = Counter()
        for s in self._sents:
            for k in range(len(s) - 1):
                adj[s[k:k + 2]] += 1
        out = set()
        n = len(self.chars)
        for i in range(n):
            for j in range(i + 1, n):
                if K[i, j] > th:
                    ca, cb = self.chars[i], self.chars[j]
                    out.add(ca + cb if adj[ca + cb] >= adj[cb + ca] else cb + ca)
        return out

    def why(self, c, k=3):
        """'为什么'——沿 K_cause 上行（因果检索——stage41）"""
        if c not in self.ci:
            return []
        i = self.ci[c]
        col = self.K[REL_IDX["cause"]][:, i].copy()
        top = np.argsort(col)[::-1][:k]
        return [(self.chars[j], col[j]) for j in top if col[j] > 0.01]

    def relate(self, a, b):
        """关系提取（四河道强度——'农业'-'产业' 的 isa/act/cause）"""
        if a not in self.ci or b not in self.ci:
            return {}
        i, j = self.ci[a], self.ci[b]
        return {r: float(self.K[REL_IDX[r]][i, j]) for r in RELS}

    def generate_chain(self, seed, words, max_len=5):
        """生成（词组链——组合性生成面——stage46）"""
        chain = [seed]
        wpairs = Counter()
        for s in self._sents:
            seq = []
            i = 0
            while i < len(s) - 1:
                if s[i:i + 2] in words:
                    seq.append(s[i:i + 2]); i += 2
                else:
                    seq.append(s[i]); i += 1
            if i < len(s):
                seq.append(s[i])
            for a in range(len(seq)):
                for b in range(a + 1, len(seq)):
                    if seq[a] in words and seq[b] in words:
                        wpairs[(seq[a], seq[b])] += 1.0 / (b - a)
        for _ in range(max_len):
            cands = [(b, c) for (a, b), c in wpairs.items() if a == chain[-1] and b not in chain]
            cands += [(a, c) for (a, b), c in wpairs.items() if b == chain[-1] and a not in chain]
            if not cands:
                break
            chain.append(max(cands, key=lambda x: x[1])[0])
        return " → ".join(chain)


def run():
    print("=== M5 阶段 53：综合链路（加法——组装集成——端到端） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=900)
    wiki = load_corpus(os.path.join(base, "corpus_wiki_filtered.txt"), n=1800)
    sents = simple + wiki
    freq = Counter("".join(sents))
    chars = [c for c, _ in freq.most_common(300)]
    w = IntegratedLake(chars)
    w._sents = sents
    t0 = time.perf_counter()
    blocks = [wiki[i:i + 600] for i in range(0, 1800, 600)]
    for block in blocks:
        for ep in range(10):
            w.learn_epoch(simple + block)
    print(f"训练完成——{time.perf_counter()-t0:.0f}s")
    words = w.words()
    print(f"词组单元 {len(words)} 个")
    # ---- 端到端演示：理解一个句子 ----
    print("\n[端到端] 输入句子（注入→湖激活→关系→检索→生成）:")
    test_sents = ["因为下雨所以带伞", "农业属于第一级产业", "苹果很甜", "技术进步改变世界"]
    for s in test_sents:
        w.inject_sentence(s)
        print(f"\n  输入: '{s}'")
        # 关系提取（句子首尾词）
        cs = [c for c in s if c in w.ci]
        if len(cs) >= 2:
            a, b = cs[0], cs[-1]
            rels = w.relate(a, b)
            strong = {r: f"{v:.3f}" for r, v in rels.items() if v > 0.01}
            print(f"    关系（{a}-{b}）: {strong}")
        # 因果检索
        if "伞" in w.ci:
            why = w.why("伞")
            if why:
                print(f"    '为什么伞？' → {[(c, f'{v:.2f}') for c, v in why[:2]]}")
        # 词组识别 + 生成
        seq = []
        i = 0
        while i < len(s) - 1:
            if s[i:i + 2] in words:
                seq.append(s[i:i + 2]); i += 2
            else:
                seq.append(s[i]); i += 1
        if i < len(s):
            seq.append(s[i])
        wd_in = [x for x in seq if x in words]
        if wd_in:
            print(f"    词组识别: {wd_in}")
            print(f"    生成（'{wd_in[-1]}' 链）: {w.generate_chain(wd_in[-1], words)}")
    print("\n[done] stage53 integration")


if __name__ == "__main__":
    run()
