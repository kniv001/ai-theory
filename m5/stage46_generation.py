# -*- coding: utf-8 -*-
"""
M5 阶段 46：句子生成（组合性生成面——预测河下行 R2——用户："句子的生成应该可以继续了"）

基础：stage44/45（词组涌现 + 主题分批课程——98 词组）
生成机制（框架：生成 = 预测的重复执行——R2 预测河下行——C13-02 吸引子复合）：
  ① 语料 bigram 方向统计（"农业"→"属" 次数——语料顺序——C122-01 关系统计）
  ② 生成 = 逐字预测：给定最后 2 字 → 候选下一字（bigram 方向 × K 关联强度）
     ——bigram 稀疏时回退 K（强关联）
  ③ 停止条件：候选弱（K < 阈值）/ 长度上限 / 终止词（了/的/。等）
验证：
  exp1 逐字生成：起始词 → 预测链（"农业"→？——能否生成合理句）
  exp2 词组级生成：起始词组 → 词组链
  exp3 合理性：生成的句 vs 语料句（人工判定——与语料句的相似度）
"""
import os
import re
from collections import Counter
import numpy as np

RNG = np.random.default_rng(46)
DT = 0.05
GAMMA = 0.8
OMEGA_LO, OMEGA_HI = 0.5, 4.0
AMP_IN = 1.2
PULSE_STEPS = 5
EPS_K = 0.02
LAMBDA_K = 0.01
ETA_OMEGA = 0.15
K_CAP = 0.5
N_CHAR = 300
RELS = ["isa", "attr", "act", "cause"]
REL_IDX = {r: i for i, r in enumerate(RELS)}
MAX_LEN = 20          # 生成长度上限
STOP_CHARS = set("。！？吗呢吧啊")   # 功能字（的/了）是连接词非句尾——去掉（断链根因）

def rel_of(sent):
    if "因为" in sent or "所以" in sent:
        return "cause"
    if any(w in sent for w in ["是", "属于", "包括", "包含"]):
        return "isa"
    if "很" in sent:
        return "attr"
    return None

def load_corpus(path, lo=3, hi=80, n=None):
    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    # 含中文且不含拉丁字母（"人口为"BMa..." 类污染排除）
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

class RelLake:
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

    def step(self, drive):
        self.z = step_dynamics(self.z, self.omega, self.gamma, self.K, self.rowsum, drive, DT)
        self.t += DT
        return self.z

    def inject_sentence(self, sent):
        drive = np.zeros(len(self.chars), dtype=complex)
        for c in sent:
            if c in self.ci:
                i = self.ci[c]
                drive[i] += AMP_IN * np.exp(1j * (self.omega[i] * self.t))
        for _ in range(PULSE_STEPS):
            self.step(drive)
        for _ in range(3):
            self.step(np.zeros(len(self.chars), dtype=complex))
        return np.abs(self.z)

    def learn_epoch(self, sents):
        n = len(self.chars)
        for sent in sents:
            rel = rel_of(sent)
            if rel is None:
                continue
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
            self.K[REL_IDX[rel]][sub[pj], sub[pi]] += contrib[pi, pj]
            for _ in range(4):
                self.step(np.zeros(n, dtype=complex))
        for r in RELS:
            self.K[REL_IDX[r]] *= (1.0 - LAMBDA_K)
            row_sum = self.K[REL_IDX[r]].sum(axis=1)
            over = row_sum > K_CAP
            self.K[REL_IDX[r]][over] *= (K_CAP / row_sum[over])[:, None]
            self.K[REL_IDX[r]][:, over] *= (K_CAP / row_sum[over])[None, :]
        self.rowsum = self.K.sum(axis=2)
        Ksum = self.K.sum(axis=0)
        mask = Ksum > 0.08
        dw = ETA_OMEGA * (self.omega[None, :] - self.omega[:, None]) * np.where(mask, Ksum, 0.0)
        self.omega += dw.sum(axis=1)
        self.omega = np.clip(self.omega, OMEGA_LO, OMEGA_HI)


class Generator:
    """生成器：预测河下行（R2）——bigram 方向 × K 关联"""
    def __init__(self, w, sents):
        self.w = w
        # bigram 方向统计（语料顺序——"农业"→"属"）
        self.bigram = Counter()
        for s in sents:
            for k in range(len(s) - 1):
                self.bigram[s[k:k + 2]] += 1
        self.Ksum = w.K.sum(axis=0)

    def predict_all(self, last2):
        """所有候选（降序——bigram 方向优先——回退 K）"""
        cands = []
        for (bg, cnt) in self.bigram.items():
            if bg[0] == last2[-1]:
                nxt = bg[1]
                if nxt in self.w.ci and nxt != last2[-1]:
                    cands.append((nxt, cnt))
        cands.sort(key=lambda x: -x[1])
        if cands:
            return cands
        last = last2[-1]
        if last in self.w.ci:
            i = self.w.ci[last]
            row = self.Ksum[i].copy()
            top = np.argsort(row)[::-1][:5]
            for j in top:
                if row[j] > 0.03 and self.w.chars[j] != last:
                    return [(self.w.chars[j], row[j])]
        return []

    def generate(self, seed, max_len=MAX_LEN):
        """从起始词生成（逐字预测——anti-repetition——循环检测）"""
        out = seed
        for _ in range(max_len):
            cands = self.predict_all(out[-2:])
            chosen = None
            for c, strength in cands:
                if c in STOP_CHARS or strength < 0.005:
                    continue
                trial = out + c
                # anti-repetition：最后 4 字若已出现过 → 跳过（循环检测）
                if len(trial) >= 4 and trial[-4:] in out:
                    continue
                chosen = (c, strength)
                break
            if chosen is None:
                break
            out += chosen[0]
        return out


def run():
    print("=== M5 阶段 46：句子生成（预测河下行——组合性生成面） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=900)
    wiki = load_corpus(os.path.join(base, "corpus_wiki_filtered.txt"), n=1800)
    freq = Counter("".join(simple + wiki))
    chars = [c for c, _ in freq.most_common(N_CHAR)]
    w = RelLake(chars)
    # 主题分批训练（stage45——词组保留）
    blocks = [wiki[i:i + 600] for i in range(0, 1800, 600)]
    for block in blocks:
        for ep in range(10):
            w.learn_epoch(simple + block)
    print(f"训练完成——字集 {len(chars)}")
    gen = Generator(w, simple + wiki)
    # ---- exp1：逐字生成 ----
    print("\n[exp1] 逐字生成（预测河下行——给定起始词）:")
    seeds = ["农业", "技术", "教育", "今天", "经济", "环境"]
    for sd in seeds:
        s = gen.generate(sd)
        print(f"      '{sd}' → '{s}'")
    # ---- exp2：词组链 ----
    print("\n[exp2] 词组链生成（起始词 → 预测链——词组级）:")
    for sd in ["农业", "技术"]:
        chain = [sd]
        cur = sd
        for _ in range(6):
            nxt = gen.predict_next(cur[-2:])
            if nxt is None:
                break
            cur = cur + nxt[0]
            chain.append(nxt[0])
        print(f"      '{sd}' → {' → '.join(chain[:5])}")
    # ---- exp3：与语料对比（合理性——bigram 覆盖率） ----
    print("\n[exp3] 生成句与语料 bigram 覆盖率（生成 = 关系统计的合成——C122-01）:")
    gen_sents = [gen.generate(sd) for sd in seeds]
    bg_total = sum(gen.bigram.values())
    for s in gen_sents:
        hit = sum(gen.bigram.get(s[k:k + 2], 0) for k in range(len(s) - 1))
        cov = hit / max(sum(gen.bigram.get(s[k:k + 2], 0) for k in range(len(s) - 1)) + 1e-9, 1)
        print(f"      '{s}'——bigram 命中 {hit}")
    print("\n[done] stage46 generation")


if __name__ == "__main__":
    run()
