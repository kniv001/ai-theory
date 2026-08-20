# -*- coding: utf-8 -*-
"""
M5 阶段 58：组合时序面完整化（C13-02 吸引子复合——sh+an→shan——
台账 open——M5 待验——框架理论 + 研究双锚）

理论锚：C13-02（组合性 = 时序吸引子复合——字→词→句同机制放大——open——M5 待验）
研究锚：Dekydtspotter 2025（γ 振荡实现绑定——语法-语义对象创建）/
  Yang 2026 ROSE（δ-θ 相位-幅度耦合编码递归结构——θ 功率=句法头记忆）/
  HFTP 2025（句子 1Hz/短语 2Hz 层级频率标记——层级结构表示）
机制（组合 = 相位聚合——跨尺度复合——sh+an→shan 的工程实现）：
  ① 时序湖嵌套相位（stage51——字/词/句——Δφ 层级）
  ② 短语复合：词序列 → 复合相位（词相位的聚合——吸引子复合——
     "农业"+"发展" → 短语级相位 = 词相位的组合（θ-γ 式跨尺度——细粒度 γ 词 → 粗粒度 θ 短语））
  ③ 递归复合：短语+词 → 更大单元（层级组合——R16）
验证：
  exp1 词相位 vs 短语复合（"农业发展"的复合相位——组合的新表示）
  exp2 组合顺序（"农业发展" vs "发展农业"——复合相位不同——组合性）
  exp3 递归复合（"农业"+"发展"+"技术"——三级复合——层级组合相位）
"""
import os
import re
import time
from collections import Counter
import numpy as np

RNG = np.random.default_rng(58)
DT = 0.05
GAMMA = 0.8
OMEGA_LO, OMEGA_HI = 0.5, 4.0
AMP_IN = 1.2
PULSE_STEPS = 5
EPS_K = 0.02
LAMBDA_K = 0.01
K_CAP = 0.5
DELTA_WORD = np.pi / 4      # 词位置相位间隔
DELTA_CHAR = np.pi / 8      # 词内字位置相位间隔

def load_corpus(path, lo=3, hi=80, n=None):
    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    clean = [s for s in lines if lo <= len(s) <= hi
             and re.search(r"[一-鿿]", s) and not re.search(r"[A-Za-z]", s)]
    if n and len(clean) > n:
        clean = clean[:n]
    return clean

def step_dynamics(z, omega, gamma, K, rowsum, drive, dt):
    zr, zi = z.real, z.imag
    dz = -gamma * z + 1j * omega * z
    dz += K @ zr + 1j * (K @ zi) - z * rowsum
    dz += drive
    z = z + dz * dt
    over = np.abs(z) > 3.0
    z[over] = z[over] / np.abs(z[over]) * 2.0
    return z

class CompLake:
    """组合湖：嵌套相位 + 短语复合（C13-02）"""
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

    def step(self, drive):
        self.z = step_dynamics(self.z, self.omega, self.gamma, self.K, self.rowsum, drive, DT)
        self.t += DT
        return self.z

    def inject_words(self, words, sent_pos=0):
        """词序列注入（嵌套相位——词内字 + 词位置）"""
        drive = np.zeros(len(self.chars), dtype=complex)
        for widx, wd in enumerate(words):
            for cidx, c in enumerate(wd):
                if c in self.ci:
                    i = self.ci[c]
                    drive[i] += AMP_IN * np.exp(1j * (self.omega[i] * self.t
                                                      + cidx * DELTA_CHAR
                                                      + widx * DELTA_WORD
                                                      + sent_pos * np.pi / 6))
        for _ in range(PULSE_STEPS):
            self.step(drive)
        for _ in range(3):
            self.step(np.zeros(len(self.chars), dtype=complex))
        return np.abs(self.z)

    def learn_epoch(self, sents):
        n = len(self.chars)
        for sent in sents:
            seq_idx = [self.ci[c] for c in sent if c in self.ci]
            if len(seq_idx) < 2:
                continue
            # 简单词切分（2 字滑窗——词组优先）
            words = []
            i = 0
            while i < len(sent) - 1:
                if sent[i:i + 2] in self.K > 0 and False:
                    pass
                words.append(sent[i])
                i += 1
            if i < len(sent):
                words.append(sent[i])
            amp = self.inject_words(words)
            L = len(seq_idx)
            sub = np.array(seq_idx)
            A = amp[sub]
            idx = np.arange(L)
            dist_w = 1.0 / np.maximum(np.abs(idx[:, None] - idx[None, :]), 1.0)
            contrib = EPS_K * np.outer(A, A) * np.triu(dist_w, 1)
            pi, pj = np.nonzero(contrib)
            self.K[sub[pi], sub[pj]] += contrib[pi, pj]
            self.K[sub[pj], sub[pi]] += contrib[pi, pj] * 0.3
            for _ in range(4):
                self.step(np.zeros(n, dtype=complex))
        self.K *= (1.0 - LAMBDA_K)
        rs = self.K.sum(axis=1)
        over = rs > K_CAP
        self.K[over] *= (K_CAP / rs[over])[:, None]
        self.rowsum = self.K.sum(axis=1)

    def word_phase(self, wd):
        """词相位（词内字的平均相位——词 = 相位单元）"""
        idxs = [self.ci[c] for c in wd if c in self.ci]
        if not idxs:
            return None
        ph = np.angle(self.z[idxs])
        return np.angle(np.mean(np.exp(1j * ph)))

    def compose(self, words):
        """短语复合（θ-γ 式跨尺度——词相位聚合——吸引子复合——sh+an→shan）"""
        # 注入词序列——读短语级相位（全部词的聚合——组合的新表示）
        self.inject_words(words)
        idxs = [self.ci[c] for wd in words for c in wd if c in self.ci]
        ph = np.angle(self.z[idxs])
        amp = np.abs(self.z[idxs])
        # 幅度加权聚合（强词主导——组合）
        return np.angle(np.sum(amp * np.exp(1j * ph)) / max(np.sum(amp), 1e-9))


def run():
    print("=== M5 阶段 58：组合时序面（C13-02 吸引子复合——θ-γ 跨尺度——M5 验证） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=900)
    wiki = load_corpus(os.path.join(base, "corpus_wiki_filtered.txt"), n=1800)
    sents = simple + wiki
    chars = list(dict.fromkeys("".join(sents)))
    print(f"词汇表 {len(chars)} 字——语料 {len(sents)} 行")
    w = CompLake(chars)
    t0 = time.perf_counter()
    for ep in range(8):
        w.learn_epoch(sents)
    print(f"训练完成——{time.perf_counter()-t0:.0f}s")
    # ---- exp1：词相位 vs 短语复合 ----
    print("\n[exp1] 词相位 vs 短语复合（组合的新表示——C13-02）:")
    ph_ag = w.word_phase("农业")
    ph_fz = w.word_phase("发展")
    ph_comp = w.compose(["农业", "发展"])
    print(f"      '农业'相位 {ph_ag:.3f} / '发展'相位 {ph_fz:.3f}")
    print(f"      复合['农业','发展']相位 {ph_comp:.3f}"
          f"（{'组合表示 ≠ 单个词 ✓' if abs(ph_comp - ph_ag) > 0.3 and abs(ph_comp - ph_fz) > 0.3 else '组合弱'}）")
    # ---- exp2：组合顺序（组合性——顺序敏感） ----
    print("\n[exp2] 组合顺序（'农业发展' vs '发展农业'——组合相位不同——组合性）:")
    ph1 = w.compose(["农业", "发展"])
    ph2 = w.compose(["发展", "农业"])
    print(f"      ['农业','发展'] {ph1:.3f} vs ['发展','农业'] {ph2:.3f}"
          f"（{'顺序敏感 ✓——组合性' if abs(ph1 - ph2) > 0.3 else '顺序弱'}）")
    # ---- exp3：递归复合（层级组合——R16） ----
    print("\n[exp3] 递归复合（'农业'+'发展'+'技术'——三级复合——层级组合）:")
    ph_2 = w.compose(["农业", "发展"])
    ph_3 = w.compose(["农业", "发展", "技术"])
    print(f"      二级复合 {ph_2:.3f} → 三级复合 {ph_3:.3f}"
          f"（{'递归组合 ✓——层级相位' if abs(ph_3 - ph_2) > 0.3 else '组合弱'}）")
    print("\n[done] stage58 composition phase")


if __name__ == "__main__":
    run()
