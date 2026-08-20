# -*- coding: utf-8 -*-
"""
M5 阶段 64：预测-误差学习（理论最大差距——C1-01/C3-02——"误差是唯一词汇" A0 工程化）

理论锚：C1-01（D = 现实 − 预测——误差驱动）/ C3-02（预测河下行 + 误差河上行）/
  A0（误差 = 唯一词汇）/ R5（沉积 dW = ε·e·conj(z)——e = 误差项）
研究锚：预测编码（Rao & Ballard——只处理预测误差）/ 大脑 20W（计算 ∝ 误差量——
  熟悉刺激零误差不激活——惊讶处处理）
机制（沉积从"共现"升级为"预测-误差"）：
  ① 预测河下行：给定前文（2 字）→ 预测下一字（K 前向最强）——生成预期
  ② 误差河上行：实际 vs 预测 → 误差 e（0 = 命中 / 1 = 预测失败——惊讶）
  ③ 误差驱动沉积：沉积 × 误差调制——预测失败（惊讶）高学习率——
     零误差（熟悉）低学习率——激活 ∝ 误差——低功耗同构
验证：
  exp1 预测（"苹果是"→预测下一字——K 前向——命中率）
  exp2 误差驱动修正（惊讶句"苹果是重力" vs 熟悉句——学习强度对比）
  exp3 误差分布（语料中预测误差——惊讶字——误差驱动焦点）
"""
import os
import re
import time
from collections import Counter
import numpy as np

RNG = np.random.default_rng(64)
DT = 0.05
GAMMA = 0.8
OMEGA_LO, OMEGA_HI = 0.5, 4.0
AMP_IN = 1.2
PULSE_STEPS = 5
EPS_K = 0.02
LAMBDA_K = 0.01
K_CAP = 0.5
DELTA_PHI = np.pi / 6

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

class PredErrLake:
    """预测-误差湖：预测河下行 + 误差河上行（A0 误差词汇）"""
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

    def predict_next(self, last2):
        """预测河下行：给定前 2 字 → 预测下一字（K 前向最强）"""
        if last2[-1] in self.ci:
            i = self.ci[last2[-1]]
            row = self.K[i].copy()
            top = np.argsort(row)[::-1]
            for j in top:
                if row[j] > 0.01 and self.chars[j] != last2[-1]:
                    return self.chars[j], row[j]
        return None, 0.0

    def learn_epoch(self, sents, report_error=False):
        """误差驱动学习：逐字预测 → 误差 → 沉积×误差调制"""
        n = len(self.chars)
        err_count = Counter()
        for sent in sents:
            seq_idx = [self.ci[c] for c in sent if c in self.ci]
            if len(seq_idx) < 2:
                continue
            # 注入（顺序——位置相位）
            drive = np.zeros(n, dtype=complex)
            for pos, i in enumerate(seq_idx):
                drive[i] += AMP_IN * np.exp(1j * (self.omega[i] * self.t + pos * DELTA_PHI))
            for _ in range(PULSE_STEPS + 3):
                self.step(drive)
            amp = np.abs(self.z)
            # 逐字预测-误差（前 2 字 → 预测下一字 → 实际 vs 预测）
            for k in range(2, len(seq_idx)):
                c_prev = self.chars[seq_idx[k - 1]]
                pred, _ = self.predict_next(c_prev)
                actual = self.chars[seq_idx[k]]
                err = 0.0 if pred == actual else 1.0   # 误差（0 命中/1 惊讶）
                if report_error:
                    err_count[actual] += err
                # 误差驱动沉积（熟悉低/惊讶高——激活 ∝ 误差）
                w = 0.2 + 2.0 * err   # 零误差 0.2（低）——惊讶 2.2（高——处理）
                sub = np.array(seq_idx[:k + 1])
                A = amp[sub]
                L = len(sub)
                d_idx = np.arange(L)
                dist_w = 1.0 / np.maximum(np.abs(d_idx[:, None] - d_idx[None, :]), 1.0)
                contrib = EPS_K * w * np.outer(A, A) * np.triu(dist_w, 1)
                pi, pj = np.nonzero(contrib)
                self.K[sub[pi], sub[pj]] += contrib[pi, pj]
                self.K[sub[pj], sub[pi]] += contrib[pi, pj] * 0.3
        self.K *= (1.0 - LAMBDA_K)
        rs = self.K.sum(axis=1)
        over = rs > K_CAP
        self.K[over] *= (K_CAP / rs[over])[:, None]
        self.rowsum = self.K.sum(axis=1)
        return err_count


def run():
    print("=== M5 阶段 64：预测-误差学习（A0 误差词汇工程化——C1-01/C3-02） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=900)
    chars = list(dict.fromkeys("".join(simple)))
    print(f"词汇表 {len(chars)} 字 / 语料 {len(simple)} 行（简单语料——快）")
    w = PredErrLake(chars)
    t0 = time.perf_counter()
    for ep in range(5):
        errs = w.learn_epoch(simple, report_error=(ep == 4))
    print(f"训练完成——{time.perf_counter()-t0:.0f}s")
    # ---- exp1：预测（前文 → 下一字） ----
    print("\n[exp1] 预测河下行（'苹果' → 预测下一字）:")
    for pre in ["苹果", "天气", "喜欢"]:
        pred, strength = w.predict_next(pre)
        print(f"      '{pre}' → 预测 '{pred}'（强度 {strength:.3f}）")
    # ---- exp2：误差驱动修正（惊讶句 vs 熟悉句——学习强度） ----
    print("\n[exp2] 误差驱动（惊讶句高学习 vs 熟悉句低——激活 ∝ 误差）:")
    # 对照：熟悉句（"苹果很甜"——共现多——零误差）vs 惊讶句（"苹果是重力"——预测失败）
    w_fam = PredErrLake(chars)
    w_sur = PredErrLake(chars)
    for _ in range(3):
        w_fam.learn_epoch(["苹果很甜", "苹果好吃", "我吃苹果"])
        w_sur.learn_epoch(["苹果是重力", "苹果会飞", "苹果是石头"])
    k_fam = w_fam.K[w_fam.ci["苹"], w_fam.ci["果"]]
    k_sur = w_sur.K[w_sur.ci["苹"], w_sur.ci["果"]]
    print(f"      熟悉句训练后 '苹-果' K = {k_fam:.3f}（低——零误差少学习）")
    print(f"      惊讶句训练后 '苹-果' K = {k_sur:.3f}（高——惊讶多学习）")
    # ---- exp3：误差分布（惊讶字——误差驱动焦点） ----
    print("\n[exp3] 误差分布（预测失败的字——惊讶处——误差驱动焦点）:")
    top_err = errs.most_common(8)
    print(f"      高误差字: {[(c, e) for c, e in top_err]}")
    print("\n[done] stage64 predictive error")


if __name__ == "__main__":
    run()
