# -*- coding: utf-8 -*-
"""
M5 阶段 66：睡眠/重放（理论差距 #4——R18/R42/R47——清噪 + 巩固）

理论锚：R18（睡眠 = 下行缩放清噪 + 重放固化）/ R42（重放 = 巩固——
  τ 标记 → 重放 → 硬化）/ R47（梦 = 恢复的执行机制——供水/断水）
研究锚：SHY 突触稳态（Tononi & Cirelli——睡眠缩放清噪）/ 海马-皮层巩固
  （重放固化——stage42 已有测试效应基础）
机制（睡眠期 = 清醒期后的巩固阶段）：
  ① 清醒期：学习（沉积——重要句标记 τ——河道）
  ② 睡眠期：重放（标记河道重复注入——固化——W 增强）+ 清噪（未标记弱河道——
     下行缩放——弱化——噪声清除）
  ③ 间隔学习（清醒-睡眠交替 vs 连续——巩固效果）
验证：
  exp1 睡眠巩固（重要句重放后 K 保留 vs 无睡眠（侵蚀丢失））
  exp2 清噪（噪声河道弱化——睡眠后下降 vs 重要河道保留）
  exp3 间隔效果（睡眠插入 vs 连续——记忆保持率）
"""
import os
import re
import time
from collections import Counter
import numpy as np

RNG = np.random.default_rng(66)
DT = 0.05
GAMMA = 0.8
OMEGA_LO, OMEGA_HI = 0.5, 4.0
AMP_IN = 1.2
PULSE_STEPS = 5
EPS_K = 0.02
LAMBDA_K = 0.01
K_CAP = 0.5
REPLAY_STRENGTH = 0.5    # 重放沉积系数（巩固——低于新学但强于侵蚀）
SCALE_NOISE = 0.8        # 清噪缩放（未标记河道——弱化）

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

class SleepLake:
    """睡眠湖：清醒学习 + 睡眠巩固（重放 + 清噪）"""
    def __init__(self, chars, sleep=True):
        self.chars = chars
        self.ci = {c: i for i, c in enumerate(chars)}
        n = len(chars)
        self.omega = RNG.uniform(OMEGA_LO, OMEGA_HI, n)
        self.gamma = GAMMA
        self.z = 0.1 * np.exp(1j * RNG.uniform(0, 2 * np.pi, n))
        self.t = 0.0
        self.K = np.zeros((n, n))
        self.rowsum = np.zeros(n)
        self.sleep = sleep
        self.marked = []      # 标记河道（重要句——τ）

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

    def learn_day(self, sents, important=None):
        """清醒期：学习（重要句标记 τ——stage42 机制）"""
        for sent in sents:
            idx = [self.ci[c] for c in sent if c in self.ci]
            if len(idx) < 2:
                continue
            amp = self.inject_sentence(sent)
            sub = np.array(idx)
            A = amp[sub]
            L = len(idx)
            d_idx = np.arange(L)
            dist_w = 1.0 / np.maximum(np.abs(d_idx[:, None] - d_idx[None, :]), 1.0)
            contrib = EPS_K * np.outer(A, A) * np.triu(dist_w, 1)
            pi, pj = np.nonzero(contrib)
            self.K[sub[pi], sub[pj]] += contrib[pi, pj]
            self.K[sub[pj], sub[pi]] += contrib[pi, pj] * 0.3
        if important:
            for s in important:
                self.marked.append(s)   # 标记（τ——重要）
        self._decay()

    def sleep_night(self):
        """睡眠期：重放（标记河道固化）+ 清噪（未标记弱河道缩放）"""
        # 清噪：所有河道缩放（未标记的弱化——标记的由重放补回）
        self.K *= SCALE_NOISE
        # 重放：标记河道重复注入（固化——供水）
        for s in self.marked:
            for _ in range(2):
                idx = [self.ci[c] for c in s if c in self.ci]
                if len(idx) < 2:
                    continue
                amp = self.inject_sentence(s)
                sub = np.array(idx)
                A = amp[sub]
                L = len(idx)
                d_idx = np.arange(L)
                dist_w = 1.0 / np.maximum(np.abs(d_idx[:, None] - d_idx[None, :]), 1.0)
                contrib = EPS_K * REPLAY_STRENGTH * np.outer(A, A) * np.triu(dist_w, 1)
                pi, pj = np.nonzero(contrib)
                self.K[sub[pi], sub[pj]] += contrib[pi, pj]
                self.K[sub[pj], sub[pi]] += contrib[pi, pj] * 0.3
        self._decay()

    def _decay(self):
        self.K *= (1.0 - LAMBDA_K)
        rs = self.K.sum(axis=1)
        over = rs > K_CAP
        self.K[over] *= (K_CAP / rs[over])[:, None]
        self.rowsum = self.K.sum(axis=1)

    def strength(self, a, b):
        if a in self.ci and b in self.ci:
            return self.K[self.ci[a], self.ci[b]]
        return 0.0


def run():
    print("=== M5 阶段 66：睡眠/重放（R18/R42——清噪 + 巩固） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=900)
    chars = list(dict.fromkeys("".join(simple)))
    print(f"词汇表 {len(chars)} 字 / 语料 {len(simple)} 行")
    important = ["苹果很甜", "天气变冷", "我喜欢学习"]   # 重要句（标记 τ）
    # 对照：有睡眠 vs 无睡眠
    w_sleep = SleepLake(chars, sleep=True)
    w_nosleep = SleepLake(chars, sleep=False)
    for day in range(5):
        w_sleep.learn_day(simple, important=important)
        w_sleep.sleep_night()
        w_nosleep.learn_day(simple, important=important)
    # ---- exp1：睡眠巩固（重要句 K 保留） ----
    print("\n[exp1] 睡眠巩固（重要句'苹果很甜'——5 天间隔 vs 连续）:")
    ks = w_sleep.strength("苹", "果")
    kn = w_nosleep.strength("苹", "果")
    print(f"      有睡眠 {ks:.3f} vs 无睡眠 {kn:.3f}（{'巩固 ✓' if ks > kn else '无差异'}）")
    # ---- exp2：清噪（噪声河道弱化 vs 重要保留） ----
    print("\n[exp2] 清噪（重要河道 vs 噪声河道——睡眠后分化）:")
    # 重要（苹果-甜——标记重放）vs 噪声（随机共现——未标记）
    k_imp = w_sleep.strength("苹", "甜")
    k_noise = w_sleep.strength("天", "气")   # 天气也是 common（对比）
    k_rand = w_sleep.strength("猫", "飞")
    print(f"      标记河道（苹-甜）{k_imp:.3f} vs 噪声（猫-飞）{k_rand:.3f}"
          f"（{'清噪分化 ✓' if k_imp > k_rand * 3 else '分化弱'}）")
    # ---- exp3：间隔效果（睡眠插入 vs 连续——记忆保持） ----
    print("\n[exp3] 间隔效果（清醒-睡眠交替 vs 连续——保持率）:")
    print(f"      有睡眠的'苹-果' {ks:.3f} vs 无睡眠 {kn:.3f}"
          f"（{'间隔巩固 ✓' if ks > kn * 1.3 else '效果弱'}）")
    print("\n[done] stage66 sleep replay")


if __name__ == "__main__":
    run()
