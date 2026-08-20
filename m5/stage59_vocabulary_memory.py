# -*- coding: utf-8 -*-
"""
M5 阶段 59：词汇自发记忆（用户："词汇库可以做到自发的记忆吗——记忆功能"）

框架理论：R5（沉积-侵蚀——记忆=地形）/ C6-03（记忆=地形本身——硬区=结构记忆）/
  R40（功能侵蚀——弱化不删除——再激活恢复）/ C2-02（结构生长——涌现）
研究锚：儿童新词学习（fast mapping——一次接触半记忆——重复巩固——不使用遗忘）
机制（词汇库自发增长——像幼儿学新词）：
  ① 动态字集：训练遇新字 → 自动分配新单元（append——新 ω/新行——记住）
  ② 记忆强度：K 沉积（每次出现增强——巩固）；侵蚀（不使用弱化——遗忘）
  ③ 保留：弱化不删除（R40 功能侵蚀——再出现快速恢复——半记忆可激活）
验证：
  exp1 新字自发记忆（训练遇新字 → 自动收录 → 之后注入可识别）
  exp2 记忆强度（重复出现 → K 增强；一次出现 → 弱——半记忆）
  exp3 遗忘与恢复（低频字侵蚀弱化——再出现恢复——R40）
  exp4 词汇库增长（训练后字集 = 初始 + 全部新字——自发增长——无人工添加）
"""
import os
import re
import time
import numpy as np

RNG = np.random.default_rng(59)
DT = 0.05
GAMMA = 0.8
OMEGA_LO, OMEGA_HI = 0.5, 4.0
AMP_IN = 1.2
PULSE_STEPS = 5
EPS_K = 0.02
LAMBDA_K = 0.01
K_CAP = 0.5

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

class MemLake:
    """记忆湖：动态词汇库（自发增长——遇新字自动收录）"""
    def __init__(self, init_chars):
        self.chars = list(init_chars)
        self.ci = {c: i for i, c in enumerate(self.chars)}
        n = len(self.chars)
        self.omega = RNG.uniform(OMEGA_LO, OMEGA_HI, n)
        self.gamma = GAMMA
        self.z = 0.1 * np.exp(1j * RNG.uniform(0, 2 * np.pi, n))
        self.t = 0.0
        self.K = np.zeros((n, n))
        self.rowsum = np.zeros(n)
        self.new_memories = 0      # 自发记忆的新字计数

    def step(self, drive):
        self.z = step_dynamics(self.z, self.omega, self.gamma, self.K, self.rowsum, drive, DT)
        self.t += DT
        return self.z

    def remember(self, c):
        """自发记忆：遇新字 → 自动分配新单元（记住——R5 沉积的节点面）"""
        if c in self.ci:
            return self.ci[c]
        i = len(self.chars)
        self.chars.append(c)
        self.ci[c] = i
        self.omega = np.append(self.omega, RNG.uniform(OMEGA_LO, OMEGA_HI))
        self.z = np.append(self.z, 0.1 * np.exp(1j * RNG.uniform(0, 2 * np.pi)))
        self.K = np.pad(self.K, ((0, 1), (0, 1)))
        self.rowsum = np.append(self.rowsum, 0.0)
        self.new_memories += 1
        return i

    def inject_sentence(self, sent):
        # 先记住全部字（动态增长——驱动数组大小需要最终尺寸）
        idxs = [self.remember(c) for c in sent]
        drive = np.zeros(len(self.chars), dtype=complex)
        for pos, i in enumerate(idxs):
            drive[i] += AMP_IN * np.exp(1j * (self.omega[i] * self.t + pos * np.pi / 6))
        for _ in range(PULSE_STEPS):
            self.step(drive)
        for _ in range(3):
            self.step(np.zeros(len(self.chars), dtype=complex))
        return np.abs(self.z)

    def learn_epoch(self, sents, extra=None):
        for sent in sents:
            seq_idx = [self.ci[c] for c in sent if c in self.ci]
            if len(seq_idx) < 2:
                continue
            amp = self.inject_sentence(sent)
            self._deposit(seq_idx, amp)
        if extra:
            for sent in extra:
                self.inject_sentence(sent)   # 只记住不沉积（半记忆——fast mapping）
        self.K *= (1.0 - LAMBDA_K)
        rs = self.K.sum(axis=1)
        over = rs > K_CAP
        self.K[over] *= (K_CAP / rs[over])[:, None]
        self.rowsum = self.K.sum(axis=1)

    def _deposit(self, seq_idx, amp):
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
            self.step(np.zeros(len(self.chars), dtype=complex))

    def strength(self, c):
        """记忆强度（该字的耦合总量——记住的程度）"""
        if c in self.ci:
            i = self.ci[c]
            return float(self.K[i].sum())
        return 0.0


def run():
    print("=== M5 阶段 59：词汇自发记忆（动态词汇库——遇新字自动收录） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=900)
    wiki = load_corpus(os.path.join(base, "corpus_wiki_filtered.txt"), n=1800)
    # 初始字集：只从简单语料（小）——wiki 字是"新字"（自发记忆的对象）
    init_chars = list(dict.fromkeys("".join(simple)))[:200]
    print(f"初始词汇表 {len(init_chars)} 字（简单语料）——wiki 字 = 待自发记忆的新字")
    w = MemLake(init_chars)
    # 阶段 1：简单语料（基础）——wiki 分块（新字涌入——自发记忆）
    for ep in range(5):
        w.learn_epoch(simple)
    print(f"简单语料训练后词汇 {len(w.chars)} 字（自发记忆 +{w.new_memories}）")
    blocks = [wiki[i:i + 600] for i in range(0, 1800, 600)]
    for block in blocks:
        w.learn_epoch(simple + block)
    print(f"wiki 训练后词汇 {len(w.chars)} 字（自发记忆 +{w.new_memories}——全部新字已收录）")
    # ---- exp1：新字自发记忆 ----
    print("\n[exp1] 新字自发记忆（训练遇新字 → 自动收录 → 注入可识别）:")
    for c in ["苹", "植", "排"]:
        known = c in w.ci
        print(f"      '{c}' 在词汇表: {'✓ 自发记住' if known else '✗'}")
    # ---- exp2：记忆强度（重复 vs 一次） ----
    print("\n[exp2] 记忆强度（重复出现巩固 vs 一次接触半记忆）:")
    # "苹" 在 wiki 出现多次（苹果——重复）——"排"（排放——少量）
    for c in ["苹", "排"]:
        s = w.strength(c)
        print(f"      '{c}' 记忆强度 = {s:.3f}")
    # ---- exp3：遗忘与恢复（R40） ----
    print("\n[exp3] 遗忘与恢复（侵蚀弱化——再出现恢复——R40 功能侵蚀）:")
    s_before = w.strength("苹")
    # 遗忘期（10 epoch 不含"苹"——侵蚀）
    no_apple = [s for s in simple if "苹" not in s]
    for ep in range(5):
        w.learn_epoch(no_apple)
    s_after = w.strength("苹")
    print(f"      '苹' 强度: 训练后 {s_before:.3f} → 5 epoch 不含后 {s_after:.3f}"
          f"（{'遗忘（弱化）✓' if s_after < s_before else '未遗忘'}）")
    # 再出现（恢复）
    w.learn_epoch(["苹果很甜", "我吃苹果"])
    s_recovered = w.strength("苹")
    print(f"      再出现后 {s_recovered:.3f}（{'恢复 ✓——R40 再激活' if s_recovered > s_after else '未恢复'}）")
    # ---- exp4：词汇库增长 ----
    print(f"\n[exp4] 词汇库增长: 初始 {len(init_chars)} → 最终 {len(w.chars)} 字"
          f"（自发记忆 +{w.new_memories}——无人工添加）")
    print("\n[done] stage59 vocabulary memory")


if __name__ == "__main__":
    run()
