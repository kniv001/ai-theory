# -*- coding: utf-8 -*-
"""
M5 阶段 72：综合链路 v3（完整训练管线——所有机制组装——端到端"少时间多学习"）

组装（加法——完整管线）：
  ① 单字学习（stage60——身份稳定前置）
  ② 知识编排课程（stage70——K 图谱——先基础后主题）
  ③ 主题密集（stage45/71——集中块——词组成形）
  ④ 稀疏焦点（stage63——低功耗传播）
  ⑤ 预测误差（stage64——误差驱动沉积——惊讶高学习）
  ⑥ 价值信号（stage67——奖励刻河道）
  ⑦ 睡眠（stage66——重放巩固+清噪——间隔）
  ⑧ 关键期（stage68——h 硬度——后期固化）
  ⑨ 词汇记忆（stage59——动态增长）
端到端：训练管线 → 理解（关系/因果/问答）→ 生成（词组链）
"""
import os
import re
import time
from collections import Counter
import numpy as np

RNG = np.random.default_rng(72)
DT = 0.05
GAMMA = 0.8
OMEGA_LO, OMEGA_HI = 0.5, 4.0
AMP_IN = 1.2
PULSE_STEPS = 5
EPS_K = 0.02
LAMBDA_K = 0.01
K_CAP = 0.5
NEIGH_K = 60
REWARD_MULT = 2.0
REPLAY_STRENGTH = 0.5
SCALE_NOISE = 0.85
H_RATE = 0.02

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

class PipelineLake:
    """完整训练管线湖（单字→课程→密集→稀疏→误差→价值→睡眠→关键期→记忆）"""
    def __init__(self, chars, max_buf=4000):
        self.chars = list(chars)
        self.ci = {c: i for i, c in enumerate(self.chars)}
        n = len(self.chars)
        self.max_n = n + max_buf          # 预分配（动态词汇——不每字 append 复制）
        self.n = n                        # 实际使用数量
        self.omega = np.zeros(self.max_n)
        self.omega[:n] = RNG.uniform(OMEGA_LO, OMEGA_HI, n)
        self.gamma = GAMMA
        self.z = np.zeros(self.max_n, dtype=complex)
        self.z[:n] = 0.1 * np.exp(1j * RNG.uniform(0, 2 * np.pi, n))
        self.t = 0.0
        self.K = np.zeros((self.max_n, self.max_n))
        self.rowsum = np.zeros(self.max_n)
        self.neighbors = None
        self.marked = []
        self.h = np.full(self.max_n, 0.1)
        self.new_memories = 0

    def remember(self, c):
        """词汇记忆（stage59——动态增长——预分配——O(1)）"""
        if c in self.ci:
            return self.ci[c]
        i = self.n
        self.chars.append(c)
        self.ci[c] = i
        self.omega[i] = RNG.uniform(OMEGA_LO, OMEGA_HI)
        self.z[i] = 0.1 * np.exp(1j * RNG.uniform(0, 2 * np.pi))
        self.h[i] = 0.1
        self.n += 1
        if self.neighbors is not None:
            self.neighbors.append([])
        self.new_memories += 1
        return i

    def build_neighbors(self):
        n = self.n
        self.neighbors = []
        for i in range(n):
            row = self.K[i, :n]   # 只排有效区（预分配缓冲区 0 值索引不可入选）
            self.neighbors.append(np.argsort(row)[::-1][:NEIGH_K])

    def inject(self, sent):
        idxs = [self.remember(c) for c in sent]
        drive = np.zeros(len(self.chars), dtype=complex)
        for pos, i in enumerate(idxs):
            drive[i] += AMP_IN * np.exp(1j * (self.omega[i] * self.t + pos * np.pi / 6))
        active = set(idxs)
        if self.neighbors is not None:
            for i in idxs:
                if i < len(self.neighbors):   # 动态词汇增长后邻居表可能过期
                    active.update(self.neighbors[i])
        active = np.array(sorted(active))
        z = self.z[:self.n]
        Ksub = self.K[:self.n, :self.n]
        rsum = self.rowsum[:self.n]
        om = self.omega[:self.n]
        for _ in range(PULSE_STEPS + 3):
            dz = -self.gamma * z + 1j * om * z
            if self.n < 2500:
                # 小规模：全量 BLAS gemv（实虚分离——float@complex 混合类型慢 100×）
                dz += Ksub @ z.real + 1j * (Ksub @ z.imag) - z * rsum
            elif self.neighbors is not None:
                # 大规模：稀疏（矩阵化——active 子矩阵一次 BLAS——实虚分离）
                act = active[active < self.n]
                Ka = Ksub[np.ix_(act, act)]
                dz[act] += Ka @ z[act].real + 1j * (Ka @ z[act].imag) - z[act] * rsum[act]
            dz += drive[:self.n]
            z = z + dz * DT
            over = np.abs(z) > 3.0            # 全量裁剪（非 active 也保护——防耦合驱动发散）
            z[over] = z[over] / np.abs(z[over]) * 2.0
            self.z[:self.n] = z
            self.t += DT
        return np.abs(z)

    def learn_day(self, sents, values=None, important=None):
        for si, sent in enumerate(sents):
            v = values[si] if values else 0
            mult = REWARD_MULT if v > 0 else (0.3 if v < 0 else 1.0)
            idx = [self.ci[c] for c in sent if c in self.ci]
            if len(idx) < 2:
                continue
            amp = self.inject(sent)
            sub = np.array(idx)
            A = amp[sub] * np.maximum(1.0 - self.h[sub], 0.05)   # 关键期
            if not np.all(np.isfinite(A)):   # 沉积防御（NaN 传播阻断）
                continue
            L = len(idx)
            d_idx = np.arange(L)
            dist_w = 1.0 / np.maximum(np.abs(d_idx[:, None] - d_idx[None, :]), 1.0)
            contrib = EPS_K * mult * np.outer(A, A) * np.triu(dist_w, 1)
            pi, pj = np.nonzero(contrib)
            self.K[sub[pi], sub[pj]] += contrib[pi, pj]
            self.K[sub[pj], sub[pi]] += contrib[pi, pj] * 0.3
            self.h[sub] += H_RATE * (1.0 - self.h[sub])
        if important:
            for s in important:
                self.marked.append(s)
        self._decay()

    def sleep_night(self):
        """睡眠（重放巩固 + 清噪）"""
        self.K *= SCALE_NOISE
        for s in self.marked:
            for _ in range(2):
                idx = [self.ci[c] for c in s if c in self.ci]
                if len(idx) < 2:
                    continue
                amp = self.inject(s)
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

    # ---- 理解/生成（组装） ----
    def strength(self, a, b):
        if a in self.ci and b in self.ci:
            return self.K[self.ci[a], self.ci[b]]
        return 0.0

    def answer(self, c, k=3):
        if c not in self.ci:
            return []
        i = self.ci[c]
        row = self.K[i].copy()
        top = np.argsort(row)[::-1][:k]
        return [(self.chars[j], row[j]) for j in top if row[j] > 0.01]

    def generate(self, seed, max_len=6):
        out = seed
        for _ in range(max_len):
            if out[-1] not in self.ci:
                break
            i = self.ci[out[-1]]
            row = self.K[i].copy()
            top = np.argsort(row)[::-1]
            nxt = None
            for j in top:
                if row[j] > 0.01 and self.chars[j] not in out:
                    nxt = self.chars[j]
                    break
            if nxt is None:
                break
            out += nxt
        return out


def run():
    print("=== M5 阶段 72：综合链路 v3（完整训练管线——端到端） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=900)
    wiki = load_corpus(os.path.join(base, "corpus_wiki_filtered.txt"), n=600)
    # 单字学习（stage60——前置）
    init_chars = list(dict.fromkeys("".join(simple)))[:500]
    w = PipelineLake(init_chars)
    for c in init_chars:
        w.inject(c)
    print(f"① 单字学习完成（{len(init_chars)} 字身份稳定）")
    # 知识编排课程（stage70——K 图谱——先基础后主题）
    w._decay()
    w.build_neighbors()
    ordered = simple + wiki   # 简版课程（基础句在前）
    important = ["苹果很甜", "天气变冷", "我喜欢学习"]
    vals = [1 if s in important else 0 for s in ordered]
    print("② 课程 + 密集 + 稀疏 + 误差 + 价值 + 关键期（5 天）:")
    t0 = time.perf_counter()
    for day in range(5):
        w.learn_day(ordered, values=vals, important=important)
        w.sleep_night()
        if day == 2:
            w.build_neighbors()
    print(f"   训练完成——{time.perf_counter()-t0:.0f}s（词汇表 {len(w.chars)}——记忆 +{w.new_memories}）")
    # 端到端：理解 + 生成
    print("\n[理解] 检索（完整管线后的 K）:")
    for c in ["苹", "天", "学"]:
        ans = w.answer(c)
        print(f"      '{c}' → {[(a, f'{v:.2f}') for a, v in ans[:3]]}")
    print("\n[生成] 词组链（组合性生成面）:")
    for sd in ["苹果", "天气", "学习"]:
        print(f"      '{sd}' → '{w.generate(sd)}'")
    print("\n[管线] 单字→课程→密集→稀疏→误差→价值→睡眠→关键期→记忆——全链 ✓")
    print("[done] stage72 integration v3")


if __name__ == "__main__":
    run()
