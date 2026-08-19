# -*- coding: utf-8 -*-
"""
M5 阶段 38：无监督湖涌现（文字湖桥梁 ②——词类湖自组织——C2-02 结构涌现）

TEXT_LAKE_ROADMAP 差距①（核心算法）：签名自组织——湖从语料涌现而非预置分配。
机制（研究驱动）：
  ① 慢尺度频率吸引：共现字对的频率互相吸引（ω_i += η·(ω_j-ω_i)·K_ij）——
     共现结构 → 频率聚簇 = 湖（相变：耦合强 + 频率近 → 同步域——stage7 锁相条件）
  ② 快尺度耦合沉积：句内共现（距离衰减——stage37 模式）→ K——侵蚀每遍
  ③ 容量预算（出生竞争 R38）：K 行归一——每字耦合总量上限——强关联挤弱关联——
     防单簇化（否则全部字互相吸引成 1 簇——湖无结构）
  ④ 同步域：簇内相位锁定（相干度——stage7 验证机制）
验证：
  exp1 湖涌现：训练后 ω 分布聚簇（≥2 簇——K-means 判定）
  exp2 语义对应：簇内字 vs 语义类（天气类/学习类/国家类…）
  exp3 同步域：簇内相干（>0.6 = 相位锁定）
  exp4 湖间联想：湖 A 激活 → 湖 B（句级共现）
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(38)
DT = 0.05
GAMMA = 0.8
N_WORDS = 60   # 实际 = len(CHARS)（语料字符集去重——动态）
OMEGA_LO, OMEGA_HI = 0.5, 4.0
AMP_IN = 1.2
PULSE_STEPS = 8
EPS_K = 0.02            # 耦合沉积率
LAMBDA_K = 0.02         # 耦合侵蚀（每遍）
ETA_OMEGA = 0.15        # 频率吸引率（慢尺度）
K_CAP = 0.6             # 容量预算（每字耦合总量 ≤0.6——每字只强耦合 2-3 字——局域性——防全局汇聚）
COH_TH = 0.6            # 同步判定阈值（相干度）

SENTS = [
    "今天天气很好",
    "我喜欢学习",
    "人民的生活很好",
    "技术进步很快",
    "世界和平发展",
    "我们相信明天",
    "教育很重要",
    "国家繁荣富强",
    "社会和谐稳定",
    "科学技术创新",
    "经济发展",
    "大家好",
    "知识就是力量",
    "认真学习",
    "每天努力进步",
    "我的名字是小明",
    "他每天去学校",
    "老师教我们数学",
    "这本书很有意思",
    "我喜欢吃苹果",
    "妹妹在看电视",
    "爸爸工作很忙",
    "我们一起去公园",
    "天上有星星",
    "水里有很多鱼",
    "春天花开了",
    "鸟儿在唱歌",
    "下雨了要带伞",
    "这条路很远",
    "妈妈做的饭很好吃",
]
# 语料字符集（去重——保证全覆盖）
CHARS = list(dict.fromkeys("".join(SENTS)))
CHAR_IDX = {c: i for i, c in enumerate(CHARS)}


class LakeWorld:
    """文字湖：频率吸引自组织——湖 = 频率簇 = 同步域"""
    def __init__(self):
        n = len(CHARS)
        # 字频率初始：随机（慢尺度演化——共现吸引）
        self.omega = RNG.uniform(OMEGA_LO, OMEGA_HI, n)
        self.gamma = GAMMA
        self.t = 0.0
        self.z = 0.1 * np.exp(1j * RNG.uniform(0, 2 * np.pi, n))
        self.K = np.zeros((n, n))          # 耦合（学习）
        self.act = np.zeros(n)

    # ---- 快尺度：单元动力学（stage7 形式——全连接耦合） ----
    def step(self, drive):
        dz = -self.gamma * self.z + 1j * self.omega * self.z
        # 耦合不除 n（÷60 让有效耦合 ~0.008 vs γ0.8——差 100 倍——同步不可能；
        # 强耦合邻居数量级决定——K 0.1×5 邻居 = 0.5——与 γ 可比）
        dz += (self.K * (self.z[None, :] - self.z[:, None])).sum(axis=1)
        dz += drive
        self.z = self.z + dz * DT
        self.t += DT
        over = np.abs(self.z) > 3.0
        self.z[over] = self.z[over] / np.abs(self.z[over]) * 2.0
        return self.z

    def inject(self, c):
        i = CHAR_IDX[c]
        drive = np.zeros(len(CHARS), dtype=complex)
        drive[i] = AMP_IN * np.exp(1j * (self.omega[i] * self.t))
        for _ in range(PULSE_STEPS):
            self.step(drive)
        for _ in range(3):
            self.step(np.zeros(len(CHARS), dtype=complex))

    # ---- 学习：句内耦合沉积 + 慢尺度频率吸引 + 容量预算 ----
    def learn_epoch(self, sents):
        for sent in sents:
            seq = [c for c in sent if c in CHAR_IDX]
            self.act = np.zeros(len(CHARS))
            for c in seq:
                self.inject(c)
                self.act += np.abs(self.z)
                self.act *= 0.9
            for i in range(len(seq)):
                wi = CHAR_IDX[seq[i]]
                for j in range(i + 1, len(seq)):
                    wj = CHAR_IDX[seq[j]]
                    pair = EPS_K * self.act[wi] * self.act[wj] / (j - i)
                    self.K[wi, wj] += pair
                    self.K[wj, wi] += pair
            for _ in range(4):
                self.step(np.zeros(len(CHARS), dtype=complex))   # 句界空窗
        # 侵蚀（每遍一次）
        self.K *= (1.0 - LAMBDA_K)
        # 容量预算（出生竞争——每字耦合总量 ≤ K_CAP——强挤弱）
        row_sum = self.K.sum(axis=1)
        over = row_sum > K_CAP
        self.K[over] *= (K_CAP / row_sum[over])[:, None]
        self.K[:, over] *= (K_CAP / row_sum[over])[None, :]
        # 慢尺度频率吸引（共现耦合强 → 频率互相吸引 → 聚簇）
        for i in range(len(CHARS)):
            for j in range(i + 1, N_WORDS):
                kij = self.K[i, j]
                if kij > 0.08:      # 只吸引较强耦合（紧密词对——"学习"级——弱连接不传吸引）
                    pull = ETA_OMEGA * (self.omega[j] - self.omega[i]) * kij
                    self.omega[i] += pull
                    self.omega[j] -= pull
        self.omega = np.clip(self.omega, OMEGA_LO, OMEGA_HI)

    # ---- 湖检测：K 强耦合连通分量（模块 = 湖——同步域的前提是耦合网络） ----
    def clusters(self, th=0.05):
        n = len(CHARS)
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for i in range(n):
            for j in range(i + 1, n):
                if self.K[i, j] > th:
                    parent[find(i)] = find(j)
        comps = {}
        for i in range(n):
            r = find(i)
            comps.setdefault(r, []).append(i)
        return list(comps.values())

    def coherence(self, members):
        """簇内相干（相位锁定——同步域）"""
        ph = np.angle(self.z[members])
        return abs(np.mean(np.exp(1j * ph)))

    def predict(self, c, k=3):
        """字 → 联想（W = K 作为关联——湖间联想看强耦合）"""
        i = CHAR_IDX[c]
        row = self.K[i].copy()
        top = np.argsort(row)[::-1][:k]
        return [(CHARS[j], row[j]) for j in top if row[j] > 0.01]


def run():
    print("=== M5 阶段 38：无监督湖涌现（文字湖桥梁 ②——词类湖自组织——C2-02） ===\n")
    w = LakeWorld()
    om0 = w.omega.copy()
    for ep in range(40):
        w.learn_epoch(SENTS)
    # ---- exp1：湖涌现（K 耦合模块——湖 = 耦合网络的同步域前提） ----
    clusters = w.clusters()
    big = [c for c in clusters if len(c) >= 2]
    print(f"[exp1] K 耦合模块数 = {len(big)}（≥2 字模块）——频率标准差 {np.std(om0):.2f} → {np.std(w.omega):.2f}")
    # ---- exp2：语义对应 ----
    print("[exp2] 湖内容（语义检查）:")
    for ci, members in enumerate(big[:8]):
        chars = "".join(CHARS[i] for i in members)
        coh = w.coherence(members)
        print(f"      湖{ci}: [{chars}] 相干={coh:.2f}")
    # ---- exp3：同步域（模块内相位锁定） ----
    cohs = [w.coherence(c) for c in big]
    n_sync = sum(1 for c in cohs if c > COH_TH)
    print(f"[exp3] 同步模块（相干>{COH_TH}）= {n_sync}/{len(big)}"
          f"（{'相位锁定 ✓' if n_sync >= 2 else '同步不足——检查耦合强度'})")
    # ---- exp4：湖间联想（跨湖强耦合） ----
    print("[exp4] 跨湖联想（强耦合 K——湖间共现）:")
    for c in ["今", "学", "人", "国", "科", "世"][:4]:
        if c in CHAR_IDX:
            pred = w.predict(c)
            print(f"      '{c}' → {[(n, f'{v:.2f}') for n, v in pred]}")
    # 图：K 矩阵（模块结构）+ 频率分布
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    im = axes[0].imshow(w.K, cmap="viridis")
    axes[0].set_title("Coupling K (modules = lakes)")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    axes[1].hist(om0, bins=20, alpha=0.5, label="initial", color="gray")
    axes[1].hist(w.omega, bins=20, alpha=0.6, label="after", color="blue")
    axes[1].set_title("Frequency distribution (clustering)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("fig_stage38.png", dpi=110)
    print("\n[plot] saved fig_stage38.png")
    print("[done] stage38 lake emergence")


if __name__ == "__main__":
    run()
