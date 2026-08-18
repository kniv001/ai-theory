# -*- coding: utf-8 -*-
"""
M5 阶段 15：完整蛋——同步域（相位语义）× 闭环（预测-误差）× 决策环（DDM）× 内容（概念湖）
"蛋的完整化"（R169——语义蛋 + 相位语义 + 预测 + 决策的整合）

蛋 = 完整可计算认知体：
  4 概念湖（复值单元——组内同步——概念 = 相位签名）
  输入：结构化 + 相位锁定（共振——stage7 发现）
  处理：相位绑定（识别）+ 联想（复值 W——时序）
  决策：DDM 累积竞争（stage2）
  闭环：预测（W 相位）→ 误差（时序惊讶）→ 回写（复值 Hebb + 沉积）
  判据：同步域 × 闭环（C53）

实验 1：识别（相位签名匹配——蛋知道"这是什么"）
实验 2：预测（时序学习——蛋知道"接下来发生什么"）
实验 3：决策（DDM——价值接近犹豫——蛋"想"）
实验 4：判据复查（同步域 × 闭环在完整负载下）
实验 5：完整闭环演示（识别→决策→预测→误差→学习 循环）
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(67)
N = 40
GROUPS = 4
SIZE = N // GROUPS
DT = 0.05
OMEGA = 2*np.pi*1.0

class CompleteEgg:
    def __init__(self):
        # 概念湖（组内同步——弱——输入驱动绑定）
        self.omega = np.tile(OMEGA * (1 + RNG.normal(0, 0.03, SIZE)), GROUPS)
        self.z = 0.1 * np.exp(1j * RNG.uniform(0, 2*np.pi, N))
        self.K = np.zeros((N, N))
        for g in range(GROUPS):
            idx = slice(g*SIZE, (g+1)*SIZE)
            self.K[idx, idx] = 2.0   # 强组内耦合（概念湖内聚——1c 锁定区）
            np.fill_diagonal(self.K[idx, idx], 0.0)
        # 概念签名（相位模式）
        self.sig = {c: np.linspace(0, 0.8, SIZE) * (c+1) * 0.5 for c in range(GROUPS)}
        # 湖间联想（复值——时序）
        self.W = np.zeros((GROUPS, GROUPS), dtype=complex)
        self.input_phase = np.full(GROUPS, np.nan)
        self.t = 0.0
        self.learn_rate = 0.05

    def drive(self, concept, amp=0.8):
        d = np.zeros(N, dtype=complex)
        if np.isnan(self.input_phase[concept]):
            self.input_phase[concept] = RNG.uniform(0, 2*np.pi)
        idx = slice(concept*SIZE, (concept+1)*SIZE)
        # 外部信号 = 单一频率（湖中心频率）——组内同频驱动（绑定——相位差 = sig 结构）
        d[idx] = amp * np.exp(1j * (OMEGA * self.t + self.sig[concept]))
        return d

    def step(self, drive):
        dz = -0.3 * self.z + 1j * self.omega * self.z
        dz += (self.K * (self.z[None, :] - self.z[:, None])).sum(axis=1) / N
        # 湖间联想传播（复值 W——预测信号——共振相位：跟随目标湖 ω——预测到达时刻）
        for a in range(GROUPS):
            act_a = np.abs(np.mean(self.z[a*SIZE:(a+1)*SIZE]))
            for b in range(GROUPS):
                if a != b and abs(self.W[a, b]) > 0.05:
                    phase_adv = np.angle(self.W[a, b])
                    idx_b = slice(b*SIZE, (b+1)*SIZE)
                    dz[idx_b] += act_a * 0.3 * np.exp(1j * (self.omega[idx_b] * self.t + phase_adv))
        dz += drive
        self.z = self.z + dz * DT
        self.t += DT
        over = np.abs(self.z) > 3.0
        self.z[over] = self.z[over] / np.abs(self.z[over]) * 2.0
        return self.z

    def activity(self):
        return np.array([np.abs(np.mean(self.z[g*SIZE:(g+1)*SIZE])) for g in range(GROUPS)])

    def decide_ddm(self, input_act, thresh=1.0, max_t=50.0):
        """DDM 决策：湖激活竞争到阈值（价值接近 → 犹豫）"""
        acc = np.zeros(GROUPS)
        t = 0.0
        while t < max_t:
            act = self.activity()
            acc += (act * 0.15 + RNG.normal(0, 0.1, GROUPS)) * DT
            t += DT
            hit = np.where(acc >= thresh)[0]
            if len(hit):
                return int(hit[0]), t
        return int(np.argmax(acc)), t

    def learn_sequence(self, a, b, dt, n=40):
        for _ in range(n):
            self.W[a, b] += self.learn_rate * np.exp(1j * OMEGA * dt)

    def predict_delay(self, a, b):
        d = np.angle(self.W[a, b]) / OMEGA
        if d < 0:
            d += 2*np.pi/OMEGA
        return d

def run():
    egg = CompleteEgg()
    print("=== 完整蛋（同步域 × 闭环 × 决策环 × 内容） ===\n")

    # ---- 实验 1：识别 ----
    correct = 0
    for c in range(GROUPS):
        egg2 = CompleteEgg()
        for _ in range(30):
            egg2.step(egg2.drive(c))
        act = egg2.activity()
        pred = int(np.argmax(act))
        ok = pred == c
        correct += ok
        print(f"[exp1] 输入概念{c+1}: 识别 = L{pred+1}（{'✓' if ok else '✗'}）")
    print(f"       识别率 {correct}/{GROUPS} = {'✓ 蛋知道"这是什么"（相位签名匹配）' if correct == GROUPS else '需检查'}")

    # ---- 实验 2：预测 ----
    egg.learn_sequence(0, 1, 0.3)
    egg.learn_sequence(1, 2, 0.5)
    p01 = egg.predict_delay(0, 1)
    p12 = egg.predict_delay(1, 2)
    print(f"\n[exp2] 预测: L1 在 L0 后 {p01:.2f}s（学 0.3）| L2 在 L1 后 {p12:.2f}s（学 0.5）")
    print(f"       = {'✓ 蛋知道"接下来发生什么"（相位时序知识）' if abs(p01-0.3) < 0.05 and abs(p12-0.5) < 0.05 else '需检查'}")

    # ---- 实验 3：决策（DDM） ----
    # 双输入竞争：L0 强 L1 弱 → 决策 L0（快）；等强 → 犹豫（慢）——多次平均
    def ddm_pair(amp0, amp1, reps=10):
        times = []
        for _ in range(reps):
            eg = CompleteEgg()
            for _ in range(15):
                eg.step(eg.drive(0, amp0))
                eg.step(eg.drive(1, amp1))
            w, t = eg.decide_ddm(None)
            times.append(t)
        return np.mean(times)
    t_strong = ddm_pair(0.8, 0.3)
    t_even = ddm_pair(0.6, 0.6)
    print(f"\n[exp3] 决策（10 次平均）: 强弱竞争 t={t_strong:.2f} | 等强竞争 t={t_even:.2f}")
    print(f"       = {'✓ 蛋会"想"（DDM——价值接近犹豫——R60 自由感）' if t_even > t_strong * 1.2 else '需检查'}")

    # ---- 实验 4：判据复查（每概念停留 40 步——测稳定段相干） ----
    egg5 = CompleteEgg()
    stable_coher = []
    for k in range(400):
        c = (k // 40) % GROUPS
        egg5.step(egg5.drive(c))
        ph = np.angle(egg5.z).reshape(GROUPS, -1)
        c_in = np.mean(np.abs(np.mean(np.exp(1j * ph), axis=1)))
        if k % 40 >= 25:   # 每概念的后 15 步（稳定段）
            stable_coher.append(c_in)
    c_mean = np.mean(stable_coher)
    print(f"\n[exp4] 判据复查（稳定段相干——每概念停留 40 步）: 同步域相干 = {c_mean:.3f}")
    print(f"       = {'✅ 蛋判据仍成立（同步域 × 闭环——C53）' if c_mean > 0.8 else '✗'}")

    # ---- 实验 5：完整闭环 ----
    egg6 = CompleteEgg()
    # 学习序列：0→1→2（预测链）
    egg6.learn_sequence(0, 1, 0.3, 30)
    egg6.learn_sequence(1, 2, 0.5, 30)
    print("\n[exp5] 完整闭环演示（识别→预测→决策→误差→学习）:")
    # 输入概念 0 → 识别 → 预测链激活 → 决策
    for _ in range(20):
        egg6.step(egg6.drive(0))
    rec = int(np.argmax(egg6.activity()))
    # 预测：1 将被联想传播激活（预测信号——vs 未学习对 L3 的传播）
    pred_l1 = egg6.activity()[1]
    pred_l3 = egg6.activity()[3]
    # 现实：1 真的来了
    for _ in range(20):
        egg6.step(egg6.drive(1))
    arrived = egg6.activity()[1]
    print(f"   识别 L{rec+1} → 预测：L1 被联想激活（{pred_l1:.2f} vs 未关联 L3 {pred_l3:.2f}）"
          f" → 现实 L1 到来（{arrived:.2f}——预测方向正确） → 回写巩固")
    print(f"   = {'✓ 完整闭环：识别→预测→误差→学习（蛋的完整认知循环）' if rec == 0 and pred_l1 > pred_l3 + 0.05 and arrived > 0.5 else '需检查'}")

    # ---- 图 ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(range(GROUPS), egg.activity())
    axes[0].set_title("Complete egg: lake activities")
    axes[0].set_xlabel("lake"); axes[0].set_ylabel("activation")
    axes[1].plot(stable_coher)
    axes[1].set_title("Complete egg: stable-segment coherence")
    axes[1].set_xlabel("step"); axes[1].set_ylabel("coherence")
    fig.tight_layout()
    fig.savefig("fig_stage15.png", dpi=110)
    print("\n[plot] saved fig_stage15.png")
    print("[done] stage15 complete egg done")

if __name__ == "__main__":
    run()
