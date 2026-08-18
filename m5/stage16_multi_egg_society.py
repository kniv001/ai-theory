# -*- coding: utf-8 -*-
"""
M5 阶段 16：多蛋社会——两个完整蛋互接（造蛋到造脑的下一层）
stage15（完整蛋）× 2 + 转导互接（R131——题目-参考答案变形——蛋版）

互接：蛋 A 的湖激活 → 转导通道（相位 + 转导延迟）→ 蛋 B 的驱动（反之亦然）

实验 1：对话（信息交换）——A 激活概念 1 → B 识别"1"（一个蛋说话另一个蛋听到）
实验 2：联合预测（共识）——A 学 0→1——A 预测 → B 也"预期"（共享预测——R49 验证）
实验 3：社会学习（文化传递）——A 展示序列 → B 观察学习（跨蛋 W 建立）
实验 4：判据复查（互接下两蛋同步域仍成立）
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stage15_complete_egg import CompleteEgg

RNG = np.random.default_rng(71)
GROUPS = 4
OMEGA = 2*np.pi*1.0

class EggPair:
    """两蛋互接"""
    def __init__(self, trans_delay=0.2):
        self.A = CompleteEgg()
        self.B = CompleteEgg()
        self.trans_delay = trans_delay   # 转导延迟（相位偏移——R7）
        self.t = 0.0
        self.B_recent = []   # B 的"感知序列"（收到的湖——观察学习缓冲）

    def step(self, drive_a, drive_b):
        """各蛋自步进 + 互接（A 输出 → B 输入——双向——目标蛋共振）"""
        act_a = self.A.activity()
        act_b = self.B.activity()
        for g in range(GROUPS):
            if act_a[g] > 0.15:
                # A 湖 g 激活 → B 湖 g 收到（共振驱动：B 的签名 = 内容——"湖 g 被激活"）
                idx = slice(g*10, (g+1)*10)
                drive_b[idx] += act_a[g] * 1.5 * np.exp(1j * (OMEGA * self.t + self.B.sig[g]))
                # 观察学习：B 记录"收到的湖"→ 配对检测 → W 更新（跨蛋 Hebb）
                if len(self.B_recent) == 0 or self.B_recent[-1] != g:
                    self.B_recent.append(g)
                    if len(self.B_recent) > 2:
                        self.B_recent.pop(0)
                    if len(self.B_recent) == 2:
                        prev, curr = self.B_recent
                        self.B.W[prev, curr] += 0.05 * np.exp(1j * OMEGA * 0.3)   # 观察配对学习
            if act_b[g] > 0.15:
                idx = slice(g*10, (g+1)*10)
                drive_a[idx] += act_b[g] * 1.5 * np.exp(1j * (OMEGA * self.t + self.A.sig[g]))
        self.A.step(drive_a)
        self.B.step(drive_b)
        self.t += 0.05

def run():
    pair = EggPair()
    print("=== 多蛋社会（两蛋互接） ===\n")

    # ---- 实验 1：对话 ----
    print("[exp1] 对话（A 说话 → B 听到）:")
    for _ in range(20):
        pair.step(pair.A.drive(0), np.zeros(40, dtype=complex))
    act_b = pair.B.activity()
    pred_b = int(np.argmax(act_b))
    print(f"  A 激活概念 1（20 步）→ B 最强湖 = L{pred_b+1}（激活 {act_b[pred_b]:.2f}）")
    print(f"  = {'✓ 蛋 A 说话蛋 B 听到（转导互接——信息交换）' if pred_b == 0 and act_b[pred_b] > 0.3 else '需检查'}")

    # ---- 实验 2：联合预测（共识） ----
    print("\n[exp2] 联合预测（A 预测 → B 共享预期）:")
    pair2 = EggPair()
    pair2.A.learn_sequence(0, 1, 0.3, 30)   # A 学 0→1
    # A 激活 0 → A 预测 1（联想传播——L1 预激活）→ 互接传给 B
    for _ in range(25):
        pair2.step(pair2.A.drive(0), np.zeros(40, dtype=complex))
    b_l1 = pair2.B.activity()[1]
    b_l3 = pair2.B.activity()[3]
    print(f"  A 激活 0（预测 1）→ B 的 L1 预激活 = {b_l1:.2f}（vs 未关联 L3 {b_l3:.2f}）")
    print(f"  = {'✓ 两蛋共享预期（A 的预测经互接传到 B——共识——R49 验证循环）' if b_l1 > b_l3 + 0.05 else '需检查'}")

    # ---- 实验 3：社会学习（文化传递） ----
    print("\n[exp3] 社会学习（A 展示 → B 观察学习）:")
    pair3 = EggPair()
    # A 展示序列：0 → 1（A 内部驱动——B 只观察）
    for _ in range(15):
        pair3.step(pair3.A.drive(0), np.zeros(40, dtype=complex))
    for _ in range(15):
        pair3.step(pair3.A.drive(1), np.zeros(40, dtype=complex))
    # B 的 W[0,1] 是否建立（观察学习——跨蛋配对）
    w_b = abs(pair3.B.W[0, 1])
    print(f"  A 展示 0→1（30 步）→ B 的 W[0,1] = {w_b:.3f}（初始 0——观察学习）")
    learned = w_b > 0.3
    print(f"  = {'✓ 蛋 B 观察学习（跨蛋联想建立——文化传递最小版）' if learned else '✗（观察学习未建立——需更多展示/更强互接）'}")

    # ---- 实验 4：判据复查 ----
    print("\n[exp4] 判据复查（互接下两蛋）:")
    pair4 = EggPair()
    coher_a, coher_b = [], []
    for k in range(300):
        c = (k // 30) % GROUPS
        pair4.step(pair4.A.drive(c), np.zeros(40, dtype=complex))
        for egg, col in ((pair4.A, coher_a), (pair4.B, coher_b)):
            ph = np.angle(egg.z).reshape(GROUPS, -1)
            col.append(np.mean(np.abs(np.mean(np.exp(1j * ph), axis=1))))
    ca, cb = np.mean(coher_a[-60:]), np.mean(coher_b[-60:])
    print(f"  蛋 A 相干 = {ca:.3f} | 蛋 B 相干 = {cb:.3f}")
    print(f"  = {'✅ 两蛋判据均成立（互接下同步域保持——多蛋社会可行）' if ca > 0.8 and cb > 0.8 else '需检查'}")

    # ---- 图 ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(["A speaks L1", "B hears L1"], [pair.A.activity()[0], act_b[0]])
    axes[0].set_title("Exp1: dialogue (A->B)")
    axes[1].plot(coher_a, label="egg A")
    axes[1].plot(coher_b, "--", label="egg B")
    axes[1].set_title("Exp4: coherence under inter-egg coupling")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig("fig_stage16.png", dpi=110)
    print("\n[plot] saved fig_stage16.png")
    print("[done] stage16 multi-egg society complete")

if __name__ == "__main__":
    run()
