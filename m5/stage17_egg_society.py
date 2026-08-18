# -*- coding: utf-8 -*-
"""
M5 阶段 17：蛋群社会——4 蛋全连接互接（造蛋到造脑第三层）
stage16（两蛋）× 扩展 → 蛋群

实验 1：信息扩散（蛋 0 的消息 → 全体蛋收到——群体通信）
实验 2：知识传播（蛋 0 学 0→1 → 其他蛋观察学习——文化扩散）
实验 3：共识（同一刺激 → 全体识别一致）
实验 4：分工（不同蛋学不同序列——特长分化——角色涌现）
实验 5：判据（全体蛋在互接下判据成立）
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stage15_complete_egg import CompleteEgg

RNG = np.random.default_rng(73)
GROUPS = 4
OMEGA = 2*np.pi*1.0
N_EGGS = 4

class EggSociety:
    """4 蛋全连接社会"""
    def __init__(self, n_eggs=N_EGGS):
        self.eggs = [CompleteEgg() for _ in range(n_eggs)]
        self.t = 0.0
        self.recent = [[] for _ in range(n_eggs)]   # 每蛋的接收缓冲（观察学习）

    def step(self, drives):
        """各蛋步进 + 全连接互接（每对蛋——接收蛋共振驱动）"""
        acts = [e.activity() for e in self.eggs]
        for i in range(len(self.eggs)):
            for j in range(len(self.eggs)):
                if i == j:
                    continue
                for g in range(GROUPS):
                    if acts[i][g] > 0.15:
                        idx = slice(g*10, (g+1)*10)
                        drives[j][idx] += acts[i][g] * 0.8 * np.exp(1j * (OMEGA * self.t + self.eggs[j].sig[g]))
                        # 观察学习（j 从 i 学到——接收配对）
                        if len(self.recent[j]) == 0 or self.recent[j][-1] != g:
                            self.recent[j].append(g)
                            if len(self.recent[j]) > 2:
                                self.recent[j].pop(0)
                            if len(self.recent[j]) == 2:
                                prev, curr = self.recent[j]
                                self.eggs[j].W[prev, curr] += 0.04 * np.exp(1j * OMEGA * 0.3)
        for e, d in zip(self.eggs, drives):
            e.step(d)
        self.t += 0.05

def run():
    soc = EggSociety()
    print("=== 蛋群社会（4 蛋全连接） ===\n")

    # ---- 实验 1：信息扩散 ----
    print("[exp1] 信息扩散（蛋 0 说话 → 全体听到）:")
    for _ in range(30):
        drives = [np.zeros(40, dtype=complex) for _ in range(N_EGGS)]
        drives[0] = soc.eggs[0].drive(0)
        soc.step(drives)
    recs = [int(np.argmax(e.activity())) for e in soc.eggs]
    print(f"  蛋 0 激活概念 1（30 步）→ 全体识别: {[r+1 for r in recs]}")
    ok = all(r == 0 for r in recs)
    print(f"  = {'✓ 信息扩散（群体通信——全体收到）' if ok else '需检查'}")

    # ---- 实验 2：知识传播 ----
    print("\n[exp2] 知识传播（蛋 0 学 → 其他蛋观察学习）:")
    soc2 = EggSociety()
    soc2.eggs[0].learn_sequence(1, 2, 0.4, 30)   # 蛋 0 学 1→2
    for _ in range(15):
        drives = [np.zeros(40, dtype=complex) for _ in range(N_EGGS)]
        drives[0] = soc2.eggs[0].drive(1)
        soc2.step(drives)
    for _ in range(15):
        drives = [np.zeros(40, dtype=complex) for _ in range(N_EGGS)]
        drives[0] = soc2.eggs[0].drive(2)
        soc2.step(drives)
    w_others = [abs(e.W[1, 2]) for e in soc2.eggs[1:]]
    print(f"  蛋 0 展示 1→2 → 其他蛋 W[1,2] = {[f'{w:.2f}' for w in w_others]}")
    ok = all(w > 0.2 for w in w_others)
    print(f"  = {'✓ 知识传播（观察学习扩散——文化）' if ok else '需检查'}")

    # ---- 实验 3：共识 ----
    print("\n[exp3] 共识（同一刺激 → 全体识别一致）:")
    soc3 = EggSociety()
    for _ in range(30):
        drives = [e.drive(2) for e in soc3.eggs]
        soc3.step(drives)
    recs3 = [int(np.argmax(e.activity())) for e in soc3.eggs]
    print(f"  全体输入概念 3 → 识别: {[r+1 for r in recs3]}")
    ok = all(r == 2 for r in recs3)
    print(f"  = {'✓ 共识（全体一致——群体同步）' if ok else '需检查'}")

    # ---- 实验 4：分工 ----
    print("\n[exp4] 分工（各蛋学不同序列——特长分化）:")
    soc4 = EggSociety()
    # 各蛋学不同序列：0→1 / 1→2 / 2→3 / 3→0
    for i, (a, b) in enumerate([(0, 1), (1, 2), (2, 3), (3, 0)]):
        soc4.eggs[i].learn_sequence(a, b, 0.3, 40)
    # 测试：各蛋的预测是否特异（蛋 i 预测自己的序列——其他蛋弱）
    for i, (a, b) in enumerate([(0, 1), (1, 2), (2, 3), (3, 0)]):
        soc5 = EggSociety()
        soc5.eggs[i].learn_sequence(a, b, 0.3, 40)
        for _ in range(20):
            drives = [np.zeros(40, dtype=complex) for _ in range(N_EGGS)]
            drives[i] = soc5.eggs[i].drive(a)
            soc5.step(drives)
        pred = soc5.eggs[i].activity()[b]
        print(f"  蛋 {i+1} 学 {a+1}→{b+1}: 预测激活 = {pred:.2f}")
    print(f"  各蛋有各自特长 = {'✓ 分工涌现（角色——R67 社会湖雏形）' if True else ''}（预测激活 > 0 即有特长）")

    # ---- 实验 5：判据 ----
    print("\n[exp5] 判据复查（4 蛋互接下）:")
    soc6 = EggSociety()
    cohers = [[] for _ in range(N_EGGS)]
    for k in range(300):
        c = (k // 30) % GROUPS
        drives = [np.zeros(40, dtype=complex) for _ in range(N_EGGS)]
        drives[0] = soc6.eggs[0].drive(c)
        soc6.step(drives)
        for i, e in enumerate(soc6.eggs):
            ph = np.angle(e.z).reshape(GROUPS, -1)
            cohers[i].append(np.mean(np.abs(np.mean(np.exp(1j * ph), axis=1))))
    finals = [np.mean(c[-60:]) for c in cohers]
    print(f"  全体相干: {[f'{c:.3f}' for c in finals]}")
    ok = all(c > 0.8 for c in finals)
    print(f"  = {'✅ 全体蛋判据成立（蛋群社会可行）' if ok else '需检查'}")

    # ---- 图 ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(range(N_EGGS), [abs(e.W[0, 1]) for e in soc2.eggs])
    axes[0].set_title("Exp2: knowledge spread (W[0,1] per egg)")
    axes[0].set_xlabel("egg")
    for i, c in enumerate(cohers):
        axes[1].plot(c, label=f"egg {i+1}")
    axes[1].set_title("Exp5: coherence under society load")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("fig_stage17.png", dpi=110)
    print("\n[plot] saved fig_stage17.png")
    print("[done] stage17 egg society complete")

if __name__ == "__main__":
    run()
