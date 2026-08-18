# -*- coding: utf-8 -*-
"""
M5 阶段 5：语义蛋（R161 #15——蛋从"空壳"升级为"有内容"）
把决策环（阶段 2）叠入体验蛋（阶段 4）：同步域 × 维持闭环 × 语义内容

语义蛋 = 蛋 + 内容：
  4 组振荡器 = 4 个概念湖（C1-C4——组内同步）
  组间耦合 W(4×4) = 可学习联想（关联河道——R5 沉积）
  输入 = 概念激活（外部信息经转导——结构化）
  决策环 = 组间激活竞争（联想展开 → 竞争 → 选择——阶段 2 机制）
  闭环 = 输入→处理→输出→误差→回写（阶段 4 机制 + 联想学习）

实验 1：内容编码（输入概念 → 对应湖激活可读——信息进蛋）
实验 2：联想学习（配对输入 → 组间耦合增强 → 联想建立——蛋学习）
实验 3：决策（双输入竞争 → 选择胜出湖——蛋决策）
实验 4：蛋判据复查（同步域 × 闭环在语义负载下仍成立）
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(19)
N = 40
GROUPS = 4
SIZE = N // GROUPS
DT = 0.05

class SemanticEgg:
    def __init__(self, K_in=3.0, K_out0=0.02, gamma=0.3):
        # 组内强耦合（同步域）+ 组间弱耦合（联想——可学习）
        self.K = np.zeros((N, N))
        for g in range(GROUPS):
            idx = slice(g*SIZE, (g+1)*SIZE)
            self.K[idx, idx] = K_in
            np.fill_diagonal(self.K[idx, idx], 0.0)
        self.gamma = gamma
        self.omega = RNG.normal(1.0, 0.2, N)
        self.z = np.ones(N, dtype=complex)
        self.W = np.full((GROUPS, GROUPS), K_out0)   # 组间联想（可学习）
        np.fill_diagonal(self.W, 0.0)
        self.t = 0.0
        self.learn_rate = 0.02
        self.input_phase = np.full(GROUPS, np.nan)   # 输入相位锁定（稳定外部信号）

    def step(self, input_act):
        """输入（概念激活向量）→ 处理 → 输出（选择）→ 误差 → 回写"""
        # ① 输入：概念激活 → 驱动对应组（结构化——组内同相位——相位锁定：稳定外部信号）
        drive = np.zeros(N, dtype=complex)
        for g in range(GROUPS):
            if input_act[g] > 0:
                if np.isnan(self.input_phase[g]):
                    self.input_phase[g] = RNG.uniform(0, 2*np.pi)   # 信号到达时锁定相位
                drive[g*SIZE:(g+1)*SIZE] = input_act[g] * 0.8 * np.exp(1j * self.input_phase[g])
            else:
                self.input_phase[g] = np.nan   # 信号消失——相位解锁
        # ② 处理：组内同步 + 组间联想传播
        dz = -self.gamma * self.z + 1j * self.omega * self.z
        dz += (self.K * (self.z[None, :] - self.z[:, None])).sum(axis=1) / N
        # 组间联想：组激活 → 传播到关联组（W 耦合）
        for a in range(GROUPS):
            for b in range(GROUPS):
                if a != b and self.W[a, b] > 0.01:
                    act_a = np.abs(np.mean(self.z[a*SIZE:(a+1)*SIZE]))
                    dz[b*SIZE:(b+1)*SIZE] += self.W[a, b] * act_a * 0.3
        dz += drive
        self.z = self.z + dz * DT
        # 只防发散（幅度 > 3 才归一）——保留组间差异（信息编码）
        over = np.abs(self.z) > 3.0
        self.z[over] = self.z[over] / np.abs(self.z[over]) * 2.0
        # ③ 输出（T_out）：组激活度 → 选择（决策环——竞争胜者）
        act = np.array([np.abs(np.mean(self.z[g*SIZE:(g+1)*SIZE])) for g in range(GROUPS)])
        winner = int(np.argmax(act))
        # ④ 误差：输出选择 vs 输入激活（预测误差）
        error = np.sum(np.abs(input_act - act * 0.5))
        # ⑤ 回写：共现组间联想增强（Hebb 式配对学习——R5 沉积——一起激活的组耦合增强）
        in_groups = [g for g in range(GROUPS) if input_act[g] > 0]
        for a in range(len(in_groups)):
            for b in range(a+1, len(in_groups)):
                self.W[in_groups[a], in_groups[b]] += self.learn_rate * max(error, 0.1)
                self.W[in_groups[b], in_groups[a]] += self.learn_rate * max(error, 0.1)
        self.t += DT
        return winner, act, error

def run_all():
    egg = SemanticEgg()
    # ---- 实验 1：内容编码（每概念测试前重置基线——测稳态响应） ----
    def reset(egg):
        egg.z = np.full(N, 0.1 + 0j)

    acts_by_input = {}
    for c in range(GROUPS):
        reset(egg)
        inp = np.zeros(GROUPS); inp[c] = 1.0
        for _ in range(40):
            w, act, err = egg.step(inp)
        acts_by_input[c] = act
        print(f"[exp1] 输入概念{c+1}: 湖激活 = " +
              " ".join(f"L{g+1}:{act[g]:.2f}" for g in range(GROUPS)))
    enc_ok = all(acts_by_input[c][c] > acts_by_input[c][(c+1) % GROUPS] + 0.15 for c in range(GROUPS))
    print(f"       内容编码（输入概念 → 对应湖最强）={'✓' if enc_ok else '✗'}")

    # ---- 实验 2：联想学习 ----
    # 配对训练：输入 1 时同时输入 2（共现——配对学习）——重复 50 次
    for _ in range(50):
        inp = np.zeros(GROUPS); inp[0] = 1.0; inp[1] = 0.8
        egg.step(inp)
    w01 = egg.W[0, 1]
    print(f"[exp2] 配对训练后 W(1→2) = {w01:.3f}（初始 0.02——增强 = 联想建立）")
    # 测试：只输入 1 → 湖 2 是否被激活（联想检索）
    reset(egg)
    inp = np.zeros(GROUPS); inp[0] = 1.0
    for _ in range(40):
        w, act, err = egg.step(inp)
    l2 = act[1]
    print(f"       只输入 1: 湖 2 激活 = {l2:.3f}（vs 未关联湖 L4={act[3]:.2f}——联想检索——{'✓ 蛋学习' if l2 > act[3] + 0.1 else '✗'}）")

    # ---- 实验 3：决策 ----
    # 双输入竞争：1 强 2 弱 → 选择 1
    inp = np.zeros(GROUPS); inp[0] = 1.0; inp[2] = 0.4
    wins = [egg.step(inp)[0] for _ in range(10)]
    sel = max(set(wins), key=wins.count)
    print(f"[exp3] 双输入竞争（L1 强/L3 弱）: 10 次选择 = {wins} → 选择 {sel+1}")
    print(f"       选择强输入湖 = {'✓ 蛋决策（决策环工作）' if sel == 0 else '✗'}")

    # ---- 实验 4：蛋判据复查（语义负载下） ----
    # 持续信息流下同步域与闭环
    egg2 = SemanticEgg()
    coher, outs = [], []
    for k in range(600):
        inp = np.zeros(GROUPS)
        inp[RNG.integers(0, GROUPS)] = 1.0
        w, act, err = egg2.step(inp)
        g_phase = np.angle(egg2.z).reshape(GROUPS, -1)
        coher.append(np.mean(np.abs(np.mean(np.exp(1j * g_phase), axis=1))))
        outs.append(err)
    c_mean = np.mean(coher[-100:])
    print(f"[exp4] 语义负载下: 同步域相干 = {c_mean:.3f} | 误差活动 = {np.std(outs):.5f}")
    print(f"       判据复查 = {'✅ 蛋判据仍成立（同步域×闭环×内容）' if c_mean > 0.8 else '✗'}")
    return egg, acts_by_input, coher

if __name__ == "__main__":
    egg, acts, coher = run_all()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].imshow(egg.W, cmap="hot")
    axes[0].set_title("Learned group coupling W (associations)")
    axes[1].plot(coher)
    axes[1].set_title("Coherence under semantic load")
    axes[1].set_xlabel("step"); axes[1].set_ylabel("within-group coherence")
    fig.tight_layout()
    fig.savefig("fig_stage5.png", dpi=110)
    print("[plot] saved fig_stage5.png")
    print("[done] stage5 semantic egg complete")
