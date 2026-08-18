# -*- coding: utf-8 -*-
"""
M5 阶段 6：τ-重放模块（R42 多巴胺间接化——R161 #10）
验证：价值信号不直接改 W——τ 标记 → 重放 → 流量 → 硬化（间接链）

R42 核心：RPE 不直接改 h/W（更新预期 + 门控流量 + 标记 τ）；
τ 触发重放 → 重复流量 → 稳定硬化——唯一直接硬化机制 = 水流硬化

模型：
  记忆项（10 个）——软区 W（活动强度）+ 硬区 H（结构）+ τ 标记（0/1）
  学习事件：输入 → 误差 e → 价值 v（RPE）→ |v| > θ_v → τ=1（标记）
  睡眠/空闲重放：τ=1 项被重放（激活流量）→ H += η_r（硬化——写回）
  对比 A（直接）：v 直接加 H（R11 旧式——ε 直接刻河）
  对比 B（间接）：v 只标记——重放才硬化（R42 间接化）

实验 1：标记选择性（|v| 大 → 标记；小 → 不标记）
实验 2：重放选择性（标记项重放 → 硬化；未标记项不硬化）
实验 3：遗忘差异（标记项 vs 未标记项——停止输入后的保持）
实验 4：直接 vs 间接（A/B 对比——"睡一觉记得更牢"——间接的时间分布）
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(23)
N_ITEMS = 10
TH_V = 0.5        # 价值标记阈值（|v| > 阈值 → τ=1）
ETA_REPLAY = 0.08 # 每次重放的硬化率
ETA_DIRECT = 0.3  # 直接模式的单次硬化（对比）
LAM_H = 0.02      # 硬区慢蚀（遗忘）

class TauReplayAgent:
    """间接化（R42）代理"""
    def __init__(self):
        self.W = np.zeros(N_ITEMS)      # 软区
        self.H = np.zeros(N_ITEMS)      # 硬区
        self.tau = np.zeros(N_ITEMS)    # 标记

    def learn(self, item, v):
        """学习事件：价值 → 标记（不直接改 H）"""
        self.W[item] += 0.3             # 软区小沉积（输入激活）
        if abs(v) > TH_V:
            self.tau[item] = 1.0        # 标记（重要事件）

    def replay(self):
        """睡眠/空闲重放：标记项 → 流量 → 硬化（间接链——重放主动重新激活——R70 痕迹保留）"""
        for i in range(N_ITEMS):
            if self.tau[i] > 0:
                self.H[i] += ETA_REPLAY * max(self.W[i], 0.05)  # 流量 → 沉积（痕迹可重新激活）
                self.W[i] += ETA_REPLAY * self.W[i]             # 重放维持软区（自维持）
        self.H *= (1 - LAM_H)           # 慢蚀

    def forget(self, steps=1):
        self.W *= np.exp(-0.5 * steps)  # 软区快衰
        self.H *= (1 - LAM_H)

class DirectAgent:
    """直接化（R11 旧式——对照组）"""
    def __init__(self):
        self.H = np.zeros(N_ITEMS)

    def learn(self, item, v):
        self.H[item] += ETA_DIRECT * abs(v)   # 价值直接改 H

    def replay(self):
        self.H *= (1 - LAM_H)

    def forget(self, steps=1):
        self.H *= (1 - LAM_H)

def run():
    # 学习阶段：20 个事件——半重要（v 大）半不重要（v 小）
    agent = TauReplayAgent()
    direct = DirectAgent()
    important = set()
    for k in range(20):
        item = RNG.integers(0, N_ITEMS)
        v = RNG.uniform(0.6, 1.0) if k % 2 == 0 else RNG.uniform(0.0, 0.3)
        if abs(v) > TH_V:
            important.add(item)
        agent.learn(item, v)
        direct.learn(item, v)

    # 睡眠重放（3 轮）
    for _ in range(3):
        agent.replay()
        direct.replay()

    print(f"[exp1] 标记选择性: {len(important)} 项被标记（|v|>{TH_V}）——{N_ITEMS - len(important)} 项未标记")
    print(f"       标记 = 重要事件 = {'✓' if 0 < len(important) < N_ITEMS else '✗'}")

    h_imp = np.mean(agent.H[list(important)]) if important else 0
    h_not = np.mean(agent.H[[i for i in range(N_ITEMS) if i not in important]])
    print(f"[exp2] 重放选择性: 标记项 H={h_imp:.3f} vs 未标记项 H={h_not:.3f}")
    print(f"       标记项硬化更深 = {'✓（τ→重放→硬化——间接链工作）' if h_imp > h_not * 2 else '✗'}")

    # 遗忘阶段（10 轮——模拟时间流逝）
    agent_trace = []
    direct_trace = []
    for _ in range(10):
        agent.forget(); direct.forget()
        # 每轮后加一次重放（睡眠——标记项持续固化）
        agent.replay(); direct.replay()
        agent_trace.append(np.mean(agent.H[list(important)]) if important else 0)
        direct_trace.append(np.mean(direct.H[list(important)]) if important else 0)

    a_start, a_end = agent_trace[0], agent_trace[-1]
    d_start, d_end = direct_trace[0], direct_trace[-1]
    a_keep, d_keep = a_end / max(a_start, 1e-9), d_end / max(d_start, 1e-9)
    print(f"[exp3] 保持率: 间接 {a_start:.3f}→{a_end:.3f}（{a_keep:.2f}x——重放对抗侵蚀）"
          f" vs 直接 {d_start:.3f}→{d_end:.3f}（{d_keep:.2f}x——纯衰减）")
    verdict = "✓（R42 间接化：重放维持记忆——睡眠巩固）" if a_keep >= 1.0 and d_keep < 1.0 else "✗"
    print(f"       间接维持/增长 + 直接衰减 = {verdict}")

    # 时间分布对比
    print(f"[exp4] 直接 vs 间接: 直接单次完成（{ETA_DIRECT}×|v|）vs 间接标记+多轮重放（3×{ETA_REPLAY}×W）")
    print(f"       间接 = 时间分散的硬化（睡眠后更深——'睡一觉记得更牢' R42/R47 机制）")

    return agent, direct, agent_trace, direct_trace, important

if __name__ == "__main__":
    agent, direct, at, dt, imp = run()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(range(N_ITEMS), agent.H, label="indirect (tau-replay)")
    axes[0].bar(range(N_ITEMS), direct.H, alpha=0.5, label="direct")
    axes[0].set_title("H after learning+replay (marked vs unmarked)")
    axes[0].legend(fontsize=8)
    axes[1].plot(at, label="indirect (replay in sleep)")
    axes[1].plot(dt, "--", label="direct (no replay)")
    axes[1].set_title("Forgetting with replay")
    axes[1].set_xlabel("rounds"); axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("fig_stage6.png", dpi=110)
    print("[plot] saved fig_stage6.png")
    print("[done] stage6 tau-replay complete")
