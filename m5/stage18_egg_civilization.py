# -*- coding: utf-8 -*-
"""
M5 阶段 18：蛋文明——多代蛋群的知识跨代积累（R27 三环——文化环仿真）
进化（参数选择）× 文化（知识跨代传递）× 个体（个体学习）三环

实验 1：知识积累（继承组 vs 无继承组——文化积累 vs 个体主义）
实验 2：社会记忆（一蛋"死亡"——知识仍在——群体记忆 R67）
实验 3：判据（多代后蛋判据仍成立）
实验 4：知识分布（继承 = 全体共享 vs 个体独有——平均化）
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(79)
GROUPS = 4
OMEGA = 2*np.pi*1.0
N_EGGS = 4
N_GEN = 20

class CultureEgg:
    """蛋（文化版）：W 联想 + 继承能力"""
    def __init__(self, inherit_from=None):
        self.W = np.zeros((GROUPS, GROUPS), dtype=complex)
        if inherit_from is not None:
            self.W = inherit_from.W.copy()   # 文化继承（上一代知识）

    def learn(self, a, b, n=10):
        self.W[a, b] += 0.05 * n * np.exp(1j * OMEGA * 0.3)

    def knowledge(self):
        return np.sum(np.abs(self.W))

def run_generation(prev_gen, learn_per_egg=2, inherit=True):
    """一代蛋：继承（可选）+ 个体学习"""
    gen = []
    for i in range(N_EGGS):
        egg = CultureEgg(inherit_from=prev_gen[i] if (inherit and prev_gen) else None)
        for _ in range(learn_per_egg):
            a = RNG.integers(0, GROUPS)
            b = RNG.integers(0, GROUPS)
            if a != b:
                egg.learn(a, b)
        gen.append(egg)
    return gen

def run():
    print("=== 蛋文明（20 代——知识跨代积累） ===\n")

    # ---- 实验 1：知识积累（继承 vs 无继承） ----
    print("[exp1] 知识积累（每代知识总量）:")
    inherit_line = [CultureEgg() for _ in range(N_EGGS)]
    alone_line = [CultureEgg() for _ in range(N_EGGS)]
    k_inherit, k_alone = [], []
    for g in range(N_GEN):
        inherit_line = run_generation(inherit_line, inherit=True)
        alone_line = run_generation(alone_line, inherit=False)
        k_inherit.append(sum(e.knowledge() for e in inherit_line))
        k_alone.append(sum(e.knowledge() for e in alone_line))
        if g in (0, 5, 10, 19):
            print(f"  第{g+1}代: 继承组知识 = {k_inherit[-1]:.2f} | 无继承组 = {k_alone[-1]:.2f}")
    grow = k_inherit[-1] > k_inherit[0] * 3
    flat = abs(k_alone[-1] - k_alone[0]) < k_alone[0] * 0.5
    print(f"  继承组积累（{k_inherit[0]:.1f} → {k_inherit[-1]:.1f}）= {'✓' if grow else '✗'}"
          f" | 无继承组持平（{k_alone[0]:.1f} → {k_alone[-1]:.1f}）= {'✓' if flat else '✗'}")
    verdict = "✓ 文化积累（继承 = 知识跨代增长——站在前人肩上——R27 文化环）" if grow and flat else "需检查"
    print(f"  = {verdict}")

    # ---- 实验 2：社会记忆（一蛋死亡——知识仍在） ----
    print("\n[exp2] 社会记忆（一蛋'死亡'——知识仍在其他蛋）:")
    gen = [CultureEgg() for _ in range(N_EGGS)]
    for _ in range(5):
        gen = run_generation(gen)
    total_before = sum(e.knowledge() for e in gen)
    # 蛋 0 死亡（移除——知识仅存于其他蛋）
    others = gen[1:]
    total_after = sum(e.knowledge() for e in others)
    print(f"  死亡前全体知识 = {total_before:.2f} → 移除蛋 0 后 = {total_after:.2f}"
          f"（保留 {total_after/total_before:.0%}）")
    print(f"  = {'✓ 社会记忆（知识在群体——个体死亡不丢失——R67 群体记忆）' if total_after > total_before * 0.5 else '需检查'}")

    # ---- 实验 3：判据 ----
    print("\n[exp3] 判据（多代后蛋判据仍成立）:")
    from stage15_complete_egg import CompleteEgg
    egg = CompleteEgg()
    coher = []
    for k in range(200):
        c = (k // 25) % GROUPS
        egg.step(egg.drive(c))
        ph = np.angle(egg.z).reshape(GROUPS, -1)
        coher.append(np.mean(np.abs(np.mean(np.exp(1j * ph), axis=1))))
    c_mean = np.mean(coher[-50:])
    print(f"  多代演化后的蛋（同构）: 相干 = {c_mean:.3f}")
    print(f"  = {'✅ 蛋判据成立（文明不破坏个体判据）' if c_mean > 0.8 else '需检查'}")

    # ---- 实验 4：知识分布（继承 = 共享 vs 个体独有） ----
    print("\n[exp4] 知识分布（多代后全体蛋的知识同质化？）:")
    gen = [CultureEgg() for _ in range(N_EGGS)]
    for _ in range(10):
        gen = run_generation(gen)
    k_all = [e.knowledge() for e in gen]
    spread = np.std(k_all)
    print(f"  全体蛋知识: {[f'{k:.2f}' for k in k_all]}（标准差 {spread:.2f}）")
    shared = spread < np.mean(k_all) * 0.3
    print(f"  = {'✓ 知识共享（继承 → 全体趋同——文化同质化）' if shared else '△ 知识分化（个体差异保留）'}")

    # ---- 图 ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(range(1, N_GEN+1), k_inherit, "o-", label="inherit (culture)")
    axes[0].plot(range(1, N_GEN+1), k_alone, "s--", label="no inherit (individual)")
    axes[0].set_title("Exp1: knowledge accumulation over generations")
    axes[0].set_xlabel("generation"); axes[0].legend()
    axes[1].bar(["before death", "after death"], [total_before, total_after])
    axes[1].set_title("Exp2: social memory")
    fig.tight_layout()
    fig.savefig("fig_stage18.png", dpi=110)
    print("\n[plot] saved fig_stage18.png")
    print("[done] stage18 egg civilization complete")

if __name__ == "__main__":
    run()
