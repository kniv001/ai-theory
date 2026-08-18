# -*- coding: utf-8 -*-
"""
M5 阶段 8：多层联想（R161 #9——候选重生成/联想树多层）
验证 R120/R122（联想展开 = 树——多层）+ C105（截止——逐层成本交叉）

三层联想网络：
  概念层（4）——属性层（8）——情境层（6）
  概念→属性：每个概念关联 2-3 个属性（what it has）
  属性→情境：每个属性关联 1-2 个情境（where it appears）

实验 1：多层传播——输入概念 → 属性层激活（其属性）→ 情境层激活（其情境）——联想链完整
实验 2：候选重生成——决策逐层展开——候选随层增加（需求高 → 展开深）
实验 3：截止（C105）——需求阈值高 → 展开深；低 → 浅层即止
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(31)
N_CON, N_ATTR, N_SIT = 4, 8, 6

def make_net():
    """三层关联（概念→属性→情境）——半随机预置"""
    C2A = np.zeros((N_CON, N_ATTR))
    for c in range(N_CON):
        for a in RNG.choice(N_ATTR, size=RNG.integers(2, 4), replace=False):
            C2A[c, a] = RNG.uniform(0.6, 1.0)
    A2S = np.zeros((N_ATTR, N_SIT))
    for a in range(N_ATTR):
        for s in RNG.choice(N_SIT, size=RNG.integers(1, 3), replace=False):
            A2S[a, s] = RNG.uniform(0.6, 1.0)
    return C2A, A2S

def propagate(C2A, A2S, concept, depth=1):
    """多层联想传播：概念 → 属性（depth=1）→ 情境（depth=2）"""
    attr_act = C2A[concept]
    sit_act = np.zeros(N_SIT)
    if depth >= 2:
        sit_act = A2S.T @ attr_act
    return attr_act, sit_act

def run():
    C2A, A2S = make_net()
    print("关联结构（概念→属性）:")
    for c in range(N_CON):
        attrs = np.where(C2A[c] > 0)[0]
        print(f"  概念{c+1} → 属性 {attrs + 1}")

    # ---- 实验 1：多层传播 ----
    print("\n[exp1] 多层联想传播:")
    for c in range(2):
        attr_act, sit_act = propagate(C2A, A2S, c, depth=2)
        top_attr = np.argsort(-attr_act)[:3]
        top_sit = np.argsort(-sit_act)[:2]
        print(f"  概念{c+1} → 属性层 {top_attr + 1}（激活 {np.round(attr_act[top_attr], 2)}）"
              f" → 情境层 {top_sit + 1}（激活 {np.round(sit_act[top_sit], 2)}）")
    # 验证：概念 → 直接关联属性（第一层非零）→ 属性关联情境（第二层非零）
    ok1 = True
    for c in range(N_CON):
        aa, sa = propagate(C2A, A2S, c, depth=2)
        if np.sum(aa > 0) == 0 or np.sum(sa > 0) == 0:
            ok1 = False
    print(f"  联想链完整（概念→属性→情境 非零）= {'✓' if ok1 else '✗'}")

    # ---- 实验 2/3：候选重生成 + 截止（C105） ----
    print("\n[exp2/3] 候选重生成与截止:")
    # 决策：输入概念 → 逐层展开候选——每层评估（候选价值 = 激活）——最佳 > 需求阈值则停
    # 需求阈值低 → 第一层（属性）即满足；高 → 展开到情境层
    depth_stats = []
    for demand in (0.3, 0.6, 0.9):
        depths = []
        for c in range(N_CON):
            attr_act, sit_act = propagate(C2A, A2S, c, depth=2)
            # 层 1 候选：属性激活
            best1 = np.max(attr_act) if len(attr_act) else 0
            if best1 >= demand:
                depths.append(1)
            else:
                # 重生成：层 2 候选（情境——组合价值）
                best2 = np.max(sit_act) if len(sit_act) else 0
                depths.append(2 if best2 >= demand else 3)   # 3 = 均不满足（放弃）
        depth_stats.append(depths)
        print(f"  需求阈值={demand}: 决策深度分布 = {depths}（1=属性层/2=情境层/3=放弃）")
    # 验证：需求越高展开越深（C105 成本交叉的树版本——边际收益 vs 需求的比较）
    avg_depths = [np.mean(d) for d in depth_stats]
    mono = avg_depths[0] <= avg_depths[1] <= avg_depths[2]
    print(f"  平均深度: {np.round(avg_depths, 2)}——需求↑深度↑ = {'✓ 候选重生成+截止（C105）' if mono else '✗'}")

    # ---- 图 ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].imshow(C2A, cmap="hot", aspect="auto")
    axes[0].set_title("Concept->Attribute links")
    axes[0].set_xlabel("attribute"); axes[0].set_ylabel("concept")
    axes[1].bar(["demand 0.3", "demand 0.6", "demand 0.9"], avg_depths)
    axes[1].set_title("Decision depth vs demand (C105)")
    fig.tight_layout()
    fig.savefig("fig_stage8.png", dpi=110)
    print("\n[plot] saved fig_stage8.png")
    print("[done] stage8 multi-level association complete")

if __name__ == "__main__":
    run()
