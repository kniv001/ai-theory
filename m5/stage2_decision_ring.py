# -*- coding: utf-8 -*-
"""
M5 阶段 2：S1 决策环最小仿真（叠在物理层上）
验证 R120/R129 决策环五层 + C104（候选生成）/ C105（截止）/ P6（注意力=硬化控制杆）

模型：
  地形：N 个记忆项，强度 W（河道）
  联想：关联矩阵 A[q, :]（联想树——query 的候选域）
  决策环：
    ① 输入：query q（价值注入）
    ② 数据层：联想展开（A[q,:] 激活——候选 = 激活超阈项）
    ③ 服务层：价值评估（v_i = W_i + 噪声）+ 竞争（累积到阈值——决策时间 = 到达时刻）
    ④ 行为层：选择（第一个到阈值的候选胜出）
    ⑤ 反馈层：回写 W_winner += η（沉积——聚焦硬化）

实验 1：联想展开顺序（强河道先行——C72/C104）
实验 2：决策时间 vs 价值接近度（DDM——R60 自由感——价值接近=决策慢）
实验 3：截止（C105 成本交叉——边际收益 < 搜索成本 → 停止）
实验 4：回写（P6——聚焦的河道沉积更深——重测同 query 决策更快/更偏）
实验 5：闭环演化（1000 决策——高频项增强——习惯形成 C77）

输出：控制台摘要 + fig_stage2.png + npz
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(7)
N = 12
THRESH = 1.0          # 决策阈值
ETA = 0.05            # 回写率（沉积）
ACT_TH = 0.3          # 联想激活阈值（候选生成）

def make_env():
    """地形：W 强度 + 关联矩阵"""
    W = np.ones(N) + RNG.normal(0, 0.2, N)
    # 稀疏关联：每个 query 关联 3-5 项（联想树）
    A = np.zeros((N, N))
    for q in range(N):
        nbrs = RNG.choice(N, size=RNG.integers(3, 6), replace=False)
        nbrs = nbrs[nbrs != q]
        A[q, nbrs] = RNG.uniform(0.5, 1.0, len(nbrs))
    return W, A

def candidates(W, A, q):
    """② 数据层：联想展开——激活超阈候选（按激活强度排序）"""
    act = A[q] * W           # 激活 = 关联 × 目标强度（共现×锚定 R30）
    cands = np.where(act > ACT_TH)[0]
    order = cands[np.argsort(-act[cands])]   # 强河道先行
    return order, act[order]

def decide(W, A, q, dt=0.01, noise=0.15):
    """③④ 服务层+行为层：累积竞争——第一个到阈值者胜出——返回 (胜者, 决策时间)"""
    order, act = candidates(W, A, q)
    if len(order) == 0:
        return None, np.inf
    # 累积：drift = 价值（激活）——竞争到阈值
    acc = np.zeros(len(order))
    t = 0.0
    while t < 50.0:
        acc += (act * 0.15 + RNG.normal(0, noise, len(order))) * dt
        t += dt
        hit = np.where(acc >= THRESH)[0]
        if len(hit):
            return order[hit[0]], t
    return order[np.argmax(acc)], t

# ---------- 实验 1：联想展开顺序 ----------
def run_exp1():
    W, A = make_env()
    q = 3
    order, act = candidates(W, A, q)
    # 展开顺序是否 = 激活排序（强河道先行）
    sorted_ok = all(act[i] >= act[i+1] for i in range(len(act)-1))
    print(f"[exp1] query={q}: 候选={order.tolist()} | 激活排序={sorted_ok}")
    print(f"       激活值: {np.round(act, 3).tolist()}")
    return order, act

# ---------- 实验 2：决策时间 vs 价值接近度 ----------
def run_exp2():
    W, A = make_env()
    # 构造双候选（q 关联到 i,j——调节价值差）
    q = 5
    i, j = 0, 1
    A[q, i] = 1.0; A[q, j] = 1.0
    deltas = [0.02, 0.1, 0.3, 0.6, 1.0]   # 价值差（通过 W 差）
    times = []
    for d in deltas:
        W[i] = 1.0; W[j] = 1.0 + d
        _, t = decide(W, A, q)
        times.append(t)
    print(f"[exp2] 决策时间 vs 价值差: " + " ".join(f"Δv={d}:t={t:.2f}" for d, t in zip(deltas, times)))
    # DDM 预测：t ∝ 1/Δv（近似——决策时间随 Δv 减小而增长）
    trend = np.all(np.diff(times) >= 0) or np.all(np.diff(times) <= 0)
    print(f"       趋势 = {'✓ 决策时间随价值接近（Δv↓）而增长（DDM/R60 自由感）' if times[0] > times[-1] else '需检查'}")
    return deltas, times

# ---------- 实验 3：截止（C105 成本交叉） ----------
def run_exp3():
    W, A = make_env()
    # 模拟"继续展开"决策：每次展开新增候选的边际价值 vs 搜索成本
    costs = [0.80, 0.90, 0.97, 1.05]   # 覆盖边际收益区间（候选激活 ~0.87-1.0）
    results = []
    for c in costs:
        # 展开序列（act 递减——边际收益递减）
        order, act = candidates(W, A, 2)
        expanded = 0
        marginal = act.copy()
        for k in range(len(act)):
            # 边际收益（当前最强未选候选）vs 搜索成本
            if len(marginal) and marginal[0] > c:
                expanded += 1
                marginal = marginal[1:]
            else:
                break
        results.append(expanded)
        print(f"[exp3] 搜索成本={c}: 展开深度={expanded}（边际收益 > 成本则继续）")
    # 截止：成本越高 → 展开越浅（C105 交叉）
    print(f"       成本↑ → 展开↓ = {'✓ 截止机制（C105 成本交叉）' if results == sorted(results, reverse=True) else '需检查'}")
    return costs, results

# ---------- 实验 4：回写（P6——聚焦硬化） ----------
def run_exp4():
    W, A = make_env()
    q = 8
    # 第一次决策（记录胜者 W）
    w_before = W.copy()
    winner, t1 = decide(W, A, q)
    # 回写（⑤ 反馈层——选择项沉积）
    W[winner] += ETA
    # 重测同 query：决策是否更快/更偏
    winner2, t2 = decide(W, A, q)
    # 胜者的胜率（多次重测）
    wins = sum(1 for _ in range(20) if decide(W, A, q)[0] == winner)
    print(f"[exp4] 回写后: W[{winner}] {w_before[winner]:.3f} → {W[winner]:.3f} (+{ETA})")
    print(f"       首次决策 t={t1:.2f} → 重测 t={t2:.2f} | 20 次重测胜率={wins}/20")
    print(f"       聚焦硬化 = {'✓ 回写增强选择项（P6 注意力=硬化控制杆）' if wins >= 12 else '需检查'}")
    return winner, t1, t2, wins

# ---------- 实验 5：闭环演化（习惯形成） ----------
def run_exp5():
    W, A = make_env()
    hist = []
    for step in range(1000):
        q = RNG.integers(0, N)
        winner, _ = decide(W, A, q)
        if winner is not None:
            W[winner] += ETA * 0.5   # 慢回写
        if step % 200 == 0:
            hist.append(W.copy())
    # 演化：W 分布偏斜（高频项增强）
    spread0 = np.std(hist[0])
    spreadN = np.std(hist[-1])
    print(f"[exp5] 1000 决策闭环: W 标准差 {spread0:.3f} → {spreadN:.3f}（偏斜度{'+' if spreadN > spread0 else '-'}）")
    print(f"       分化 = {'✓ 闭环演化（高频项增强——习惯形成 C77 的雏形）' if spreadN > spread0 * 1.2 else '需检查'}")
    return hist

# ---------- 绘图 ----------
def plot_all(e2, e3, e5):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    deltas, times = e2
    axes[0].plot(deltas, times, "o-")
    axes[0].set_title("Exp2: decision time vs value gap")
    axes[0].set_xlabel("dv"); axes[0].set_ylabel("decision time")

    costs, results = e3
    axes[1].plot(costs, results, "o-")
    axes[1].set_title("Exp3: cutoff (search cost vs depth)")
    axes[1].set_xlabel("search cost"); axes[1].set_ylabel("expansion depth")

    hist = e5
    for i, w in enumerate(hist):
        axes[2].plot(sorted(w), label=f"step {i*200}")
    axes[2].set_title("Exp5: W distribution evolution")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("fig_stage2.png", dpi=110)
    print("[plot] saved fig_stage2.png")

if __name__ == "__main__":
    e1 = run_exp1()
    e2 = run_exp2()
    e3 = run_exp3()
    e4 = run_exp4()
    e5 = run_exp5()
    plot_all(e2, e3, e5)
    np.savez("stage2_data.npz", deltas=e2[0], times=e2[1], costs=e3[0], results=e3[1])
    print("[done] stage2 complete")
