# -*- coding: utf-8 -*-
"""
M5 阶段 11：P12 先验裁决——语言结构先验（h₀）是否必要（C15-01 剩余压力点）

对比：无先验统计学习器（bigram——纯统计）vs 含 h₀ 学习器（bigram + 层级组合偏好）
任务 1：简单模板（Saffran 类——相邻转移统计——可纯统计学习）
任务 2：嵌套递归（中心嵌入——远端依赖——"A B C ... C' B' A'"——需层级）

预测（R149）：
  简单模板：两者都能学（无差异——统计够）
  嵌套递归：h₀ 者胜（bigram 无法捕获远端依赖——层级偏好必要）
  → 有差异 = h₀ 必要（C15-01 补完确认）；无差异 = 纯统计（压力点消除）
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(47)

# ---------- 数据生成 ----------
SYM = list("ABCcba")   # 6 符号（大小写 = 配对：A-a, B-b, C-c）
ABC = list("ABC")
BC, CA, AB = list("BC"), list("CA"), list("AB")

def gen_simple(n_seq=200, L=10):
    """简单模板：AB/BC/CA 转移统计（Saffran 类——相邻依赖）"""
    seqs = []
    for _ in range(n_seq):
        s = [RNG.choice(ABC)]
        for _ in range(L):
            s.append({"A": RNG.choice(BC), "B": RNG.choice(CA), "C": RNG.choice(AB)}[s[-1]])
        seqs.append("".join(s))
    return seqs

def gen_nested(n_seq=200, max_depth=3):
    """嵌套递归：开放段随机排列 → 闭合段 = 逆序配对（真远端依赖——闭合段转移不固定——
    bigram 无法学——LIFO 层级可学）"""
    seqs = []
    for _ in range(n_seq):
        opens = RNG.permutation(ABC).tolist()
        s = opens + [RNG.choice(ABC)] + [c.lower() for c in reversed(opens)]
        seqs.append("".join(s))
    return seqs

# ---------- 学习器 ----------
class BigramLearner:
    """无先验：bigram 转移统计"""
    def __init__(self):
        self.count = {a: {} for a in SYM}
    def learn(self, seqs):
        for s in seqs:
            for i in range(len(s) - 1):
                self.count[s[i]][s[i+1]] = self.count[s[i]].get(s[i+1], 0) + 1
    def predict(self, prefix):
        if not prefix:
            return RNG.choice(SYM)
        last = prefix[-1]
        if last not in self.count or not self.count[last]:
            return RNG.choice(SYM)
        items = list(self.count[last].items())
        tot = sum(v for _, v in items)
        return RNG.choice([k for k, _ in items], p=[v/tot for _, v in items])

class StackLearner(BigramLearner):
    """含 h₀：bigram + 层级组合偏好（栈——嵌套配对——递归模板先验）"""
    def __init__(self):
        super().__init__()
        self.stack = []
    def reset(self):
        self.stack = []
    def predict(self, prefix):
        # 层级偏好：闭合阶段（出现小写）→ LIFO 配对闭合（层级模板——最近开放先闭合）
        if any(ch in "abc" for ch in prefix):
            opens = [ch for ch in prefix if ch in "ABC"]
            closed = {ch.lower() for ch in opens if ch.lower() in prefix}
            remaining = [ch for ch in reversed(opens) if ch.lower() not in closed]
            if remaining:
                return remaining[0].lower()
            return RNG.choice(list("abc"))
        # 开放阶段：bigram（统计）
        return super().predict(prefix)
    def learn(self, seqs):
        super().learn(seqs)
        # 同时学习配对（层级组合——"开放→闭合"模板）
        for s in seqs:
            st = []
            for ch in s:
                if ch in "ABC":
                    st.append(ch)
                elif ch in "abc" and st:
                    top = st[-1]
                    if top.lower() == ch:
                        self.count[top][ch] = self.count[top].get(ch, 0) + 5  # 配对加强

def evaluate(learner_cls, seqs, train_frac=0.7):
    """学习 → 预测测试（next-symbol 准确率——远端依赖考验）"""
    n_train = int(len(seqs) * train_frac)
    train, test = seqs[:n_train], seqs[n_train:]
    learner = learner_cls()
    learner.learn(train)
    correct, total = 0, 0
    for s in test:
        if hasattr(learner, "reset"):
            learner.reset()
        for i in range(1, len(s)):
            pred = learner.predict(s[:i])
            if pred == s[i]:
                correct += 1
            total += 1
    return correct / max(total, 1)

def run():
    REPS = 10
    print(f"重复 {REPS} 次（统计稳定性）:")
    diffs_s, diffs_n = [], []
    acc_bg_s, acc_h0_s = [], []
    acc_bg_n, acc_h0_n = [], []
    for _ in range(REPS):
        simple = gen_simple()
        nested = gen_nested()
        a, b = evaluate(BigramLearner, simple), evaluate(StackLearner, simple)
        c, d = evaluate(BigramLearner, nested), evaluate(StackLearner, nested)
        acc_bg_s.append(a); acc_h0_s.append(b)
        acc_bg_n.append(c); acc_h0_n.append(d)
        diffs_s.append(abs(a - b)); diffs_n.append(d - c)

    ms, mh_s = np.mean(acc_bg_s), np.mean(acc_h0_s)
    nb, nh = np.mean(acc_bg_n), np.mean(acc_h0_n)
    print(f"\n[任务1] 简单模板: bigram {ms:.3f}±{np.std(acc_bg_s):.3f} | h₀ {mh_s:.3f}±{np.std(acc_h0_s):.3f}")
    print(f"        差异均值 {np.mean(diffs_s):.3f}——{'无显著差异（统计够用——Saffran 一致）' if np.mean(diffs_s) < 0.1 else '有差异'}")
    print(f"\n[任务2] 嵌套递归: bigram {nb:.3f}±{np.std(acc_bg_n):.3f} | h₀ {nh:.3f}±{np.std(acc_h0_n):.3f}")
    diff_n = np.mean(diffs_n)
    # 统计：h₀ > bigram 的次数
    wins = sum(1 for d in diffs_n if d > 0)
    print(f"        差异均值 {diff_n:.3f} | h₀ 胜 {wins}/{REPS} 次")
    if diff_n > 0.05 and wins >= 8:
        verdict = "✓ h₀ 必要（层级先验在嵌套上稳定优于纯统计）——C15-01 补完确认"
    elif diff_n > 0 and wins >= 8:
        verdict = "△ 弱支持（方向一致但差异小——h₀ 有帮助但非决定性）"
    else:
        verdict = "✗ 无差异（纯统计足够——压力点消除）"
    print(f"        裁决: {verdict}")

    # 图
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(["bigram", "h0-stack"], [ms, mh_s], yerr=[np.std(acc_bg_s), np.std(acc_h0_s)])
    axes[0].set_title("Task1: simple template (adjacent)")
    axes[0].set_ylim(0, 1)
    axes[1].bar(["bigram", "h0-stack"], [nb, nh], yerr=[np.std(acc_bg_n), np.std(acc_h0_n)])
    axes[1].set_title("Task2: nested recursion (remote)")
    axes[1].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig("fig_stage11.png", dpi=110)
    print("\n[plot] saved fig_stage11.png")
    print("[done] stage11 prior test complete")

if __name__ == "__main__":
    run()
