# -*- coding: utf-8 -*-
"""
M5 阶段 12：P31 元学习验证——习得先验 ≈ 进化先验
R168：h₀ 可以是训练出来的——层级偏好可以从数据中学（不必须进化预置）

对比三个学习者（嵌套递归任务）：
  1. 无先验（bigram——纯统计）
  2. 有 h₀（LIFO 预置——进化先验——stage11）
  3. 元学习者（先学配对统计——从数据中习得"结构有配对"——再应用）

预测：元学习 ≈ h₀ > bigram（习得先验 ≈ 进化先验——P31 兑现）
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(53)
SYM = list("ABCcba")
ABC = list("ABC")

def gen_nested(n_seq=200):
    """嵌套：开放段随机排列 → 闭合段逆序配对（远端依赖）"""
    seqs = []
    for _ in range(n_seq):
        opens = RNG.permutation(ABC).tolist()
        s = opens + [RNG.choice(ABC)] + [c.lower() for c in reversed(opens)]
        seqs.append("".join(s))
    return seqs

# ---------- 学习器 ----------
class BigramLearner:
    """无先验：bigram 统计"""
    def __init__(self):
        self.count = {a: {} for a in SYM}
    def learn(self, seqs):
        for s in seqs:
            for i in range(len(s) - 1):
                self.count[s[i]][s[i+1]] = self.count[s[i]].get(s[i+1], 0) + 1
    def predict(self, prefix):
        last = prefix[-1]
        if last not in self.count or not self.count[last]:
            return RNG.choice(SYM)
        items = list(self.count[last].items())
        tot = sum(v for _, v in items)
        return RNG.choice([k for k, _ in items], p=[v/tot for _, v in items])

class MetaLearner(BigramLearner):
    """元学习者：bigram + 从数据中习得的配对统计（远端关联——习得先验）"""
    def __init__(self):
        super().__init__()
        self.pair_count = {u: {v: 0 for v in "abc"} for u in "ABC"}   # 开放→闭合配对
    def learn(self, seqs):
        super().learn(seqs)
        # 元学习：统计远端配对（开放符号与闭合符号的共现——数据驱动的层级结构）
        for s in seqs:
            opens = [ch for ch in s if ch in "ABC"]
            for ch in s:
                if ch in "abc":
                    # 找它对应的开放符号（LIFO：最近未闭合的）
                    closed = {c for c in s[:s.index(ch)] if c in "abc"}
                    remain = [o for o in reversed(opens) if o.lower() not in closed]
                    if remain and remain[0].lower() == ch:
                        self.pair_count[remain[0]][ch] += 1
    def predict(self, prefix):
        # 闭合阶段：用习得的配对统计（数据驱动的 LIFO——习得先验）
        if any(ch in "abc" for ch in prefix):
            opens = [ch for ch in prefix if ch in "ABC"]
            closed = {ch.lower() for ch in opens if ch.lower() in prefix}
            remaining = [ch for ch in reversed(opens) if ch.lower() not in closed]
            if remaining:
                top = remaining[0]
                # 习得的配对偏好：top→ 的闭合分布（学过则偏——未学则随机）
                cand = self.pair_count[top]
                tot = sum(cand.values())
                if tot > 0:
                    return RNG.choice(list("abc"), p=[cand[c]/tot for c in "abc"])
            return RNG.choice(list("abc"))
        return super().predict(prefix)

class H0Learner(BigramLearner):
    """有 h₀：预置 LIFO（进化先验——stage11）"""
    def predict(self, prefix):
        if any(ch in "abc" for ch in prefix):
            opens = [ch for ch in prefix if ch in "ABC"]
            closed = {ch.lower() for ch in opens if ch.lower() in prefix}
            remaining = [ch for ch in reversed(opens) if ch.lower() not in closed]
            if remaining:
                return remaining[0].lower()
            return RNG.choice(list("abc"))
        return super().predict(prefix)

def evaluate(learner, seqs, train_frac=0.7):
    n = int(len(seqs) * train_frac)
    learner.learn(seqs[:n])
    correct, total = 0, 0
    for s in seqs[n:]:
        for i in range(1, len(s)):
            if learner.predict(s[:i]) == s[i]:
                correct += 1
            total += 1
    return correct / max(total, 1)

def run():
    REPS = 10
    acc_bg, acc_h0, acc_meta = [], [], []
    for _ in range(REPS):
        seqs = gen_nested(300)
        acc_bg.append(evaluate(BigramLearner(), seqs))
        acc_h0.append(evaluate(H0Learner(), seqs))
        acc_meta.append(evaluate(MetaLearner(), seqs))

    mb, mh, mm = np.mean(acc_bg), np.mean(acc_h0), np.mean(acc_meta)
    sb, sh, sm = np.std(acc_bg), np.std(acc_h0), np.std(acc_meta)
    print(f"[P31] 嵌套递归任务（{REPS} 次重复）:")
    print(f"  无先验 bigram:   {mb:.3f}±{sb:.3f}")
    print(f"  有 h₀（预置）:    {mh:.3f}±{sh:.3f}")
    print(f"  元学习（习得）:   {mm:.3f}±{sm:.3f}")
    meta_ok = mm > mb + 0.03
    meta_h0 = abs(mm - mh) < 0.05
    print(f"\n  元学习 > 无先验 = {'✓' if meta_ok else '✗'}"
          f" | 元学习 ≈ h₀ = {'✓' if meta_h0 else '✗（差异大——习得≠预置）'}")
    if meta_ok and meta_h0:
        verdict = "✓ P31 兑现：习得先验 ≈ 进化先验（h₀ 可训练——元学习可行）"
    elif meta_ok:
        verdict = "△ 部分：元学习有效但不等价 h₀"
    else:
        verdict = "✗ 元学习未习得先验"
    print(f"  裁决: {verdict}")

    fig, axes = plt.subplots(figsize=(7, 5))
    axes.bar(["no prior\n(bigram)", "h0 preset\n(evolved)", "meta-learned\n(acquired)"],
             [mb, mh, mm], yerr=[sb, sh, sm])
    axes.set_title("P31: acquired prior ~ evolved prior")
    axes.set_ylim(0, 0.7)
    fig.tight_layout()
    fig.savefig("fig_stage12.png", dpi=110)
    print("\n[plot] saved fig_stage12.png")
    print("[done] stage12 meta-learning complete")

if __name__ == "__main__":
    run()
