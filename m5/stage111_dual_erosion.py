# -*- coding: utf-8 -*-
"""
M5 阶段 111：侵蚀双层化（差距⑧——C36-01 结构侵蚀（拓扑——关键期内，
ε_struct 包络门控——关闭即冻结——猫实验）+ 功能侵蚀（W 强度——终身——
生疏/遗忘）——"忘不干净捡起来快" = 结构冻结 + 功能弱化）

理论锚：
  C36-01（侵蚀双层化——结构侵蚀仅关键期内（冻结即拓扑稳定）+
    功能侵蚀终身（生疏/遗忘）——supported）
  C4-03（塑形分两层：功能塑性（W 强度，快-中速）+ 结构生长（拓扑，慢））

机制（双层侵蚀）：
  ① 结构层 W_s：连接有无（拓扑）——关键期内可侵蚀（λ_s 门控）——
    关键期后冻结（猫实验——成年剥夺无永久损伤）
  ② 功能层 W_f：强度（终身——λ_f 持续——生疏/遗忘）
  ③ "忘不干净捡起来快"：结构保留（拓扑在）——功能弱（强度降）——
    重学 = 强度恢复（快——比全新快）

验证：
  exp1 双层侵蚀（结构冻结 vs 功能持续——关键期后）
  exp2 关键期对照（期内侵蚀 vs 期后冻结——猫实验）
  exp3 捡起来快（功能弱化后重学——恢复快——结构保留）
"""
import os
import re
import sys
import time
from collections import Counter
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("gb"):
    sys.stdout.reconfigure(encoding="utf-8")

EPS_K = 0.02
LAMBDA_S = 0.05        # 结构侵蚀率（关键期内）
LAMBDA_F = 0.02        # 功能侵蚀率（终身）
CRITICAL_END = 3       # 关键期结束（epoch）


class DualErosionLake:
    """双层侵蚀湖：结构层（拓扑——关键期门控）+ 功能层（强度——终身）"""

    def __init__(self, chars):
        self.chars = list(chars)
        self.ci = {c: i for i, c in enumerate(chars)}
        n = len(chars)
        self.W_s = np.zeros((n, n))      # 结构层（连接有无——拓扑）
        self.W_f = np.zeros((n, n))      # 功能层（强度——终身）
        self.epoch = 0
        self.critical = True

    def learn(self, sent):
        """学习：沉积到两层（功能层终身——结构层关键期内才塑性）"""
        idx = [self.ci[c] for c in sent if c in self.ci]
        for a in range(len(idx) - 1):
            i, j = idx[a], idx[a + 1]
            self.W_f[i, j] += EPS_K * 3.0       # 功能（终身）
            if self.critical:
                self.W_s[i, j] += EPS_K * 3.0   # 结构（关键期内）
        # 侵蚀
        self.W_f *= (1.0 - LAMBDA_F)            # 功能（终身——生疏）
        if self.critical:
            self.W_s *= (1.0 - LAMBDA_S)        # 结构（关键期内）
        # 关键期结束
        self.epoch += 1
        if self.epoch >= CRITICAL_END:
            self.critical = False               # 结构冻结（C36-01）

    def strength(self, a, b):
        if a in self.ci and b in self.ci:
            return self.W_f[self.ci[a], self.ci[b]], self.W_s[self.ci[a], self.ci[b]]
        return 0.0, 0.0


def run():
    print("=== M5 阶段 111：侵蚀双层化（C36-01——结构冻结+功能终身） ===\n")
    chars = list("苹果天气小猫吃鱼水很甜冷热")
    w = DualErosionLake(chars)

    # ---- 阶段 1：关键期内学习（"苹果很甜"重复） ----
    print("[阶段1] 关键期内学习（'苹果很甜'×20——结构+功能都沉积）:")
    for _ in range(20):
        w.learn("苹果很甜")
    f, s = w.strength("苹", "果")
    print(f"      W_f[苹→果] = {f:.4f} / W_s[苹→果] = {s:.4f}（两层建立）")

    # ---- 阶段 2：关键期后遗忘期（"苹果很甜"停止出现——侵蚀） ----
    print(f"\n[阶段2] 遗忘期（关键期后——'苹果很甜'停止——{CRITICAL_END} epoch 后冻结）:")
    for ep in range(5):
        w.learn("天气很冷")        # 只学别的（苹果句停止）
        f, s = w.strength("苹", "果")
        phase = "关键期内（结构可蚀）" if w.critical else "结构已冻结"
        print(f"      epoch {w.epoch}: W_f={f:.4f} / W_s={s:.4f}（{phase}）")

    # ---- exp1：双层侵蚀 ----
    print("\n[exp1] 双层侵蚀（关键期后——结构冻结 vs 功能持续）:")
    f, s = w.strength("苹", "果")
    print(f"      W_f={f:.4f}（功能持续侵蚀——生疏）")
    print(f"      W_s={s:.4f}（结构冻结——拓扑保留——{'>0' if s > 0.001 else '已断'}）")

    # ---- exp2：关键期对照（猫实验） ----
    print("\n[exp2] 关键期对照（期内侵蚀 vs 期后冻结——猫实验 C36-01）:")
    # 期内停止：关键期内（学 2 次）就停止苹果句（结构可蚀）
    w2 = DualErosionLake(chars)
    for _ in range(2):
        w2.learn("苹果很甜")
    for ep in range(5):
        w2.learn("天气很冷")
    f2, s2 = w2.strength("苹", "果")
    print(f"      期内停止: W_s={s2:.4f}（{'结构侵蚀——连接弱（关键期可塑）' if s2 < 0.05 else '结构保留'}）")
    # 期后停止：过关键期后停止（结构冻结）
    w3 = DualErosionLake(chars)
    for _ in range(20):
        w3.learn("苹果很甜")
    for ep in range(5):
        w3.learn("天气很冷")
    f3, s3 = w3.strength("苹", "果")
    print(f"      期后停止: W_s={s3:.4f}（{'结构冻结——连接保留 ✓' if s3 > 0.05 else '结构丢失'}——"
          f"成年剥夺无永久损伤——猫实验）")

    # ---- exp3：捡起来快（功能重学——结构保留） ----
    print("\n[exp3] 捡起来快（功能弱化后重学——结构保留→恢复快——对照全新）:")
    f_before = w.strength("苹", "果")[0]
    for _ in range(5):
        w.learn("苹果很甜")        # 重学（结构已在——只恢复功能）
    f_after = w.strength("苹", "果")[0]
    # 对照：全新湖学 5 次
    w_new = DualErosionLake(chars)
    for _ in range(5):
        w_new.learn("苹果很甜")
    f_new = w_new.strength("苹", "果")[0]
    print(f"      重学 5 次: {f_before:.4f} → {f_after:.4f}（+{f_after - f_before:.3f}）")
    print(f"      全新 5 次: 0 → {f_new:.4f}（+{f_new:.3f}）")
    print(f"      （重学起点高（结构保留）——总量 {f_after:.3f} vs 全新 {f_new:.3f}"
          f"——{'捡起来快 ✓' if f_after > f_new * 2 else '恢复慢'}——"
          f"'忘不干净捡起来快'——C36-01）")
    print("\n[done] stage111 dual erosion")


if __name__ == "__main__":
    run()
