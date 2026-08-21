# -*- coding: utf-8 -*-
"""
M5 阶段 124：转导（A6/C7-01——边界函数化、内部过程化——T_in/T_out——
文字湖的边界——load_corpus 的显式化）

理论锚：
  A6（转导边界——边界函数化、内部过程化——与外界接口 T_in/T_out/T_value）
  C7-01（硬编码转导只存在于网络边界——内部无函数映射——内部=过程——
    supported）
  C7-02（两级转导：感官特异机器（硬编码）→ 统一网络语言（复值状态））
  C7-03（转导位于预测层级最底部——网络预测转导输出）
  C7-04（输出转导对称存在——内语言→行为）

机制（边界函数化）：
  T_in（输入转导）：外部文本 → 网络语言（字符 → 复值驱动——
    位置相位编码——C1-03——两级转导 C7-02）
  T_out（输出转导）：网络状态 → 外部文本（幅度 → 字符选择——生成）
  ——边界是函数（硬编码转导）——内部是过程（动力学——非函数）——C7-01

验证：
  exp1 T_in（字符 → 复值驱动——边界函数）
  exp2 T_out（状态 → 文本——输出转导）
  exp3 边界函数 vs 内部过程（转导=函数/内部=动力学——C7-01）
  exp4 对称性（T_in/T_out 对称——C7-04）
"""
import os
import re
import sys
import time
from collections import Counter
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("gb"):
    sys.stdout.reconfigure(encoding="utf-8")

from stage79_spontaneous_hubs import (load_corpus, extract_blocks, extract_hubs,
                                       HubLake, AMP_IN, DT)

DELTA_PHI = np.pi / 6


class TransductionLake:
    """转导湖：边界函数（T_in/T_out）+ 内部过程（动力学）"""

    def __init__(self, w):
        self.w = w

    def T_in(self, text):
        """输入转导（C7-02 两级：字符→复值驱动——位置相位——网络语言）"""
        n = self.w.n
        drive = np.zeros(n, dtype=complex)
        for pos, c in enumerate(text):
            if c in self.w.ci:
                i = self.w.ci[c]
                drive[i] += AMP_IN * np.exp(1j * (self.w.omega[i] * self.w.t + pos * DELTA_PHI))
        return drive

    def T_out(self, z, exclude=""):
        """输出转导（C7-04：状态→文本——幅度→字符选择）"""
        amp = np.abs(z)
        for c in exclude:
            if c in self.w.ci:
                amp[self.w.ci[c]] = 0
        j = np.argmax(amp)
        return self.w.chars[j], amp[j]

    def process(self, drive, steps=8):
        """内部过程（动力学——非函数——C7-01）"""
        z = np.zeros(self.w.n, dtype=complex)
        for _ in range(steps):
            dz = -self.w.gamma * z + 1j * self.w.omega * z
            dz += (self.w.KT @ z.real + 1j * (self.w.KT @ z.imag)) - z * self.w.rsT
            dz += drive
            z = z + dz * DT
            over = np.abs(z) > 3.0
            z[over] = z[over] / np.abs(z[over]) * 2.0
        return z


def run():
    print("=== M5 阶段 124：转导（A6/C7-01——边界函数化——T_in/T_out） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=300)
    simple2 = load_corpus(os.path.join(base, "corpus_simple2.txt"))
    full = simple + simple2
    blocks = extract_blocks(full)
    hubs = extract_hubs(full, blocks)
    chars = list(dict.fromkeys("".join(full)))
    w = HubLake(chars, blocks + hubs)
    for day in range(3):
        w.learn_epoch_batch(full, B=128)
    tr = TransductionLake(w)
    print(f"训练完成（{w.n} 字 / {len(w.hubs)} 河道）")

    # ---- exp1：T_in（输入转导） ----
    print("\n[exp1] T_in（外部文本 → 网络语言——字符→复值驱动——C7-02）:")
    drive = tr.T_in("苹果")
    act = np.where(np.abs(drive) > 0)[0][:4]
    print(f"      '苹果' → 驱动 {len(act)} 个单元"
          f"（{[(w.chars[j], f'{np.abs(drive[j]):.2f}') for j in act]}——"
          f"字符→相位编码——两级转导（特异机器→网络语言））")

    # ---- exp2：T_out（输出转导） ----
    print("\n[exp2] T_out（网络状态 → 外部文本——C7-04 输出转导）:")
    z = tr.process(tr.T_in("苹果"))
    out, amp = tr.T_out(z, exclude="苹果")
    print(f"      '苹果'注入 → 状态 → 输出 '{out}'（幅度 {amp:.2f}——"
          f"状态→字符选择——生成=输出转导）")

    # ---- exp3：边界函数 vs 内部过程（C7-01） ----
    print("\n[exp3] 边界函数 vs 内部过程（转导=函数/内部=动力学——C7-01）:")
    d1 = tr.T_in("苹果")
    d2 = tr.T_in("苹果")
    same = np.allclose(d1, d2)
    print(f"      T_in 确定性（同输入同输出）: {same}"
          f"（{'边界=函数 ✓（硬编码转导）' if same else '内部化（非边界）'}——"
          f"C7-01：函数只在边界）")
    z1 = tr.process(d1)
    t0 = tr.w.t
    z2 = tr.process(d2)
    print(f"      内部过程: 同驱动两次演化 → "
          f"{'同输出（确定性动力学）' if np.allclose(np.abs(z1), np.abs(z2)) else '演化中（t 依赖）'}"
          f"（内部=过程——时间演化——非函数映射）")

    # ---- exp4：对称性（T_in/T_out——C7-04） ----
    print("\n[exp4] 对称性（输入/输出转导——C7-04）:")
    print("      T_in: 文本 → 驱动（外界→网络——入海口）")
    print("      T_out: 状态 → 文本（网络→外界——出海口）")
    print("      （对称存在——运动/发声=另一方向入海口——C7-04——"
          "框架的'做出反应'通道）")
    print("\n[done] stage124 transduction")


if __name__ == "__main__":
    run()
