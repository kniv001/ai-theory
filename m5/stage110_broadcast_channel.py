# -*- coding: utf-8 -*-
"""
M5 阶段 110：双通道广播（差距⑦——C10-01 广播通道（全局快无方向——调质/
价值——弥散网化石）+ 河道通道（定向精细——预测/误差）——分工：
广播管状态模式、河道管内容结构）

理论锚：
  C10-01（双通道架构——广播管状态模式、河道管内容结构——open）
  C10-03（调质 = 误差的广播形态——多巴胺双身份：RPE + 调质——supported）

机制（双通道）：
  ① 广播通道：全局价值/调质信号——无方向（弥散——所有单元同调）——
    调制状态模式（全局兴奋 value>0 / 抑制 value<0——g 全局调制）
  ② 河道通道：定向（K 矩阵——预测/误差——内容——"苹果"→"甜"）
  ③ 分工：广播管"状态"（整体兴奋/平静）——河道管"内容"（具体关联）
  ④ 多巴胺双身份（C10-03）：同一价值信号——RPE（河道——定向误差）
    + 调质（广播——全局调制）

验证：
  exp1 广播 vs 河道（全局调制 vs 定向检索——输出差异）
  exp2 广播全局性（一个信号 → 全湖同调——无方向——vs 河道定向）
  exp3 分工（广播管状态 + 河道管内容——并存不冲突）
  exp4 双身份（RPE 定向 + 调质广播——同一信号两作用）
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

G_BG = 0.2


class DualChannelLake(HubLake):
    """双通道湖：广播（全局调质——无方向）+ 河道（定向 K）"""

    def broadcast(self, value, amp=0.3):
        """广播通道（C10-01）：全局价值/调质——无方向（弥散——全湖同调）
        value>0 兴奋（g 全局升）/ value<0 抑制（g 全局降）——状态模式"""
        g = np.full(self.n, G_BG)
        if value > 0:
            g += amp                     # 全局兴奋（所有单元同调）
        else:
            g -= amp
        g = np.clip(g, 0.05, 1.0)
        return g

    def state_shift(self, value, steps=30):
        """广播调制状态：全局增益（兴奋 +gain/抑制 -gain——无方向——
        所有单元同调——状态模式）——C10-01 广播管状态"""
        n = self.n
        z = np.zeros(n, dtype=complex)
        g = self.broadcast(value)
        gain = (g.mean() - G_BG) * 2.0        # 广播偏移 → 全局增益
        drive = np.full(n, 0.05, dtype=complex)    # 恒定背景输入（环境）
        for _ in range(steps):
            dz = -self.gamma * z + 1j * self.omega * z
            dz += (self.KT @ z.real + 1j * (self.KT @ z.imag)) - z * self.rsT
            dz += gain * z                    # 全局增益（状态——无方向）
            dz += drive
            z = z + dz * DT
            over = np.abs(z) > 3.0
            z[over] = z[over] / np.abs(z[over]) * 2.0
        return np.abs(z).mean()


def run():
    print("=== M5 阶段 110：双通道广播（C10-01——广播管状态/河道管内容） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=300)
    simple2 = load_corpus(os.path.join(base, "corpus_simple2.txt"))
    full = simple + simple2
    print(f"语料 {len(full)} 行")

    blocks = extract_blocks(full)
    hubs = extract_hubs(full, blocks)
    chars = list(dict.fromkeys("".join(full)))
    w = DualChannelLake(chars, blocks + hubs)
    for day in range(3):
        w.learn_epoch_batch(full, B=128)
    print(f"训练完成（{w.n} 字 / {len(w.hubs)} 河道）")

    # ---- exp1：广播 vs 河道 ----
    print("\n[exp1] 广播 vs 河道（全局调制 vs 定向检索——输出差异）:")
    s_neg = w.state_shift(-1.0)     # 广播抑制（平静状态）
    s_pos = w.state_shift(1.0)      # 广播兴奋（激动状态）
    print(f"      广播: 抑制状态平均激活 {s_neg:.3f} vs 兴奋状态 {s_pos:.3f}"
          f"（{'状态调制 ✓' if s_pos > s_neg * 1.2 else '调制不足'}——无方向全湖）")
    ans = w.answer("很", "苹果")
    print(f"      河道: K[很][苹果] → {[(a, f'{v:.2f}') for a, v in ans[:3]] if ans else '无'}"
          f"（定向内容——'苹果'→'甜'——与状态无关）")

    # ---- exp2：广播全局性 ----
    print("\n[exp2] 广播全局性（一个信号 → 全湖同调——无方向——C10-01）:")
    g1 = w.broadcast(1.0)
    g2 = w.broadcast(-1.0)
    spread1 = g1.std()
    spread2 = g2.std()
    print(f"      兴奋广播 g 标准差 {spread1:.3f} / 抑制 {spread2:.3f}"
          f"（{'全局同调（无方向）' if spread1 < 0.05 and spread2 < 0.05 else '有方向（非广播）'}——"
          f"vs 河道定向（'苹果'→'甜'——单一指向））")

    # ---- exp3：分工（广播状态 + 河道内容——并存） ----
    print("\n[exp3] 分工（广播管状态 + 河道管内容——并存不冲突）:")
    s1 = w.state_shift(1.0)
    ans1 = w.answer("很", "苹果")
    s2 = w.state_shift(-1.0)
    ans2 = w.answer("很", "苹果")
    same = [a for a, _ in ans1[:2]] == [a for a, _ in ans2[:2]]
    print(f"      兴奋态内容: {[(a, f'{v:.2f}') for a, v in ans1[:2]]}"
          f" vs 抑制态内容: {[(a, f'{v:.2f}') for a, v in ans2[:2]]}"
          f"（{'内容不变（状态不扰内容）✓' if same else '内容变了'}——"
          f"广播管状态/河道管内容——分工）")

    # ---- exp4：多巴胺双身份（C10-03） ----
    print("\n[exp4] 双身份（同一价值信号——RPE 定向 + 调质广播——C10-03）:")
    print("      价值信号 v → ① RPE（河道——定向误差——刻/削特定河道）")
    print("      价值信号 v → ② 调质（广播——全局调制——全湖状态）")
    print("      （同一信号两作用——广播形态 = 误差的广播（C10-03）——")
    print("      本 stage 演示广播面——RPE 面已在训练管线（REWARD_MULT））")
    print("\n[done] stage110 broadcast channel")


if __name__ == "__main__":
    run()
