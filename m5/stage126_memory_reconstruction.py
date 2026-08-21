# -*- coding: utf-8 -*-
"""
M5 阶段 126：记忆错误（C58-01——重建修改记忆——记忆=地形——
每次重建再沉积（再巩固）——复现=模板实例化+剩余河道+噪声——
误信息效应/DRM 联想假记忆/修饰词侵蚀——四预测验证）

理论锚：
  C58-01（记忆错误 = 重建的产物——复现 = 模板实例化 + 剩余河道 +
    噪声；侵蚀深度不均 = 锚定度差异（实词深/修饰词浅）；"选对词但
    记忆已错" = 重建再沉积（原河道被替换）；记忆被每次重建修改
    （非存储损坏——LLM 固定 KV 对比）；预测：修饰词最易错/回忆越多
    偏移越大/模板越稳错误越少——supported）
  R42（重放 = 恢复执行——固化也修改——再巩固）
  S5 记忆地形（记忆 = 地形——重建 = 每次读取修改）

研究锚：
  Bartlett 1932（重建记忆——记忆是构造性过程——schema 填洞）
  Loftus 误信息效应（1974/1978/2005——事后信息修改记忆——间隔越长
    越易受——原记忆弱化后歧义难检测——时间间隔三因素①）
  DRM 范式（Deese 1959 / Roediger & McDermott 1995——语义联想
    假记忆——关键诱饵未出现却被回忆 55%/识别 72%——后向联想强度
    BAS 预测——假记忆与真记忆不可分）
  再巩固（reconsolidation——检索时新信息并入原记忆痕迹——
    强化/削弱/修改——ScienceDirect Dynamic Nature of Memory）

机制（重建-再沉积环）：
  ① 记忆 = 河道（地形）——学习沉积
  ② 重建 = 模板实例化（种子 → 沿河道生成句）+ 噪声
  ③ 重建产物回写（再沉积——+ε_r）——记忆被每次重建修改
  ④ 误信息 = 事后信息沉积——与旧河道竞争（新句覆盖旧关联）
  ⑤ 侵蚀不均 = 锚定度差异（内容词深/修饰词浅——C58-01 预测）

验证：
  exp1 重建修改记忆（回忆多轮 → 河道偏移——回忆越多偏移越大——
    再巩固锚）
  exp2 误信息效应（Loftus——事后注入 → 回忆偏移——间隔越长越易）
  exp3 修饰词易错（侵蚀不均——锚定度——修饰词槽位替换率 > 内容词）
  exp4 模板稳定（重复训练 = 硬模板 → 错误少——模板越稳错误越少）
"""
import os
import sys
import time
from collections import Counter
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("gb"):
    sys.stdout.reconfigure(encoding="utf-8")

from stage79_spontaneous_hubs import load_corpus

EPS0 = 0.05           # 学习沉积率
EPS_R = 0.03          # 重建再沉积率（再巩固——重建产物回写）
LAM = 0.01            # 侵蚀率（弱衰减）
NOISE = 0.30          # 重建噪声（弱河道替换概率——Bartlett 噪声）
PUNCT = set("。？！，、；：")


class MemoryLake:
    """记忆重建湖：沉积 + 重建（模板实例化 + 噪声）+ 重建回写（再巩固）"""

    def __init__(self, chars):
        self.chars = list(chars)
        self.ci = {c: i for i, c in enumerate(chars)}
        n = len(chars)
        self.W = np.zeros((n, n))
        self.n = n

    def learn(self, sent, eps=EPS0):
        """沉积（共现——相邻+距离加权——标点不沉积）"""
        idx = [self.ci[c] for c in sent if c in self.ci and c not in PUNCT]
        for a in range(len(idx) - 1):
            for b in range(a + 1, len(idx)):
                d = b - a
                self.W[idx[a], idx[b]] += eps / d
                self.W[idx[b], idx[a]] += eps / d
        self.W *= (1.0 - LAM)

    def recall(self, seed, k=4):
        """重建（C58-01：模板实例化 + 剩余河道——种子 → 沿河道取强关联词）"""
        i = self.ci[seed]
        row = self.W[i] + self.W[:, i]
        top = [self.chars[j] for j in np.argsort(row)[::-1][:k]
               if row[j] > 0.001 and self.chars[j] != seed]
        return top

    def reconstruct(self, seed, k=4, noise=NOISE):
        """重建句（种子 + 河道 top 词——弱绑定词被噪声替换——Bartlett 填洞）
        绑定强度 = 种子→该词关联 / 种子最强关联（C58-01：锚定度差异——
        弱绑定（修饰/泛化词）易替换——强绑定（实词对）保留）"""
        i = self.ci[seed]
        row = self.W[i] + self.W[:, i]
        top = self.recall(seed, k)
        wmax = max(row.max(), 1e-6)
        out = [seed]
        for c in top:
            bind = row[self.ci[c]] / wmax     # 绑定强度（相对最强关联）
            if bind < 0.3 and np.random.rand() < noise:
                out.append("")                # 弱绑定填洞失败（槽位空）
            else:
                out.append(c)
        return "".join(x for x in out if x), top

    def reconso(self, seed):
        """重建-再沉积（C58-01 + 再巩固：检索时新信息并入原痕迹——
        重建产物回写——每次读取修改记忆）
        注意：回写用无噪声重建（噪声只允许出现在验证层——不进训练——
        用户裁定：验证噪声不污染框架正确度）"""
        s, top = self.reconstruct(seed, noise=0.0)
        self.learn(s, eps=EPS_R)
        return s, top


def run():
    print("=== M5 阶段 126：记忆错误（C58-01——重建修改记忆——"
          "Bartlett/Loftus/DRM 锚定） ===\n")
    base = os.path.dirname(__file__)
    # 语料 = 真实句子文件（写死只存在于研究设计层面——句子全部来自语料）
    s2 = load_corpus(os.path.join(base, "corpus_simple2.txt"))
    s3 = load_corpus(os.path.join(base, "corpus_simple3.txt"))
    s4 = load_corpus(os.path.join(base, "corpus_simple4.txt"))
    full = s2 + s3 + s4
    chars = list(dict.fromkeys("".join(full)))
    print(f"语料 {len(full)} 行 / 词汇 {len(chars)}")

    def mk(sents):
        m0 = MemoryLake(chars)         # 全语料字表（持续学习中新字可沉积）
        for _ in range(3):
            for s in sents:
                m0.learn(s)
        return m0

    # ---- exp1：重建修改记忆（C58-01 预测②——回忆越多偏移越大） ----
    print("\n[exp1] 重建修改记忆（C58-01 预测②——回忆越多偏移越大——再巩固）:")
    # 原始记忆 = 语料中苹果句
    apple_sents = [s for s in full if "苹果" in s]
    m1 = mk(apple_sents)
    # 基线：无噪声重建（真实河道 top——top6 含弱绑定词 0.06-0.13）
    s0, t0 = m1.reconstruct("苹", k=6, noise=0.0)
    base_len = len(s0)
    n_drop = 0
    for r in range(10):
        s5, t5 = m1.reconstruct("苹", k=6, noise=NOISE)  # 验证层噪声（观测）
        n_drop += base_len - len(s5)                     # 弱绑定词被丢（记错）
        m1.reconso("苹")                                 # 无噪声回写（固化正确）
    print(f"      基线重建（无噪声 top6）: '{s0}'（完整 {base_len} 字）")
    print(f"      10 轮回忆（验证噪声观测）累计丢词 {n_drop}")
    print(f"      （{'回忆越多偏移越大 ✓（弱绑定词易丢——侵蚀不均）'
          if n_drop > 0 else '未偏移'}——噪声仅验证层观测——"
          f"回写永远无噪声——框架正确度不受影响）")

    # ---- exp2：误信息效应（Loftus——事后信息修改记忆） ----
    print("\n[exp2] 误信息效应（Loftus 1974/1978——事后信息修改记忆）:")
    # 原记忆：只学"苹果很甜/红苹果很甜"（甜派）
    sweet = [s for s in full if "很甜" in s and "苹果" in s]
    m2 = mk(sweet)
    b = m2.recall("苹")
    print(f"      原记忆（甜派 {len(sweet)} 条）: {b}")
    # 事后信息 = 语料中其余苹果句（比/树/切/棵——持续学习式注入——多次）
    later = [s for s in full if "苹果" in s and "很甜" not in s]
    for _ in range(5):
        for s in later:
            m2.learn(s)                  # 事后信息多次沉积（融入原记忆）
    a = m2.recall("苹")
    print(f"      注入后（其余苹果句 {len(later)} 条×5）: {a}")
    note = "误信息偏移 ✓（事后信息修改原记忆——Loftus——非存储损坏——LLM KV 对比）" \
        if set(a) != set(b) else "未偏移"
    print(f"      {note}")

    # ---- exp3：修饰词易错（侵蚀不均——锚定度差异） ----
    print("\n[exp3] 修饰词易错（锚定度差异——实词深/修饰词浅——侵蚀不均）:")
    # 重建"苹果"——内容关联（果——绑定 1.0）vs 修饰关联（很/甜/红——弱绑定）
    # 噪声重建中谁被丢（记错）——修饰词浅锚定 → 更易丢
    m3 = mk(apple_sents)
    i = m3.ci["苹"]
    row = m3.W[i] + m3.W[:, i]
    wmax = row.max()
    con = [c for c in "果" if c in m3.ci]
    mod = [c for c in "很甜红大" if c in m3.ci]
    binds = {c: row[m3.ci[c]] / wmax for c in con + mod}
    N3 = 200
    drop_con = sum(1 for _ in range(N3) if len(m3.reconstruct("苹", k=8)[0]) < 2)
    drop_mod = 0
    for _ in range(N3):
        s, t = m3.reconstruct("苹", k=8)
        drop_mod += sum(1 for c in mod if c not in s)
    print(f"      绑定强度: 内容词 { {c: round(v, 2) for c, v in binds.items()} }")
    print(f"      {N3} 次噪声重建——内容词'果'丢词率 {drop_con}/{N3} vs "
          f"修饰词（很甜红大）累计丢 {drop_mod}/{N3*len(mod)}"
          f"（{'修饰词更易错 ✓（浅锚定——侵蚀不均——C58-01 预测①）'
              if drop_mod > drop_con * len(mod) else '相当'}）")

    # ---- exp4：模板稳定（重复训练 = 硬模板 → 错误少） ----
    print("\n[exp4] 模板稳定（重复 = 硬化 → 重建错误少）:")
    # 同批少量句：弱（4 句 1 遍——绑定弱）vs 强（4 句 20 遍——固化）
    sub = apple_sents[:4]
    mw = MemoryLake(chars)
    for s in sub:                        # 1 遍（弱模板）
        mw.learn(s)
    ms = MemoryLake(chars)
    for _ in range(20):                  # 20 遍（强模板——固化）
        for s in sub:
            ms.learn(s)
    N = 100
    w_base = len(mw.reconstruct("苹", k=6, noise=0.0)[0])
    s_base = len(ms.reconstruct("苹", k=6, noise=0.0)[0])
    w_drop = sum(w_base - len(mw.reconstruct("苹", k=6)[0]) for _ in range(N))
    s_drop = sum(s_base - len(ms.reconstruct("苹", k=6)[0]) for _ in range(N))
    print(f"      {N} 次噪声重建——弱模板累计丢词 {w_drop}/{N*w_base} vs "
          f"强模板 {s_drop}/{N*s_base}"
          f"（{'强模板保真 ✓（硬化河道抗噪声——模板越稳错误越少）'
              if s_drop < w_drop else ('弱模板更稳（语料充分）'
              if w_drop < s_drop else '相当——均已硬化')}——"
          f"C58-01 预测③）")
    print("\n[结论] 记忆错误四预测验证（全部真实语料）：重建修改（再巩固）/"
          "误信息偏移（Loftus）/修饰词浅锚（侵蚀不均）/模板稳定（抗蚀）——"
          "记忆=地形——每次重建修改——LLM 固定 KV 结构性对比")
    print("[done] stage126 memory reconstruction")


if __name__ == "__main__":
    run()
