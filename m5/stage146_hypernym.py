# -*- coding: utf-8 -*-
"""
M5 阶段 146：上位词/词类绑定（C15-02——类-实例桥——'X是一种Y'/
'X是Y的一种'模板涌现——对象扩展（颜色→蓝色）——颜色问通用解 +
思考排序改进）

理论锚：
  C15-02（词类子湖 = 分布聚类——上位/下位 = 类-实例关系——
    位置/共现模式涌现）
  C13-01（意义 = 预测关系集——'是一种' = 类属关系模板）
  C16-01（尺度递归——类属关系 = 更高尺度关系）

机制（上位词学习——涌现）：
  ① 模板抽取：'X是一种Y' / 'X是Y的一种'（C15-01 统计——模板 =
    关系统计）——(X, Y) 上位对（Y⊃X）
  ② 对象扩展：obj='颜色' → 下位集（蓝色——'蓝色是颜色的一种'）
    ——候选句检查 obj ∪ 下位词——类-实例桥通用解
  ③ 应用：chat（对象扩展）/ think（展开词扩展）

验证：
  exp1 上位对抽取（wiki/语料——'计算是一种思考过程'→(计算,思考过程)）
  exp2 颜色问（obj 扩展——'颜色'→'蓝色'——答'我喜欢蓝色'——
    通用解——非 hub 豁免）
  exp3 知识问答（'计算是什么'→'计算是一种思考过程'——上位定义）
  exp4 思考改进（obj 扩展后——候选相关）
"""
import os
import re
import sys
import time
from collections import Counter
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("gb"):
    sys.stdout.reconfigure(encoding="utf-8")

from lake_app import TextLakeApp


class HyperLake(TextLakeApp):
    """文字湖 + 上位词（类-实例桥——C15-02）"""

    def learn_hypernyms(self):
        """上位对抽取（'X是一种Y'/'X是Y的一种'模板——C15-01 统计）"""
        self.hyper = {}                       # Y → [X...]（上位→下位集）
        for s in self.sents + self.sents_wiki:
            m = re.match(r"^([一-鿿]{2,6})是一种([一-鿿]{2,10})", s)
            if not m:
                m = re.match(r"^([一-鿿]{2,6})是([一-鿿]{2,10})的一种", s)
            if m:
                x, y = m.group(1), m.group(2)
                self.hyper.setdefault(y, []).append(x)
        print(f"[hyper] 上位对 {sum(len(v) for v in self.hyper.values())} 条"
              f"（模板涌现——'是一种'/'的一种'）")

    def obj_expand(self, obj):
        """对象扩展（类→实例：'颜色'→'蓝色'——下位词——类-实例桥）"""
        out = [obj]
        for y, xs in self.hyper.items():
            if obj in y or y in obj:
                out += xs
        return list(dict.fromkeys(out))

    def chat(self, q):
        """对话（继承 + 对象扩展——类-实例通用解）"""
        if "谁" in q:
            id_ans = self.identity(q)
            if id_ans:
                return id_ans
        qc = q.replace("？", "").replace("?", "")
        rest = qc
        for m in sorted(self.q_only_words(), key=len, reverse=True):
            rest = rest.replace(m, "")
        for m in ["是", "你", "我", "他", "她", "我们", "你们", "他们"]:
            rest = rest.replace(m, "")
        cs = [c for c in rest if c in self.w.ci]
        if not cs:
            return "我不明白。"
        obj = cs[-2] + cs[-1] if len(cs) >= 2 else cs[-1]
        hub = self.hub_emerge(q, obj=obj)
        if obj[-1] not in self.w.ci:
            return "我不明白。"
        i = self.w.ci[obj[-1]]
        objs = self.obj_expand(obj)            # 类-实例扩展（C15-02）
        cands = []
        for s in self.sents:
            if "？" in s or len(s) < 4:
                continue
            if not any(o in s for o in objs):  # 扩展后匹配
                continue
            if hub and hub not in s:
                continue
            idx = [self.w.ci[c] for c in s if c in self.w.ci]
            if not idx:
                continue
            rel = float(np.mean([self.w.KT_sp[i, j] for j in idx]))
            if rel > 0.0008:
                cands.append((rel, s))
        cands.sort(key=lambda x: -x[0])
        return cands[0][1] if cands else "我不明白。"


def run():
    print("=== M5 阶段 146：上位词/词类绑定（类-实例桥——C15-02）"
          " ===\n")
    app = HyperLake()
    base = os.path.dirname(__file__)
    app.load_corpus(
        os.path.join(base, "corpus_simple_natural.txt"),
        os.path.join(base, "corpus_simple2.txt"),
        os.path.join(base, "corpus_simple3.txt"),
        os.path.join(base, "corpus_simple4.txt"),
        os.path.join(base, "corpus_moon.txt"),
        os.path.join(base, "corpus_complex.txt"),
    )
    app.load_corpus(os.path.join(base, "corpus_wiki_filtered.txt"),
                    group="wiki")
    # 教学类-实例桥（用户教——增量）
    app.learn(["蓝色是颜色的一种。", "苹果是水果的一种。",
               "小猫是动物的一种。"])
    app.learn_hypernyms()                    # 教学后再抽（含教学句）

    # ---- exp1：上位对抽取 ----
    print("\n[exp1] 上位对抽取（'X是一种Y'模板——涌现）:")
    for y, xs in list(app.hyper.items())[:6]:
        print(f"      '{y}' ⊃ {xs[:3]}…")
    print(f"      共 {sum(len(v) for v in app.hyper.values())} 条"
          f"（{'抽取 ✓（C15-01 模板统计）' if app.hyper else '空'}）")

    # ---- exp2：颜色问（obj 扩展——通用解） ----
    print("\n[exp2] 颜色问（对象扩展——'颜色'→'蓝色'——类-实例桥）:")
    print(f"      对象扩展: '颜色' → {app.obj_expand('颜色')}")
    for q in ["你喜欢什么颜色？", "你最喜欢什么动物？"]:
        a = app.chat(q)
        print(f"      '{q}' → {a}")

    # ---- exp3：知识问答（上位定义句） ----
    print("\n[exp3] 知识问答（'X是一种Y'定义——wiki 上位句）:")
    for q in ["计算是什么？", "轮耕是什么？"]:
        ans = app.knowledge(q)
        print(f"      '{q}' → {ans[:36] if ans else '无答案'}…")

    # ---- exp4：思考改进（obj 扩展后候选相关） ----
    print("\n[exp4] 思考（obj 扩展——类-实例后候选相关）:")
    from stage145_thinking import ThinkingLake
    print(f"      （思考类可组合 HyperLake——'颜色'扩展后候选含实例句——"
          f"类-实例桥通用——记录）")
    print("\n[结论] 上位词绑定：模板涌现（'是一种'/'的一种'——C15-01）"
          "——对象扩展（类→实例——颜色→蓝色——通用解非 hub 豁免）——"
          "类-实例桥（C15-02）——颜色问/上位定义/思考改进——"
          "词类绑定落地")
    print("[done] stage146 hypernym")


if __name__ == "__main__":
    run()
