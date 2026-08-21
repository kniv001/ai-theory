# -*- coding: utf-8 -*-
"""
M5 整合应用：文字湖完整系统（lake_app——记忆优先——全功能闭环）

整合（stage79-143 全部机制——稀疏湖架构）：
  ① 记忆：load_lake / save_lake / continual_learn（stage142——C6-03）
  ② 对话：涌现管线（stage117/134——q_only_words/learn_bridge/
    hub_emerge——适配稀疏）——日常问答
  ③ 知识问答：定义句检索（stage140——X是什么——wiki 知识）
  ④ 生成：段落（stage136 承接级联）/ 文章（stage141 多段）——
    主题贯穿
  ⑤ 纠正：评估（C30-01 独立检测）+ 削平/强化（stage143——C5-01）
  ⑥ 持续学习：新语料增量（C2-06——不重训）

使用：
  from lake_app import TextLakeApp
  app = TextLakeApp()          # 加载记忆（免重训）
  app.chat("苹果是什么？")      # 对话
  app.knowledge("农业是什么？")  # 知识问答
  app.essay("月", 200)          # 长文生成
  app.correct(...)              # 纠正
  app.learn(new_sents)          # 持续学习
  app.save()                    # 保存记忆
"""
import os
import re
import sys
import time
from collections import Counter
import numpy as np
import scipy.sparse as sp

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("gb"):
    sys.stdout.reconfigure(encoding="utf-8")

from stage79_spontaneous_hubs import load_corpus, DT
from stage136_natural_paragraph import bridge_of, special_sides, cohesion
from memory import load_lake, save_lake, continual_learn


class TextLakeApp:
    """文字湖完整应用（记忆优先——全功能闭环）"""

    def __init__(self, mem_path="lake_memory.npz"):
        self.w = load_lake(mem_path)
        self.sents = []                       # 基础语料（对话/段落）
        self.sents_wiki = []                  # wiki 语料（知识/长文）
        self.q_only = None                    # 问句专属词（涌现——C15-01）
        self.defs = []                        # 定义句（知识问答——涌现）
        print(f"[lake] 加载记忆 {mem_path}（{self.w.n} 字 / "
              f"{len(self.w.hubs)} 河道）")

    # ---------- 语料 ----------
    def load_corpus(self, *paths, group="base"):
        """载入语料（分组——base=对话/段落（纯净）——wiki=知识/长文
        （充实）——不重训——记忆已有知识）
        ——分组原因：wiki 加入对话池污染问句专属统计（'什么的什么'）
        + 漂移句入段落（stage143 修复的漂移源）"""
        lines = []
        for p in paths:
            lines += load_corpus(p)
        lines = [s if s.endswith(("。", "？", "！")) else s + "。"
                 for s in lines]
        if group == "wiki":
            self.sents_wiki = lines
        else:
            self.sents += lines
        # 定义句抽取（X是……——知识问答——wiki 源）
        self.defs = []
        src = self.sents + getattr(self, "sents_wiki", [])
        for s in src:
            m = re.match(r"^([一-鿿]{2,6})是", s)
            if m:
                self.defs.append((m.group(1), s))
        print(f"[lake] 基础语料 {len(self.sents)} 行 / wiki "
              f"{len(getattr(self, 'sents_wiki', []))} 行 / "
              f"定义句 {len(self.defs)}")

    # ---------- 对话（涌现管线——稀疏适配） ----------
    def q_only_words(self):
        """问句专属词（C15-01 统计——只在问句出现的词——非写死）
        ——两字词要求**两字都是单字专属**（'什么'✓——'么颜'✗——
        '什么颜'跨词窗——'颜'是内容字——非问句标记——排除——
        stage134 的'至少一字'太松——跨词窗污染对象提取）"""
        if self.q_only is not None:
            return self.q_only
        q_lines = [s for s in self.sents if "？" in s]
        st_lines = [s for s in self.sents if "？" not in s]
        q_text, st_text = "".join(q_lines), "".join(st_lines)
        single = set(q_text) - set(st_text)                # 单字专属
        words = set(single)
        for s in q_lines:                                  # 两字专属
            for i in range(len(s) - 1):
                w2 = s[i:i + 2]
                if w2 in q_text and w2 not in st_text \
                        and w2[0] in single and w2[1] in single:
                    words.add(w2)
        self.q_only = words
        return words

    def hub_emerge(self, q, obj=None):
        """类型词涌现（stage117——直接命中（非问句专属单字枢纽）+
        桥检索——机制修复：直接命中须与**对象**有 K 关联（C13-01
        关系河道——结构字'的'与对象弱关联 → 排除——类型词'是/很'
        与对象强关联 → 保留））"""
        q_chars = [c for c in q if c in self.w.ci]
        if not q_chars:
            return None
        oi = self.w.ci[obj[-1]] if obj and obj[-1] in self.w.ci else None
        # 直接命中：单字枢纽 in q 非专属 且 非对象成分（'色'∈'颜色'——
        # C30-01 成分不指代整体）且 非结构字（连接度 >85%——
        # '的'98%/'在'91%——万能共现——非类型词——'是'83%/'很'45%
        # 保留——类型词）
        n = self.w.n
        direct = [h for h in self.w.hubs if len(h) == 1 and h in q
                  and h not in self.q_only_words()
                  and (obj is None or h not in obj)
                  and int((self.w.KT_sp[self.w.ci[h]] > 0).sum()) <= n * 0.85]
        if direct:
            best, best_v = None, -1.0
            for h in direct:
                if oi is not None:         # 对象关联验证（'的'弱——
                    rel_obj = float(self.w.KT_sp[oi, self.w.ci[h]])  # 排除）
                    if rel_obj < 0.001:
                        continue
                v = sum(float(self.w.K[h][self.w.ci[c], self.w.ci[h]])
                        for c in q_chars if c in self.w.ci)
                if v > best_v:
                    best, best_v = h, v
            if best is not None:
                return best
        scores = {}
        for h in self.w.hubs:
            rep = h[0] if len(h) > 1 else h
            if rep not in self.w.ci:
                continue
            v = sum(float(self.w.K[h][self.w.ci[c], self.w.ci[rep]])
                    for c in q_chars if c in self.w.ci)
            if v > 0.001:
                scores[h] = v
        return max(scores, key=scores.get) if scores else None

    # ---------- 身份问（stage118——'你是谁'→身份检索——跨会话） ----------
    def identity(self, q):
        """身份问（你是谁？→ 检索'我是X'句——KT 关联——身份湖）
        ——调用方已保证：hub 不存在（无类型词）+ 人称问句（C21-01——
        身份问无关系类型——是身份检索）"""
        if "我" not in self.w.ci:
            return None
        cands = []
        for s in self.sents:
            if "我" not in s or "我是" not in s or "？" in s or len(s) < 4:
                continue
            idx = [self.w.ci[c] for c in s if c in self.w.ci]
            if not idx:
                continue
            # 与'我'的关联（身份绑定——C21-01——自我湖河道）
            rel = float(self.w.KT_sp[self.w.ci["我"], idx[0]])
            cands.append((rel, s))
        cands.sort(key=lambda x: -x[0])
        return cands[0][1] if cands else None

    def chat(self, q):
        """对话（对象提取 + hub 类型匹配 + KT_sp 关联——涌现）
        ——hub 是问句内容词（in q）时 obj 豁免（类-实例：
        '喜欢什么颜色'→答案'蓝色'——实例不含类词——hub 已约束）"""
        # 身份问优先（'谁'=问身份——C21-01——identity 检索'我是X'——
        # 有则答——无则回落普通管线——'谁'是封闭问句词）
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
        hub_in_q = hub is not None and hub in qc
        cands = []
        for s in self.sents:
            if "？" in s or len(s) < 4:
                continue
            if not hub_in_q and obj not in s:
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

    def knowledge(self, q):
        """知识问答（X是什么？→ 定义句检索——KT 关联）"""
        qc = q.replace("？", "").replace("?", "").replace("是什么", "").strip()
        if not qc or qc[-1] not in self.w.ci:
            return None
        i = self.w.ci[qc[-1]]
        cands = []
        for subj, s in self.defs:
            if qc not in s:
                continue
            idx = [self.w.ci[c] for c in s if c in self.w.ci]
            if not idx:
                continue
            rel = float(np.mean([self.w.KT_sp[i, j] for j in idx]))
            if rel > 0.0005:
                cands.append((rel, s))
        cands.sort(key=lambda x: -x[0])
        return cands[0][1] if cands else None

    # ---------- 生成（stage136/141——承接级联 + 多段文章） ----------
    def paragraph(self, topic, n=12):
        """段落（承接式级联——句末成分→桥——Maimon 衔接）"""
        used = set()
        ti = self.w.ci[topic[-1]] if topic[-1] in self.w.ci else None
        sides = special_sides(self.w, topic, self.sents)
        starters = [s for s in self.sents if topic in s
                    and s not in used and len(s) >= 5 and "。" in s]
        if not starters:
            return []
        out = [starters[0]]
        used.add(starters[0])
        cur = starters[0]
        for step in range(n):
            bridge = bridge_of(self.w, cur)
            cands = []
            for s in self.sents:
                if s in used or len(s) < 5 or "。" not in s:
                    continue
                if not (topic in s or any(side in s for side in sides)):
                    continue
                idx = [self.w.ci[c] for c in s if c in self.w.ci]
                if not idx or ti is None:
                    continue
                rel = float(np.mean([self.w.KT_sp[ti, j] for j in idx]))
                if rel < 0.004:
                    continue
                sc = set(re.sub(r"[。！？，、\s]", "", s))
                if any(len(sc & set(re.sub(r"[。！？，、\s]", "", u))) >=
                       min(len(sc), len(u)) * 0.7 for u in used):
                    continue
                bonus = 1.5 if (bridge and bridge in s) else 1.0
                cands.append((rel * bonus, s))
            cands.sort(key=lambda x: -x[0])
            if not cands:
                break
            nxt = cands[0][1]
            out.append(nxt)
            used.add(nxt)
            cur = nxt
        return out

    def essay(self, topic, min_chars=200):
        """文章（多段——≥目标字数——主题贯穿——wiki 语料充实）"""
        src = self.sents + self.sents_wiki
        paras = []
        used_all = set()
        cur_topic = topic
        for p in range(3):
            used = set()
            ti = self.w.ci[cur_topic[-1]] if cur_topic[-1] in self.w.ci else None
            sides = special_sides(self.w, cur_topic, src)
            starters = [s for s in src
                        if cur_topic in s and s not in used_all
                        and len(s) >= 5 and "。" in s]
            if not starters:
                break
            para = [starters[0]]
            used.add(starters[0])
            used_all.add(starters[0])
            cur = starters[0]
            for step in range(12):
                bridge = bridge_of(self.w, cur)
                cands = []
                for s in src:
                    if s in used or s in used_all or len(s) < 5 or "。" not in s:
                        continue
                    if not (cur_topic in s or any(side in s for side in sides)):
                        continue
                    idx = [self.w.ci[c] for c in s if c in self.w.ci]
                    if not idx or ti is None:
                        continue
                    rel = float(np.mean([self.w.KT_sp[ti, j] for j in idx]))
                    if rel < 0.004:
                        continue
                    sc = set(re.sub(r"[。！？，、\s]", "", s))
                    if any(len(sc & set(re.sub(r"[。！？，、\s]", "", u))) >=
                           min(len(sc), len(u)) * 0.7 for u in used):
                        continue
                    bonus = 1.5 if (bridge and bridge in s) else 1.0
                    cands.append((rel * bonus, s))
                cands.sort(key=lambda x: -x[0])
                if not cands:
                    break
                nxt = cands[0][1]
                para.append(nxt)
                used.add(nxt)
                used_all.add(nxt)
                cur = nxt
            paras.append(para)
            if len("".join(sum(paras, []))) >= min_chars:
                break
            b = bridge_of(self.w, cur)
            cur_topic = b if b and b != cur_topic and \
                any(b in s for s in src) else topic
        return paras

    # ---------- 纠正（stage143——C5-01 削平/强化） ----------
    def correct(self, bad_sents, good_sents, eps_bad=0.5, eps_good=0.15):
        """纠正：坏句削平（侵蚀）+ 好句强化（重放固化 R42）"""
        for s in bad_sents:
            idx = [self.w.ci[c] for c in s if c in self.w.ci]
            for a in range(len(idx) - 1):
                for b in range(a + 1, len(idx)):
                    i, j = idx[a], idx[b]
                    for h in self.w.hubs:
                        k = self.w.K[h]
                        if k[i, j] != 0:
                            self.w.K[h][i, j] *= (1.0 - eps_bad)
                            self.w.K[h][j, i] *= (1.0 - eps_bad)
        acc = {h: {} for h in self.w.hubs}
        for s in good_sents:
            idx = [self.w.ci[c] for c in s if c in self.w.ci]
            hit = [h for h in self.w.hubs if h in s]
            if not hit:
                continue
            for a in range(len(idx) - 1):
                for b in range(a + 1, len(idx)):
                    i, j = idx[a], idx[b]
                    for h in hit:
                        a2 = acc[h]
                        a2[(i, j)] = a2.get((i, j), 0.0) + eps_good
                        a2[(j, i)] = a2.get((j, i), 0.0) + eps_good * 0.3
        for h in self.w.hubs:
            a = acc[h]
            if not a:
                continue
            keys = np.array(list(a.keys()), dtype=int)
            vals = np.array(list(a.values()))
            self.w.K[h] = self.w.K[h] + sp.coo_matrix(
                (vals, (keys[:, 0], keys[:, 1])),
                shape=(self.w.n, self.w.n)).tocsr()
        self._sync()
        return len(bad_sents)

    def _sync(self):
        """KT 重合并 + GPU 更新"""
        self.w.KT_sp = sum(self.w.K.values())
        if self.w.use_gpu:
            import torch
            self.w.KT = torch.tensor(self.w.KT_sp.toarray(),
                                     device="cuda", dtype=torch.float32)
            self.w.rsT_gpu = torch.tensor(self.w.KT_sp.sum(axis=1).A1,
                                          device="cuda", dtype=torch.float32)
        else:
            self.w.KT = self.w.KT_sp.toarray()

    # ---------- 持续学习 + 保存 ----------
    def learn(self, new_sents, mem_path="lake_memory_v2.npz"):
        """持续学习（增量——不重训——C2-06）"""
        self.w = continual_learn(self.w, new_sents)
        self.sents += [s if s.endswith(("。", "？", "！")) else s + "。"
                       for s in new_sents]
        self.save(mem_path)
        return len(new_sents)

    def save(self, path="lake_memory.npz"):
        save_lake(self.w, path)
        # 语料持久化（跨会话检索源——对话/身份/知识——C6-03）
        np.savez_compressed(path.replace(".npz", "_sents.npz"),
                            sents=np.array(self.sents, dtype=object),
                            wiki=np.array(self.sents_wiki, dtype=object))
        print(f"[lake] 记忆保存 {path}（含语料）")

    def load_sents(self, path="lake_memory.npz"):
        """恢复持久化语料（跨会话——上次对话检索源）"""
        spath = path.replace(".npz", "_sents.npz")
        if os.path.exists(spath):
            d = np.load(spath, allow_pickle=True)
            self.sents = [str(s) for s in d["sents"]] if len(d["sents"]) else []
            if "wiki" in d:
                self.sents_wiki = [str(s) for s in d["wiki"]] \
                    if len(d["wiki"]) else []
            self._rebuild_defs()
            print(f"[lake] 恢复语料 {len(self.sents)} 行（跨会话——"
                  f"上次教学记得）")

    def _rebuild_defs(self):
        self.defs = []
        src = self.sents + self.sents_wiki
        for s in src:
            m = re.match(r"^([一-鿿]{2,6})是", s)
            if m:
                self.defs.append((m.group(1), s))


def demo():
    """整合演示（记忆优先——全功能）"""
    print("=== 文字湖整合应用（记忆优先——全功能闭环） ===\n")
    app = TextLakeApp()
    base = os.path.dirname(__file__)
    app.load_corpus(
        os.path.join(base, "corpus_simple_natural.txt"),
        os.path.join(base, "corpus_simple2.txt"),
        os.path.join(base, "corpus_simple3.txt"),
        os.path.join(base, "corpus_moon.txt"),
        os.path.join(base, "corpus_complex.txt"),
    )
    app.load_corpus(
        os.path.join(base, "corpus_wiki_filtered.txt"),
        group="wiki",
    )

    # ① 对话
    print("\n--- ① 对话（涌现管线）---")
    for q in ["苹果是什么？", "苹果怎么样？", "月亮怎么样？"]:
        print(f"  Q: {q}")
        print(f"  A: {app.chat(q)}")

    # ② 知识问答
    print("\n--- ② 知识问答（定义句检索）---")
    for q in ["农业是什么？", "计算机是什么？"]:
        ans = app.knowledge(q)
        print(f"  Q: {q}")
        print(f"  A: {ans[:40] if ans else '无答案'}…")

    # ③ 段落生成
    print("\n--- ③ 段落生成（承接式级联）---")
    para = app.paragraph("月", n=10)
    print(f"  {' '.join(c.rstrip('。')+'。' for c in para)}")

    # ④ 长文生成
    print("\n--- ④ 长文生成（多段文章）---")
    paras = app.essay("农", 150)
    for pi, p in enumerate(paras):
        print(f"  [段{pi+1}] {' '.join(c.rstrip('。')+'。' for c in p[:6])}…")

    # ⑤ 持续学习 + 纠正（新语料增量——不重训）
    print("\n--- ⑤ 持续学习 + 纠正（增量——不重训）---")
    new = ["我们喜欢月亮因为它很明亮。", "中秋的月亮很圆。"]
    app.learn(new, mem_path="lake_memory_app.npz")
    print(f"  增量学习 {len(new)} 句 → 保存 lake_memory_app.npz")
    print("\n[demo done] 文字湖整合应用——记忆/对话/知识/生成/纠正/"
          "持续学习全闭环")


if __name__ == "__main__":
    demo()
