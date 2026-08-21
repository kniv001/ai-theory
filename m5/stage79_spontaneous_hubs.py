# -*- coding: utf-8 -*-
"""
M5 阶段 79：自发枢纽 v2（写死词表清零——hardcoded_audit 待办①②——用户：
"工程要靠着理论——惯性思维在本框架内不成立——随时记住理论"）

v2 修正（diag 根因——"因"单配"为"→ 词块先行）：
  ① 词块先行：相邻共现率>阈值的字对 → 词块（"因为"=94/94=1.0）——
     "因"未涌现为单字枢纽是正确统计结论（它单配"为"）——涌现对象是词块"因为"
  ② 词块覆盖单字：句子含词块 → 只沉积词块河道（K["因为"] 纯因果——不污染 K["为"]）
  ③ 多字对象检索：answer 对对象的所有字索引求和（ci 是单字——"苹果"要取 苹+果 两行）
  ④ 动力学线性合并：多河道线性叠加 = K_total@z（一次矩阵乘——河道数不增计算量）

写死替换（对照 hardcoded_audit.md）：
  FUNC_WORDS（stage47/48——是/很/有/因为…词表）→ 自发枢纽（词块+单字——
      频率×上下文熵×位置集中度——C15-01 模板锚从统计生成）
  REL_IDX 4层 + 触发词 if-else（stage61）→ 枢纽河道（K={枢纽: 矩阵}——C22-01）
  问句解析 if-else（stage61）→ 问句驱动枢纽检索（C74-01/C122-01 零河道）
  模板匹配 FUNC 找（stage47）→ 枢纽序列模板（C20-01/02）

理论锚：C15-01 / C15-02 / C22-01 / C122-01 / C74-01

验证：
  exp1 自发枢纽（词块+单字——对照人工功能词 是/很/有/因为/所以/属于/包括）
  exp2 枢纽河道（K[是][苹果]→水果；K[很][苹果]→甜；K[因为][带伞]→下雨）
  exp3 自发问答（无 if-else——"苹果是什么？"→是河道；"为什么带伞？"→为→因为河道）
  exp4 枢纽序列模板 + 词块展示
"""
import os
import re
import sys
import time
from collections import Counter, defaultdict
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("gb"):
    sys.stdout.reconfigure(encoding="utf-8")

RNG = np.random.default_rng(79)
DT = 0.05
GAMMA = 0.8
OMEGA_LO, OMEGA_HI = 0.5, 4.0
AMP_IN = 1.2
PULSE_STEPS = 5
EPS_K = 0.02
LAMBDA_K = 0.01
K_CAP = 0.5
DELTA_PHI = np.pi / 6


def load_corpus(path, lo=3, hi=80, n=None):
    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    clean = [s for s in lines if lo <= len(s) <= hi
             and re.search(r"[一-鿿]", s) and not re.search(r"[A-Za-z]", s)]
    if n and len(clean) > n:
        clean = clean[:n]
    return clean


# ---------- exp1：自发词块 + 单字枢纽（纯统计——C15-01/C15-02/C16-01 尺度递归） ----------

def extract_blocks(sents, min_co=8, ratio=0.5):
    """词块提取（自发分词）：相邻共现率 > ratio → 词块
    相邻率 = 相邻共现 / min(单字出现)——"因为" 201/min(257,383)=0.78——纯统计
    ratio=0.5（2026-08-21 两次修正收敛——stage86 曾调 0.85 防"很甜"绑架"很"
      ——但 0.85 也滤掉"因为"(0.78)——因果河道消失——过度修复——正解是
      extract_hubs 的块内豁免（很 100-13=87 豁免恢复单字枢纽——"很甜"块
      存在无害）——0.5 回原值：因为✓/很甜✓（豁免保很）/没有✓/什么✓）
    纯汉字块（标点/拉丁相邻对——'，但'/',0'——过滤）
    C16-01 尺度递归：字→词——词块 = 字级统计的递归应用"""
    freq = Counter()
    pairs = Counter()
    for s in sents:
        for i in range(len(s)):
            freq[s[i]] += 1
            if i < len(s) - 1:
                pairs[s[i] + s[i + 1]] += 1
    blocks = []
    for (a, b), cnt in pairs.most_common():
        if cnt < min_co:
            continue
        if freq[a] > 0 and freq[b] > 0 and cnt / min(freq[a], freq[b]) > ratio:
            if re.fullmatch(r"[一-鿿]{2}", a + b):
                blocks.append(a + b)
    return blocks


def extract_hubs(sents, blocks, top_n=50, min_freq=30):
    """单字枢纽：频率 × 上下文熵 × 位置集中度——词块内字排除（词块已覆盖）
    频率：结构锚候选；上下文熵：内容词受限/功能词弥散（C15-02）；
    位置集中度：结构位（"是"句中/"很"句中——句内 std 小）
    块内豁免（2026-08-21——stage86："很甜"块（率0.87）绑架"很"——结构字
      被内容块锁死——C15-02 违反——豁免：块外出现仍 ≥ min_freq 的字
      保留单字枢纽（"很"100-13=87 ✓——"甜"15-13=2 留块内——属性词正确）"""
    in_block = set("".join(blocks))
    block_cnt = Counter()
    for b in blocks:
        block_cnt[b[0]] += 1
        block_cnt[b[1]] += 1
    # 第一遍：总频率（不排除——豁免判定用）
    freq_all = Counter()
    for s in sents:
        freq_all.update(s)
    # 豁免：块外出现仍 ≥ min_freq 的字保留单字枢纽（"很"100-13=87 ✓）
    for c in list(in_block):
        if freq_all[c] - block_cnt[c] >= min_freq:
            in_block.discard(c)
    # 第二遍：排除剩余 in_block 统计
    freq = Counter()
    ctx_pre = defaultdict(set)
    ctx_post = defaultdict(set)
    pos = defaultdict(list)
    for s in sents:
        L = len(s)
        for i, c in enumerate(s):
            if c in in_block:
                continue
            freq[c] += 1
            pos[c].append(i / L)
            if i > 0:
                ctx_pre[c].add(s[i - 1])
            if i < L - 1:
                ctx_post[c].add(s[i + 1])
    score = {}
    for c, n in freq.items():
        if n < min_freq or not re.match(r"[一-鿿]", c):
            continue
        ctx_div = np.log2(len(ctx_pre[c]) + 1) + np.log2(len(ctx_post[c]) + 1)
        p = np.array(pos[c])
        pos_std = p.std() + 0.05
        score[c] = np.log(1 + n) * ctx_div / pos_std
    ranked = sorted(score.items(), key=lambda kv: -kv[1])
    return [c for c, _ in ranked[:top_n]]


# ---------- exp2：枢纽河道（K={枢纽: 矩阵}——C22-01 关系湖） ----------

class HubLake:
    """枢纽湖：每个枢纽（词块或单字）一条河道——无 RELS 列表——河道数=涌现数
    学习：句子命中词块 → 词块河道（覆盖——块内单字不单独沉积）；
      否则按单字枢纽沉积——"因为下雨所以带伞" → K[因为] + K[所以]（纯因果）
    动力学：多河道线性叠加 = K_total@z（合并一次矩阵乘）"""

    def __init__(self, chars, hubs):
        self.chars = list(chars)
        self.ci = {c: i for i, c in enumerate(chars)}
        self.hubs = [h for h in hubs if all(c in self.ci for c in h)]
        n = len(chars)
        self.n = n
        self.omega = RNG.uniform(OMEGA_LO, OMEGA_HI, n)
        self.gamma = GAMMA
        self.z = 0.1 * np.exp(1j * RNG.uniform(0, 2 * np.pi, n))
        self.t = 0.0
        self.K = {h: np.zeros((n, n)) for h in self.hubs}
        self.rowsum = {h: np.zeros(n) for h in self.hubs}
        self.KT = np.zeros((n, n))      # 合并河道（动力学用）
        self.rsT = np.zeros(n)

    def _sync_total(self):
        self.KT = sum(self.K.values())
        self.rsT = sum(self.rowsum.values())

    def remember(self, c):
        """动态词汇扩展（stage87——持续学习——C2-06 结构永不冻结：
        训练中遇新字 → 湖扩展（所有河道矩阵扩列）——O(n²) 拼接——新字少可接受）"""
        if c in self.ci:
            return self.ci[c]
        n = self.n
        for h in self.hubs:
            Kh = self.K[h]
            Kh2 = np.zeros((n + 1, n + 1))
            Kh2[:n, :n] = Kh
            self.K[h] = Kh2
            rs2 = np.zeros(n + 1)
            rs2[:n] = self.rowsum[h]
            self.rowsum[h] = rs2
        self.chars.append(c)
        self.ci[c] = n
        self.omega = np.append(self.omega, RNG.uniform(OMEGA_LO, OMEGA_HI))
        self.z = np.append(self.z, 0.1 * np.exp(1j * RNG.uniform(0, 2 * np.pi)))
        self.n += 1
        return n

    def add_hub(self, h):
        """新枢纽河道（持续学习——新语料的新词块/单字——C2-02 结构生长：
        湖结构是训练产物——持续学习中新河道加入）"""
        if h in self.K or not all(c in self.ci for c in h):
            return
        n = self.n
        self.hubs.append(h)
        self.K[h] = np.zeros((n, n))
        self.rowsum[h] = np.zeros(n)
        self._sync_total()

    def learn_epoch_batch(self, sents, B=128):
        n = len(self.chars)
        for start in range(0, len(sents), B):
            batch = sents[start:start + B]
            # 动态词汇扩展（batch 内新字先入湖——持续学习——C2-06）
            for sent in batch:
                for c in sent:
                    if c not in self.ci:
                        self.remember(c)
            n = self.n
            self._sync_total()
            Z = np.zeros((len(batch), n), dtype=complex)
            drives = np.zeros((len(batch), n), dtype=complex)
            seqs = []
            for bi, sent in enumerate(batch):
                hit = [h for h in self.hubs if h in sent]   # 命中的枢纽（自发）
                idx = [self.ci[c] for c in sent if c in self.ci]
                if len(idx) < 2 or not hit:
                    continue
                seqs.append((bi, idx, hit))
                for posi, i in enumerate(idx):
                    drives[bi, i] += AMP_IN * np.exp(1j * (self.omega[i] * self.t + posi * DELTA_PHI))
            for _ in range(PULSE_STEPS + 3):
                dz = -self.gamma * Z + 1j * self.omega * Z
                dz += (self.KT @ Z.T).T - Z * self.rsT   # 合并河道——一次矩阵乘
                dz += drives
                Z = Z + dz * DT
                over = np.abs(Z) > 3.0
                Z[over] = Z[over] / np.abs(Z[over]) * 2.0
            amp = np.abs(Z)
            for bi, idx, hit in seqs:
                sub = np.array(idx)
                A = amp[bi, sub]
                L = len(idx)
                di = np.arange(L)
                dist_w = 1.0 / np.maximum(np.abs(di[:, None] - di[None, :]), 1.0)
                contrib = EPS_K * np.outer(A, A) * np.triu(dist_w, 1)
                pi, pj = np.nonzero(contrib)
                for h in hit:
                    self.K[h][sub[pi], sub[pj]] += contrib[pi, pj]
                    self.K[h][sub[pj], sub[pi]] += contrib[pi, pj] * 0.3
        for h in self.hubs:
            self.K[h] *= (1.0 - LAMBDA_K)
            rs = self.K[h].sum(axis=1)
            over = rs > K_CAP
            self.K[h][over] *= (K_CAP / rs[over])[:, None]
            self.K[h][:, over] *= (K_CAP / rs[over])[None, :]
            self.rowsum[h] = self.K[h].sum(axis=1)
        self._sync_total()

    def answer(self, hub, c, k=3):
        """沿枢纽河道检索（多字对象——所有字索引行求和——双向）
        无 rel 类型判断（cause 上行自动被双向和覆盖）"""
        if hub not in self.K:
            return []
        idx = [self.ci[ch] for ch in c if ch in self.ci]
        if not idx:
            return []
        row = np.zeros(len(self.chars))
        for i in idx:
            row += self.K[hub][i] + self.K[hub][:, i]
        row[idx] = 0                          # 排除自身
        top = np.argsort(row)[::-1][:k]
        return [(self.chars[j], row[j]) for j in top if row[j] > 0.002]

    def ask(self, q):
        """自发问答（无 if-else）：问句 → 去标记 → 关系河道检索
        ① 问标记块 = 与"？"紧邻的词块（"什么"若在语料成块——自发位置判定
           ——非写死问词表）；单字枢纽（"是"）从对象移除——内容块（"苹果"）
           保留（对象词）
        ② 河道 = 单字枢纽（"是"）；无单字 → 字级反查（"为什么"→"为"→
           "因为"块——因果河道——max sum 自然避开无沉积的标记块）
        ③ 对象 = 词块位置最前（主语位"苹果"）→ cs 尾双字（"带伞"）→ cs 首字
        零河道（C122-01）：语料无问标记词（什/吗≈0 次）→ "怎么样"类问句
          无解析 = 统计诚实（不是写死失败）"""
        hit_single = [h for h in self.hubs if len(h) == 1 and h in q]
        mark_blocks = []
        if "？" in q:
            iq = q.index("？")
            for h in self.hubs:
                if len(h) >= 2 and iq >= len(h) and q.startswith(h, iq - len(h)):
                    mark_blocks.append(h)
        rest = q
        for h in mark_blocks + hit_single:
            rest = rest.replace(h, "")
        cs = [c for c in rest if c in self.ci]
        if not cs:
            return None, None, []
        if hit_single:
            hub = max(hit_single, key=lambda h: self.K[h].sum())
        else:
            cand = set()
            for c in q:
                if c not in self.ci:
                    continue
                covers = [h for h in self.hubs if len(h) > 1 and c in h]
                if covers:
                    cand.add(max(covers, key=len))
            if not cand:
                return None, None, []
            hub = max(cand, key=lambda h: self.K[h].sum())   # 因果块有沉积——胜出
            for c in q:
                for hh in cand:
                    if c in hh:
                        rest = rest.replace(c, "")           # 反查覆盖字移除
            cs = [c for c in rest if c in self.ci]
            if not cs:
                return None, None, []
        # 外来标记残余：紧跟在非 ci 字后的 ci 字（"什么"——"什"不在语料——
        # "么"是外来残余——从对象排除——零河道：非语料标记不构成对象）
        foreign = set()
        for i in range(1, len(rest)):
            if rest[i - 1] not in self.ci and rest[i] in self.ci and rest[i - 1] != "？":
                foreign.add(rest[i])
        cs = [c for c in rest if c in self.ci and c not in foreign]
        if not cs:
            return None, None, []
        rest_clean = rest.replace("？", "").replace("?", "")
        blks = [h for h in self.hubs if len(h) >= 2 and h in rest_clean]
        if blks:
            obj = min(blks, key=rest_clean.index)   # 主语位优先（"苹果"）
        elif len(cs) >= 2:
            obj = cs[-2] + cs[-1]                    # 尾双字（"带伞"）
        else:
            obj = cs[0]                              # 首字（"猫"）
        return hub, obj, self.answer(hub, obj)


# ---------- exp4：枢纽序列模板（替换 stage47 FUNC 匹配） ----------

def template_of(sent, hubs):
    """句子 → 枢纽序列模板（X[枢纽]Y——槽位=枢纽之间）——替换 stage47 FUNC_WORDS"""
    hits = []
    i = 0
    while i < len(sent):
        m = next((h for h in hubs if sent.startswith(h, i)), None)
        if m:
            hits.append((i, m))
            i += len(m)
        else:
            i += 1
    if not hits:
        return None
    tmpl, slots = [], []
    prev = 0
    for i, h in hits:
        if i > prev:
            slots.append(sent[prev:i][:6])
            tmpl.append("X")
        tmpl.append(h)
        prev = i + len(h)
    if prev < len(sent):
        slots.append(sent[prev:][:6])
        tmpl.append("X")
    return (tuple(tmpl), slots)


def run():
    print("=== M5 阶段 79：自发枢纽 v2（写死词表清零——词块先行——枢纽河道——自发问答） ===\n")
    base = os.path.dirname(__file__)
    simple = load_corpus(os.path.join(base, "corpus_simple_natural.txt"), n=900)
    wiki = load_corpus(os.path.join(base, "corpus_wiki_filtered.txt"), n=600)
    attr = load_corpus(os.path.join(base, "corpus_attr_cause.txt"))
    isa_sents = ["苹果是水果", "香蕉是水果", "西瓜是水果", "葡萄是水果",
                 "猫是动物", "狗是动物", "鸟是动物", "鱼是动物",
                 "水是液体", "冰是固体", "雪是白色的", "天空是蓝色的",
                 "老虎是动物", "树是植物", "花是植物", "石头是固体",
                 "苹果可以吃", "水可以喝", "雨是从云落下来的"]
    sents = simple + wiki + attr + isa_sents

    # ---- exp1：自发词块 + 单字枢纽 ----
    blocks = extract_blocks(sents)
    hubs = extract_hubs(sents, blocks)
    all_hubs = blocks + hubs
    print(f"[exp1] 自发词块（相邻共现率>{0.5}——{len(blocks)} 个）:")
    print(f"      {blocks}")
    print(f"[exp1] 单字枢纽（词块外——频率×上下文熵×位置集中度——{len(hubs)} 个）:")
    print(f"      {''.join(hubs)}")
    print("      人工功能词对照（应被统计涌现——非词表）:")
    for fw in ["是", "很", "有", "因为", "所以", "属于", "包括"]:
        print(f"      '{fw}' {'[涌现]' if fw in all_hubs else '[未涌现]'}")

    # ---- exp2：枢纽河道 ----
    chars = list(dict.fromkeys("".join(sents)))
    print(f"\n词汇表 {len(chars)} 字 / 枢纽 {len(all_hubs)} 个（词块 {len(blocks)} + 单字 {len(hubs)}）/ 语料 {len(sents)} 行")
    w = HubLake(chars, all_hubs)
    t0 = time.perf_counter()
    for ep in range(6):
        w.learn_epoch_batch(sents, B=128)
    print(f"训练完成——{time.perf_counter()-t0:.0f}s（{len(all_hubs)} 河道——无 RELS 列表——合并动力学）")
    print("\n[exp2] 枢纽河道（无触发词——K[枢纽] 直接检索——多字对象）:")
    for hub, obj in [("是", "苹果"), ("很", "苹果"), ("因为", "带伞"),
                     ("是", "猫"), ("很", "天气"), ("因为", "穿棉衣"), ("有", "苹果")]:
        ans = w.answer(hub, obj)
        if ans:
            print(f"      K['{hub}']['{obj}'] → {[(a, f'{v:.2f}') for a, v in ans[:3]]}")

    # ---- exp3：自发问答（无 if-else） ----
    print("\n[exp3] 自发问答（问句 → 命中枢纽 → 河道检索——零人工映射）:")
    for q in ["苹果是什么？", "苹果怎么样？", "为什么带伞？", "猫是什么？", "天气怎么样？", "为什么穿棉衣？"]:
        hub, obj, ans = w.ask(q)
        if ans:
            print(f"      Q: '{q}' → 枢纽'{hub}' '{obj}' → {[(a, f'{v:.2f}') for a, v in ans[:3]]}")
        else:
            print(f"      Q: '{q}' → （问句无命中枢纽——语料覆盖不足）")

    # ---- exp4：枢纽序列模板 ----
    tmpl_count = Counter()
    for s in sents:
        m = template_of(s, all_hubs)
        if m:
            tmpl_count[m[0]] += 1
    print("\n[exp4] 枢纽序列模板（替换 FUNC_WORDS 查找——频率 ≥20）:")
    for tmpl, cnt in tmpl_count.most_common(12):
        if cnt >= 20:
            print(f"      {list(tmpl)} × {cnt}")
    print("\n[done] stage79 spontaneous hubs v2")


if __name__ == "__main__":
    run()
