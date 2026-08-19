# -*- coding: utf-8 -*-
"""
M5 阶段 37：字符转导器（文字湖路线图——桥梁三步 ①——符号湖第一块砖）

TEXT_LAKE_ROADMAP 差距②：字符转导器 T_in——文字（字符/词）→ 复值动力场。
框架版（词嵌入的替代——不是查表向量，是动力学原语）：
  T_in：字 → 该字本征频率 ω 的振荡信号（驱动相位 = ω·t——共振才进入——stage7 发现）
  字身份 = 频率（ω_i 唯一）——单元 = 复值振荡器（γ 泄漏 + ω 本征 + 驱动 D）
  T_out：单元幅度读出 → 字识别（幅度最高 = 当前字）
  关联：沉积-侵蚀（R5——dW/dt = ε·e·conj(z) − λW）——序列共现 → 字间 W → 联想

验证（转导正确性）：
  exp1 共振识别：单字注入 → 对应单元幅度最高（识别率）
  exp2 频率分辨率：邻近频率区分度（共振带宽——stage7 不完善处 #3 顺带测量）
  exp3 关联学习：共现对（学习/天气…）→ W 增长 → 联想（字A → 预测字B）
  exp4 字序（相位差）：AB 序列 → 相位差编码顺序（stage7 exp2 字符版）
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(37)
DT = 0.05
GAMMA = 0.8    # 0.3 衰减太慢（13 步后残留 82%——前字干扰识别）→ 0.8（快衰减——注入间隔干净）
N_WORDS = 200
OMEGA_LO, OMEGA_HI = 0.5, 4.0        # 频率范围（rad/DT 归一化前）
AMP_IN = 1.2
PULSE_STEPS = 10                     # 输入脉冲宽度（步）
EPS = 0.02                           # 沉积率
LAMBDA = 0.005                       # 侵蚀率

# 200 常用汉字（高频——覆盖简单句子）
_CHARS_RAW = (
    "的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高"
    "自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题党程展五果料象员革位入常文总次品式活设及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运农指几九区强放决西被干做必战先回则任取据处队南给色光门即保治北造百规热领七海口东导器压志世金增争济阶油思术极交受联什认六共权收证改清己美再采转更单风切打白教速花带安场身车例真务具万"
    "每目至达走积示议声报斗完类八离华名确才科张信马节话米整空元况今集温传土许步群广石记需段研界拉林律叫且究观越织装影算低持音众书布复容儿须际商非验连断深难近矿千周委素技备半办青省列习响约支般史感劳便团往酸历市克何除消构府称太准精值号率族维划选标写存候毛亲快效斯院查江型眼王按格养易置派层片始却专状育厂京识适属圆包火住调满县局照参红细引听该铁价严龙飞"
)
# 语料字符集并入字表（语料字优先收录——保证训练字全部在表——过滤丢字破坏共现）
_CORPUS_CHARS = "今天天气很好我喜欢学习人民的生活很好技术进步很快世界和平发展我们相信明天教育很重要国家繁荣富强社会和谐稳定科学技术创新经济发展大家好知识就是力量认真学习每天努力进步"
CHARS = list(dict.fromkeys(_CORPUS_CHARS + _CHARS_RAW))[:N_WORDS]
CHAR_IDX = {c: i for i, c in enumerate(CHARS)}


class TextLake:
    """字符湖：字 = 频率（身份）——单元 = 复值振荡器——共振转导 + 沉积-侵蚀关联"""
    def __init__(self, omega_lo=OMEGA_LO, omega_hi=OMEGA_HI):
        self.omega = np.linspace(omega_lo, omega_hi, N_WORDS)   # 字 = 唯一频率（身份）
        self.gamma = GAMMA
        self.t = 0.0
        self.z = 0.1 * np.exp(1j * RNG.uniform(0, 2 * np.pi, N_WORDS))
        self.W = np.zeros((N_WORDS, N_WORDS))                   # 字间关联（沉积-侵蚀）
        self.act = np.zeros(N_WORDS)                            # 短时活跃度（学习窗口）

    # ---- T_in：字 → 振荡信号（共振准入） ----
    def input_signal(self, c):
        i = CHAR_IDX[c]
        drive = np.zeros(N_WORDS, dtype=complex)
        drive[i] = AMP_IN * np.exp(1j * (self.omega[i] * self.t))
        return drive

    def inject(self, c, steps=PULSE_STEPS):
        """注入一个字（脉冲）——读前加 3 步无驱动（前字残留衰减——稳定读出）"""
        for _ in range(steps):
            self.step(self.input_signal(c))
        for _ in range(3):
            self.step(np.zeros(N_WORDS, dtype=complex))
        return self.read()

    # ---- 单元动力学 ----
    def step(self, drive):
        dz = -self.gamma * self.z + 1j * self.omega * self.z
        dz += drive
        self.z = self.z + dz * DT
        self.t += DT
        over = np.abs(self.z) > 3.0
        self.z[over] = self.z[over] / np.abs(self.z[over]) * 2.0
        return self.z

    # ---- T_out：幅度读出 → 字识别 ----
    def read(self):
        amp = np.abs(self.z)
        best = int(np.argmax(amp))
        return CHARS[best], amp[best], amp

    # ---- 沉积-侵蚀（字间关联学习——R5） ----
    def inject_gap(self):
        """句界空窗（无驱动步——跨句不沉积）"""
        self.step(np.zeros(N_WORDS, dtype=complex))

    def learn(self, sents):
        """逐句注入——句内共现窗口沉积 W（句间 5 步空窗——侵蚀每遍一次）"""
        self.act = np.zeros(N_WORDS)          # 活跃度每遍清零（窗口语义——防跨遍膨胀）
        for sent in sents:
            seq = [c for c in sent if c in CHAR_IDX]
            for c in seq:
                self.inject(c)
                amp = np.abs(self.z)
                self.act += amp                               # 短时活跃累积
                self.act *= 0.9
            # 句内全对沉积（距离衰减——相邻强/远弱——短句无需滚动窗口）
            for i in range(len(seq)):
                wi = CHAR_IDX[seq[i]]
                for j in range(i + 1, len(seq)):
                    pair = EPS * self.act[wi] * self.act[CHAR_IDX[seq[j]]] / (j - i)
                    self.W[wi, CHAR_IDX[seq[j]]] += pair
            for _ in range(5):                                # 句界：5 步空窗
                self.inject_gap()
        self.W *= (1.0 - LAMBDA)                              # 侵蚀：每遍一次（低频关联消退）

    # ---- 联想：字 A → 预测字 B ----
    def predict(self, c, k=3):
        i = CHAR_IDX[c]
        row = self.W[i].copy()
        top = np.argsort(row)[::-1][:k]
        return [(CHARS[j], row[j]) for j in top if row[j] > 0]


def corpus():
    """正常的简单语句（用户指示——训练先建立在正常简单语句上——短句/日常/完整）"""
    return [
        "今天天气很好",
        "我喜欢学习",
        "人民的生活很好",
        "技术进步很快",
        "世界和平发展",
        "我们相信明天",
        "教育很重要",
        "国家繁荣富强",
        "社会和谐稳定",
        "科学技术创新",
        "经济发展",
        "大家好",
        "知识就是力量",
        "认真学习",
        "每天努力进步",
    ]


def run():
    print("=== M5 阶段 37：字符转导器（文字湖桥梁 ①——符号湖第一块砖） ===\n")
    lake = TextLake()
    sents = corpus()

    # ---- exp1：共振识别（T_in→T_out 正确性） ----
    test_chars = "学习天气人民"
    hits = 0
    for c in test_chars:
        if c in CHAR_IDX:
            out, amp, _ = lake.inject(c)
            ok = out == c
            hits += ok
            print(f"[exp1] 注入'{c}' → 读出'{out}' 幅度={amp:.2f} {'✓' if ok else '✗'}")
    print(f"       识别率 {hits}/{len([c for c in test_chars if c in CHAR_IDX])}")

    # ---- exp2：频率分辨率（共振带宽） ----
    # 注入字 0——观察近邻单元的激活（区分度）
    lake2 = TextLake()
    out, amp0, amps = lake2.inject(CHARS[0])
    order = np.argsort(amps)[::-1][:5]
    nb = [f"{CHARS[i]}:{amps[i]:.2f}" for i in order]
    sep = (amps[order[0]] - amps[order[1]]) / max(amps[order[0]], 1e-9)
    print(f"[exp2] 注入'{CHARS[0]}' 激活排序: {nb}")
    print(f"       主峰-次峰分离度 = {sep:.2f}（>0.2 = 频率分辨率足够——混淆风险低）")

    # ---- exp3：关联学习（沉积-侵蚀——共现 → 联想） ----
    for _ in range(12):                       # 语料重复注入（学习）
        lake.learn(sents)
    pairs = [(a, b) for (a, b) in [("学", "习"), ("天", "气"), ("人", "民"), ("世", "界"), ("技", "术")]
             if a in CHAR_IDX and b in CHAR_IDX]
    print("[exp3] 联想（学习 12 遍后——共现对应激活）:")
    for a, b in pairs:
        pred = lake.predict(a, k=3)
        names = [p[0] for p in pred]
        rank = names.index(b) + 1 if b in names else 99
        print(f"       '{a}' → {pred} —— '{b}' 排名 {rank}{' ✓' if rank <= 2 else ''}")

    # ---- exp4：字序（相位差编码顺序——AB vs BA 区分） ----
    lake4 = TextLake()
    lake4.inject("学"); ph_after_a = np.angle(lake4.z)
    lake4.inject("习"); ph_after_b = np.angle(lake4.z)
    # 相位差 = 注入间隔 × ω 差——顺序信息在相对相位
    d_ab = np.angle(np.exp(1j * (ph_after_b - ph_after_a)))
    i_a, i_b = CHAR_IDX["学"], CHAR_IDX["习"]
    delta = np.abs(d_ab[i_a] - d_ab[i_b])
    print(f"[exp4] '学'→'习' 相位差（学/习单元间）= {delta:.3f} rad"
          f"（非零 = 顺序信息承载——注入顺序改变相位关系）")

    # 图：学习前后 W 热图
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    im = ax.imshow(lake.W, cmap="viridis")
    ax.set_title("W (after learning)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax2 = axes[1]
    amp = np.abs(lake.z)
    ax2.bar(range(20), amp[:20])
    ax2.set_title("Unit amplitudes (first 20 chars)")
    fig.tight_layout()
    fig.savefig("fig_stage37.png", dpi=110)
    print("\n[plot] saved fig_stage37.png")
    print("[done] stage37 text transduction")


if __name__ == "__main__":
    run()
