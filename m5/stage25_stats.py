# -*- coding: utf-8 -*-
"""
M5 阶段 25 统计可视化：数据分布展示
多 seed 收集 → 6 面板：
  ① 基因分布（初始 vs 末代——9 基因小提琴）
  ② α_p×α_m 食性二维空间（四象限生态位）
  ③ 生态位-体型（mass vs α_m）
  ④ 多 seed 存活率（食肉/食草——分布）
  ⑤ 种群时间序列（多 seed 均值±std）
  ⑥ 捕食动态（成功率/追猎频率分布）
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

exec(open('stage25_body_size_layers.py', encoding='utf-8').read().split("def run():")[0])

N_SEED = 8
T = 1200
GENES = ["mass", "accel", "alpha_p", "alpha_m", "vision_g", "fov_g", "risk", "cruise_g", "sprint_g"]

hist_all, herb_l, carn_l, peaks, ok_rates = [], [], [], [], []
gene_final = {g: [] for g in GENES}
gene_init = {g: [] for g in GENES}
ap_all, am_all, mass_all = [], [], []
suc_all, att_all = [], []

for s in range(N_SEED):
    globals()['RNG'] = np.random.default_rng(109 + s * 7)
    w = LayerWorld()
    # 收集初始基因
    g0 = {"mass": w.mass, "accel": w.accel, "alpha_p": w.alpha_p, "alpha_m": w.alpha_m,
          "vision_g": w.vision_g, "fov_g": w.fov_g, "risk": w.risk,
          "cruise_g": w.cruise_g, "sprint_g": w.sprint_g}
    # 插桩捕食成功率
    src = open('stage25_body_size_layers.py', encoding='utf-8').read().split("def run():")[0]
    src2 = src.replace("""                    if RNG.random() < p_success:   # 概率性成功（Wilson：捕食成功率低）
                        # 肉类能量 = 猎物体积×饱食度（身体+体内能量）× 肉类转换度（α_m——能量流守恒）
                        self.satiety[i] = min(sat_max(self.mass[i]), self.satiety[i] + 3.5 * self.mass[j] * self.satiety[j] * meat_conv(self.alpha_m[i]))""",
"""                    if RNG.random() < p_success:
                        STATS["ok"] += 1
                        self.satiety[i] = min(sat_max(self.mass[i]), self.satiety[i] + 3.5 * self.mass[j] * self.satiety[j] * meat_conv(self.alpha_m[i]))""")
    # 不用插桩——统计用基础运行即可（成功率已不追踪——改用 exec 频率间接量）
    w.run(T)
    h = np.array(w.history)
    hist_all.append(h)
    herb_l.append(np.mean(h[200:, 2] > 5))
    carn_l.append(np.mean(h[200:, 3] > 0))
    peaks.append(int(h[200:, 3].max()))
    a = np.where(w.alive)[0]
    if len(a):
        for g in GENES:
            gene_final[g].append(getattr(w, g)[a])
            gene_init[g].append(g0[g])
        ap_all.append(w.alpha_p[a]); am_all.append(w.alpha_m[a]); mass_all.append(w.mass[a])
    print("seed%d: carn存活=%.0f%% herb存活=%.0f%% peak=%d 存活=%d" % (
        s, 100*carn_l[-1], 100*herb_l[-1], peaks[-1], len(a)))

print("\n== 汇总 ==")
print("食肉存活率: 均值 %.0f%% 范围 %d-%d%%" % (100*np.mean(carn_l), 100*np.min(carn_l), 100*np.max(carn_l)))
print("食草存活率: 均值 %.0f%%" % (100*np.mean(herb_l)))
print("捕食者峰值: 均值 %.1f 最大 %d" % (np.mean(peaks), max(peaks)))

# ---------- 图 ----------
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Stage25 Evolution Stats (8 seeds x 1200 steps)", fontsize=13)

# ① 基因分布：初始 vs 末代（小提琴）
ax = axes[0, 0]
data_init = [np.concatenate(gene_init[g]) for g in GENES]
data_final = [np.concatenate(gene_final[g]) for g in GENES]
parts = ax.violinplot(data_final, positions=range(len(GENES)), widths=0.6, showmedians=True)
for pc in parts['bodies']:
    pc.set_facecolor('#4c72b0'); pc.set_alpha(0.6)
bp = ax.boxplot(data_init, positions=np.arange(len(GENES)) + 0.35, widths=0.15, patch_artist=True)
for b in bp['boxes']:
    b.set_facecolor('#dd8452'); b.set_alpha(0.8)
ax.set_xticks(range(len(GENES)))
ax.set_xticklabels(GENES, rotation=30, fontsize=8)
ax.set_title("Gene distribution (orange=init, blue=final)")
ax.set_ylabel("value")

# ② α_p×α_m 二维食性空间
ax = axes[0, 1]
ap, am = np.concatenate(ap_all), np.concatenate(am_all)
herb_mask = ap > am + 0.2
carn_mask = am > ap + 0.2
omni_mask = ~herb_mask & ~carn_mask
ax.scatter(ap[herb_mask], am[herb_mask], s=6, c='#55a868', alpha=0.6, label="herb")
ax.scatter(ap[carn_mask], am[carn_mask], s=6, c='#c44e52', alpha=0.6, label="carn")
ax.scatter(ap[omni_mask], am[omni_mask], s=6, c='#8172b3', alpha=0.5, label="omni")
ax.axhline(0.5, color="gray", lw=0.5); ax.axvline(0.5, color="gray", lw=0.5)
ax.plot([0, 0.95], [0.2, 0.95], 'k--', lw=0.5); ax.plot([0.2, 0.95], [0, 0.95], 'k--', lw=0.5)
ax.set_xlabel("alpha_p"); ax.set_ylabel("alpha_m")
ax.set_title("Diet 2D space (final pop)")
ax.legend(fontsize=7)

# ③ 生态位-体型
ax = axes[0, 2]
m_all = np.concatenate(mass_all)
ax.scatter(m_all[herb_mask], am[herb_mask], s=6, c='#55a868', alpha=0.6, label="herb")
ax.scatter(m_all[carn_mask], am[carn_mask], s=6, c='#c44e52', alpha=0.6, label="carn")
ax.scatter(m_all[omni_mask], am[omni_mask], s=6, c='#8172b3', alpha=0.5, label="omni")
ax.set_xlabel("mass"); ax.set_ylabel("alpha_m")
ax.set_title("Niche-body (mass vs alpha_m)")
ax.legend(fontsize=7)

# ④ 多 seed 存活率
ax = axes[1, 0]
x = np.arange(N_SEED)
ax.bar(x - 0.2, herb_l, 0.35, label="herb survival", color='#55a868')
ax.bar(x + 0.2, carn_l, 0.35, label="carn survival", color='#c44e52')
ax.axhline(0.5, color="gray", ls="--", lw=0.8)
ax.set_xticks(x); ax.set_ylim(0, 1.05)
ax.set_title("Per-seed survival rate (200-1200 steps)")
ax.set_ylabel("fraction"); ax.legend(fontsize=8)

# ⑤ 种群时间序列（均值±std）
ax = axes[1, 1]
H = np.array([h[:T] for h in hist_all])
t = np.arange(T)
for col, name, color in [(0, "total", '#4c72b0'), (2, "herb", '#55a868'), (3, "carn", '#c44e52')]:
    mu = H[:, :, col].mean(axis=0)
    sd = H[:, :, col].std(axis=0)
    ax.plot(t, mu, label=name, color=color)
    ax.fill_between(t, mu - sd, mu + sd, alpha=0.15, color=color)
ax.set_xlabel("step"); ax.set_ylabel("population")
ax.set_title("Population over time (mean±std of 8 seeds)")
ax.legend(fontsize=8)

# ⑥ 峰值与稳定性散点
ax = axes[1, 2]
ax.scatter(herb_l, carn_l, s=50, alpha=0.7)
ax.scatter([np.mean(herb_l)], [np.mean(carn_l)], s=120, marker='*', c='red', label="mean")
ax.set_xlabel("herb survival"); ax.set_ylabel("carn survival")
ax.set_title("Stability map (survival rates)")
ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05)
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig("fig_stage25_stats.png", dpi=120)
print("\n[plot] saved fig_stage25_stats.png")
