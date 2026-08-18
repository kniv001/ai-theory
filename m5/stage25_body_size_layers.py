# -*- coding: utf-8 -*-
"""
M5 阶段 25：体型分层 v2——用户速度修正（"小体型速度是优势——大体型难以捕食"）
stage24 捕食冻结的解法：捕食 = 快吃慢（狮子-牛模式——小快捕食者吃大慢猎物）

捕食条件（用户修正）：
  捕食者速度 > 猎物速度（追得上——小快捕食大慢）
  猎物体积 > 捕食者体积 × 0.5（吞得下——不能太小——回报 = 猎物体积×10）
猎物逃跑：检测到威胁（附近捕食者朝自己来）→ 朝反方向跑（速度 = 自己速度）
体型-生态位：大体积 = 食草（慢——植物充足——防御=体型）小体积 = 食肉（快——追大猎物）

实验 1：体型-食性关联（大体积食草/小体积食肉——生态位涌现）
实验 2：捕食链（小快捕食者吃大慢食草者——捕食不再冻结）
实验 3：α 与体积共演化（双生态位）
实验 4：对比 stage24（捕食冻结 vs 捕食链）
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(109)
W, H = 200.0, 200.0   # 100→200（地图 4 倍——pezzza 空间缓冲思路——生态有空间分层）
CONTACT = 2.0
PLANT_REWARD = 16.0   # 植物基础饱食（× 转换度 × intake——12→16 补偿 intake 均值的削减）
# 肉类能量 = 猎物饱食度（其体内储存的能量——能量流守恒：植物→猎物→捕食者——
# 饱而大的猎物 = 大能量包——饿空的猎物 = 没什么可吃——用户：捕食能量与猎物数值挂钩）
FOV = 120.0   # 扇形视野角（度）——生物只看前方——背后突袭/盲区
FOV_HALF = np.radians(FOV / 2.0)
MEM_CAP = 8      # 记忆容量（条/生物）
MEM_TTL = 60     # 记忆遗忘时间（步——超时未刷新 = 遗忘）
PLANT_MATURE = 10
PLANT_REPRO = 0.12   # 0.08→0.12（植物产量支撑种群——否则被吃光灭绝）
PLANT_SPREAD = 0.04
MAX_PLANTS = 300   # 500→300（植物稀缺化——采食竞争出现——速度有价值——形态分化压力）
REPRO_TH = 120.0
MAX_POP = 500
THREAT_R = 8.0   # 20→8（猎物警觉距离——捕食者隐蔽接近后突袭——Wilson 2013：捕食=短距爆发非长追）
VISION = 30.0   # 视野半径（感知范围——发现猎物/察觉威胁）

def tradeoff(mass):
    return np.clip(1.5 - 0.8 * mass, 0.1, 1.5)

def max_speed(mass, accel):
    """极速上限 = 体积 trade-off × 加速度 trade-off（用户设计：食肉加速快极速低——
    猫科 vs 羚羊——Wilson 2015：质量增强速度削弱——形态分化靠此涌现）"""
    return tradeoff(mass) * np.clip(1.35 - 0.22 * accel, 0.7, 1.35)

def acceleration(speed, accel_gene=1.0):
    """加速度 = 独立进化基因（用户设计：食肉加速快速度上限低——猫科 vs 羚羊）：
    捕食形态（α 低）被选择出高 a（冲刺）；猎物形态（α 高）被选择出高 v（极速）——
    形态分化进化涌现（a 和 v 独立变异——不通过体积绑定）"""
    return accel_gene

def decay_rate(mass):
    return 0.4 + 0.35 * mass

def plant_conv(alpha):
    """植物转换度（α 高 = 食草倾向 = 植物全转换——α 低保底 0.15——饿不死但效率极低）"""
    return 0.15 + 0.85 * alpha

def meat_conv(alpha):
    """肉类转换度（α 低 = 食肉倾向 = 肉类全转换——α 高保底 0.05——吃草者吃肉几乎无收益——
    0.15→0.05：更极端专业化——否则 α>0.7 者吃肉仍净正——全体追猎）"""
    return 0.05 + 0.85 * (1.0 - alpha)

def sat_max(mass):
    """storage（能量储备）：饱食度上限随质量——大动物耐饿（脂肪储备——现实：骆驼/熊）——
    120+60×mass：繁殖线 120 需要余量（100+50×mass 时 mass0.5 → 125——只剩 5 点——繁殖率趋零）"""
    return 120.0 + 60.0 * mass

def intake(mass):
    """intake（摄入能力）：单次进食量随质量——大动物一口顶小动物几口（现实：消化能力∝体型）——
    0.7-1.4 范围（mass 0.3-1.7）——0.7+0.4m（多 seed 验证：0.6+0.6m 破坏捕食链——食草暴涨）"""
    return 0.7 + 0.4 * mass

class LayerWorld:
    def __init__(self, n0=120, plants0=300):   # 密度对齐 pezzza（0.003/格²——相遇率——捕食频率）
        self.n = n0
        self.x = RNG.uniform(0, W, n0)
        self.y = RNG.uniform(0, H, n0)
        self.mass = np.clip(0.5 + RNG.normal(0, 0.25, n0), 0.3, 1.7)   # 质量（体型——速度/代谢/防御/能量包）
        # 加速度 = 进化基因（0.3-2.5——与极速 trade-off：加速快 = 极速低）
        self.accel = np.clip(1.0 + RNG.normal(0, 0.4, n0), 0.3, 2.5)
        self.speed = max_speed(self.mass, self.accel)
        self.heading = RNG.uniform(0, 2*np.pi, n0)   # 朝向（扇形视野方向）
        self.memory = [{} for _ in range(n0)]   # 空间记忆：{(qx,qy): [kind, age]}——视野的缓存
        self.target_lock = np.full(n0, -1, dtype=int)   # 追猎锁定（猎物索引——持续追一只）
        self.lock_age = np.zeros(n0)   # 锁定时长（超时解锁——追丢）
        self.vx = np.zeros(n0)   # 速度（运动学积分——不是瞬时）
        self.vy = np.zeros(n0)
        self.satiety = np.full(n0, 60.0)
        self.alive = np.ones(n0, dtype=bool)
        # α 初始 0.6±0.2（食肉种子少量——5-10%——否则初始食肉 22 只吃光猎物——
        # 专业化应进化涌现而非预置大军——生态学捕食者/猎物 1:10+）
        self.alpha = np.clip(0.6 + RNG.normal(0, 0.2, n0), 0.05, 0.95)
        self.px = RNG.uniform(0, W, plants0)
        self.py = RNG.uniform(0, H, plants0)
        self.page = np.zeros(plants0)
        self.history = []

    def move_with_dynamics(self, i, dx, dy, chase=False):
        """移动（瞬时——效率——速度上限 vmax）：
        追逐模式（捕食者追猎物）：追速 = 极速 + 加速度加成——
        Wilson 2013：短程追逐速度由加速度决胜（猎豹爆发）——没有它捕食者与猎物
        同速——猎物 flee 后永远追不回——冲刺永远无法发动"""
        vmax = self.speed[i] + (1.0 * self.accel[i] if chase else 0.0)   # 追猎 = 持续爆发（短程）
        self.x[i] = np.clip(self.x[i] + dx * vmax, 0, W)
        self.y[i] = np.clip(self.y[i] + dy * vmax, 0, H)

    def local_density(self, idx):
        if len(self.px) <= 1:
            return 0.0
        ds = np.hypot(self.px - self.px[idx], self.py - self.py[idx])
        return np.sum((ds < 15.0) & (ds > 1e-6))

    def plant_grow(self):
        # 外生补充（pezzza 核心机制：随机长新植物——防灭绝保底）
        if len(self.px) < MAX_PLANTS and RNG.random() < 0.15:
            self.px = np.append(self.px, RNG.uniform(0, W))
            self.py = np.append(self.py, RNG.uniform(0, H))
            self.page = np.append(self.page, 0)
        if len(self.px) == 0:
            return
        self.page += 1
        mature = np.where(self.page > PLANT_MATURE)[0]
        new_x, new_y, new_age = [], [], []
        for idx in mature:
            dens = self.local_density(idx)
            if RNG.random() < PLANT_REPRO / (1 + 0.2 * dens):
                new_x.append(np.clip(self.px[idx] + RNG.uniform(-10, 10), 0, W))
                new_y.append(np.clip(self.py[idx] + RNG.uniform(-10, 10), 0, H))
                new_age.append(0)
            if RNG.random() < PLANT_SPREAD / (1 + 0.1 * dens):
                ang = RNG.uniform(0, 2*np.pi)
                dist = RNG.uniform(10, 50)
                new_x.append(np.clip(self.px[idx] + dist*np.cos(ang), 0, W))
                new_y.append(np.clip(self.py[idx] + dist*np.sin(ang), 0, H))
                new_age.append(0)
        if new_x:
            room = MAX_PLANTS - len(self.px)
            if room > 0:
                new_x, new_y, new_age = new_x[:room], new_y[:room], new_age[:room]
                self.px = np.append(self.px, new_x)
                self.py = np.append(self.py, new_y)
                self.page = np.append(self.page, new_age)

    def step(self):
        alive = np.where(self.alive)[0]
        if len(alive) == 0:
            self.plant_grow()
            self.history.append((0, len(self.px), 0, 0, 0, 0))
            return
        urge_full = np.zeros(len(self.satiety))
        urge_full[alive] = 1.0 + (1.0 - np.clip(self.satiety[alive] / 100, 0, 1)) * 2.0
        # 吃饱的生物不觅食（satiety ≥ 储备上限——休息——采食压力饱和）
        hungry = alive[self.satiety[alive] < sat_max(self.mass[alive])]
        # 记忆衰减（age+1——超过 TTL 遗忘——缓存驱逐）
        for i in alive:
            mem = self.memory[i]
            for k in list(mem):
                mem[k][1] += 1
                if mem[k][1] > MEM_TTL:
                    del mem[k]
        # 追猎锁定期限（超时 = 追丢——解锁）+ 目标死亡释放
        self.lock_age[alive] += 1
        for i in alive:
            if self.target_lock[i] != -1 and \
                    (self.lock_age[i] > 40 or not self.alive[self.target_lock[i]]):
                self.target_lock[i] = -1
                self.lock_age[i] = 0
        targets = []
        # 扇形视野：只看到前方视野锥内的目标（发现——捕食第一环节）——看到即写入记忆
        for fx, fy in zip(self.px, self.py):
            for i in hungry:
                dx, dy = fx - self.x[i], fy - self.y[i]
                d = np.hypot(dx, dy)
                if d < 1e-6:
                    d = 1e-6
                ang = np.arctan2(dy, dx) - self.heading[i]   # 目标在朝向的哪个方向
                ang = (ang + np.pi) % (2*np.pi) - np.pi   # 归一化 [-π, π]
                if d < VISION and abs(ang) < FOV_HALF:   # 前方锥内才可见
                    # 摄入量随质量（大动物一口多——intake 维度）
                    targets.append((i, fx, fy, PLANT_REWARD / d * urge_full[i] * plant_conv(self.alpha[i]) * intake(self.mass[i]), "food"))
                    # 写入记忆（写回缓存——看到的位置记下来——转头后仍可寻）
                    self.memory[i][(int(fx), int(fy))] = ["food", 0]
        # 捕食目标（用户设计：冲刺预判——冲刺能覆盖距离才生成（避免失败冲刺亏空）+
        # 价值含净收益（回报 - 冲刺代价））
        for i in hungry:
            for j in alive:
                if i == j:
                    continue
                # 捕食资格：accel 差 1.1 倍（资格宽松——专业化靠目标价值的风险调整
                # （×p_success——Δa 小者成功率低——追猎价值低——不追）自然涌现——
                # 1.3 门槛太严：α 低与 accel 高两基因耦合——食草种群突变的捕食者
                # accel 不够格——饿死——生态位永不恢复）
                if self.accel[i] > self.accel[j] * 1.1 \
                        and self.mass[j] > self.mass[i] * 0.5 \
                        and self.mass[j] < self.mass[i] * 1.6:
                    d = np.hypot(self.x[j] - self.x[i], self.y[j] - self.y[i])
                    if d < 1e-6:
                        d = 1e-6
                    ang = np.arctan2(self.y[j] - self.y[i], self.x[j] - self.x[i]) - self.heading[i]
                    ang = (ang + np.pi) % (2*np.pi) - np.pi
                    # 冲刺距离（Wilson 2013：加速度决胜——爆发覆盖距离 ∝ 加速度差——
                    # 最小 3 格（≥移动步长——否则离散步进跳过冲刺窗口——永远无法发动））
                    d_charge = max((self.accel[i] - self.accel[j]) * 4.0, 3.0)
                    # 冲刺成功率（Wilson 2013：加速决胜——捕食成功率低 30-70%——
                    # 体积防御：猎物比捕食者大 → 挣脱力强 → 成功率降（Wilson 2018：质量=防御））
                    p_success = np.clip(0.4 + 0.4 * min(1.0, (self.accel[i] - self.accel[j]) / 2.0)
                                        - 0.5 * max(0.0, self.mass[j] - self.mass[i]), 0.05, 0.85)
                    # 扇形视野内即可追——不要求速度优势（Wilson：捕食 = 突袭——接近靠隐蔽
                    # 不靠速度——追逐速度优势是猎物"逃脱"的武器——追不上的自动追丢
                    # （d2 超出冲刺范围）——approach>0 曾误滤（accel 补偿在追逐中不存在））
                    if d < VISION and abs(ang) < FOV_HALF:
                        cost = 3.0 + (self.accel[i] - self.accel[j])   # 冲刺代价（研究：失败亏空但非致命）
                        # 肉类能量 = 猎物体积×饱食度（身体+体内能量——能量包）× 转换度——预期净收益
                        # 系数 3.0 + 风险调整（价值 ×p_success 挡业余者）：专业者回报 48/次
                        # （1.6/步 > 消耗 0.8——可繁殖）——1.5 时期望只略高于采食——方差杀小种群
                        net = p_success * 3.0 * self.mass[j] * self.satiety[j] * meat_conv(self.alpha[i]) \
                              - cost - 0.6 * d / max(self.speed[i] + self.accel[i], 0.5)
                        if net > 0:   # 预期净收益 > 0 才捕（多次尝试的期望——捕食 = 风险投资）
                            # 追猎锁定：持续追同一只（价值 ×3——否则每帧换目标——距离永不收敛）
                            locked = self.target_lock[i] == j
                            # 风险调整：价值 ×p_success（成功概率）——Δa 小者（杂食/业余）成功率
                            # 低——追猎价值低于采食——不追；Δa 大者（专业形态）p 高——价值高——追——
                            # 专业化由形态自然分化（无人工门槛——决策环的风险厌恶）
                            targets.append((i, self.x[j], self.y[j],
                                            net / d * urge_full[i] * (3.0 if locked else 1.0) * p_success,
                                            "prey", j, p_success))
                            # 捕食者记忆：看到猎物位置（猎物会动——仅弱记忆——价值低）
                            self.memory[i][(int(self.x[j]), int(self.y[j]))] = ["prey", 0]
        # 记忆目标（决策读取缓存：记得的植物/猎点——新鲜度打折——
        # 记忆意义：转头后仍知道去哪找——视野目标已全价值——记忆目标只补盲区）
        for i in hungry:
            for (qx, qy), (kind, age) in list(self.memory[i].items()):
                dx, dy = qx - self.x[i], qy - self.y[i]
                d = np.hypot(dx, dy)
                if d < 1e-6:
                    d = 1e-6
                fresh = 1.0 - age / MEM_TTL   # 新鲜度 0-1（越新越可信）
                if kind == "food":
                    targets.append((i, qx, qy, PLANT_REWARD / d * urge_full[i] * self.alpha[i] * intake(self.mass[i]) * (0.3 + 0.7 * fresh),
                                    "memfood"))
                else:   # prey 记忆（弱——猎物已移动——只当方向提示）
                    targets.append((i, qx, qy, 0.2 * PLANT_REWARD / d * urge_full[i] * (0.3 + 0.7 * fresh),
                                    "memprey"))
        # 威胁检测（扇形警觉：捕食者进入冲刺覆盖内且猎物朝它看才察觉——
        # 背对捕食者 = 盲区（看不到威胁）——突袭窗口：捕食者从背后接近不被发现——
        # 扇形视野让"伏击"成为捕食通道（真实捕食——背后突袭））
        # 追猎中的捕食者不逃（专注猎物——威胁让位——否则捕食者互相 flee——永无法完成冲刺）
        hunters = {t[0] for t in targets if t[4] == "prey"}
        flee = {}
        for i in alive:
            if i in hunters:
                continue
            threats = []
            for j in alive:
                if i == j or self.alive[j] == False:
                    continue
                if self.accel[j] > self.accel[i] * 1.1 and self.mass[i] > self.mass[j] * 0.5:
                    d = np.hypot(self.x[j] - self.x[i], self.y[j] - self.y[i])
                    ang = np.arctan2(self.y[j] - self.y[i], self.x[j] - self.x[i]) - self.heading[i]
                    ang = (ang + np.pi) % (2*np.pi) - np.pi
                    d_charge_j = max((self.accel[j] - self.accel[i]) * 4.0, 3.0)
                    if d < d_charge_j and abs(ang) < FOV_HALF:   # 冲刺覆盖内 + 视野内 = 危险
                        threats.append((j, d))
            if threats:
                # 最近威胁 → 朝反方向跑
                j, d = min(threats, key=lambda t: t[1])
                dx = self.x[i] - self.x[j]
                dy = self.y[i] - self.y[j]
                nd = np.hypot(dx, dy)
                if nd > 1e-6:
                    flee[i] = (dx/nd, dy/nd)
        best = {}
        for t in targets:
            if t[0] not in best or t[3] > best[t[0]][3]:
                best[t[0]] = t
        moved = set()
        for i, (i0, tx, ty, val, kind, *rest) in best.items():
            # 逃跑优先（威胁存在 → 逃跑而非觅食）
            if i in flee:
                fx, fy = flee[i]
                self.move_with_dynamics(i, fx, fy)
                self.heading[i] = np.arctan2(fy, fx)   # 朝向 = 移动方向
                moved.add(i)
                continue
            dx, dy = tx - self.x[i], ty - self.y[i]
            d = np.hypot(dx, dy)
            if d > 1e-6:
                self.move_with_dynamics(i, dx/d, dy/d, chase=(kind == "prey"))
                if kind == "prey":
                    self.satiety[i] -= 0.6   # 追逐燃能（Wilson 2018：追逐 = 肌肉燃烧——
                    # 失败追逐的机会成本——杂食者追猎转亏——只有专业食肉者承担得起）
                self.heading[i] = np.arctan2(dy, dx)   # 朝向 = 移动方向
                moved.add(i)
            if kind in ("food", "memfood") and d < CONTACT and len(self.px) > 0:
                # 植物转换度 = α（食草者全转换——食肉者吃草几乎无饱食）
                ds = np.hypot(self.px - self.x[i], self.py - self.y[i])
                idx = int(np.argmin(ds))
                if ds[idx] < CONTACT:
                    # 摄入随质量（intake）+ 上限随质量（storage——耐饿）
                    self.satiety[i] = min(sat_max(self.mass[i]), self.satiety[i] + PLANT_REWARD * plant_conv(self.alpha[i]) * intake(self.mass[i]))
                    if kind == "memfood":
                        self.memory[i][(int(tx), int(ty))] = ["food", 0]   # 记忆命中——刷新（写回）
                    self.px = np.delete(self.px, idx)
                    self.py = np.delete(self.py, idx)
                    self.page = np.delete(self.page, idx)
                elif kind == "memfood":
                    del self.memory[i][(int(tx), int(ty))]   # 记忆落空（植物已被吃）——遗忘
            elif kind == "memprey" and d < CONTACT:
                del self.memory[i][(int(tx), int(ty))]   # 猎点记忆到达——遗忘（猎物已移动）
            elif kind == "prey":
                j = rest[0]
                p_success = rest[1] if len(rest) > 1 else 0.5
                charge_cost = 3.0 + (self.accel[i] - self.accel[j])
                # 冲刺判定用本帧开始距离（猎物反应延迟：捕食者进入冲刺范围瞬间发动——
                # 猎物下帧才逃——逃脱概率已在 p_success——移动后距离判定 = 竞态
                # （猎物 flee 同帧抢先 → 永远追不到）——追丢留给失败后的持续追逐）
                if self.alive[j] and d < d_charge:   # 本帧开始时已在冲刺范围内才掷骰子
                    self.satiety[i] -= charge_cost   # 冲刺消耗（无论成败）
                    if RNG.random() < p_success:   # 概率性成功（Wilson：捕食成功率低）
                        # 肉类能量 = 猎物体积×饱食度（身体+体内能量）× 转换度（能量流守恒——
                        # 大而饱的猎物 = 大餐——小饿猎物 = 零食）——上限随质量（storage）
                        self.satiety[i] = min(sat_max(self.mass[i]), self.satiety[i] + 3.0 * self.mass[j] * self.satiety[j] * meat_conv(self.alpha[i]))
                        self.alive[j] = False
                        self.target_lock[i] = -1   # 捕食成功——解锁
                        self.lock_age[i] = 0
                    else:
                        self.satiety[j] -= 1.0   # 猎物逃跑代价（小——主要是时间成本）
                        self.target_lock[i] = -1   # 冲刺失败——猎物逃脱——解锁重选
                        self.lock_age[i] = 0
                else:
                    self.target_lock[i] = j   # 追不上——保持锁定同一只（持续追——距离收敛）
                    self.lock_age[i] = 0
            # 捕食丢失：猎物逃出捕食者视野（d > VISION——下步重新决策时自动丢失——
            # 此处追踪：猎物逃跑速度 > 捕食者 → 距离拉大 → 超出视野 → 追丢）
        # 无目标（也不逃）的生物：扫视转头（视野动态覆盖——否则扇形视野 = 呆站饿死）
        for i in alive:
            if i not in moved:
                self.heading[i] += 0.785   # 45°/步（8 步扫一圈）
        # 消耗 = 体积 + 加速度成本（a 高 = 冲刺肌肉 = 高代谢——Wilson 2018：捕食者维持成本高）
        self.satiety[alive] -= decay_rate(self.mass[alive]) + self.accel[alive] * 0.15
        self.alive[np.where(self.satiety <= 0)[0]] = False
        alive = np.where(self.alive)[0]
        new = []
        for i in alive:
            if self.satiety[i] > REPRO_TH and len(alive) + len(new) < MAX_POP:
                v = np.clip(self.mass[i] * RNG.uniform(0.9, 1.1), 0.3, 1.7)
                # 偶发大突变（3% ±0.3 跳跃——宏突变/基因重组——低 α 种子重注入——
                # 否则捕食者灭绝后 α 漂不回——生态位不可恢复——真实进化靠突变保持多样性）
                if RNG.random() < 0.03:
                    a = np.clip(self.alpha[i] + RNG.uniform(-0.3, 0.3), 0.05, 0.95)
                else:
                    a = np.clip(self.alpha[i] * RNG.uniform(0.9, 1.1), 0.05, 0.95)
                ac = np.clip(self.accel[i] * RNG.uniform(0.9, 1.1), 0.3, 2.5)   # 加速度基因变异
                new.append((self.x[i] + RNG.uniform(-3, 3), self.y[i] + RNG.uniform(-3, 3),
                            v, max_speed(v, ac), 40.0, a, ac, RNG.uniform(0, 2*np.pi)))
                self.satiety[i] -= 60.0
        for nx, ny, nv, ns, nsat, na, nac, nh in new:
            self.x = np.append(self.x, np.clip(nx, 0, W))
            self.y = np.append(self.y, np.clip(ny, 0, H))
            self.mass = np.append(self.mass, nv)
            self.speed = np.append(self.speed, ns)
            self.accel = np.append(self.accel, nac)
            self.heading = np.append(self.heading, nh)
            self.memory.append({})   # 新生物空记忆
            self.target_lock = np.append(self.target_lock, -1)
            self.lock_age = np.append(self.lock_age, 0)
            self.vx = np.append(self.vx, 0.0)
            self.vy = np.append(self.vy, 0.0)
            self.satiety = np.append(self.satiety, nsat)
            self.alpha = np.append(self.alpha, na)
            self.alive = np.append(self.alive, True)
        self.plant_grow()
        a = np.where(self.alive)[0]
        herb = np.sum(self.alpha[a] > 0.7) if len(a) else 0
        carn = np.sum(self.alpha[a] < 0.3) if len(a) else 0
        vol_herb = np.mean(self.mass[a][self.alpha[a] > 0.7]) if herb else 0
        vol_carn = np.mean(self.mass[a][self.alpha[a] < 0.3]) if carn else 0
        self.history.append((np.sum(self.alive), len(self.px), herb, carn, vol_herb, vol_carn))

    def run(self, T):
        for _ in range(T):
            self.step()
        return np.array(self.history)

def run():
    print("=== 体型分层 v2（用户速度修正：小快难捕食——快吃慢） ===\n")

    # ---- 实验 1/2/3：生态位涌现 ----
    print("[exp1-3] 生态位 + 捕食链（600 步——体型多样性起点）:")
    w = LayerWorld()
    h = w.run(600)
    herb, carn = h[-1, 2], h[-1, 3]
    vh, vc = h[-1, 4], h[-1, 5]
    print(f"  末代: 食草 {herb}（体积 {vh:.2f}）| 食肉 {carn}（体积 {vc:.2f}）")
    # 生态金字塔：食肉者天然稀少（能量损耗——1:10 是常态）——共存即涌现
    print(f"  = {'✓ 双生态位涌现（食草+食肉共存——捕食链自持）' if herb > 5 and carn >= 2 and herb > carn * 2 else '✗ 单生态位'}")

    # ---- 实验 2：捕食链 ----
    print("\n[exp2] 捕食链（小快捕食者吃大慢猎物）:")
    if vh > 0 and vc > 0:
        print(f"  食草体积 {vh:.2f} > 食肉体积 {vc:.2f} = "
              f"{'✓ 体型分层（大食草被小快食肉捕食——狮子-牛模式）' if vh > vc + 0.1 else '分层弱（防御压力不足——需增大）'}")
    else:
        print(f"  生态位未涌现（需检查参数）")

    # ---- 实验 3：α-体积共演化 ----
    a = np.where(w.alive)[0]
    corr = np.corrcoef(w.alpha[a], w.mass[a])[0, 1] if len(a) > 2 else 0
    print(f"\n[exp3] α-体积相关: {corr:+.2f}"
          f"（{'✓ 负相关——小体积食肉/大体积食草——生态位-体型绑定' if corr < -0.3 else '关联弱'}）")

    # ---- 实验 4：对比 stage24（捕食冻结） ----
    print("\n[exp4] 对比 stage24（捕食冻结 vs 捕食链）:")
    print(f"  stage24: 食肉 1（冻结）| 本阶段: 食肉 {carn}"
          f"——{'✓ 捕食冻结解除（捕食链涌现）' if carn >= 2 else '✗'}")

    # ---- 图 ----
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].hist(w.alpha[a], bins=15, alpha=0.7, label="alpha")
    axes[0].set_title("Exp1: alpha distribution (both niches)")
    axes[0].set_xlabel("alpha")
    axes[1].scatter(w.mass[a], w.alpha[a], s=15)
    axes[1].set_title("Exp3: mass-alpha (niche-body binding)")
    axes[1].set_xlabel("mass"); axes[1].set_ylabel("alpha")
    axes[2].plot(h[:, 0], label="total")
    axes[2].plot(h[:, 2], label="herb")
    axes[2].plot(h[:, 3], "--", label="carn")
    axes[2].set_title("Exp: population over time")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("fig_stage25.png", dpi=110)
    print("\n[plot] saved fig_stage25.png")
    print("[done] stage25 body size layers complete")

if __name__ == "__main__":
    run()
