# PRE_REGISTRATION_V07 — 结构化命令覆盖 × 动作交互建模

> 冻结日期：2026-09-24 · 本文件在**正式采集前**写定并 commit
> 依据：`tasks/EA_WM_V07与Risk_R01_并行实验执行包/01_CVPR_V07_ClaudeCode执行任务书.md`
> 本机事实依据：`V0_6_2_DECISION.md` / `V0_6_2_FULL_REPORT.md` §2–§6
> 配套冻结文件：`command_cells.json`、`split_plan.json`、`r0_vs_r1_marginals.json`

## 0. 研究问题（只回答两个）

- **Q-data**：总数据预算相同时，单轴数据 **R0** 与"单轴 + 部分二元组合"数据 **R1** 的区别有多大？
- **Q-model**：在**完全相同**的 R1 数据与预测任务上，动作因素交互模型 **I** 是否优于 **Direct D**？

2×2：{R0, R1} × {D, I}，每格 3 seed，共 **12 个主 run**。
误差降低定义：Δ_data = E(D,R0) − E(D,R1)；Δ_arch = E(D,R1) − E(I,R1)。两者分别报告。

## 1. 动作空间与命令单元（cells）

Stage 0 实测 controller ranges：`lin_vel_x/lin_vel_y/ang_vel_z = ±1.0`（三轴相同）。

```
A = 0.35, B = 0.65                      （相对 limit 的比例，全部三轴）
VALUES = [-0.65, -0.35, 0.0, +0.35, +0.65]   索引 0..4，索引 2 == 0
cell_id = c{ix}{iy}{iw}                例 c202 = (vx=+0.35, vy=0, wz=-0.65)
单位：vx, vy = m/s；wz = rad/s
```

### 1.1 cell 类型与分配（`command_cells.json`）

| 类型 | 数量 | 分配 |
|---|---:|---|
| zero | 1 | 全部 regime |
| single（单轴非零） | 12 | **全部进 R0 与 R1 的 train** |
| double（两轴非零） | 每类 16（XY / XW / YW 各 16） | 每类 **10 train / 3 val / 3 test** |
| triple（三轴非零） | 64 中 **14** 个满足"三个二元投影都在 R1 train" | **5 val / 9 test** |

- double 分配用穷举搜索取"每轴取值在 val/test 中最均匀"。**3 个 cell 对 4 个取值，
  最优不平衡度 6.0 无法为 0** —— 这是不可避免的，逐值频数见 `count_check`。
- triple 的 14/64 可用率来自 §4.2 的要求（三个二元投影都必须在 R1 train）；
  其余 50 个 triple 因其某个二元投影未在训练出现而**不纳入**本轮。

### 1.2 联合可执行性核验（§4.1 要求）

各轴独立合法 ≠ 三轴联合可执行。本轮用两条实测证据代替猜测：
1. Stage 0 `combo_xw` wave（vx=+0.65, wz=+0.65 同时）实测：tilt p95 = 4.84°、max = 5.69°，
   全部 tick 在 valid-planar 阈值内，未触发终止；
2. 正式采集时**逐 episode** 记录 `termination_reason` 与最大 tilt，任何 cell 若系统性
   触发终止或 tilt 超限，在 `metrics/termination_and_masks.csv` 中单列，不静默删除。
   主实验只使用数值上位于正常控制包络的 cells（本设计全部 ∈ [−0.65, 0.65] ⊂ ±1.0）。

## 2. 三种泛化层级（不混为一项）

| 层级 | 定义 | 来源 |
|---|---|---|
| **P0** seen-cell control | cell 在训练出现，新 anchor / 新时序模板 | singles + train doubles |
| **P1** held-out pair cells | 二元轴类型在训练出现，**具体符号/幅值元组从未出现**；每个单轴取值都在训练出现过 | val/test doubles |
| **P2** triple composition | 三轴同时**未训练**；其三个二元投影都在 R1 train 出现 | 14 个可用 triple 的 val/test 部分 |

P1 是主要结构泛化测试；P2 是更强的外推测试，但**测试侧只有 9 个不同的 triple cell**，
故 P2 结论标注 EXPLORATORY，不作为决策主依据。

## 3. 数据预算与 group 结构（`split_plan.json`）

```
24 个独立 anchor group：12 train / 6 val / 6 test
每 group：12 条脚本 × 3 个摩擦工况（nominal 1.0 / mid 0.6 / low 0.3）
正式采集 = 24 × 3 × 12 = 864 episodes（+ smoke ≤12 → 总上限 900）
reset_seed = 7000 + 37·g           （新 namespace，与 V0.6 的 AQ/SP 不重叠）
每条 episode：1.0 s settle + 12.6 s schedule = 13.6 s（50 Hz 原始，20 Hz 派生）
```

- **train group 脚本**：sc00–sc05 = R0 regime，sc06–sc11 = R1 regime（等脚本、等回合、等窗口）。
- **val/test group 脚本**：sc00–03 = P0，sc04–07 = P1，sc08–11 = P2（各 4 条）。
- 每个 reset/group 的所有工况、所有脚本、所有时间版本属于**同一 split**。

### 3.1 R0 vs R1 边际对照（§4.3 强制报告）

R1 数据集 = R0 脚本 + R1 脚本（R1 ⊇ R0）。为匹配每轴占空比 d：

```
d_R0 = s1 / (3·72)          d_R1 = (s1 + a2) / (3·144)
令 d_R0 = d_R1  ⟹  a2 = s1
30 个 train double 本身贡献 60 次轴激活 ⟹ a2 ≥ 60 ⟹ s1 = 60
取：R0 脚本 = 12 个 single 各 5 次 (60) + 12 zero
    R1 脚本 = 30 double + 0 single + 42 zero
```

实测（`r0_vs_r1_marginals.json`）：

| 量 | R0 数据集 | R1 数据集 |
|---|---:|---:|
| slots | 864 | 1728 |
| 每轴占空比 vx / vy / wz | 0.2778 / 0.2778 / 0.2778 | 0.2778 / 0.2778 / 0.2778 |
| **共激活 slot 占比** | **0.0000** | **0.2083** |
| zero slot 占比 | 0.1667 | 0.3750 |
| 段数与 dwell 多重集 | 完全相同 | 完全相同 |

**无法同时匹配的量**：zero 占比（0.167 vs 0.375）。这是"匹配每轴占空比"与"R0 共激活为 0"
两个约束的必然结果：R1 用 double 覆盖同等轴激活量时每个 slot 激活两轴，故必须用更多
zero slot 摊平。**如实报告，不声称"其它全部相同"。**

### 3.2 时序模板

6 个模板 T0–T5 使用**同一个 dwell 多重集**（12 段：0.6/0.6/0.9/0.9/1.2/1.2/1.5/1.5/0.6/0.9/1.2/1.5，
合计 12.6 s），只是顺序不同 → **任意 split 子集的驻留分布与切换次数完全相同**，
从构造上排除"训练短驻留、测试长驻留"。

train 用 T0–T2，val 用 T3，test 用 T4–T5（模板 ID 逐 episode 落盘）。

## 4. 时间、命令与状态口径（继承 V0.6.1 并复核）

- 时间身份 = **整数 control tick**（50 Hz，dt=0.02）；20 Hz 派生用有理数关系
  `hold_tick = (5k)//2`，禁用浮点秒等式。
- `u_requested`：写入 command term 的计划命令（**模型主输入**）
- `u_consumed`：policy 实际读到的命令（**诊断字段**；Stage 0 复核：精确滞后 1 tick，
  三个 wave 分别 199/199、199/199、224/224 tick 成立）
- `joint_command`：低层 policy 输出的关节动作（12 维）
- `execution`：实际物理运动（**不叫"修正后的 command"**）
- 输入所有观测源时间 ≤ prediction origin；未来 command 只来自 origin 时已生成的计划，
  **不从未来反馈控制日志偷取实际已执行命令**。

标签：`e_label = F(e_raw)`（抗混叠，独立于 command）→ `u_ref = S(u)`（事件表，整数 tick）
→ `r_label = e_label − u_ref`，全量断言 `u_ref + r_label == e_label`，评价真值直接读 `e_label`。
姿态：四元数先符号连续 + 归一化，相对位姿用 SLERP 或等价合法方法；不逐分量滤波。
特征：`FeatureSchema` 提取 vx/vy/wz（索引 [0,1,5]），禁止 `proprio[..., :3]`。
摩擦：读 static/dynamic 分别 + combine mode + 绑定 prim（Stage 0 实测 ground combine = **multiply**，
故有效值 = 写入值）；配置名 nominal/mid/low，不把写入值当"真实摩擦"。

**主输入采用 simulated deployable-candidate（去 GT 速度/角速度通道）**，沿用 V0.6.2 的去 GT 版本；
不宣称已完成真机验证。

## 5. 模型（最多两个主架构）

### 5.1 Direct D
复用 M0 的 history encoder（GRU）+ 完整未来命令输入，预测 [H,3] residual。

### 5.2 Interaction I（零锚定、每因素输出完整 [H,3]）

```
Ê = b(h) + Σ_i f_i(h,U_i) + Σ_{i<j} f_ij(h,U_i,U_j)
f_i(h,U_i)      = φ_i(h,U_i) − φ_i(h,0)
f_ij(h,U_i,U_j) = φ_ij(h,U_i,U_j) − φ_ij(h,U_i,0) − φ_ij(h,0,U_j) + φ_ij(h,0,0)
```

**0 表示整条未来该轴命令序列为零；h 不置零**（系统有惯性与历史作用，当前零命令不强制零响应）。
每个 φ 输出完整 `[H,3]`，**不是只输出自己的轴** —— 否则预先禁止了真实的 cross-axis 效应。

> **与 V0.6.2 的区别（§3.4 核实）**：V0.6.2 的 additive baseline 是**轴受限**形式
> （每个 g_a 只输出 a 轴 1 维），不能表达跨轴效应。V0.6.2 的负结果只否证了轴受限加性分解，
> **不**能推广为"所有加性/交互模型不可能有效"。本轮的 I 用完整输出形式。

实现：每因素 2 层 MLP（temporal block 用同一 MLP 一次性输出 H 步）。
不加显式 context、oracle 摩擦、概率头、语言模型；第一轮不加只对 I 生效的额外损失。

### 5.3 公平性
- 参数量目标 ±10%；做不到则报告参数量/FLOPs 并核算预算（见 `metrics/param_check.json`）。
- 同一数据、窗口、输入、horizon、标准化、优化器、早停、训练步数上限、seed。
- 共同验证指标（val FDE_xy）选 checkpoint，**不按测试成绩选**。

### 5.4 简单基线
Command-copy、正确轴 Persistence（schema [0,1,5]）、RidgeMultiOutput（各自 R0/R1 独立拟合）。
旧 M1 只作只读参考，不与新数据 I 比较后归因架构。

## 6. 预测目标、损失与轨迹

```
ê = u_ref + r̂ ,   L_E = Huber((ê − e)/s_E) ,  s_E 只从训练池拟合，全模型一致
可选：L = L_E + λ_xy L_xy + λ_ψ L_ψ ，λ 在 val 冻结，本轮最多一个共同配置
```

轨迹：平面任务用预测 vx/vy/wz 自回归积分，初值 = origin 已知姿态；**之后不读未来 GT yaw**。
SE(2) 相对位姿 `T_t^{-1} T_{t+k}`，世界姿态 GT 来自合法 raw quaternion。
同时审计 tilt 与 `∫ω_z dt` vs 真实净 yaw 的近似误差。平面假设不满足的窗口**保留并单列**。
用未来真实 yaw 积分预测速度的结果只标 `ORACLE_YAW_DIAGNOSTIC`，**不是主轨迹能力**。

积分单测先行：直线、纯旋转、恒定转弯圆弧、非零起始 yaw、零命令惯性、给 GT 速度的误差下界。

## 7. 评价（主次明确）

**主终点**：2 s FDE_xy（m）；2 s ADE_xy（m）；净 yaw error（圆周最短差，rad，独立报告）；
fixed-lead execution MAE（0.25/0.5/1/2 s × vx/vy/wz）。
**主比较**：I/R1 vs D/R1 在 **P1 held-out pair cells** 的 FDE_xy；P2 为次要强外推（EXPLORATORY）；
P0 seen-cell 结果用于检查代价。

统计单元 = 独立 anchor/session group；先 group 内平均，再跨 group 配对 bootstrap；
3 个模型 seed 分别报告，不伪装成额外物理环境。报告有效独立 group 数、均值、median、
seed 范围、group-level CI。绝不把窗口或多摩擦变体当独立 n。

**预先锁定的实用改善门**（研究投入决策阈值，非安全认证）：
主要 FDE 相对改善 ≥5% **且** group CI 方向为改善 **且** 至少 2/3 seed 同向；
seen-cell FDE 恶化 ≤5%；yaw 不得出现未报告的显著恶化。CI 跨零 → 写 INCONCLUSIVE。

## 8. 计算预算

主 run 上限 12；神经训练总额 ≤40 GPU-h；整个流程 ≤150 GPU-h。
先 D/R0 与 D/R1 各 seed42 实测 wall time / 峰值显存 / GPU-h，再放开其余。
预计超额则保存检查点并输出 BUDGET_REVIEW，不擅自扩 run。

## 9. 本轮不做

Risk 场景、极端摩擦扫描、路由器、真机动作、MDA、LLM agent、HumanoidVLN、V-JEPA 大训练、
策略重训。显式 c_t 不作为强制结构或必须胜出的主贡献。M0 为主干基线，M1 仅保留旧结果作参考。
脚本默认 `allow_real_robot: false`，检测到硬件控制接口时中止。

## 10. 变更控制

本文件 commit 后，采集期间**不改**：值集与 a/b、cell 分配、group 切分、时序模板与模板 split、
R0/R1 构造、窗口 manifest 规则、损失与评价口径、改善门。
任何偏离必须写入 `PRE_REGISTRATION_DEVIATIONS.md` 并说明原因与影响范围。
