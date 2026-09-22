# PRE_REGISTRATION — V0.6 阶段B/C 协议冻结

> 日期：2026-09-22 · 本文件在任何新数据采集/训练前冻结。
> 冻结后只允许追加"偏差记录"，不允许静默修改；旧 v0/v0_5 全部数据视为 development 材料。

## 1. 科学问题（预登记）

Q-A：在相同当前状态与相同未来命令下，来自**同条件不同命令** support 的 context 是否改善
query 预测（相对 zero/shuffle context 与等输入 Direct）？
Q-B：context 是否跨 support 命令族与 query 状态迁移（held-out 命令族、held-out anchor）？
Q-C：learned context 与 privileged friction 输入的差距有多大（M3 仅为诊断，非上界保证）？

## 2. 工况（写入值命名，readback 在 smoke 验证）

| 名称 | friction 写入 | actuator_scale | 用途 |
|---|---:|---:|---|
| normal | 1.0 | 1.0 | train/eval |
| friction_mid | 0.6 | 1.0 | train/eval |
| friction_low | 0.3 | 1.0 | train/eval |

0.15（旧 OOD）不进入本轮训练与调参。外力/actuator 条件本轮不用。

## 3. 命令族（固定段表，秒；幅值为绝对值 m/s、rad/s）

settle 段（所有 query 共有）：1.0s [0,0,0]（站立稳定，提供 query origin 的 history）。

| family | 段表 (duration_s, [vx,vy,wz]) |
|---|---|
| Q1_straight | (0.5,[0,0,0]) (0.5,[0.4,0,0]) (1.0,[0.8,0,0]) (0.5,[0.4,0,0]) (0.5,[0,0,0]) |
| Q2_lateral | (0.5,[0,0,0]) (1.0,[0,0.5,0]) (1.0,[0,-0.5,0]) (0.5,[0,0,0]) |
| Q3_turn | (0.5,[0,0,0]) (1.0,[0,0,0.8]) (1.0,[0,0,-0.8]) (0.5,[0,0,0]) |
| Q4_combo | (0.5,[0,0,0]) (1.0,[0.6,0,0.6]) (1.0,[-0.6,0,-0.6]) (0.5,[0,0,0]) |

Support（random dwell，与 query 族不同的实现）：8 段 × 1.0s，
每段 cmd = U(0.4,0.8)·limits·sign（sign 均匀，25% 概率单轴置零），由 support seed 决定。
**Q4_combo 为 held-out 命令族：不出现在任何训练窗口，只在 test。**

## 4. Anchor / seed / split

- query anchor seed：`AQ[i] = 1000 + 17*i, i=0..7`（8 个独立 anchor group）。
  repetition 1 = 原始段表；repetition 2 = 段边界加 U(±0.1s) jitter（jitter seed = AQ[i]+5000），
  用于分布统计；同 seed 重放只用于 reproducibility 核验，不计为独立样本。
- support seed：`SP[j] = 5000 + 31*j, j=0..15`（16 条/条件）。
- split（anchor group 级，整体切分，含其全部窗口与 pair）：
  train = AQ[0..4]，val = AQ[5]，**test = AQ[6..7]（训练与参数选择不可见）**。
- 每个 (condition, anchor, rep) 一次 env.reset()（torch.manual_seed(anchor_seed)），
  记录 reset 后完整状态与跨 condition 复原差（approximate-paired，不声称精确反事实）。

## 5. 预算（冻结）

- smoke：3 条件 × {Q1,Q3} × 2 anchor = 12 episodes（先跑，验证注入/记录/restore）。
- pilot：4 族 × 8 anchor × 3 条件 × 2 rep = **192 query** + 16 seed × 3 条件 = **48 support**，
  合计 240；smoke+pilot = 252 ≤ 300。
- 训练（阶段C）：4 配置 × 3 seed = 12 run；先实测 2 个 run 的 GPU·h 再决定是否全跑；
  单轮总预算 ≤120 RTX4090 GPU·h（本机卡型，不与 5090 折算）。

## 6. 指标（冻结主指标；旧 overall 混合单位 L2 仅作审计同口径）

- 主：per-axis MAE @ fixed leads {0.25s, 0.5s, 1s, 2s}（fixed-origin，3 轴分开）；
  per-axis squared error 差（可加口径）。
- context 迁移：squared_target_gain、parallel/orthogonal/overshoot 分解、
  swap-vs-native 直接距离。
- **等效容差（预登记）**：swap-vs-native 距离 ≤ 0.10 × target native error 才称"等效"；
  此前任何"统计不可分"措辞禁用。
- 统计：two-way bootstrap（source×target 或 support×query group），2000 次，seed=42；
  equal-anchor 估计并列；CI 跨 0 → INCONCLUSIVE，不允许改称 PASS/FAIL。
- 模型选择：val split（AQ[5]）决定 early stop / scaler / 任何校准；test 只评价。

## 7. 数据记录（B5）

- 50Hz 原始：全部 proprio + applied cmd + control event（段边界）+ termination + 外力(=0)。
- 20Hz 派生两版并存：**inputs 用因果一阶低通（fc=8Hz，只向前传播）后采样**；
  **labels/分析用非因果抗混叠（resample_poly 2/5）**。两版各自标注，不混用。
- 摩擦/执行器 readback：写入后 get_material_properties 读回记录进 meta；
  地面 material 与 combine rule 一并记录（smoke 验证项）。
- RGB/depth：本轮不阻塞主审计，仅保留接口；作为单独 smoke 候选，不计入本轮结果。

## 8. 负结果与禁令（重申）

不删负结果；不为过门调阈值；不以 test 挑 checkpoint；不手选 pair；
context 不必胜出——INCONCLUSIVE 是合法结论；不自动进入 Context Adaptation。
