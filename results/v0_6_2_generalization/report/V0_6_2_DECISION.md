# V0_6_2_DECISION — Command Generalization & Context Disentanglement Audit

> 2026-09-23 · 依据 `tasks/v0.6.2.md`
> 数据：V0.6.1 的 240 ep + 修复标签（**未采新数据**）；WindowManifest 与 V0.6.1 完全一致
> 预测：全部读 V0.6.1 的单次推理缓存（sha256 校验），本轮不重新推理已有模型
> 新训练：仅 Task 5 的 12 run（M0/M1 × {privileged, deployable} × 3 seeds），0.0028 GPU·h
> 目录：`results/v0_6_2_generalization/`；范围限定 CVPR 主线
> **Compositional Execution-Aware World Modeling**，未开展 Risk-Gated 方向

```text
ACTION_SPACE_OOD:          STRUCTURAL（train 0% / B/C 100% 窗口含多轴同时非零命令）
                           metric 距离同时超界（Mahalanobis 4.0 -> 57.6-59.0）
UNSEEN_FAILURE_EXPLAINED:  是——但**不是**"离 train 远的程度"问题，而是**该族整体**
                           落在未覆盖的输入组合上；族内 novelty 与误差几乎不相关
ACTION_COMPOSITION_SIGNAL: PARTIAL（additive primitive baseline 在 B/C 名义最优，
                           但配对 CI 跨 0、相对改善 <1% -> 未"明显改善"Q4）
CONTEXT_IS:                DYNAMICS/CONDITION-LEANING, but LOSSY & MIXED
                           （c_t 保留摩擦可解码性远多于保留 anchor 身份/当前速度）
M1_STABLE_GAIN:            NO（27 个 split×lead×axis 单元中 0 次胜出；B/C 上劣于 M0）
INST_TO_TRAJ_TRANSFER:     PARTIAL（A 上一致；B/C 上模型整体优于 ccopy 约 40%，
                           但模型内部排序与瞬时排序不一一对应）
DEPLOYABLE_GAP:            SMALL_BUT_SIGNIFICANT（混合单位 MAE 绝对差 <=0.001，
                           相对最坏 +0.6%（M1/C）；vx 上甚至为负）
RECOMMENDED_DIRECTION:     D（保留 direct EA-WM 主干，去掉"显式 context 是主要贡献"的定位）
```

## 六个问题的回答

**Q1 unseen-family failure 是否主要由 action-space OOD 解释？** —— **是，且是结构性的。**
train 的 119 个不同 future-command 向量中，多轴同时非零的数量为 **0**；B/C 的 36/22 个向量
**100%** 含多轴同时非零（vx∧wz 共激活占 86% 时间步）。这不是"离流形较远"，而是**该组合
从未出现**。metric 距离同样越界（Mahalanobis：train 8.6 / A 12.9 / B 57.6 / C 59.0；
5-PC 重建误差 3.95 / 3.99 / 8.76 / 8.79）。但**族内** novelty 与误差几乎不相关
（Spearman −0.42~+0.36 且符号随模型翻转）→ 失效是**族级**的，不是渐变式的。

**Q2 simple compositional baseline 是否改善 Q4？** —— **没有明显改善（PARTIAL）。**
additive primitive（每轴独立 MLP，无 interaction）在 B/C 上名义最优（0.1513 / 0.1592），
但相对最好的现有模型 M0 只改善 **0.75% / 0.37%**，且配对 episode-cluster bootstrap
CI 跨 0（[−0.0029,+0.0007] / [−0.0042,+0.0037]）。**结论：把 vx 与 wz 两个已见 primitive
相加，并不能解释也修不好 Q4**——瓶颈不在"缺少加法分解"这一层。

**Q3 c_t 更像 dynamics context 还是 state/history embedding？** —— **偏 dynamics/condition，
但它是有损混合瓶颈。**
- 摩擦工况可解码性：c_t（8 维）balanced acc 0.80/0.75/0.80/0.75（train/val/A/B），
  远高于 chance 0.333；其父 GRU 隐状态 0.986/0.924/0.858/0.900；M0 的 fast state
  0.946/0.924/0.854/0.800；raw 末帧仅 0.624/0.528/0.608/0.529。
- anchor（episode）身份可解码性（train 内 5 折 CV，chance 0.2）：c_t **0.32**，
  其父 GRU **0.69**，M0 fast state 0.61，raw 0.21。
- 当前速度回归 RMSE（B split，目标 std vx 0.22）：c_t 0.190、M1 state latent 0.168、
  M1 GRU 0.198、M0 fast state 0.144、raw 0.044。
→ 相对它的父表示，c_t **优先保留工况信息、丢弃了大部分 episode 身份与状态细节**；
但它并非纯工况编码。**关键限定：M0 的 fast state 同样能解码工况（0.85/0.80），
所以"工况信息只存在于显式 context"这一说法不成立。**

**Q4 M1 是否在任何公平 setting 下提供稳定、可重复增益？** —— **否。**
V0.6.1 修复后的 27 个 (split × lead × axis) 单元中 M1 胜出 **0** 次（**连 split A 的
9 个单元也一次没赢**：A 上逐轴最优分别落在 M0/M2/M3）。混合单位 aggregate 上 A 为
−0.0011 [−0.0016,−0.0006]（M1 微弱领先），但同一边上 M2/M3 与 ridge 都追平或反超；
B/C 上 M1 **显著劣于 M0**（+0.0043 / +0.0041，CI 不跨 0）。唯一稳定信号是
"M1 略优于 command-copy"，但那在 B/C 上 CI 跨 0。Task 5 训练出的无特权输入版本
同样没有改变这一排序（gap 与排序均不变）。

**Q5 instantaneous velocity improvement 是否转化为 trajectory improvement？** —— **部分转化。**
- A：转化一致（M1/M0 在瞬时与轨迹上同时最好：traj 0.0389/0.0399 vs ccopy 0.1014）。
- B/C：**模型整体优于 command-copy 约 40%**（traj 0.168–0.179 vs 0.281/0.288）——
  即便某些单轴瞬时误差上 ccopy 更好（如 vy@2s）。说明 ccopy 的瞬时优势是"看起来好"，
  在轨迹层面失效。
- 但模型**内部**排序不一一对应：C 上 M1 瞬时 vx 优于 M0（0.1627 vs 0.1748），
  净 yaw 却更差（0.1157 vs 0.0961）。
- 净 yaw 上没有任何模型稳定优于 ccopy（B: M2 0.1044 < ccopy 0.1098，但 A 上 ccopy 最好）。
- 平面近似已核验：tilt p95 3.8–4.4°（阈值 10°，**全部窗口**合法），
  |∫wz dt − Δyaw_quat| 均值 0.008–0.017 rad。

**Q6 移除 privileged state 后性能下降多少？** —— **几乎不下降。**
混合单位 MAE 的绝对差 ≤ 0.0010（相对最坏 M1/C +0.6%），vx 通道上甚至**负**（deployable 更好，
−2.4%~+1.6%）；最大单点效应是 M1 在 C 的 vy@2s：0.0605→0.0660（+9.0% 的相对变化，
绝对 +0.0054 m/s）。配对 bootstrap 显示差值为统计上非零但**实践上可忽略**。
→ 现有结论**不是** privileged 状态的产物；但也说明模型没有真正利用 GT 速度通道。

## 推荐方向（只推荐一个）

### **D. Drop explicit context as main contribution and keep direct EA-WM**

依据（三条独立证据交汇，均来自本轮）：
1. **M1 从未稳定胜出**（Q4：27 单元 0 胜；B/C 上显著劣于 M0；Task 5 复现同序）；
2. **context 不是必要的工况载体**（Q3：M0 的 fast state 同样解码工况 0.85/0.80；
   c_t 反而丢掉了 anchor 身份与状态细节）；
3. **Q4 的失败与"有没有 context"无关**（Q1/Q2：结构性 OOD；加性 primitive 无改善），
   把资源投在 B（重设 fast/slow 分解）或 C（primitive+interaction 预测器）之前，
   应先接受"显式 context 目前不是有效贡献"这一否证结果。

**次选（若必须继续 context 路线）**：A. 采集结构化 compositional V0.7 数据集 ——
但本轮证据**不支持**它作为第一优先：Task 5 表明去掉 privileged 输入几乎无损，
说明现有数据并不缺"可观测性"，缺的是**命令组合的覆盖**；若采，应专门覆盖
"多轴同时非零"的组合网格而非同族加量。

**明确不推荐**：B 与 C 作为下一步主方向（它们是"改架构"，而当前证据指向"输入覆盖 + 定位问题"，
不是容量/结构问题：M0/M1/ridge/ccopy 在 B/C 上的差距都在同一量级）。

## 本轮成功标准自评

任务要求"围绕 command novelty / compositional generalization / context disentanglement /
trajectory consequence 展开"，并"完成后停止"。已达成：
- 4 个主题各有一组可复算的量化结论 + 图；
- 未采新数据、未接 V-JEPA、未做 correction、未改大模型、未开 Risk-Gated 线；
- 唯一新训练是 Task 5 的 12 个小 run（0.0028 GPU·h）；
- 所有"未执行/不适用"项如实标注（如完整相对位姿的 SE(2) rollout 已做，但
  valid-planar 是**核验过**的而非假设；anchor 探针只在 train 内 CV）。
