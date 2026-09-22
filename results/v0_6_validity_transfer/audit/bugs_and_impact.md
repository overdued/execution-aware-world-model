# Bugs and Impact — V0.6 阶段A 审计发现汇总

> 每条按 evidence / impact / fix 记录。P0=影响训练标签或输入因果；P1=影响指标解释；P2=口径/验证缺口。
> **未发现 P0。** 审计代码自身的错误也如实记录（第 10 条）。

## 1. [P1 时间语义] e/u 一步滞后（定义性，非随机 bug）

- **Evidence**: collector.py 主循环先 `capture_step`（step 前状态）后 `env.step`；记录的 e_t 是区间
  [t−Δt,t] 的执行，u_t 是 [t,t+Δt] 将施加的命令。数据驱动验证 corr(e_t, u_{t−1})=0.887 >
  corr(e_t, u_t)=0.835（vx；三轴一致）。window 切片 future 从 t0+1 起，部分补偿。
- **Impact**: r_t 是"当前速度 vs 当前请求"的瞬时 tracking error，命令领先执行约一步；
  transition 处的大 residual 一部分是该滞后的必然产物，不全是物理 mismatch。
  训练/评估全链一致，不构成泄漏。
- **Fix**: 不改数据；所有解释带此前提。V1 可在记录时显式对齐（applied-command-at-origin 语义），
  需单独版本化。

## 2. [P1 采样] 50Hz→20Hz 最近邻降采样，无抗混叠

- **Evidence**: save_episode `idx=round(2.5k)`（banker's rounding: 0,2,5,7,…），真实间隔
  {0.04,0.06}s 交替却标称 0.05s；无低通。实测 dt 计数 {0.04: 81, 0.06: 80}（403 步 episode），
  时钟误差 ±0.01s 内交替、无累计漂移。
- **Impact**: >10Hz 步态波纹混叠进 20Hz 数据（wz 波纹尤甚）；瞬时相位非均匀。
  wz 难预测部分可能是采样/相位问题而非"无信号"（§C4 要求）。
- **Fix**: 修改会改变数据 → 单独版本化。B5 保留 50Hz 原始日志，20Hz 另行抗混叠派生。

## 3. [P1 统计] pair-level / episode-pair-cluster bootstrap CI 作废

- **Evidence**: synthetic 验证（20 episode、随机效应 sd=1、200 窗/ep）：窗级 naive bootstrap
  覆盖率 **8%**（名义 95%），episode 级 **94%**。本数据 25772 行仅 153 unique episodes /
  147 anchor groups / N→L 67×54 episodes。
- **Impact**: V0.5 及之前所有 pair/cluster CI 系统性过窄 4–6×（如 N→L P_target：
  旧 [−0.184,−0.159]，two-way [−0.238,−0.099]）。**主要符号结论在 two-way/anchor 下仍成立**
  （S_dir、P_target 不对称、M_shift cross>same 的 CI 均不含 0）；E_swap cross−same 的
  VL→N 方向 CI 跨 0（[−0.551,+0.044]），该结论降级。
- **Fix**: 已用 two-way（episode 与 anchor 两级）+ LOEO 替代（metrics/node_aware_bootstrap.csv,
  leave_one_group_out.csv）。LOEO 显示无单一 episode 支配。

## 4. [P2 泄漏] episode-level split + 链内连续

- **Evidence**: d0 episode 间无 sim reset（A3）；split 按 episode 随机 → 同 anchor 链的
  相邻 episode 可跨 train/test。实测 2/155 链跨 split。
- **Impact**: 链内首窗与上一条末窗动力学连续 → 轻微 train/test 泄漏（估计 <2% 窗口）。
- **Fix**: 本轮不重建 split（避免改动 V0 训练史）；B/C 新数据按 anchor group 切分。

## 5. [P2 验证缺口] 摩擦注入无运行时 readback；combine rule 未定

- **Evidence**: set_friction 写 PhysX buffer 后无读回；地面 material 未改、PhysX 默认 average
  combine → 有效足-地摩擦 ≠ YAML 写入值。
- **Impact**: "friction_low=0.3" 应读作"写入值 0.3"；剂量效应的方向性结论保留，档位数值解释挂起。
- **Fix**: 阶段B smoke 必须含 get_material_properties readback + 地面 material + combine rule 记录。

## 6. [P2 口径] V0.5 transition 分组退化 + inf 误标

- **Evidence**: class_tau0.25 == class_tau0.5（25772 行全同）；每窗 min age ∈ {0,1,inf}；
  10682 行全 inf（episode 起点到首次观测变化之间）被当 steady。
- **Impact**: 0.25/0.5s 不是两次独立敏感性证据；"steady −0.269"混入 unknown 窗口。
- **Fix**: per-timestep 重算（τ=0.25s，含 unknown 类）：N→L P_target 贡献
  transition +0.001 / steady −0.022 / unknown −0.039（metrics/transition_timestep_metrics.csv）。
  unknown 占 48% timestep，旧"steady 主因"说法需修正为"steady+unknown（非瞬变）主因"。

## 7. [P2 口径] 校准回归方向误读 → "校准失败"论据部分撤回

- **Evidence**: V0.5 的 calib_slope=Cov(pred,true)/Var(true) 是 pred-on-truth；MSE 意义下即使
  条件均值无偏也有 slope=Var(m)/(Var(m)+Var(ε))<1。本轮 truth-on-pred（部署方向）：
  friction_low vx **1.049**、vy 1.015；friction_vlow vx **0.986**；normal vx 0.959 ——接近 1。
  wz：low 0.724、vlow 0.791、normal 0.946。
- **Impact**: "vx 幅值需放大 1.2–1.6×"不成立；**vz/vy 的均值校准在 test 上接近无偏**；
  wz 仍有 ~0.72–0.79 的收缩。且 V0.5 val-affine 在 test 上多数轴 **变差**
  （MAE_affine > MAE_raw，如 low vx 0.158→0.170）——"6 参数 affine 修复主门"不可外推，
  当时的 P_target 改善（+0.009, CI 跨 0）应记为 inconclusive。
- **Fix**: prediction_reliability.csv 双向斜率 + 分箱 E[truth|pred] + episode CI；
  oracle gamma 仅标 ORACLE_DIAGNOSTIC。

## 8. [P2 口径] "native error > 真实工况差距"（N/L）论据撤回

- **Evidence**: 旧报告的 1.22 实为 D_before=‖e_B−pred_A‖；真实 actual-actual 距离 1.485，
  target native error 1.349 < 1.485。N/VL：1.976 > 1.885 仍成立。
- **Impact**: N/L 方向的该论据撤回（复核声明 9 复算一致）；N/VL 保留。

## 9. [P2 口径] "regime teleport 统计不可分" 撤回

- **Evidence**: swap 预测与 target 原生预测的直接距离 = native error 的 20.7%–27.5%（四方向，
  复核声明 5 复算一致）；到 GT 距离近似相等 ≠ 两预测等效。
- **Impact**: Gate M 的 teleport 证据降级为"预测显著相似"，不作等效性结论；
  等效检验需预设物理容差（B 阶段 PRE_REGISTRATION 锁定）。

## 10. [审计代码自身] 本轮审计脚本的三处错误（已修复）

- causal_audit 初版 quaternion 逆旋转公式方向错误（自测 vs 独立矩阵法发现；修复后
  max err 8.9e-16，数据侧 frame 验证通过 2.4e-07）。教训：审计代码也需自测。
- baselines 初版 context==cache 断言 atol=1e-5 过严：batched GRU eval 与单样本有 ~1e-4
  数值非确定性（非逻辑错误），放宽至 1e-3 并记录。
- dependence 初版 M_same 误用 E_native：修正为双口径（M_shift 口径复现 V0.5 数值；
  E_swap 口径为新指标），两口径在报告中分开命名。

## 11. [非 bug] batched vs single GRU 前向 ~1e-4 差异

cuDNN batch 算法差异，float32 精度内；对指标影响 <0.1%。记录以避免未来"复现不一致"误报。

## 12. [阶段B 代码] derive_20hz labels 与 grid 尾部长度差 1（已修复，未影响任何训练）

resample_poly 输出长度 ceil(n·2/5) 与 0.05s grid（arange 上界）在 episode 时长非 0.05
整数倍时相差 1 个尾样本（初版 154 条中 20 条 lb_ 比 grid 短 1）。修复：labels 一律对齐
到 inputs 网格长度（截断或末端 edge-hold），全量 240 条重派生后 misaligned=0。
修复发生在任何训练/评估之前，无结果受影响。

## 13. [阶段C 代码] train_sq/eval_sq 误从 execution_wm.models 导入 MODEL_REGISTRY（已修复）

registry 实际定义在 execution_wm.train.train_execution（其 __init__.py 为空）。
首个探测 run 启动即 ImportError，未产生任何训练结果；修正导入后 12 run 全部完成。
