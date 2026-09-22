# V0_6_FULL_REPORT — 有效性审计与 Support/Query 迁移试验全记录

> 日期：2026-09-22 · 执行依据：tasks/03_V0_6_ClaudeCode执行Prompt.md（参考 01/02）
> 范围：阶段A 离线审计 → 阶段B support/query pilot（240 ep）→ 阶段C 公平模型对照（12 run）
> 判定摘要见 report/V0_6_DECISION.md；逐题回答见 report/NEXT_ACTIONS.md

---

## 1. 总览

| 阶段 | 内容 | 状态 |
|---|---|---|
| A | 旧结论离线审计（不训练/不采数）：外部复核 10 条重算、A1 manifest、A2 因果/时间、A3 reset/注入、A4 指标/基线、A5 依赖统计、A6 校准/几何/transition | PASS（无 P0，11 项 bugs/口径记录） |
| B | PRE_REGISTRATION 冻结 → 12 组合 smoke → 240 ep pilot（192 query + 48 support，50Hz + 20Hz 派生） | PASS（2 项偏差已记录） |
| C | M0/M1/M2/M3 × seeds 42/43/44（同数据/协议），held-out anchor + held-out family 评估 + episode-cluster bootstrap | 完成（12/12 run） |

## 2. 阶段A 摘要（详见 report/A_AUDIT_DECISION.md）

- 外部复核（tasks/01 §0）10 条 31 项**全部复现一致**（audit/DIFFERENCE_REPORT.md，
  metric_equivalence.csv）。
- 6 项判定：DATA_CAUSALITY PASS（4 强制注记）/ PAIR_INDEPENDENCE PASS（旧 pair CI
  作废）/ SIMPLE_BASELINE_CHECK mixed / CONTEXT_OUTPUT_SENSITIVITY supported /
  PHYSICAL_CONTEXT_TRANSFER inconclusive / NEXT_STAGE_ALLOWED yes。
- 关键撤回（bugs_and_impact.md #6–#9）：transition 分组退化、校准回归方向误读、
  "native error > 工况差距"、"regime teleport 不可分"。

## 3. 阶段B：support/query pilot

### 3.1 协议（PRE_REGISTRATION.md，采集前冻结）

- 工况：normal(1.0) / friction_mid(0.6) / friction_low(0.3)，写 robot material；
- 4 query families × 8 anchor groups × 3 conditions × 2 reps = 192 query +
  16 support/condition × 3 = 48，合计 240 ≤ 预算 300；
- anchor seed AQ[i]=1000+17i，support seed SP[j]=5000+31j；每 wave 独立 env.reset；
- split：train AQ0–4 / val AQ5 / test AQ6–7（**held-out anchor**）；
  Q4_combo 全部 anchor **held-out family**（test-only）；
- history L=20（1s），H=40（2s），20Hz 派生：inputs 因果（一阶低通 fc=8Hz +
  previous-hold）、labels 抗混叠（resample_poly 2/5）。

### 3.2 偏差（PRE_REGISTRATION_DEVIATIONS.md）

1. smoke 实采 48 文件（num_envs=4 × 12 组合）→ smoke 目录删除，不计入数据集；
2. pilot 改 num_envs=1 逐 wave 采集以严格满足 ≤300；
3. 摩擦有效值 = (写入+1)/2（ground=1.0，PhysX average combine；写入=读回精确）；
4. anchor 跨 condition 复原 bit 级一致（max |Δpos| = 0.0，仍标 approximate-paired）。

### 3.3 数据校验

- 240/240 ep；query：3 cond × 4 family × 16，support：3 cond × 16；
- termination：schedule_end 239 / terminated 1（ep_00225，friction_low support，
  6.12s，如实保留）；
- 20Hz 派生 240/240，grid/labels 长度对齐 misaligned=0（修复 bugs #12 后重派生）；
- friction readback：写入=读回（1.0/0.6/0.3），ground static=dynamic=1.0。

## 4. 阶段C：公平模型对照

### 4.1 设置（冻结）

四组同数据/同 history/同 H/同 loss（SmoothL1 β=1）/同 early-stop（patience 15，
仅 train/val）/同预算（≤100 epoch，Adam 1e-3，bs=128）：
- **M0** direct recurrent（等输入对照，226,808 参数）
- **M1** context（c 来自自身 history，216,128）
- **M2** 同 M1 结构，训练时 c 来自同 condition 另一条 support ep（在线编码、梯度回传）
- **M3** M1 + 零初始化 W·真实摩擦标量（privileged 诊断，216,144；+16 参数已声明）

训练窗：train 552 / val 144（train+support，每 ep 4 窗；val 每 ep 8 窗）。
测试：held-anchor 384 窗（AQ6/7 × Q1–Q3）+ held-family 384 窗（Q4_combo），仅评价。

### 4.2 主结果（3 seeds 均值，MAE_all@2s；metrics/model_comparison_3seeds.csv）

| split | ccopy | persistence | ARX(ridge) | M0 | M1 | M2 | M3 |
|---|---|---|---|---|---|---|---|
| held-anchor | 0.1054 | 0.2356 | 0.0897 | 0.0878 | 0.0864 | **0.0843** | 0.0863 |
| held-family | 0.1622 | 0.3339 | 0.1363 | 0.1388 | 0.1365 | **0.1317** | 0.1365 |

val_loss（best epoch）：M0 0.0062×3；M1 0.0058/0.0058/0.0059；M2 0.0056–0.0058；
M3 0.0058/0.0058/0.0059 —— seed 间离散度极小。

### 4.3 关键差值（metrics/model_diff_bootstrap.csv，episode-cluster，2000 次）

负值 = 前者更优；**全部 CI 不跨 0**：

| diff | held-anchor | held-family |
|---|---|---|
| M1−M0 | −0.0014 [−0.0021,−0.0006] | −0.0023 [−0.0032,−0.0015] |
| M2−M1 | −0.0021 [−0.0028,−0.0013] | −0.0049 [−0.0058,−0.0040] |
| M3−M1 | −0.0001 [−0.0001,−0.0000] | −0.0001 [−0.0001,−0.0000] |
| M1−ccopy | −0.0190 [−0.0271,−0.0118] | −0.0257 [−0.0361,−0.0148] |
| M1 native−cross_support | −0.0406 [−0.0445,−0.0369] | −0.0272 [−0.0299,−0.0247] |
| M1 native−diff_condition | −0.0440 [−0.0484,−0.0400] | −0.0289 [−0.0318,−0.0258] |
| M1 native−zero | −0.0113 [−0.0146,−0.0081] | −0.0034 [−0.0048,−0.0019] |
| M1 native−shuffle | −0.0189 [−0.0216,−0.0162] | −0.0097 [−0.0117,−0.0077] |

原始每窗误差：metrics/per_window_errors.npz（含 episode/anchor/cond 映射，可复算）。

### 4.4 support/query 迁移（M1 s42，held-anchor；metrics/support_query_transfer.csv）

| variant | MAE_vx@1s | MAE_wz@1s | MAE_all@2s |
|---|---|---|---|
| native_c | 0.0653 | 0.1187 | **0.0865** |
| cross_support_c（同 cond 跨命令） | 0.1271 | 0.1365 | 0.1263 |
| diff_condition_c | 0.1274 | 0.1371 | 0.1293 |
| zero_c | 0.0826 | 0.1212 | 0.0978 |
| shuffle_c | 0.0850 | 0.1262 | 0.1052 |

k=0 首步连续性 |pred[0]−e[origin]|：native 0.1134 / cross 0.1850 / diff 0.1898 /
zero 0.1235 / shuffle 0.1336（metrics/k0_continuity.json）。
解读：c 与 episode 状态纠缠（Q6）；cross-support 部署迁移不支持，训练正则有增益。

## 5. 负结果与缺失数据（如实记录）

1. **M3 oracle 摩擦无增益**（−0.0001）——"知道真实工况"不改善预测，瓶颈不在
   context inference（Q9）。
2. **cross-support c 部署迁移失败**（+46%/+31% 相对误差）——旧"物理 context
   transfer"设想不成立。
3. **M1 相对 ARX 在 held-family 仅持平**（0.1365 vs 0.1363）——简单线性基线在
   OOD family 上仍有竞争力。
4. held-anchor 只有 **2 个 anchor group**——anchor 泛化结论为探索性强度，不可外推。
5. 缺失：无 deployable 输入对照（输入含 GT 速度/姿态）；support pool 每 cond 仅
   16 条；rep=2 仅够粗估残差不可约方差（留作下一步 b）。

## 6. 每阶段日志（命令 / 耗时 / GPU·h / 输出）

| 阶段 | 命令（要点） | 耗时 | GPU·h | 输出 |
|---|---|---|---|---|
| A | `python -m execution_wm.validity_v06.{external_review_check,inventory,manifest_build,causal_audit,reset_injection_audit,baselines,dependence,calibration_geometry}` | ~2h（离线 CPU+GPU eval） | ≈0.3 | audit/* manifests/* metrics/*(A 部分) |
| B smoke | `isaaclab.sh -p execution_wm/data/collect_support_query.py --config .../collect_v06_sq.yaml --smoke` | ~10 min | ≈0.17 | audit/smoke_validation.md（PASS） |
| B pilot | 同上 去 --smoke --num-envs 1 | ~16 min wall | ≈0.27 | v0_6_sq 240 ep + index.json |
| B derive | `python -m execution_wm.validity_v06.derive_20hz --sq-dir ...` | <2 min | 0 | *.20hz.npz ×240 |
| C train | `python -m execution_wm.validity_v06.train_sq --model {M0..M3} --seed {42,43,44}`（先实测 M0/M1 s42 ≈0.0005 GPU·h/run，确认后放开） | ~12 min 合计 | **0.0059** | checkpoints/v0_6/*/best.pt + train_log.json |
| C eval | `python -m execution_wm.validity_v06.eval_sq` | ~2 min | <0.01 | model_comparison_3seeds.csv, support_query_transfer.csv, k0_continuity.json |
| C boot | `python -m execution_wm.validity_v06.boot_models` | ~3 min | <0.01 | per_window_errors.npz, model_diff_bootstrap.csv |

全程环境：conda `isaaclab`，torch 2.7.0，Isaac Sim 5.1 / Isaac Lab 2.3.2，RTX 4090。
采集/训练总 GPU·h ≈ 0.75。

## 7. 文件清单（results/v0_6_validity_transfer/）

- audit/：source_inventory, metric_equivalence, DIFFERENCE_REPORT, causal_time_frame,
  input_roles, reset_and_injection, smoke_validation, synthetic_bootstrap_validation,
  bugs_and_impact（13 项）
- manifests/：episodes, full_pair_manifest（25772+2564）, split, duplicate_and_reuse,
  support_query_manifest（576 行 = 192 query × 3 swap types）
- metrics/：A 段 8 表 + C 段 model_comparison_3seeds / model_diff_bootstrap /
  support_query_transfer / k0_continuity / per_window_errors.npz
- raw/：audit_prediction_cache.npz（66MB 全量预测缓存）,
  representative_full_episodes/（12 ep × {50Hz,20Hz,meta}）, support_query_examples.npz
  （24 窗：history/future/GT/M1 双 c 预测/support 窗/IDs/dt）
- figures/：CAL6 可靠性 ×6, GEO6 分解， TRANS6
- code_snapshot/：validity_v06 全部 .py + collector + 2 个 config + git diff stat
- report/：A_AUDIT_DECISION / V0_6_DECISION / V0_6_FULL_REPORT / NEXT_ACTIONS
- PRE_REGISTRATION.md + PRE_REGISTRATION_DEVIATIONS.md

## 8. 结论指针

判定：report/V0_6_DECISION.md · 逐题回答与下一步：report/NEXT_ACTIONS.md。
核心一句话：模型在协议内 held-out 上显著优于 command-copy（V0.5 的"不胜基线"是
probe 分布问题）；但 context 不是可迁移物理编码，旧机制叙事撤回，主张收窄，
下一步优先 deployable 输入对照与残差可预测性上界，而非扩同类数据。
