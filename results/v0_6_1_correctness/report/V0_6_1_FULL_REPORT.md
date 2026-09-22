# V0_6_1_FULL_REPORT — 标签、Support 配对与评价协议修复全记录

> 2026-09-22 · 执行依据 `tasks/v0.6.1/`（02 任务书 + 01 验收 + 03 源码证据）
> 判定字段见 `report/V0_6_1_DECISION.md`；缺陷矩阵见 `report/BUG_IMPACT_MATRIX.md`；
> 逐题回答见 `report/NEXT_ACTIONS_V061.md`
> 旧 V0.6 结果（`results/v0_6_validity_transfer/`）原地保留未改动

## 0. 一句话

两处 P0 与四处 P1 在真实代码/数据中**全部复现**并已修复；修复后主结论方向改变——
模型只在"已见命令族"上明确优于 command-copy，在"未见命令族"上与 command-copy 相当或更差，
且 **Direct ≥ Context**。本轮不新增研究模块、未换方向、未重采数据。

## 1. 时间线（命令 / 耗时 / GPU·h / 输出）

| 步骤 | 命令（要点） | 耗时 | GPU·h | 输出 |
|---|---|---|---|---|
| 冻结+复现 P0 | `frozen_v06_manifest.json`、`p0_label_identity_reproduction.csv` | ~5 min | 0 | audit/ |
| 修复派生 | `python -m execution_wm.validity_v061.derive_targets --sq-dir .../v0_6_sq --out-dir .../v0_6_1` | ~2 min | 0 | 240×`*.20hz.npz` + `derive_identity_audit.csv` |
| 单测 T01–T08 | `python -m execution_wm.validity_v061.tests_v061` | <1 min | 0 | `unit_tests/` |
| §3 运行时审计 | `isaaclab.sh -p execution_wm/data/policy_command_audit.py --num-waves 3` | ~3 min | ≈0.02 | `audit/policy_command_timeline.json` |
| 影响量化 | `python -m execution_wm.validity_v061.impact_analysis` | ~3 min | 0 | `derived_data_summary/` |
| 摩擦/材质 | `isaaclab.sh -p execution_wm/data/friction_material_audit.py` + USD 属性读取 | ~4 min | ≈0.02 | `audit/friction_material.json`, `audit/usd_material_attrs.json` |
| R1 旧模型重评 | `python -m execution_wm.validity_v061.r1_old_model_reeval` | ~2 min | <0.01 | `metrics/r1_old_models_new_labels.csv` |
| R2/R3 训练 | 12 × `python -m execution_wm.validity_v061.train_v061 --model Mx --seed 4y` | ~15 min | **0.0058** | `checkpoints/`（12×best.pt + train_log） |
| R3/R4 评估 | `python -m execution_wm.validity_v061.eval_v061`（单次推理） | ~2 min | <0.01 | `predictions/pred_cache.npz`（sha256 固定） |
| boot/T09/T10 | `boot_v061.py`, `tests_t09_t10.py` | ~4 min | <0.01 | `metrics/v061_*.csv` |
| 姿态/净 yaw | `python -m execution_wm.validity_v061.pose_metrics` | ~2 min | 0 | `metrics/relative_yaw_*.csv` |
| 图 | `python -m execution_wm.validity_v061.figures_v061` | <1 min | 0 | `figures/F1–F6` |

环境：Isaac Sim 5.1 / Isaac Lab 2.3.2 / torch 2.7.0 / RTX 4090。
**未重采任何数据**（§3 判定旧 raw 可复用）。

## 2. 复现与修复（Q1）

### 2.1 P0-1 support 不成对

- 原函数重放：10000 次抽样中 **9859 次** hp/ha 编号不一致（98.59%）；验收报告给出的
  98.43% 属实（同一现象、不同玩具规模）。
- 修复：`SupportBank` + `SupportRecord`（一次抽 `support_window_id` → 读取全部配套字段）；
  `exclude_episodes` 实现 self-exclusion（本数据 query/support episode 不重叠 → 真空但生效）；
  donor lineage（episode_id / window_id / origin_tick / session / condition / split）逐窗保存。
- 覆盖路径：M2 train/val/test、M1 cross_support/diff_condition、bootstrap variants。

### 2.2 P0-2 标签恒等式被破坏

- 全量 240 ep：旧 `cmd20 + lb_residual` 与同文件 `lb_execution` 的最大差 **1.120208**，
  **0/240** 条成立；`ep_00162` t=2.5s wz：1.885307 vs 0.765099（差 1.120208）——与验收报告逐位一致。
- 修复流程：整数 tick 参考时钟 → `execution_label = F(e)`（只依赖物理 execution）→
  `command_reference = S(u)`（事件表，整数 tick）→ `residual_label = e_label − u_ref`。
  评估**只**读 `e_label`，不再用 `u+r` 自证。
- 结果：恒等式 max err **1.11e-16**（float64）/ 5.96e-08（float32 存储，阈值 1e-5），
  全 240 ep × 69612 点通过。
- 影响分解（`derived_data_summary/impact_summary.json`）：旧−新 = ①B3 浮点 hold 错位
  （max 1.600 / mean 0.0090）+ ②B2 命令采样 vs 滤波 `S(u)−F(u)`（max 0.480 / mean 0.0090），
  分解残差 7.9e-08 → 两成分**精确**解释全部差异。
- residual std 上升：vx ×1.042 / vy ×1.043 / wz ×1.076（即旧标签低估了残差）。

### 2.3 P1-1 时间 hold 错位

- 全 240 ep 共 **8751** 个 grid 点旧浮点 hold ≠ 正确整数 tick（均值 36.5/ep，
  与验收报告"80 点中 36 点"同量级）。
- T04 单测：真实 Q3_turn 分段下 2.5s 处新实现取 tick 125（wz −0.8），旧浮点实现取 tick 124（wz +0.8）；
  1.0/1.5/2.5/3.5s 四个事件与前一 grid 点全部右连续正确；长时程 1000 tick 与浮点扰动均通过。

### 2.4 P1-2 轴选择

- 标识值 [11,12,13,21,22,23] → 正确输出 **[11,12,23]**，旧切片给 [11,12,13]。
- 注意方向：**修正后 persistence 反而更差**（A +0.0106 / B +0.0207 / C +0.0229）——
  旧的"更准"是错误轴恰好蒙对，必须按正确轴报告。

### 2.5 P1-3 split 混叠

- 旧 held_anchor 48 ep = Q1/Q2/Q3/Q4 各 12（Q4 占 384 窗中 96），与 held_family 重叠 12 ep。
- 新三 population（互不重叠，断言通过）：A 36 ep / B 30 ep / C 12 ep，`supp_AQ5xQ4` 6 ep 单列。

### 2.6 P1-4 lead 指标

- toy：仅最后一步非零 → fixed-lead 1.0 vs prefix 1/40，二者必不相等（T07）。
- lead 索引由 timestamp 推导：0.25s→j4、0.5s→j9、1.0s→j19、2.0s→j39。
- 主表逐轴 fixed lead；prefix 单列；归一化 scalar 只用 train 拟合的容差。

### 2.7 其余（B7 / §3 / §4）

- **B7 统计单元**：固定 window manifest（`manifest_hash=245f1e23c43b1c48` 对全部 12 run 相同，
  720 训练窗），训练 seed 只影响初始化与打乱；单次推理 + `pred_cache.npz`（sha256 固定，
  bootstrap 只读缓存，T09 断言无推理调用）。
- **§3 policy 消费**：见 `audit/policy_command_timeline.md`（VERIFIED，滞后恰好 1 tick）。
- **§4 额外审计**：见 `audit/additional_audits.md`（摩擦 combine=**multiply** 纠正 V0.6 的
  average 记录；四元数合法化；rep 不可作重复；ARX 正名；oracle 分支响应）。

## 3. 主结果（Q2–Q8）

### 3.1 逐轴 fixed-lead MAE @2.0s（3-seed 均值；单位 m/s, m/s, rad/s）

**A unseen_anchor_seen_family**（36 ep / 288 窗）

| | vx | vy | wz | prefix vx |
|---|---:|---:|---:|---:|
| M0 | 0.0424 | 0.0437 | 0.0780 | 0.0407 |
| M1 | 0.0410 | 0.0419 | 0.0757 | 0.0401 |
| M2 | 0.0462 | 0.0468 | 0.0821 | 0.0413 |
| M3 | 0.0410 | 0.0418 | 0.0757 | 0.0401 |
| command-copy | 0.0655 | 0.0641 | 0.0888 | 0.0590 |
| ridge_multioutput | 0.0412 | 0.0537 | 0.0809 | 0.0393 |
| persistence | 0.1635 | 0.1675 | 0.2915 | 0.1659 |

**B seen_anchor_unseen_family**（30 ep / 240 窗）

| | vx | vy | wz |
|---|---:|---:|---:|
| M0 | 0.1925 | 0.0919 | 0.1364 |
| M1 | 0.1820 | 0.0673 | 0.1385 |
| M2 | 0.1722 | 0.0619 | 0.1341 |
| command-copy | 0.1891 | **0.0305** | 0.1160 |
| ridge | 0.1843 | 0.0548 | 0.1348 |

**C unseen_anchor_unseen_family**（12 ep / 96 窗）：同型（ccopy vy 0.0285 优于全部模型）。

### 3.2 归一化主标量（<1 优于 command-copy）

| split | M0 | M1 | M2 | M3 | ccopy | ridge |
|---|---:|---:|---:|---:|---:|---:|
| A（3-seed 范围） | 0.706–0.731 | 0.699–0.713 | 0.739–0.748 | 0.697–0.712 | 0.980 | 0.713 |
| B | 1.877–1.945 | 1.803–2.036 | 1.833–2.068 | 1.803–2.036 | 1.808 | 1.854 |
| C | 1.888–1.944 | 1.863–2.062 | 1.861–2.036 | 1.862–2.061 | 1.827 | 1.901 |
| supp_AQ5xQ4 | 1.95–2.06 | 1.92–2.12 | 1.91–2.07 | 1.92–2.12 | 1.882 | 1.958 |

### 3.3 关键差值（episode-cluster bootstrap，2000 次；**探索性**）

| diff | A (2 anchors) | B (5 anchors) | C (2 anchors) |
|---|---|---|---|
| M1 − M0 | −0.0011 [−0.0016,−0.0006] | +0.0043 [+0.0037,+0.0048] | +0.0041 [+0.0034,+0.0050] |
| M2 − M1 | +0.0038 [+0.0021,+0.0056] | −0.0008 [−0.0016,−0.0001] | −0.0021 [−0.0034,−0.0008] |
| M3 − M1 | −0.0001 | −0.0000 | −0.0001 |
| M1 − ccopy | −0.0244 [−0.0335,−0.0166] | −0.0002 **跨0** | +0.0035 **跨0** |
| M2 − ridge | +0.0015 | −0.0020 | −0.0018 **跨0** |
| persistence 正确轴 − 旧错轴 | +0.0106 | +0.0207 | +0.0229 |

**M2 donor 对照（固定 donor 表）**，负=前者更好：

| diff | A | B | C |
|---|---|---|---|
| same_cond − wrong_cond | **−0.0215** | **−0.0097** | **−0.0088** |
| native − wrong_cond | −0.0162 | −0.0076 | −0.0070 |
| native − same_cond | +0.0054 | +0.0021 | +0.0018（跨0） |
| native − zero | −0.0029 | +0.0041 | +0.0038 |

**M1 context variant**（native 为基准），负=前者更好：

| diff | A | B | C |
|---|---|---|---|
| native − cross_support(paired) | −0.0185 | −0.0043 | −0.0029 |
| native − diff_condition | −0.0202 | −0.0035 | −0.0015（跨0） |
| native − zero | −0.0089 | **+0.0023** | **+0.0033** |
| native − shuffle | −0.0158 | −0.0019（跨0） | −0.0002（跨0） |

### 3.4 同 command 不同 condition 的真实 execution 差距（正确 e，非预测误差）

192 个匹配对（同 anchor/family/rep，逐 grid 点平均绝对差）：

| 对比 | vx | vy | wz |
|---|---:|---:|---:|
| normal − friction_mid | 0.0371 | 0.0297 | 0.0694 |
| normal − friction_low | 0.0783 | 0.0582 | 0.1155 |
| friction_mid − friction_low | 0.0591 | 0.0512 | 0.0915 |

### 3.5 净 yaw（relative pose 的合法子集）

`metrics/relative_yaw_summary.csv`：A 上全部 variant 的净 yaw MAE ≈1.547–1.556 rad，
command-copy 1.554，**无模型实质性领先**；B/C 上 command-copy（0.306 / 2.648）反而最好。
→ **相对位姿排序与瞬时 wz 排序不一致**（模型在瞬时 wz 上的优势没有转化为积分后的净 yaw 优势）。
完整相对位姿轨迹 **NOT_RUN**（未预注册）。

### 3.6 R1 旧模型 + 新标签（OLD_MODEL_NEW_EVALUATION_ONLY）

`metrics/r1_old_models_new_labels.csv`：旧 checkpoint 在修复真值下仍然"可用"
（A: MAE_all 0.0735–0.0753；B: 0.1439–0.1460），且旧 M2 在 B/C 的 wz@1s 上仍略优
（0.1975/0.2000 vs M1 0.2351/0.2318）。**只能说明评价变化，不能代替正确标签重训。**

## 4. 单测（T01–T10）

| 单测 | 状态 | 关键证据 |
|---|---|---|
| T01 support 成对性 | PASS | 新实现 2000 次 0 违反；旧实现同玩具 98.3% 错配 |
| T02 privileged/标签不进入 inference | PASS | 相同输入两次前向 bit 相同；proprio 40 维无 friction/label 通道 |
| T03 恒等式全量 | PASS | 240 ep / 69612 点，max 5.96e-08 |
| T04 命令边界整数 tick | PASS | 2.5s：新 125(−0.8) / 旧 124(+0.8)；8751 点错位复现 |
| T05 schema persistence | PASS | [11,12,13,21,22,23] → [11,12,23] |
| T06 split 与 donor lineage | PASS | A/B/C 互斥；legacy overlap = 12 ep；donor 表固定可复现 |
| T07 fixed-lead vs prefix | PASS | toy 1.0 vs 0.025；索引 {4,9,19,39} |
| T08 因果性与未来独立 | PASS | 输入源 tick ≤ origin（max 0）；未来源 tick ≥ origin+2 |
| T09 缓存一致性 | PASS | cache sha256 == manifest；bootstrap 无推理调用 |
| T10 policy 消费时刻 | PASS | 3×199 tick 精确 1 tick 滞后 |

## 5. 负结果与未执行项（如实记录）

1. **M1（native Context）在 27 个 (split, lead, axis) 单元中 0 次胜出**；B/C 上 M1 劣于 M0。
2. **B/C 上模型不优于 command-copy**（归一化 1.80–2.07 vs 1.81/1.83）；vy 通道模型明显更差
   （ccopy 0.0305/0.0285 vs 模型 0.06–0.09）。
3. **M3 oracle 无增益**（−0.0001），尽管分支确实响应 friction。
4. **净 yaw 排序与瞬时 wz 排序不一致**；完整相对位姿 NOT_RUN。
5. **M1 的 cross_support 对照含义有限**：M1 encoder 训练于自身 history，support 窗口是分布外
   → "cross-support 更差"混合了信息量与分布匹配两个因素，不能单独归因。
6. held-anchor 仅 **2 个 anchor group**（A/C），B 为 5 个；episode cluster 数（36/30/12）
   不等于独立 anchor 数，所有区间只能作探索性。
7. 未执行：扩采、新结构、真机、视觉、V-JEPA（本轮禁止）。

## 6. 可独立重算清单（任务书 §7 逐条）

| 要求 | 文件 |
|---|---|
| 全部修改源码与 git diff；模型类/registry/schema | `code_snapshot/`（含 `model_sources/`）+ `report/BUG_IMPACT_MATRIX.md` 的 commit |
| 每 checkpoint 的 config/weights hash/train_log | `config/*_train_log.json`、`checkpoints/`（12×best.pt，11 MB）、hash 见 BUG_IMPACT_MATRIX |
| 240 raw + 240 derived 的 hash 与全部 metadata | `manifests/data_manifest_240.csv` |
| 每 window 的 id/origin_tick/origin_time/history 索引/future 索引/split/donor | `manifests/window_lineage.csv`、`manifests/donor_tables.json` |
| 完整 predictions（非 object、JSON/npz） | `predictions/pred_cache.npz`（160 数组，sha256 固定）+ `pred_cache_manifest.json` |
| 完整单测输出与失败最小复现 | `unit_tests/T01–T10*.json`（10/10 PASS，无失败复现） |
| 代表性 normal/mid/low × Q1–Q4（raw50Hz/derived/metadata） | `raw/representative_full_episodes/`（15 ep × 3 文件） |
| 新 summary、逐 axis/lead 表、误差按 anchor 图、修复前后例子 | `metrics/v061_main_metrics.csv`、`metrics/per_axis_lead_ranking.csv`、`metrics/per_anchor_group.csv`、`figures/F1–F6` |
| 决策与全报告 | `V0_6_1_DECISION.md`、`V0_6_1_FULL_REPORT.md`、`BUG_IMPACT_MATRIX.md` |

## 7. 与 CVPR 主线的关系

修复未更换 `Command → Execution → Future → Correction` 主线，只恢复了每步输入/监督的
物理含义。当前证据**不支持**在此方向上继续堆结构或堆同类数据；最有价值的单个下一步是
deployable 输入 + 条件推断（见 `NEXT_ACTIONS_V061.md`）。大模型训练/接视觉/真机在线纠错
均未启动，按要求停在本轮边界内。
