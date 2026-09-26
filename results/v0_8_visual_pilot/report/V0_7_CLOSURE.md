# V0.7 CLOSURE — 收尾验收（Stage A 输出）

> 2026-09-27 · 任务书 §A6 · 全部结论基于只读旧缓存复算，未重训、未重采、未改旧预注册。
> 重算脚本：`execution_wm/v08_closure/`（reconcile_v07 / purity_heterogeneity /
> regime_marginals / pose_truncation_check），可独立运行。

## 门禁字段

```text
RESULT_TABLE_CONSISTENCY: PASS
ORIGINAL_PRIMARY_EFFECT: SUPPORTED
SUBGROUP_HETEROGENEITY: SUPPORTED
INTERACTION_MODEL_STATUS: FROZEN_AS_NEGATIVE_OR_INCONCLUSIVE_BASELINE
PHYSICAL_BACKBONE: DIRECT_R1
VISUAL_SMOKE_ALLOWED: YES
```

## A1. 两张 P1 主表的对账结果（PASS）

§4.1 与 §4.4 的数值差异**完全由聚合口径造成，两处都精确可复现**（复算差 ≤ 1e-16）：

| 出处 | 口径 | D/R1 | I/R1 |
|---|---|---|---|
| §4.1 `data_vs_arch_effect.csv` | **每 seed 分别算 FDE**，group-equal | 0.0849 / 0.0838 / 0.0842 | 0.0814 / 0.0865 / 0.0831 |
| §4.4 `trajectory_metrics.csv` `*_seedmean` 行 | **3 个 seed 的预测先算术平均再算 FDE**，window-equal | 0.0809 | 0.0801 |

差异分解（D/R1）：总差 0.00338 = 集成效应 0.00338 + 聚合口径效应 0.00000
（6 个 group 各 72 窗，group-equal 与 window-equal 数值相同）。
"预测先平均"的 FDE 低于"FDE 再平均"（凸性/Jensen），属**集成效应**，不是错误。

逐项标注（任务书 §A1 清单）见 `metrics/v07_table_reconciliation_meta.json`：
指标确为 FDE_xy@2s（中点积分、自回归 wz，非 ADE/prefix/fixed-lead）；population =
test_P1 全部 432 窗（非高纯度）；窗口 ID = episode_id@origin_grid；终止处理 = 432 窗
全部落在已记录数据内（见 A5）；checkpoint = val FDE_xy 选择的 best.pt；
缓存 sha256 = `839a3dfedb6b81bf91341116c04d0c433c3f57ebe85a1cd7c930d0c7cc6eccdd`。

**处理**：两表保留，按正确命名理解——§4.1 是"每 seed 的组等权 FDE"（预注册主口径），
§4.4 的 `seedmean` 行是"三 seed 预测集成的窗等权 FDE"，后者仅作参考基线对照，
不用于门判定。一个已知缺口如实记录：pred_cache 只含预测数组，GT/姿态未入缓存；
GT 由冻结数据 + 确定性 manifest（无 RNG）重建并核对一致。

## A2. 表述勘误

见 `report/V0_7_ERRATA.md`（E1 seed 计数、E2 纯度表述、E3 基线对照范围、
E4 终止计数口径）。数值与判定不变。

## A3. 效应异质性（SUPPORTED，pilot）

d_het（低纯度 − 高纯度的 Δ_data 差）= +0.0087 / +0.0101 / +0.0069；
seed 42/43 CI 不含零，seed 44 跨零（[−0.0002, 0.0136]）；6/6 group 两个子集均非空。
**解释边界**：高/低纯度子集命令构成不同（转向占比 0–25% vs 88–100%），
d_het > 0 只是该划分下的效应异质性，不作因果证明；主终点仍是 P1 全部窗口，
高纯度仍为敏感性分析，未提升为新主终点。

## A4. R0/R1 时序边际（描述性，不重配不重训）

数据集池口径（R1 数据集 = R0 + R1 脚本），逐 episode 均值：

| 指标 | R0 数据集 | R1 数据集 |
|---|---|---|
| zero 命令时间占比 | 0.230 | 0.419 |
| 低速（\|e_xy\|≤0.05）时间占比 | 0.314 | 0.387 |
| 命令切换率（次/秒） | 0.811 | 0.697 |
| dwell 均值 [min, max]（秒） | 1.14 [0.60, 1.93] | 1.40 [0.64, 2.96] |
| 切换 \|Δu\| 均值 / max | 0.65 / 1.04 | 0.72 / 1.10 |
| settle 后初速（m/s） | 0.040 | 0.040 |
| 瞬态（切换后 0.3s 内）占比 | 0.243 | 0.209 |
| 共激活时间占比 | 0.000 | 0.196 |

（逐 episode 全表：`metrics/v07_regime_marginals.csv`；脚本 regime 级与数据集池级
两口径都在其中，0.583 vs 0.375 的 zero 占比差别来自 R1 脚本 vs R1 数据集池。）

**允许结论**：当前结构化采集方案 R1 优于 R0（Δ_data 过门）。
**不允许结论**：已严格隔离共激活而排除全部停走/时序因素——两池的停/走比例、
切换率、dwell 分布、|Δu| 均不同（zero 时间占比 0.23 vs 0.42 最显眼）。

## A5. 位姿与截断核验（通过，不影响 CORRECTNESS）

- **0.2999 的归属**：是"抗混叠滤波后、合法化前"四元数的范数偏差审计字段。
  原始 simulator 四元数实测范数偏差 max **1.92e-07**，864 episode **0 次符号跳变**。
- **处理顺序事实**：derive 是滤波在前、符号连续+归一化在后。因原始序列无符号跳变，
  滤波未造成符号混合，仅有范数收缩；归一化恢复单位范数。
- **参考子集重算**（test_P1 432 窗，yaw 换用原始合法姿态 lbra_yaw 口径）：
  逐窗 |Δyaw| max 0.011 rad（0% 窗口 >1°）；四格 FDE 变化 |Δ| ≤ 8e-5 m；
  Δ_data / Δ_arch 全部 3 seed **无符号翻转** → 处理方式不影响任何结论，不入 CORRECTNESS。
- **终止/截断**：唯一终止 episode = ep492（val/G13/low/sc00，3.92 s 提前终止）。
  终止即停止录制，`fixed_origins` 只取 origin ≤ T−H−1 → 所有窗口完整落在终止前
  数据内，**不存在把短 horizon 当 2 s FDE 的情形**；val 含其 6 窗（0.46%），
  参与过 checkpoint 选择但影响有界；test 三 split 均为 0 终止；所有 baseline 与模型
  同规则。无隐形删样本。

## A6. 继续判定

数据正确性与主表可追溯性均通过；未发现影响训练数据真实性的 P0 问题
→ **VISUAL_SMOKE_ALLOWED: YES**，进入 Stage B（RGB + 冻结 V-JEPA 2 smoke）。
Interaction I 冻结为 negative/inconclusive baseline；Direct/R1 为物理预测参考主干。
