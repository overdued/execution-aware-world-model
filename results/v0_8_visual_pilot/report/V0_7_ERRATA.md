# V0.7 ERRATA — 报告表述勘误（不改旧预注册、不改旧数值）

> 2026-09-27 · 依据任务书 §A2。旧报告 `results/v0_7_composition/report/` 保持原样，
> 勘误以本文件为准。所有数值结论不变；只修正表述与解释边界。

## E1. Δ_arch 的 seed 方向计数写错

- 旧报告 §4.1 原文：「1/3 seed 为正且仅 +4.2%」。
- 正确：Δ_arch 点估计为 **+4.2% / −3.2% / +1.3%（seed 42/43/44），2/3 为正**；
  只有 seed 42 的 CI 不含零。不过门的事实不变（未达 ≥5%、2/3 seed CI 跨零），
  INCONCLUSIVE 判定不变。

## E2. 高纯度子集 CI 跨零 ≠ 证明"收益只来自切换窗口"

- 旧报告 §6 caveats 第 3 条的表述过强。
- 正确表述：高纯度子集（purity≥0.8）CI 跨零只说明**该子集上效应不显著**，不显著不等于
  无效应，更不等于证明效应只存在于切换窗口。
- 按任务书 §A3 的正式异质性分析（`metrics/v07_purity_heterogeneity.csv`）：
  d_het = d_low − d_high = +0.0087 / +0.0101 / +0.0069（seed 42/43/44），
  seed 42/43 的 group bootstrap CI 不含零、seed 44 跨零（[−0.0002, 0.0136]）。
  即**该纯度划分下存在效应异质性**（低纯度窗口中 Δ_data 更大），但两个子集的命令构成
  本身就不同（转向占比：高纯度 0–25% vs 低纯度 88–100%），**不能作因果归因**。

## E3. "神经模型显著优于所有简单基线"的表述过宽

- 旧报告 §4.4 原文：「所有神经模型显著优于 command-copy / persistence / ridge」。
- 正确：该结论仅对 FDE_xy / ADE_xy 成立。**P1 净 yaw：D/R1 = 0.0757 ≈
  command-copy = 0.0756**，二者在 yaw 上无差别（command-copy 的"预测 yaw=命令积分"
  在合法执行下本就接近真值）。不显著不等于等效，此处仅作事实更正。

## E4（补充说明，非错误）. termination_and_masks.csv 的终止计数口径

- 该表每行 `n_terminated_episodes=1` 是**全局**同一 episode（ep492, val/G13/low/sc00）
  被填入每行，并非每个数据集各有 1 条终止。terminated episode 只在 val；
  test 三个 split 为 0。详见 `audit/termination_truncation_check.json`。
