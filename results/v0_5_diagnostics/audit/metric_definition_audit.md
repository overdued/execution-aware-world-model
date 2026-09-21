# Metric Definition Audit（V0.5 Task A）

## 数字 A：`|r_hat|_low ≈ 0.107`

- **Definition**: 预测的 mean absolute residual，pooled over 3 axes × H=40 steps
- **Source file**: `execution_wm/context_swap/diagnostics.py` → `residual_magnitude_calibration`
- **Code path**: 对 normal↔low cross pair 窗口（400 对子样本），用 c_low 预测 ê，取 `mean(|ê − u|)`
- **Dataset subset**: matched **probe** 窗口（probe_1/3/4，无 probe_2），normal↔low 两方向混合
- **Normalization**: 无（raw）。**Units**: 混合单位（vx/vy 为 m/s，wz 为 rad/s，直接平均）
- **per-window pooling over horizon**，不是 per-axis，不是 RMSE

## 数字 B：`|r|_low ≈ 0.076`

- **Definition**: 同上窗口上真实 residual 的 `mean(|e − u|)`，同样 pooled 3 axes × horizon
- **Source file**: 同上（同一函数同一批窗口）
- **Dataset subset**: 同一批 probe 窗口
- A 与 B **互为同口径**（同窗口同聚合），其比值（0.107/0.076 ≈ 1.40）在**该口径下**成立

## 数字 C：`r_vx ≈ 0.320 / r_vy ≈ 0.293 / r_wz ≈ 0.291`

- **Definition**: 真实 residual 的 **per-axis** mean |r_axis|
- **Source file**: `execution_wm/eval/quick_look.py`（residual 表）
- **Code path**: `mean(|residual[:, axis]|)` over **全部 timestep**
- **Dataset subset**: friction_low 的 **random episodes**（20s 随机命令，大量 transition）
- **Normalization**: 无。Units: m/s（vx/vy）、rad/s（wz）
- 复算确认（本脚本）：friction_low random episodes per-axis |r| =
  0.320 / 0.305 / 0.275 ✓

## 同口径复算（本次 audit，probe 窗口，per-axis）

c_low 窗口（predicted vs true）:

| axis | pred \|r̂\| | true \|r\| | ratio |
|---|---:|---:|---:|
| vx | 0.0974 | 0.0704 | 1.38 |
| vy | 0.1105 | 0.0672 | 1.64 |
| wz | 0.1118 | 0.0903 | 1.24 |
| pooled | 0.1066 | 0.0760 | 1.40 |

c_normal 窗口：pooled pred 0.0502 vs
true 0.0670
（ratio 0.75）

对照：friction_low **random** episodes pooled |r| = 0.300
（probe 窗口 pooled 真实 |r| ≈ 0.076 —— probe 命令温和，residual 天然更小）

## Are these quantities directly comparable?

**NO.**

Reason:
1. **subset 不同**：C 来自 random episodes（命令频繁跳变，transition 密集）；
   A/B 来自 probe 窗口（长恒定段）。residual 集中在 transition（见 Task D），
   random 的 |r| 天然数倍于 probe。
2. **聚合不同**：C 是 per-axis，A/B 是三轴混合 pooling（且混合 m/s 与 rad/s）。
3. A/B 是同口径的一对，二者之比（"40% overshoot"）在该口径内部自洽。

## "40% overshoot" 表述是否仍然成立？

**成立但需限定口径**：在 matched probe 窗口、pooled 3 轴上 c_low 预测幅值 / 真实幅值 ≈ 1.4。
per-axis 复算（上表）显示过冲主要在某些轴。更精确的校准描述见 Task C 的
bias / slope（per-axis、per-condition、test split 上）。
