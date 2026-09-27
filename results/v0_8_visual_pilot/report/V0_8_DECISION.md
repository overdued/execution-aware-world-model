# V0.8 DECISION — 冻结 V-JEPA 2 视觉预测 pilot

> 2026-09-27 · 证据详见 `V0_8_FULL_REPORT.md`；数值可复算（预测文件与 feature cache
> sha256 已校验，见 `manifests/hashes.json` 与 `predictions/feature_cache_manifest.json`）。

```text
V07_REPORT_RECONCILED: PASS
V07_PRIMARY_COVERAGE_STATUS: SUPPORTED
FROZEN_ENCODER_INTEGRATION: PASS
VISUAL_DATA_CAUSALITY: PASS
ACTION_DEPENDENT_VISUAL_SIGNAL: SUPPORTED
EXECUTION_CONDITIONING_INCREMENT: NOT_SUPPORTED
AUXILIARY_SUPERVISION_EFFECT: NOT_SUPPORTED
PHYSICAL_PREDICTION_RETENTION: ACCEPTABLE
STATISTICAL_SCOPE: PILOT
NEXT_MAIN_ACTION: C
```

## 证据（逐字段）

- **V07_REPORT_RECONCILED: PASS** — Stage A 完成：§4.1 与 §4.4 的 P1 均值差异
  已定位为统计口径（组级聚合 vs 窗级 pooling、纯度子集口径），只读旧缓存复算，
  未改旧结果与旧预注册；见 `V0_7_CLOSURE.md` / `V0_7_ERRATA.md` 及
  `metrics/v07_table_reconciliation*.csv`（commit `a049fb5`）。
- **V07_PRIMARY_COVERAGE_STATUS: SUPPORTED** — 复核确认 V0.7 主结论
  （R1 覆盖收益 6.3–9.4%、3/3 seed 同向、CI 不含零）在统一口径下成立，
  维持 SUPPORTED（PILOT 规模）。
- **FROZEN_ENCODER_INTEGRATION: PASS** — smoke gate 6/6：冻结校验（参数 sha、
  eval、无梯度）、特征可用性、1344 窗缓存 sha256 一致；训练中编码器全程冻结。
- **VISUAL_DATA_CAUSALITY: PASS** — T1–T4+T2b 全过；RGB 间隔恒 0.06s；
  分支 pre-action 状态逐位一致；origin 帧不读未来。
- **ACTION_DEPENDENT_VISUAL_SIGNAL: SUPPORTED** — 候选匹配 0.729–0.833
  （chance=1/3），候选间距离（0.60–0.93）2 倍于噪声底（~0.31），
  无 near-indistinguishable 组。冻结视觉表示+浅头足以分辨动作后果。
- **EXECUTION_CONDITIONING_INCREMENT: NOT_SUPPORTED** — 主终点相对改善
  +1.10 / −0.94 / +0.15%（s42/s43/s44），方向不一致、量级 ≪ 5% 门；
  CI 紧（±0.3–0.5%），非统计不足。机制解释：E 敏感性梯度弱
  （zero E 仅 +3.4–6.8%，oracle 无收益），模型主要从视觉上下文推断动态。
- **AUXILIARY_SUPERVISION_EFFECT: NOT_SUPPORTED** — V-AUX 相对 V-DIRECT
  一致劣化 −2.59 / −3.86 / −2.21%（3/3 CI 不含零），P0/P2 同向。
  辅助监督在当前设置下是负收益。
- **PHYSICAL_PREDICTION_RETENTION: ACCEPTABLE** — P0 latent（+0.5/+0.2/−0.0%）、
  P0 FDE（+1.4/+0.7/+0.4%）、P1 FDE（−1.9/−2.3/+2.0%）全部在 5% 容差内。
- **STATISTICAL_SCOPE: PILOT** — test 仅 2 场景 16 group，按预注册 §6
  不推断广泛场景总体。
- **NEXT_MAIN_ACTION: C（视觉表示缺少可利用的动作后果信号）** — 匹配证明信号
  *存在*于表示中，但回归级利用失败：显式 E 条件零增量、E 辅助监督负收益、
  全部 run 在 step 1000 早停（布局泛化瓶颈）。按任务书 C 的指向：定位相机
  配置/时域分辨率/表示瓶颈（如更密时域采样、多视角、或表示层诊断），
  **不堆大模型、不扩 backbone**。

## 不做的声明（本轮边界）

未做：Risk 场景、真机控制、V-JEPA 全参微调、在线纠正、backbone 扩展。
未删任何失败/负结果（辅助监督劣化、E 弱依赖均如实写入主结论）；
未为过门改指标；未按测试成绩选 checkpoint（全部 val 最优，实际为 step 1000）；
未修改 V0.7 旧结果与旧预注册。

> 后续备注（2026-09-27）：应用户要求，结果已推送至 GitHub 公开仓库
> `overdued/execution-aware-world-model` 的 `exp/cvpr-v08-visual-pilot-slim` 分支
> （大二进制未上传，血缘见 `BIG_FILES_LOCAL.md`）；上述实验期约束均未被违反。
