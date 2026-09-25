# V0.7 DECISION — 结构化命令覆盖 × 动作交互建模

> 2026-09-25 · 证据详见 `V0_7_FULL_REPORT.md`；数值可复算（pred_cache sha256 已校验）。

```text
CORRECTNESS: PASS
COVERAGE_CONTRIBUTION: SUPPORTED
ARCHITECTURE_INCREMENT: INCONCLUSIVE
TRAJECTORY_VALIDITY: PASS
NOMINAL_COST: ACCEPTABLE
STATISTICAL_SCOPE: PILOT
NEXT_MAIN_ACTION: A
```

## 证据（逐字段）

- **CORRECTNESS: PASS** — U1–U8、T01–T08 全 PASS；标签恒等式 1.11e-16；时间网格
  1.78e-15；checkpoint 只按 val FDE_xy 选择；评价只读 sha256 校验过的 pred_cache。
  评价期两处实现修复（postprocess 解包 bug、stats 补高纯度口径）已版本化并记入 D7，
  不触碰冻结项。
- **COVERAGE_CONTRIBUTION: SUPPORTED** — 主终点（P1 FDE_xy）：Δ_data 相对改善
  8.9% / 9.4% / 6.3%（seed 42/43/44），3/3 同向，group CI 均不含零，过预注册门
  （≥5% + CI 方向 + ≥2/3 seed）。P0、P2 上同向（7.7–12.3%）。**限定**：高纯度
  （purity≥0.8）子集上 CI 跨零，效应集中在跨 cell 切换窗口；仅 6 个 test group，
  PILOT 规模。
- **ARCHITECTURE_INCREMENT: INCONCLUSIVE** — P1 Δ_arch = +4.2 / −3.2 / +1.3%，
  未达 ≥5% 门，2/3 seed CI 跨零 → 按预注册规则写 INCONCLUSIVE。附加负证据：
  P2（EXPLORATORY）Δ_arch −28.6/−35.8/−34.9%（CI 不含零，交互模型三轴外推显著
  更差）；test_all 净 yaw 恶化 25–36%（已单列报告）。不支持"I 有稳定架构增量"。
- **TRAJECTORY_VALIDITY: PASS** — 中点法则积分单测全过（恒定转弯误差 2.6e-5 m，
  优于前向欧拉 13×）；预测侧 yaw 自回归、不读未来 GT；ORACLE_YAW 仅作诊断列；
  平面假设与 tilt 审计已在采集期通过（combo 实测 tilt p95 4.84°）。
- **NOMINAL_COST: ACCEPTABLE** — seen-cell（P0）FDE 无恶化（I/R1 反而 +5~11%）；
  参数量比 0.903 在 ±10% 内；I 推理延迟 5.8× 已披露（公平性 caveat，非门禁项）。
- **STATISTICAL_SCOPE: PILOT** — 6 个独立 test group、24 group 总量，按任务书
  §4.4 不自封 confirmatory。
- **NEXT_MAIN_ACTION: A（数据覆盖有效、架构无增量）** — 保留 Direct 作为主干，
  按任务书 §10-A 进入小规模视觉预测/动作后果验证（V-JEPA 2 接入属后续阶段，
  本轮不启动）。证据：Δ_data 过门（§4.1）；Δ_arch 不过门且 P2/yaw 为负（§4.3）；
  D/R1 已优于全部简单基线（§4.4）。

## 不做的声明（本轮边界）

未做：Risk 场景、极端摩擦扫描、路由器、真机、V-JEPA 训练、策略重训。
未把 R0/R1 收益归因于架构；未删任何失败/终止样本与负结果；未为过门改指标；
未按测试成绩选 checkpoint。
