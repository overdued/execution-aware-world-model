# §4 额外审计（不新增研究模块）

## 1. 摩擦：**combine mode 实为 multiply，V0.6 记录的 average 是错的**

引擎级读回（`audit/usd_material_attrs.json`，直接读 USD prim 属性，非代码常量）：

```
/World/ground/terrain/physicsMaterial:
  physics:staticFriction            = 1.0
  physics:dynamicFriction           = 1.0
  physics:restitution               = 0.0
  physxMaterial:frictionCombineMode     = "multiply"     <-- 关键
  physxMaterial:restitutionCombineMode  = "multiply"
  physxMaterial:dampingCombineMode      = "average"
```

robot material 写入=读回精确（`audit/friction_material.json`，27 shapes/env 全部一致），
且 **static ≠ dynamic**：

| 档位 | written_static | written_dynamic | 有效 static（×ground 1.0） | 有效 dynamic |
|---|---:|---:|---:|---:|
| normal | 1.0 | 0.70 | 1.0 | 0.70 |
| friction_mid | 0.6 | 0.42 | 0.6 | 0.42 |
| friction_low | 0.3 | 0.21 | 0.3 | 0.21 |

**修正 V0.6 偏差记录 #3**：V0.6 写"combine=average → 有效摩擦 ≈ (写入+1)/2
= 1.0/0.8/0.65"是**错误**的（既是 average 假设错误，也把 static/dynamic 混成一个数）。
按 multiply 且 ground=1.0，有效值**等于写入值**，工况分离度比 V0.6 记录**更大**
（low 档 static 0.3 而非 0.65）。
命名规则：全文只用 `written_static / written_dynamic`，不使用"真实摩擦"。

## 2. 四元数：逐分量滤波产生非法姿态（复核报告 §4 已指出，本版修复）

`lb_base_orientation` 由 resample_poly 逐分量重采样得到，**不保证单位范数**：

- 全量 240 ep / 23204 点：`max |‖q‖−1| = 0.5000`，平均 0.00798；
  **4541 / 23204 点（19.6%）违反 >1e-3**；符号翻转 0 次。
- 未滤波的 `lbra_base_orientation` 范数偏差 1.4e-07（原始数据本身合法）。

修复：`legalize_quaternion()` = 符号连续 + 归一化，合法化后 `max|‖q‖−1| = 3.3e-16`。
产物新增键：`lb_base_orientation_legal` / `lb_yaw` / `lbra_base_orientation_legal` /
`lbra_yaw`；原始四元数保留未删。

**姿态相关指标**：完整相对位姿轨迹 **NOT_RUN**（体速度+姿态联合积分未预注册）。
可合法获得的部分已计算：净 yaw = ∫wz dt（见 `metrics/relative_yaw_summary.csv`）。

## 3. 条件重复：rep1 不是同一条件的随机重复

`collect_support_query.py:306-308`，rep=1 走 `jittered(segs, rng)`，把每段时长
改 ±0.1s（seed = anchor+5000）。因此 rep0/rep1 **不是同一 (state, future command) 的
两次随机结果**，不能用于估计 `Var(E | state, U)` 或"不可约误差下界"。
本轮未使用 rep 差做噪声估计。需要噪声下界时必须另做"完全同条件、明确随机源"的重复。

## 4. ARX：准确命名

`RidgeMultiOutput`（`execution_wm/validity_v061/eval_v061.py`）：输入 = 当前
proprio(40) + 完整 future command(120)，输出 = residual(120)，ridge λ=1，
**train-only 拟合**。这是 direct multi-output ridge 回归，**不是**标准多滞后 ARX
的全阶搜索。所有表与报告中一律用该名称，不简称为"ARX"。

## 5. Oracle（M3）：分支确实响应 friction，但增益为零

`audit/oracle_branch_response.json`（对 held-anchor split，扰动 friction 输入）：

| seed | ‖W‖ | Σ|W| | Σ|b| | Δr̂ (fric 0.3 vs 1.0) max | Δr̂ mean | r̂ 量级(mean abs) |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 0.143 | 0.355 | 0.021 | 0.0115 | 0.0022 | 0.0743 |
| 43 | 0.153 | 0.372 | 0.014 | 0.0142 | 0.0026 | 0.0737 |
| 44 | 0.117 | 0.305 | 0.011 | 0.0080 | 0.0019 | 0.0724 |

结论：**分支确实响应**（零初始化 W 训练后非零，扰动 friction 使 r̂ 变化约为 r̂ 量级的
3%），但 M3−M1 ≈ −0.0001 → **仍无增益**。
替代解释（不得推导"同类数据更多无效"）：
1. 注入方式弱——零初始化 `Linear(1,8)` 加到 8 维 c 上，容量与表达力都极小；
2. learned c 可能已经吃掉了 friction 可提供的信息（但见 B/C 分 split 结果并不支持这一点）；
3. 残差中占主导的成分对 friction 不敏感（控制器瞬态、接触噪声）；
4. 训练预算（≤100 epoch，patience 15）内该分支未被有效使用。
→ 记为 **INCONCLUSIVE**，不作为上界，也不作为"扩数据无用"的证据。

## 6. 源码可复现性补齐（复核报告 §7 的要求）

已补入 `code_snapshot/`：`MODEL_REGISTRY` 与 `train_execution.py`、全部模型类
（`baseline_action` / `baseline_direct` / `execution_context_model` /
`execution_predictor` / `context_encoder`）、`dataset.py`、`schema.py`、
12 个 `train_log.json`、12 个 `best.pt`（11 MB）及其 SHA256（`config/` 与
`report/BUG_IMPACT_MATRIX.md`）。预测示例不再只给 normal：见
`raw/representative_full_episodes/`（normal/mid/low × Q1–Q4）。
