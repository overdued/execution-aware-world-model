# V0 实验总结 —— Execution-Aware World Model 第一阶段

> 2026-09-21。按 `first_work.md` §19 执行顺序完成 Step 1–12（sanity 规模：~100 随机 episodes）。
> 数据：`/media/hdd1/yuhang/datasets/execution_wm/v0`；模型与图：`/media/hdd1/yuhang/checkpoints/execution_wm/v0`。

## 1. 数据集（Step 1–4）

- **163 episodes**（123 random + 40 probe），9 个物理条件，20 Hz 输出，50 Hz 控制
- random 条件分布：normal 30 / friction_mid 11 / friction_low 10 / friction_vlow 11(OOD) /
  actuator_mid 10 / actuator_low 11(OOD) / disturbance 23
  （部分条件 +1 是单步内多 env 同时收尾的良性过采样）
- probe：4 条固定命令 × {normal, low_friction, degraded_actuator} × 3–5 repeats（matched commands）
- 扰动实现：摩擦直接写 PhysX material buffer；执行器缩 DCMotor effort_limit；外力 30–60N 随机方向 0.5–1s 重采样

### 采集期修复的关键 bug

1. **失控采集**（commit bc219cc）：`episode_length_s=25s` 与 20.4s 采集 schedule 错位，
   每个 saved episode 后 4.6s env 内建 timeout 触发 done → 短 episode 丢弃+补目标 →
   `finished`/`total_target` 同步增长近似死循环（friction_mid 曾产出 578/10）。
   修复：`episode_length_s=1e6`。
2. WindowDataset 跳过短 episode 后索引错位（IndexError）。
3. metadata 含 None（random ep 的 probe_name）导致 default_collate TypeError → `window_collate`。

## 2. Sanity checks（Step 5–6，9 项全过）

| 检查 | 结果 |
|---|---|
| residual == execution − command | PASS（atol 1e-4） |
| cmd/actual 同 frame | vx corr = **0.89** |
| **friction 造成可测 mismatch** | normal \|r_vx\|=0.1425 → low=0.3195，**+124%**（要求 >20%） |
| contact/时间戳/边界/metadata/NaN | PASS（摔倒 episode 的 contact 阈值放宽到 0.05） |

quick_look residual 表（|r| 均值）显示**随扰动单调增大**：

| 条件 | r_vx | r_vy | r_wz |
|---|---|---|---|
| normal | 0.143 | 0.140 | 0.162 |
| friction_mid (0.6) | 0.197 | 0.181 | 0.172 |
| friction_low (0.3) | 0.320 | 0.293 | 0.291 |
| friction_vlow (0.15) | 0.413 | 0.386 | 0.358 |
| actuator_mid (0.7) | 0.143 | 0.149 | 0.169 |
| actuator_low (0.5) | 0.183 | 0.219 | 0.265 |
| disturbance | 0.366 | 0.352 | 0.356 |

## 3. 三模型对比（Step 7–10，residual MAE ↓）

| model | test_id | test_ood | 备注 |
|---|---|---|---|
| action_only (baseline) | 0.1433 | 0.2294 | 只看未来命令 |
| direct | 0.1291 (−10%) | **0.1997** (−13%) | history→residual |
| context (c∈R⁸) | **0.1278** (−11%) | 0.2103 (−8%) | history→context→residual |

早停轮次：action_only@29 / direct@51 / context@71（batch 256, lr 1e-3, Huber）。

**结论**：
1. **execution mismatch 可预测** —— 用 1s proprio+action 历史，ID 误差降 10–11%，OOD 降 8–13%。
2. OOD 下所有模型退化（0.13→0.20+），符合预期；residual 仍部分可预测。
3. context ≈ direct（ID 略优、OOD 略差）——68 个训练 episode 规模下 context 结构优势尚未显现，
   符合 spec 预期（"若 context 不胜出，先检查数据量与条件覆盖"）。

## 4. 图（Step 12，`checkpoints/execution_wm/v0/figures/`）

- **figA/figB**（cmd/actual/pred、residual true/pred，normal + friction_low）：
  friction_low 下预测 residual 准确跟踪尖峰与衰减形状（vx/vy 尤其好）
- **figC**：预测误差随摩擦降低增大
- **figD（关键图）**：context PCA 按摩擦着色 —— f=1.0 与 f≤0.3 几乎线性可分，
  **8 维 context 确实编码了物理条件**（执行器/外力着色见另两张）
- **figE（关键图）**：同一 vx=0.7 直行命令下，normal 准确跟踪、low_friction 慢加速+超调 0.85
  +起步侧滑 vy≈1.0、degraded_actuator 跟踪但频繁掉速 —— matched probe 直接证明
  execution 差异来自物理条件而非命令

## 5. 结论与下一步

第一阶段核心假设全部验证：**执行失配 (r_t = e_t − u_t) 可测、随扰动单调、可学习、
且 context 表示自发编码物理条件**。

按 spec §18，进入扩大规模阶段前的改进项：
1. **数据规模** → 1000–3000 episodes（当前 163）；context 模型是数据饥渴型，规模上去后
   再判定 context vs direct
2. actuator 扰动信号偏弱（actuator_mid 的 residual 几乎=normal），考虑更低 scale 或
   加 deadband/延迟模型
3. wz 分量预测明显差于 vx/vy（RMSE/MAE 比最大），可在 loss 里加权或单独诊断
4. OOD 泛化差距（direct 0.1997 vs context 0.2103）值得在规模化后复查；
   可考虑 uncertainty head（config 已留 `predict_uncertainty`）
