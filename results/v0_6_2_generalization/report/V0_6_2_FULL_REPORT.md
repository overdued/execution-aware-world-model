# V0_6_2_FULL_REPORT — Command Generalization & Context Disentanglement Audit

> 2026-09-23 · 依据 `tasks/v0.6.2.md` · 范围限定 CVPR 主线 **Compositional Execution-Aware
> World Modeling**（未开展 Risk-Gated Extreme Execution Control 方向）
> 判定见 `report/V0_6_2_DECISION.md`
> 数据/预测：复用 V0.6.1 的 240 ep 修复标签与单次推理缓存（sha256 校验，未重新推理）
> 新训练：仅 Task 5 的 12 run（0.0028 GPU·h）

## 0. 本轮一句话

V0.6.1 的现象（seen family 有效、unseen family 退化）被定位为**结构性命令 OOD**：
train 从不含"多轴同时非零"的命令组合，而 B/C 100% 含之。加性 primitive 分解**修不好**它
（相对改善 <1%，CI 跨 0）；显式 context **不是**必要的工况载体（M0 的 fast state 同样
可解码工况，且 c_t 反而丢掉了 anchor 身份）；去掉 privileged 输入**几乎无损**。
据此推荐 **D：保留 direct EA-WM 主干，去掉"显式 context 是主要贡献"的定位**。

## 1. 环境与范围

| 项 | 值 |
|---|---|
| 数据 | `/media/hdd1/yuhang/datasets/execution_wm/v0_6_1`（V0.6.1 重派生，240 ep） |
| 窗口 | 与 V0.6.1 **完全相同**的固定 WindowManifest（`manifest_hash=245f1e23c43b1c48`） |
| 预测 | `results/v0_6_1_correctness/predictions/pred_cache.npz`（sha256 校验后只读） |
| 新训练 | `checkpoints/execution_wm/v0_6_2/{M0,M1}_{privileged,deployable}_s{42,43,44}` |
| 本轮禁止项 | 全部遵守：无 extreme friction sweep / risk router / failure gating / MPC-CEM /
disaster benchmark / HumanoidVLN / V-JEPA 训练 / 真机 correction |

## 2. Task 1 — Command-space Novelty Audit

方法：future command `U[t:t+H]` flatten 为 120 维；normalization / PCA / 正则化协方差
**只用 train** 拟合（PCA 数值秩 103/120，90% 方差 8 维，99% 方差 39 维）。
距离在白化空间用正则化特征值（λ + 1e-4·λmax）—— 未正则化时近零奇异值会把距离放大到 1e7
（本轮修正）。

### 2.1 结构 vs 距离

| split | 窗口 | 多轴同时非零窗口占比 | vx∧wz 共激活时间占比 | 不同命令向量数 | 其中含多轴 | Mahalanobis | 5-PC 重建误差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 720 | **0.0** | **0.0** | 119 | **0** | 8.6 | 3.95 |
| val | 144 | 0.0 | 0.0 | 46 | 0 | 15.3 | 4.17 |
| A | 288 | 0.0 | 0.0 | 68 | 0 | 12.9 | 3.99 |
| **B** | 240 | **1.0** | **0.865** | 36 | **36** | **57.6** | **8.76** |
| **C** | 96 | **1.0** | **0.863** | 22 | **22** | **59.0** | **8.79** |
| supp_AQ5xQ4 | 48 | 1.0 | 0.856 | 15 | 15 | 59.8 | 8.93 |

命令值域并无越界（B/C 的 vx/wz ∈ [−0.6, 0.6] ⊂ train 的 [−0.8, 0.8]）→ **新颖性是组合结构，
不是数值范围**。

### 2.2 error vs novelty

`metrics/error_vs_novelty.csv`（每 split × 模型 × 轴 × 5 种 novelty 度量）。
**族内** Spearman 相关：novelty（std 空间 NN）对 command-copy 为正（+0.09~+0.26），
对模型多为负（−0.42~+0.02）——符号随模型翻转，且量级小。
→ 结论：**B/C 失效是族级的，不是"越远越差"的渐变**。

输出：`metrics/command_novelty.csv`（每个窗口的 5 种 novelty + 结构量）、
`metrics/error_vs_novelty.csv`、`metrics/command_space_structure.csv`、
`figures/action_space_pca.png`（train-fitted PCA，A 落在 train 支持上、B/C 分离）、
`figures/F2_novelty_structure.png`。

## 3. Task 2 — Primitive Composition Baseline

    r_hat[:,k,a] = g_a(state, u[t+k,a])        a ∈ {x,y,w}

每轴一个 2 层 MLP（128-128），输入 = `[state(40), u_a(H)]`，**三轴相加、无 interaction**；
只用 train 拟合（3 seeds，early-stop 仅用 val）；另跑 ridge 变体做稳健性检查。
评估在 A/B/C **同一窗口**进行。

| split | ccopy | ridge | M0 | M1 | M2 | **composition** |
|---|---:|---:|---:|---:|---:|---:|
| A（逐轴均值 MAE） | 0.0848 | 0.0626 | 0.0590 | 0.0577 | 0.0615 | 0.0610 |
| B | 0.1625 | 0.1634 | 0.1524 | 0.1552 | 0.1554 | **0.1513** |
| C | 0.1659 | 0.1691 | 0.1598 | 0.1626 | 0.1615 | **0.1592** |

配对 episode-cluster bootstrap（composition − 对手）：
- B vs M0：−0.0011 [−0.0029, **+0.0007**]（跨 0，相对 −0.75%）
- C vs M0：−0.0006 [−0.0042, **+0.0037**]（跨 0，相对 −0.37%）
- B vs ccopy：−0.0112 [−0.0155,−0.0068]（不跨 0，相对 −6.9%）
- C vs ccopy：−0.0067 [−0.0135,+0.0003]（跨 0，相对 −4.1%）

判定阈值（预写在脚本内）：SUPPORTED 需同时满足"两个 split 均最优 + CI 不跨 0 + 相对改善 ≥5%"。
→ **ACTION_COMPOSITION_SIGNAL = PARTIAL**（名义最优但未达"明显改善"）。
ridge 变体更差（B 0.3244 / C 0.3247），说明并非"线性可分解"。

**诚实表述**：B/C 上所有方法的 vx 误差（0.15–0.27）都比 A 上（0.038–0.066）差 3–5 倍；
加性分解没有把任何方法拉回 A 的水平。输出：`metrics/composition_baseline.csv`、
`metrics/action_composition_signal.json`、`figures/F3_composition.png`。

## 4. Task 3 — Context Disentanglement Probes

固定 M1 checkpoint（3 seeds 平均表示；另单列 seed42 的 context-GRU 隐状态）。
线性探针**只用 train** 拟合：分类 = 自实现 L2 one-vs-rest 逻辑回归；
回归 = 闭式 ridge。不依赖 sklearn。

| 表示（维度） | A 摩擦工况 (chance .333) | C 命令族 (chance .333) | D anchor id (5 折 CV, chance .2) |
|---|---:|---:|---:|
| **c_t（8）** | 0.750 | 0.611 | **0.32** |
| M1 state latent（64） | 0.771 | 0.639 | 0.31 |
| M1 context-GRU hidden（128） | 0.858 | 0.646 | **0.69** |
| M0 fast state GRU（128） | 0.854 | 0.663 | 0.61 |
| raw proprio 末帧（40） | 0.608 | 0.576 | 0.21 |

（完整含 train/val/B/C 的表：`metrics/context_probe_results.csv`）

当前速度回归（B split，目标 std vx = 0.220）：identity 参照 R²=1.0；
RMSE：raw 0.044 / M0 fast state 0.144 / M1 state latent 0.168 / **c_t 0.190** /
M1 GRU 0.198。→ R² 在族内小方差下会误导，故同时报 RMSE（`..._rmse_mixed` 列）。

**解读**：
1. c_t 是**有损的 8 维瓶颈**：相对其父 GRU，工况可解码性保留 0.75/0.858 ≈ 87%，
   而 anchor 身份只保留 0.32/0.69 ≈ 46%，当前速度 RMSE 从 0.198 变差到 0.190（略好）——
   即它**优先保留工况、丢弃身份**，符合"dynamics context"而非"state embedding"。
2. 但**工况信息并非 c_t 独有**：M0 的 fast state 同样 0.854/0.800，
   所以把工况编码当作 context 的独特贡献不成立。
3. 命令族本身在 c_t 里只有 0.611（A）——远低于工况，说明 c_t 更多反映"当前动力学状态"
   而非"未来计划"。
4. 任务族探针只在 train 的 Q1–Q3 上训练，Q4 窗口无对应类 → 表中 B/C 记为 NaN；
   Q4 的预测熵已写入 CSV（`*_Q4_pred_entropy`）作为 OOD 指标。

## 5. Task 4 — Full Trajectory Consequence（SE(2)）

**不再把 body wz 直接当 world yaw rate**。三条措施：
1. 真值 yaw 一律取**合法化四元数**（V0.6.1 的 `lb_yaw`）；
2. 两条 rollout 口径：(a) velocity-only（用真值 yaw 积分预测体速度，隔离速度误差）；
   (b) full SE(2)（用 `ŷaw = yaw₀ + ∫wz_pred dt`，从真值初值起）；
3. **planar 合法性核验**（而非假设）：tilt 中位数 2.4–2.8°，p95 3.8–4.4°（阈值 10°，
   **全部窗口**合法）；|∫wz dt − Δyaw_quat| 均值 0.008–0.017 rad、p95 0.022–0.027 rad。

@2.0s（`metrics/trajectory_consequence.csv`，含 0.5/1.0/2.0s 三档）：

| split | 量 | ccopy | ridge | M0 | M1 | M2 |
|---|---|---:|---:|---:|---:|---:|
| A | inst MAE vx | 0.0655 | 0.0412 | 0.0393 | **0.0382** | 0.0433 |
| A | **traj Δxy (m)** | 0.1014 | 0.0591 | 0.0399 | **0.0389** | 0.0538 |
| A | net yaw (rad) | 0.0455 | 0.0380 | **0.0339** | 0.0386 | 0.0398 |
| B | inst MAE vx | 0.1891 | 0.1843 | 0.1878 | 0.1764 | **0.1699** |
| B | **traj Δxy (m)** | 0.2811 | 0.1939 | 0.1794 | 0.1734 | **0.1679** |
| B | net yaw (rad) | 0.1098 | 0.1095 | 0.1128 | 0.1248 | **0.1044** |
| C | **traj Δxy (m)** | 0.2884 | 0.1922 | 0.1771 | 0.1729 | **0.1682** |
| C | net yaw (rad) | 0.1083 | 0.0987 | **0.0961** | 0.1157 | 0.1012 |

**瞬时 vs 积分**：A 上二者排序一致；B/C 上模型在轨迹层面稳定优于 ccopy 约 40%
（0.168–0.179 vs 0.281/0.288），尽管单轴瞬时指标上 ccopy 有时更好（vy@2s）。
→ "瞬时改善"与"轨迹改善"**部分**传递；ccopy 的瞬时优势在轨迹层面不成立。
净 yaw 上无模型稳定优于 ccopy。
图：`figures/F5_inst_vs_traj.png`。

## 6. Task 5 — Deployable-input Ablation（secondary）

同 split / 同 window / 同协议训练 M0 与 M1 × 3 seeds × 2 组输入：
- **privileged**：现有 40 维（含 simulator GT body 速度/角速度）
- **deployable**：按 schema 字段名置零 `base_linear_velocity_body` 与
  `base_angular_velocity`；保留 projected_gravity / IMU / joint pos+vel / contact。
  schema 中**没有**可部署 odom → 不伪造替代，manifest 记 `deployable_substitute: none`。
  未加 condition classifier、未使用 GT friction。

**确定性核对**：privileged 重训结果与 V0.6.1 逐位一致（M0/s42 val=0.005528、
M1/s42 val=0.005440）→ 代码路径与随机性可控。

| split | model | priv vx@2s | depl vx@2s | gap vx@2s | gap% vx | gap% vy@2s | gap% wz@2s | gap% traj |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | M0 | 0.0393 | 0.0399 | +0.0006 | +1.56% | −2.88% | −0.18% | +0.88% |
| A | M1 | 0.0382 | 0.0380 | −0.0002 | −0.47% | −1.25% | −0.03% | −2.13% |
| B | M0 | 0.1878 | 0.1845 | −0.0033 | −1.74% | +0.25% | +3.71% | +1.37% |
| B | M1 | 0.1764 | 0.1760 | −0.0004 | −0.22% | +7.61% | +0.70% | +0.74% |
| C | M0 | 0.1748 | 0.1717 | −0.0032 | −1.82% | +0.46% | +4.14% | +1.45% |
| C | M1 | 0.1627 | 0.1587 | −0.0040 | −2.43% | **+8.98%** | −0.38% | +0.17% |

配对 bootstrap（`metrics/deployable_gap_bootstrap.csv`）：混合单位 MAE 的 gap
+0.00013 ~ +0.00104，CI 多数不跨 0 → **统计上非零、实践上可忽略**
（最大相对影响 M1/C +0.6%）。最大单点效应 M1/C 的 vy@2s：0.0605→0.0660（+0.0054 m/s）。

→ **结论**：现有全部结论**不是** privileged 状态的产物；同时说明模型并未真正依赖
GT 速度通道（GRU 从 IMU/joint 历史恢复了所需信息）。输出：`metrics/deployable_gap.csv`、
`metrics/deployable_metrics.csv`、`figures/F6_deployable_gap.png`。

## 7. 负结果与未执行项（如实记录）

1. **加性 primitive 分解未能改善 Q4**（相对 <1%，CI 跨 0）——瓶颈不在"缺少加法结构"。
2. **M1 从未在任何公平设定的单元中胜出**（27 单元 0 胜）。
3. **c_t 的工况可解码性不独特**（M0 fast state 几乎相同）。
4. **B/C 上模型仍比 A 上差 3–5 倍**，没有任何本轮方法缩小这一差距。
5. **净 yaw 上无模型稳定优于 command-copy**。
6. 探针 C（命令族）在 Q4 上无对应类，只能报预测熵，不能报准确率。
7. anchor 探针只在 train 内 5 折 CV（train 只有 AQ0–4），**不是**跨 anchor 泛化结论。
8. 未执行：新数据采集、架构改动（明确禁止）、V-JEPA、correction、Risk-Gated 线。

## 8. 可复算清单

| 内容 | 文件 |
|---|---|
| 源码（本轮全部新增） | `code_snapshot/execution_wm/validity_v062/`（common / task1-5 / 2 个 fig 脚本） |
| 预测来源 | V0.6.1 `pred_cache.npz`（sha256 记录，未重算） |
| 新 checkpoints（12） | `checkpoints/{M0,M1}_{privileged,deployable}_s{42,43,44}_best.pt` + train_log |
| Task1 | `metrics/command_novelty.csv`、`error_vs_novelty.csv`、`command_space_structure.csv`、`command_pca_meta.json` |
| Task2 | `metrics/composition_baseline.csv`、`action_composition_signal.json` |
| Task3 | `metrics/context_probe_results.csv`、`context_probe_meta.json` |
| Task4 | `metrics/trajectory_consequence.csv`、`planar_validity.json` |
| Task5 | `metrics/deployable_gap.csv`、`deployable_gap_bootstrap.csv`、`deployable_metrics.csv`、`deployable_train_logs.json` |
| 图 | `figures/`（7 张） |
| 判定 | `report/V0_6_2_DECISION.md`（本文件为全记录） |

## 9. 与上一轮的关系

V0.6.1 修复了标签恒等式、support 配对、时间刻度、schema、split 与 lead 度量，得到
"seen family 有效 / unseen family 退化"这一现象。V0.6.2 只做一件事：解释这个现象。
结论是**结构性的命令组合缺失**，且**显式 context 不是解**（也不是必要的工况载体）。
本轮结束，等待人工决定 V0.7 的架构与数据采集。
