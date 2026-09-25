# V0.7 FULL REPORT — 结构化命令覆盖 × 动作交互建模

> 完成日期：2026-09-25 · 分支 `exp/cvpr-v07-composition`（worktree `~/cvpr_embed-v07`）
> 冻结协议：`prereg/PRE_REGISTRATION_V07.md`（采集前 commit `2ecdd01`）
> 偏离记录：`prereg/PRE_REGISTRATION_DEVIATIONS.md`（D1–D7）
> 本报告所有数值可由 `predictions/pred_cache.npz`（sha256 `839a3dfedb6b81bf…`，与
> `pred_cache_manifest.json` 一致）+ `metrics/*.csv` 复算；复现步骤见 `README_REPRODUCE.md`。

---

## 0. 一句话结论

**Q-data（数据覆盖）：SUPPORTED（pilot 规模）** —— 同预算下 R1（含 30 个二元组合 cell）
相对 R0（纯单轴）在 held-out pair cell（P1）主终点 FDE_xy 上改善 6.3–9.4%，3/3 seed
同向且 group CI 不含零。
**Q-model（交互架构）：INCONCLUSIVE 偏负** —— Interaction I 相对 Direct D 在 P1 主终点
未达到预注册改善门（+4.2 / −3.2 / +1.3%，2/3 seed CI 跨零），且在 P2 三轴外推
（EXPLORATORY）上显著更差（−29 ~ −36%），test_all 净 yaw 显著恶化（25–36%）。
**下一步（任务书选项 A）**：保留 Direct，数据覆盖路线进入小规模视觉预测/动作后果验证。

---

## 1. 正确性（CORRECTNESS: PASS）

| 检查 | 结果 |
|---|---|
| 轨迹积分单测 U1–U8（直线/纯旋转/恒转弯弧/非零初 yaw/零命令惯性/误差下界/yaw wrap/因果性） | **8/8 PASS** |
| 继承单测 T01–T08（配对完整性/特权标签不入输入/标签恒等式/整数 tick/FeatureSchema/split 血缘/fixed-lead 口径/因果输入） | **8/8 PASS** |
| 派生标签恒等式 `u_ref + r_label == e_label`（864 ep 全量） | max err **1.11e-16** |
| 时间网格一致性（整数 tick，50→20 Hz `hold=(5k)//2`） | max err **1.78e-15** |
| 原始四元数范数偏差（审计字段） | max 0.2999 —— 见 §6.3 说明 |
| checkpoint 选择 | 仅用 **val FDE_xy**，未看测试成绩 |
| 评价口径 | 所有汇总只读 `pred_cache.npz`；sha256 与 manifest 一致 |

评价期发现两处实现问题并已修复（详见 D7）：`postprocess_v07.py` 的 `enumerate`
解包顺序 bug（修复前无法生成 `main_axis_lead.csv`）；`stats_v07.py` 补上预要求的
高纯度 P1 口径。两者均不改变任何冻结项。

## 2. 数据现状（Stage 2，已于交接前校验）

- **864 episodes** = 24 groups × 3 摩擦（1.0/0.6/0.3）× 12 脚本（预算上限 900 内）。
- split：train 216(R0)+216(R1)｜val 72×3(P0/P1/P2)｜test 72×3(P0/P1/P2)。
- 终止：schedule_end 863 / terminated 1 —— **原地保留**，valid-prefix mask 处理，未删样本。
- 摩擦 readback 写入=读回精确（multiply combine）；`plan_hash 6957649b0c74da06`，
  `controller_hash 98a0ed5bb1bb5a7a`。
- R0/R1 每轴占空比精确相等（0.2778）；无法同时匹配的量已如实披露：zero slot 占比
  0.167 vs 0.375（`prereg/r0_vs_r1_marginals.json`）。
- 窗口：L=20 / H=40 / 每 episode 确定性 6 个 origin，无 RNG，全模型共用
  （manifest hash `9f3238863bad34b6`）。

## 3. 训练与公平性（Stage 3）

12 个主 run 全部完成（{D,I}×{R0,R1}×seed{42,43,44}），checkpoint 仅存 val FDE_xy 最优点。

| 公平性项 | 结果 |
|---|---|
| 参数量 D / I | 226,808 / 204,723，**比 0.903（±10% 内）** |
| 推理延迟（batch64） | D 0.0014 ms/sample，**I 0.0084 ms/sample（5.8×）** —— 披露为 caveat |
| s_E | 0.2464（R0 训练池拟合，四格共用，D1） |
| 位姿损失 | λ_xy = λ_ψ = 0（仅 L_E，四格一致，D6） |
| 神经训练总 GPU·h | **≈0.006**（上限 40）；全程 ≪ 150 GPU·h 上限 |

val FDE_xy（选择依据）：D/R0 0.0875–0.0886；D/R1 0.0801–0.0819；I/R0 0.0923–0.0966；
I/R1 0.0809–0.0839。I/R0 较差符合预期（交互项在 R0 无组合数据可学，只增加方差）。

## 4. 主结果（Stage 4，6 个独立 test group，group 级配对 bootstrap）

### 4.1 主终点：P1 held-out pair cells，FDE_xy 2 s（m）

| seed | E(D,R0) | E(D,R1) | E(I,R1) | Δ_data [CI] | rel | Δ_arch [CI] | rel |
|---|---|---|---|---|---|---|---|
| 42 | 0.0932 | 0.0849 | 0.0814 | 0.0083 [0.0059, 0.0106] | **+8.9%** | 0.0036 [0.0020, 0.0053] | +4.2% |
| 43 | 0.0925 | 0.0838 | 0.0865 | 0.0087 [0.0066, 0.0110] | **+9.4%** | −0.0027 [−0.0081, 0.0015] | −3.2% |
| 44 | 0.0899 | 0.0842 | 0.0831 | 0.0057 [0.0044, 0.0078] | **+6.3%** | 0.0011 [−0.0032, 0.0042] | +1.3% |

- **Δ_data：3/3 seed 同向、CI 均不含零、相对改善均 ≥5% → 过预注册改善门。**
- **Δ_arch：1/3 seed 为正且仅 +4.2%（<5% 门），2/3 seed CI 跨零 → 不过门，INCONCLUSIVE。**

### 4.2 主比较第二版：P1 高纯度窗口（purity ≥ 0.8，135/432 窗，每 group 均有覆盖）

| seed | E(D,R0) | E(D,R1) | E(I,R1) | Δ_data [CI] | Δ_arch [CI] |
|---|---|---|---|---|---|
| 42 | 0.0368 | 0.0346 | 0.0301 | 0.0022 [−0.0020, 0.0065] | 0.0045 [0.0002, 0.0088] |
| 43 | 0.0342 | 0.0325 | 0.0307 | 0.0018 [−0.0020, 0.0055] | 0.0018 [−0.0018, 0.0047] |
| 44 | 0.0342 | 0.0331 | 0.0317 | 0.0011 [−0.0039, 0.0061] | 0.0014 [−0.0043, 0.0074] |

高纯度窗口误差整体低约 60%（单 cell 内预测本就更容易）；**Δ_data 在此子集 CI 跨零
→ 数据覆盖收益集中在跨 cell 切换的混合窗口**（§6.2 讨论）。Δ_arch 同样跨零为主。

### 4.3 P0 seen-cell（成本检查）与 P2 triple（EXPLORATORY）

| split | Δ_data rel（FDE） | Δ_arch rel（FDE） | 备注 |
|---|---|---|---|
| P0 | +12.3 / +7.8 / +12.2%（CI 不含零） | +10.6 / +9.6 / +5.2%（s44 CI 跨零） | **seen-cell 无恶化**，成本门通过 |
| P2 | +7.7 / +10.5 / +9.2%（CI 不含零） | **−28.6 / −35.8 / −34.9%（CI 不含零）** | 交互模型三轴外推**显著更差** |

净 yaw error：P1 上 Δ_data +12~15%（CI 不含零）；**test_all 上 Δ_arch −25~−36%
（CI 不含零）——I/R1 的 yaw 显著恶化，按预注册要求单列报告，未隐藏。**

### 4.4 与简单基线对照（test_P1，seed 均值）

| 模型 | FDE_xy | ADE_xy | net yaw |
|---|---|---|---|
| persistence | 0.5601 | 0.3158 | 0.3462 |
| command-copy | 0.1507 | 0.1084 | 0.0756 |
| ridge_R1 | 0.0967 | 0.0658 | 0.0780 |
| **D_R1** | **0.0809** | **0.0482** | **0.0757** |
| I_R1 | 0.0801 | 0.0484 | 0.0820 |

所有神经模型显著优于 command-copy / persistence / ridge；D/R1 与 I/R1 在 P1 上
统计不可区分（§4.1）。

## 5. 统计口径与范围（STATISTICAL_SCOPE: PILOT）

- 统计单元 = 独立 anchor group（6 个 test group），先 group 内平均再跨 group 配对
  bootstrap（2000 次）；3 个 seed 分别报告，未伪装成额外物理环境。
- **6 个 test group 撑不起强外推主张**；所有结论为 pilot 级。
- 所有模型在同一固定窗口集上评价（`same_window_set_for_all_models = True`）；
  终止样本未删（1 条）；窗口纯度分布逐窗落盘（`manifests/windows.csv`）。

## 6. 负结果与 caveats（全部保留，未删减）

1. **交互架构在 P2 triple 上显著更差**（−29~−36%，3/3 seed CI 不含零）。推测：
   零锚定交互项在训练只见过 ≤2 轴非零，三轴同时输入时多项外推失控。这是本轮
   最重要的负结果——"交互项有帮助"不能外推到未见的更高阶组合。
2. **I/R1 的净 yaw 在 test_all 显著恶化**（25–36%），尽管其 P0 yaw 改善。
   yaw 结论方向随 split 翻转，不支持"I 改善姿态预测"。
3. **Δ_data 的效应集中在跨 cell 切换窗口**：高纯度子集（purity≥0.8）中 CI 跨零。
   即 R1 的收益主要来自"命令切换附近的组合上下文"，而非单 cell 稳态段。
4. 原始四元数经抗混叠重采样后范数偏差 max 0.2999（过滤在合法化之前）——
   继承自 V0.6.1 已审 pipeline；合法化（符号连续+归一化）后的姿态才进入
   标签/yaw，原始值以 `lbra_*` 保留可复算。
5. I 的推理延迟 5.8× D（参数量相当，D3 批量化后）；若未来部署需计入。
6. R0/R1 的 zero slot 占比不同（0.167 vs 0.375）是占空比匹配的必然结果，已预披露。
7. 2 s 预测窗的 `future_cell_purity` 中位数仅 0.43–0.73（逐 split 见
   `metrics/termination_and_masks.csv`）——P1 "全部窗口"版主结果包含大量
   跨 cell 窗口，故同时给出 §4.2 高纯度版。

## 7. 交付物核对（任务书 §10）

```
prereg/ 5 文件 ✓   audit/ 3 文件 ✓   manifests/ 4 文件 ✓   metrics/ 6 文件 ✓
predictions/ pred_cache.npz + manifest(sha256 校验通过) ✓
config/runs_config.json（12 run 全配置）✓   checkpoints/ 12×best.pt ✓
train_logs/ 12×(train_log+history) ✓   code_snapshot/ + git_diff_stat ✓
figures/ 4 图 ×(PNG+PDF) ✓   report/ 本文件 + V0_7_DECISION.md ✓   README_REPRODUCE.md ✓
```

## 8. 决策字段

见 `V0_7_DECISION.md`（格式逐字按任务书 §10）。
