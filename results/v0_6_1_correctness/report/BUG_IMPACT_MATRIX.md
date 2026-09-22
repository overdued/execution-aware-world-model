# BUG_IMPACT_MATRIX — V0.6.1 缺陷—影响—处置矩阵

> 2026-09-22 · 依据：`tasks/v0.6.1/01,02,03`；全部数字来自 `results/v0_6_1_correctness/`
> 复现判定见 `report/V0_6_1_DECISION.md` Q1。**未删除任何旧结果或失败样本。**

| # | 缺陷 | 复现证据 | 影响面 | 处置 | 残余风险 |
|---|---|---|---|---|---|
| P0-1 | `SupportWindowBank.sample_batch` 分别抽 hp/ha（可跨 episode/origin） | 原函数玩具重放：10000 次中 9859 次编号不一致（98.59%）；源码 `train_sq.py:120-125` | M2 train/val/test、M1 cross_support/diff_condition、bootstrap variants、由它们推出的"纠缠/迁移否定/正则"结论 | 重写为 `SupportBank` + `SupportRecord`：一次抽 `support_window_id` 决定全部模态；donor lineage 落盘 | 旧 donor ID 未保存，**无法追溯**每条旧样本抽到了谁（不伪造） |
| P0-2 | 标签恒等式被破坏：`e_used = S(u) + F(e−u) ≠ F(e)` | 全 240 ep：最大恒等式误差 **1.120208**，成立条数 **0/240**；`ep_00162` t=2.5s wz：cmd+lb_residual=1.885307 vs lb_execution=0.765099，差 1.120208 | 全部模型的**监督目标与评估真值** | 重建为 `e_label=F(e)` 独立生成 → `u_ref` 事件采样 → `r_label=e_label−u_ref`；恒等式全量 1.11e-16 | 旧标签可解释为"滤波后的跟踪误差"，但不能再称"物理 execution" |
| P1-1 | 浮点累加时间使 hold 在整 tick 对齐点取前一 tick | 全 240 ep 共 **8751** 个 grid 点 hold 索引不一致（均值 36.5/ep）；`ep_00162` 2.5s：旧取 tick 124（+0.8），正确 tick 125（−0.8） | 输入 history、`cmd_ref`、labels、residual 的**时间语义** | 全部改整数 tick（`hold=(5k)//2`），50→20 用有理数关系；单测 T04 | 真实异步传感器时间未伪造，将来接入需带 sample_age |
| P1-2 | persistence / k0 用 `hp[:,-1,:3]` 当 [vx,vy,wz]，实为 [vx,vy,**vz**] | `PROPRIO_KEYS` 顺序 + T05：[11,12,13,21,22,23]→正确 [11,12,**23**]，旧给 [11,12,**13**] | 基线 persistence、k0_continuity、所有"当前 execution"取用点 | `FeatureSchema` + `extract_execution_from_proprio()`（索引 [0,1,5]）；k0 改名 `first_step_error` 并记录 lead | 修正后 persistence **误差更高**（A: +0.0106，B: +0.0207，C: +0.0229）——即旧错误轴恰好"更准"，必须按正确轴报告 |
| P1-3 | held-anchor 未排除 Q4，与 held_family 混叠 | 旧 `held_anchor` 48 ep = Q1/Q2/Q3/Q4 各 12；Q4 占 384 窗中的 96；与 held_family 重叠 12 ep | "仅新 anchor"与"新命令族"两类考核被混为一项 | 拆成 A/B/C 三个互不重叠 population + `supp_AQ5xQ4` 单列；断言 family/anchor 白名单、交集为空、lineage 分离 | A/C 仍只有 **2 个 anchor group** → 探索性强度 |
| P1-4 | `mae_rows` 的 `err[:, :k].mean()` 实为 prefix 平均，非 fixed lead | T07 toy：仅最后一步非零时 lead=1.0 vs prefix=1/40 | 主指标口径、所有旧 MAE_*@Xs 的含义 | `fixed_lead_mae` 与 `prefix_mae` 分离；lead 索引由 timestamp 推导；主表逐轴 fixed lead，prefix 单列 | 旧 `MAE_all` 混 m/s 与 rad/s，仅作历史兼容；新增 train-only 归一化 scalar |
| B7 | 训练 seed 同时改变数据窗口（`SQWindowData(seed=args.seed)`） | 源码 `train_sq.py:159` | "3 seeds 排除优化问题"的说法不成立 | 固定 window manifest（无 RNG，`manifest_hash` 落盘）；seed 只影响初始化与打乱；训练 seed 与数据 seed 分离 | — |
| B7b | bootstrap 与 eval 各抽一次 donor、各自推理 | 复核报告 §2 已指出 M2 held_family 两脚本值不同（0.131673 vs 0.131624） | CI 不可复算 | 单次推理写 `pred_cache.npz`（sha256 记录），bootstrap 只读缓存；T09 断言 | — |
| §3 | policy 消费的命令与写入的命令差 1 control tick | 运行时插桩：3×199 tick `u_consumed[i]=u_requested[i−1]` 精确成立；事件处滞后恒为 1 tick | `u_requested` ≠ 实际驱动命令；执行滞后共 2 tick | 直接测量替代相关性推断；保存 `cmd_consumed`；**旧 raw 无需重采**（固定规则可恢复） | 插桩第一版记录 step 后 obs 得到"零滞后"假象，已修正并记录 |
| §4a | `lb_base_orientation` 非法（逐分量滤波破坏单位范数） | 23204 点中 4541 点（19.6%）\|‖q‖−1\|>1e-3，最大 0.5000 | 任何用四元数算姿态 GT 的下游 | `legalize_quaternion`（符号连续+归一化）后 3.3e-16；原始保留 | 姿态指标仅算合法子集；完整相对位姿 NOT_RUN |
| §4b | 摩擦 combine mode 记录错误（V0.6 写 average） | USD 引擎级读回 `physxMaterial:frictionCombineMode="multiply"` | 工况命名与有效摩擦换算 | 有效值=写入值；static/dynamic 分开；命名改为 written_static/written_dynamic | 工况分离度比 V0.6 记录**更大**（low static 0.3 而非 0.65） |
| §4c | rep1 改变命令时长，不能当条件重复 | `collect_support_query.py:306-308` `jittered()` | 任何用 rep 差估不可约误差的做法 | 明确记录；本轮未使用 rep 差做噪声估计 | 需要噪声下界须另做真重复 |
| §4d | 旧 "ARX" 命名不准确 | `fit_arx` 实为 ridge direct multi-output | 与标准 ARX 的混淆 | 更名 `RidgeMultiOutput`，报告中不再简称 ARX | — |
| §4e | M3 无增益被当作"瓶颈不在 context inference"的证据 | 复核报告 §9.2 | 结论过强 | 补测分支**确实响应** friction（Δr̂ ≈ r̂ 量级的 3%），但增益仍 ≈0 → 记 **INCONCLUSIVE** | 不得推导"同类数据更多无效" |

## 影响量化（240 episodes，全部保留）

| 指标 | 值 |
|---|---|
| 旧评估真值 vs 新真值：最大逐点差 | **1.120208**（vx 1.054 / vy 1.072 / wz 1.120） |
| 平均逐点差 | vx 0.0144 / vy 0.0118 / wz 0.0171 |
| 分解①：B3 浮点 hold 错位（max / mean） | 1.600 / 0.0090 |
| 分解②：B2 命令采样 vs 滤波 `S(u)−F(u)`（max / mean） | 0.480 / 0.0090 |
| 分解恒等式残差 | 7.9e-08（即两成分精确解释全部差异） |
| residual std 变化（新/旧） | vx ×1.042 / vy ×1.043 / wz ×1.076 |
| 未保留样本 | 0（含 1 条 terminated） |

## 未被影响（明确列出，避免过度声称）

- 240 条 50Hz 原始日志本身（`residual = execution − cmd` 在原始数据中**正确**）——因此只重派生，不重采；
- anchor 复原、摩擦写入=读回、split 的 anchor/family 划分依据（metadata）；
- 模型结构、训练协议、优化器与预算；
- 图/表所依赖的 metadata 字段。
