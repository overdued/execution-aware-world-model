# CVPR 主线 V0.7：结构化命令覆盖 × 动作交互建模
## Claude Code 可直接执行的任务书

版本：v1.0｜制定日期：2026-09-24  
研究问题：少量、有结构的组合动作数据能否改善未见组合预测？显式动作交互结构在相同数据下是否提供额外收益？  
状态：以下是新的实验设计，不是已经验证的结论。执行到本轮验收结束即停止。

---

## 0. 工作范围与现有证据

你正在继续 `execution-aware-world-model` 项目。先使用本机已有仓库，读取：

- `V0_6_2_DECISION.md`、`V0_6_2_FULL_REPORT.md`；
- V0.6.1 的 correctness patch、FeatureSchema、整数 tick 时间线、数据 manifest；
- 当前 M0 Direct、M1 Context、固定预测缓存和训练配置。

若报告不在根目录，在 `results/` 查找；不要假装读过不存在的文件。

已报告的事实：
1. V0.6.2 训练窗口没有多轴同时非零命令，B/C 测试窗口全部含这种组合；单轴数值范围并未越界。
2. 当前简单 primitive baseline 相对 M0 在 B/C 只改善约 0.75% / 0.37%，区间跨零。
3. 显式 8 维 context 没有提供稳定预测优势；Direct 隐状态也有工况可解码信息。
4. 报告中的轨迹位置误差优于 command-copy，但净 yaw、瞬时误差的结果并不一致。

以上支持“命令组合覆盖缺口值得研究”，不证明它是所有误差的唯一原因，不证明任意交互结构必胜。

本轮不做：Risk 场景、极端摩擦扫描、路由器、真机动作、MDA、LLM agent、HumanoidVLN 接入、V-JEPA 大训练、策略重训。
显式 `c_t` 不作为强制结构和必须胜出的主贡献。M0 为主干基线，M1 仅保留旧结果作参考。

---

## 1. 一轮只回答两个问题

**Q-data：** 总数据预算相同时，单轴数据 R0 与“单轴 + 部分二元组合”数据 R1 的区别有多大？

**Q-model：** 在完全相同的 R1 数据和预测任务上，动作因素交互模型是否优于 Direct？

采用 2×2 设计：

| 训练分布 | Direct D | Interaction I |
|---|---|---|
| R0：单轴 + 零命令 | D/R0 | I/R0 |
| R1：单轴 + 选定二元组合 | D/R1 | I/R1 |

每格 3 个训练 seed，共 12 个主要神经模型 run。
不要比较 I/R1 与 D/R0 后，将“新增数据 + 新结构”的全部收益归于结构。

定义误差降低（以固定测试集的轨迹位置误差为例）：

\[
\Delta_{data}=E(D,R0)-E(D,R1),\qquad
\Delta_{arch}=E(D,R1)-E(I,R1).
\]

二者分别报告。R0 上的交互项未被组合输入训练，不能期待它自动识别任意未观测耦合。

---

## 2. 独立工作区和可复现性

1. 建立 `exp/cvpr-v07-composition` 分支或独立 git worktree，不修改另一条 Risk worktree。
2. 新代码建议放 `execution_wm/composition_v07/`；新结果放 `results/v0_7_composition/`。
3. 数据目录由 `DATA_ROOT` 配置，默认新目录，不覆盖 v0/v061/v062。
4. 保留 git commit、配置哈希、数据哈希、模型哈希、窗口 manifest、依赖版本。
5. 不升级 Isaac/PyTorch/CUDA，不改已有 locomotion controller 权重，不修改正常控制增益以制造失败。
6. 不打印环境变量中的 token；不改写 git 历史；推送遵守现有授权。
7. 主实验全部 simulation-only；脚本默认 `allow_real_robot: false`，检测到硬件控制接口时中止。

---

## 3. Stage 0：短回归验收，不再无限期审计

先运行 V0.6.1 单测，再做最多 12 条 smoke。复核以下必要条件：

### 3.1 时间、指令与状态

明确记录：
- `u_requested`：当前请求的高层速度命令；
- `u_consumed`：低层 policy 实际读到的命令；
- `joint_command`：低层输出；
- `execution`：实际物理运动，不能叫“修正后的 command”。

已有 1 tick 消费滞后必须在当前版本重新短测确认。它可以作为被测系统的一部分保留，但不能一边修复控制时序、一边仍使用旧延迟标签。
模型主输入采用哪种命令必须冻结：建议使用实际部署可获得的 requested/planned command，并保留 consumed 字段供时序核查。

原始日志保留实际 physics/control tick。若 control=50 Hz、dataset=20 Hz，按整数/有理数关系对齐；不使用浮点累加相等判断。

输入的所有观测源时间必须 <= prediction origin。未来 command 只能来自在 origin 已经生成的候选计划，不能从未来反馈控制日志中偷取实际已执行命令。

### 3.2 标签

独立生成真实 execution label 后再计算 residual：

\[
e^{label}=F(e^{raw}),\quad u^{ref}=S(u),\quad r^{label}=e^{label}-u^{ref}.
\]

全量断言 `u_ref + r_label == e_label`，且评价真值直接读取独立的 `e_label`。
输入滤波必须因果。标签若使用居中滤波，应记录其时间支撑与边界有效性；不能把该滤波标签回填为当前输入。

相对位姿 GT 从原始真实位置、合法姿态重采样生成。四元数先统一符号、归一化，姿态插值使用 SLERP 或等价合法方法；不要逐分量滤波后假装保持旋转几何。

### 3.3 特征与复位

使用 FeatureSchema 提取 vx/vy/wz，不能写 `proprio[..., :3]`。
记录机器人、policy hidden state、动作队列、传感器 history、随机化状态的 reset。
摩擦必须读取机器人与地面的 static/dynamic 材质、combine mode 和绑定 prim；不沿用“默认 average/multiply”假设。

### 3.4 报告口径短核对

V0.6.2 文字中“仅 Task5 新训12 run”与 Task2 “MLP 三 seed”需补齐运行清单，但不要求因此重训旧实验。
核实 V0.6.2 additive baseline 是“每轴只输出本轴”，还是“每个动作因素输出完整3维响应后相加”；这不是同一种结构，不把旧负结果扩大为“所有加性/交互模型不可能有效”。

**停止条件：** 发现标签、因果输入、命令消费或 split 的 P0 错误时，输出 BLOCKED_CORRECTNESS，修复并重跑单测后才恢复。纯文字/统计口径问题记录即可，不无限扩大审计。

---

## 4. Stage 1：采集前预注册组合协议

生成 `PRE_REGISTRATION_V07.md`、`command_cells.json`、`split_plan.json`，先保存 commit，再采正式数据。

### 4.1 动作空间

\[
u=[u_x,u_y,u_\omega]=[v_x^{cmd},v_y^{cmd},\omega_z^{cmd}].
\]

每轴从已验证的 controller 合法范围选择两个非零幅值 a_i、b_i，建议以安全工作幅值的约 0.35、0.65 为初始候选，具体由 smoke 冻结。
每轴值集为 `{-b_i,-a_i,0,a_i,b_i}`。单位分别 m/s、m/s、rad/s。

必须再检查**组合**命令的合法性、加速度/跃度约束；各轴独立合法不等于三轴联合命令可执行。超出正常控制包络的单元标记 invalid，不纳入此论文线的主实验。

### 4.2 三种泛化，不混为一项

- P0 seen-cell control：训练见过的命令元组、全新 anchor/时序；
- P1 held-out pair cells：二元轴类型在训练出现过，但具体符号/幅值元组从未出现；每个单轴取值都在训练出现；
- P2 triple composition：三轴同时作用未训练；优先选择其三个二元投影都已在 R1 训练出现的三元组合。

P1 为主要结构泛化测试；P2 是更强的外推测试。不能把“某个相互作用完全没有观察过”与“观察过因素/局部交互但没观察过完整组合”混同。

程序化构造 train/val/test cell masks。二元类型 XY、XW、YW 都在 R1 训练有覆盖；在每类中留出不同 cells 给 val/test，禁止测试后换 masks。
一个可用起点是每类16个非零二元 cells 分成10 train / 3 val / 3 test，但必须检查符号平衡及 P2 可用投影；配置可在采集前调整，不硬套无法满足的数量。
所有零命令保留，以观测惯性与制动；不能假设零命令未来 execution 必为零。

### 4.3 排除其它显著 shift

为 R0/R1 匹配或报告：每轴值域与频数、命令切换次数、驻留时间、历史长度、初始运动状态、摩擦条件、回合长度和总训练窗口数。
R0 与 R1 的共激活结构必须不同，某些边际统计不可能同时完全匹配时，明确报告差异。可通过零命令时段匹配边际活跃率；不要改标签或声称“其它全部相同”。
不要让训练全是短驻留、测试全是长驻留。时序模板和数值组合分别留出，且记录其 ID。

### 4.4 独立 group 与预算

建议上限：24 个独立 anchor/session groups × 3 个常规摩擦工况 × 12 条脚本 = 864 episodes；加 smoke 后总数上限 900。
anchor 分为12 train / 6 val / 6 test。风险线另用新的 seed/session namespace。

训练 groups 的12条脚本可分6条 R0、6条 R1，形成等回合/等窗口预算。
验证与测试 groups 的脚本均衡覆盖 seen cells、held-out pairs、triples；生成器先列出精确数量表，不为了套公式空造样本。
每条回合建议12–20秒，必须覆盖1秒历史、最多2秒预测和滤波边界。实际有效时长与窗口数落盘。
每一 reset/group 的所有工况、分支、时间 jitter 版本均属于同一 split。

24 groups 仍是 pilot 规模；6 个独立 test groups 不足以宣称广泛跨环境泛化。bootstrap 区间宽就报告 INCONCLUSIVE。

### 4.5 摩擦与场景

只使用 V0.6.1 已读回验证的正常/中等/低摩擦，建议配置名 nominal/mid/low，不把写入值当有效值。
本轮场景为平地与少量普通布局，主变量是动作组合，不加极端坡度、外力、坏电机。

### 4.6 新测试集

V0.6系列所有数据视为开发数据，可用于复现和设置协议，不冒充新确认性 test。
新 test 在配置/模型选择结束前不能查具体误差并反复调参。代码错误修复须明确版本化，修后测试仍属探索性开发结果。

---

## 5. Stage 2：采集与数据契约

每条回合保存：

```text
track_id, data_version, controller_hash, scene_id,
anchor_group_id, episode_id, reset_seed, command_seed,
split, coverage_regime, command_cell_ids, family_ids,
physics_tick, control_tick, sim_time, wall_time,
u_requested, u_consumed, joint_command,
proprio_features, feature_roles, source_timestamps,
base_position_world, base_quaternion, execution_body,
contact, camera_metadata_if_present,
written_material, readback_material, combine_mode,
termination_reason, sensor_valid_masks
```

低层 locomotion policy 冻结、不开额外自适应训练。无 donor 检索，无 GT condition 输入。
物理 GT 允许作为标签、配对和审计信息；推理输入遵守 `input_roles.json`。沿用 V0.6.2 去 GT 通道版本作主要输入，核查 projected_gravity/IMU/contact 的来源，称为 simulated deployable-candidate，不宣称已经完成真机验证。

每个样本：

```text
history: [L,D]，严格过去/当前
planned_commands: [H,3]，origin 已知
future_execution: [H,3]，仅标签
future_relative_pose: [H,...]，仅标签
origin_tick / target_ticks / masks / lineage
```

必须原地保留失败与终止样本。预测指标用明确的 valid-prefix mask，同时报告终止率及不同模型的样本集合完全一致；不能因为模型预测不准就筛除窗口。

窗口 manifest 全部模型共用；训练 seed 不改变数据窗口。group 内窗口数限额，避免长回合主导。

---

## 6. Stage 3：模型，最多两个主架构

### 6.1 Direct D

复用正确的 M0 history encoder 和完整未来命令输入，预测 [H,3] execution/residual。
每个模型必须接收相同输入 h 和 U。直接模型可以隐式编码执行条件，不宣称它假设 command=execution。

### 6.2 Interaction I：一个最小候选，不强行堆模块

将三个未来动作序列分别编码。一个允许实现的定义是：

\[
\hat E=b(h)+\sum_i f_i(h,U_i)+\sum_{i<j}f_{ij}(h,U_i,U_j).
\]

每个 f 输出完整 `[H,3]`，不是只输出自己的轴；否则预先禁止了真实 cross-axis effect。
可用零锚定实现减少重复基线：

\[
f_i(h,U_i)=\phi_i(h,U_i)-\phi_i(h,0),
\]

\[
f_{ij}=\phi_{ij}(h,U_i,U_j)-\phi_{ij}(h,U_i,0)-\phi_{ij}(h,0,U_j)+\phi_{ij}(h,0,0).
\]

这里的0表示整个未来该轴命令序列为零；history h 不置零。不要在每个当前零命令时强行令动态响应为零，因为系统有惯性与历史作用。

采用小 MLP/temporal block 即可。也可使用 action token 的小交互模块，但本轮只选一种实现；冻结在预注册中，不同时搜索多种 Transformer/FiLM/MoE。

不要加入显式 context、oracle摩擦、概率头和语言模型。第一轮不加只对 I 生效的额外损失。

### 6.3 公平性

D/I 参数量尽量在±10%内；不能做到时报告参数/FLOPs/延迟并增加等容量 D 对照前先核算预算。
同一训练数据、窗口、输入、horizon、标准化、优化器、早停、训练步数上限、seed。
R0/R1 的每个模型使用同一个共同验证指标选择 checkpoint，不按测试成绩选 checkpoint。

### 6.4 简单基线

必须补 Command-copy、正确轴 Persistence、RidgeMultiOutput。训练型 ridge 在各自 R0/R1 上独立拟合；评估同一固定 test。
可以保留旧 M1 只读参考，但不能拿其旧数据训练结果与新数据 I 比较后归因架构。

---

## 7. 预测目标、训练损失与轨迹

最终重视真实执行与轨迹后果，不只看 residual 波形。

\[
\hat e=u^{ref}+\hat r,\qquad
\mathcal L_E=\operatorname{Huber}\big((\hat e-e)/s_E\big).
\]

s_E 只从训练池按预注册方法拟合，全模型一致，原单位逐轴指标仍必须报告。
若加入相对位姿 loss，D/I 全部一致：

\[
\mathcal L=\mathcal L_E+\lambda_p\mathcal L_{xy}+\lambda_\psi\mathcal L_\psi.
\]

lambda 在 val 冻结，本轮最多一个共同候选配置，不为每个架构独立广搜。

平面任务中可用预测 vx/vy/wz 自回归积分，初值是 origin 的已知姿态或局部坐标零姿态；之后不能读取未来 GT yaw。
SE(2) 位姿使用 \(T_t^{-1}T_{t+k}\)。世界姿态 GT 来自合法 raw quaternion。
同时审计 tilt 和 \(\int\omega_zdt\) 与真实净 yaw 的近似误差。平面假设不满足的窗口保留并单列，不悄悄删除。
用未来真实 yaw 积分预测速度的结果只能标 `ORACLE_YAW_DIAGNOSTIC`，不是主轨迹能力。

先跑积分单测：直线、纯旋转、恒定转弯圆弧、非零起始 yaw、零命令惯性、给 GT 速度的误差下界检查。
V0.6.1/6.2 的净 yaw 数值曾大幅变化，本轮要求统一源标签/单位/wrap后再写结论。

---

## 8. Stage 4：评价，主次明确

### 8.1 主要终点

- 2秒 FDE_xy：预测终点位置误差，单位 m；
- 2秒 ADE_xy：全预测轨迹平均位置误差，单位 m；
- 净 yaw error：圆周最短差，rad，独立报告，不与 m 直接相加；
- fixed-lead execution MAE：0.25/0.5/1/2s × vx/vy/wz。

如报告单一归一化分数，必须使用事先固定的物理容差或 train-only尺度。分数<1不自动等于“优于 command-copy”，只有明确除以该基线的同口径分数才成立。

主比较：I/R1 vs D/R1 在 held-out pair cells 的 FDE_xy；P2 triple为次要强外推结果；seen-cell结果用于检查代价。

### 8.2 2×2 完整结果

每个 split、摩擦层、command type、anchor、seed 单列。报告 Delta_data、Delta_arch，不只展示最有利轴。
每个cell是否训练过、其单轴和二元投影覆盖情况均记录。

### 8.3 统计

主要统计单元为独立 anchor/session group；保留模型seed配对，不能把窗口或多摩擦变体当独立n。
先group内平均，再跨group配对bootstrap；3个模型seed分别报告，汇总不要把它们伪装成额外物理环境。
至少报告有效独立groups数、均值、median、seed范围、group-level CI。
六个 test groups的窄窗口级CI不能支撑强外推主张。存在共享源/目标的配对需要node-aware处理，不再做episode-pair伪独立统计。

### 8.4 预先锁定实用改善门

默认作为pilot决策参考：主要FDE相对改善>=5%，group CI方向为改善，至少2/3seed同向；seen-cell FDE恶化不超过5%，yaw不得出现未报告的显著恶化。
这些是研究投入决策阈值，不是安全认证。阈值需采集前锁定；CI跨零就写INCONCLUSIVE，不为了过门改指标。

---

## 9. 计算预算与执行顺序

1. smoke最多12条；正式数据预计864条，含所有补采总上限900。
2. 先 D/R0 与 D/R1 两个seed42 smoke训练估算wall time、实际峰值显存和GPU-hours。
3. 预算上限：12个主run、神经训练总额40 GPU-hours、整个流程150 GPU-hours（规划上限，不是性能估计）。预计超额则保存检查点并输出BUDGET_REVIEW，不擅自扩run。
4. 默认最多占2张由用户指定的闲置GPU。不得抢占Risk或其他正在跑的任务；4090/5090分别记卡型与卡时。
5. 先完成2×2主实验，再决定是否需要视觉；本轮不要求下载基模，更不能边跑边加入新架构。

---

## 10. 必须交付

```text
results/v0_7_composition/
  prereg/PRE_REGISTRATION_V07.md
  prereg/command_cells.json
  prereg/split_plan.json
  audit/time_label_feature_material_tests.json
  audit/information_roles.json
  manifests/episodes.csv
  manifests/windows.csv
  manifests/command_coverage.csv
  metrics/main_axis_lead.csv
  metrics/trajectory_metrics.csv
  metrics/data_vs_arch_effect.csv
  metrics/per_anchor_seed.csv
  metrics/termination_and_masks.csv
  predictions/pred_cache.npz
  predictions/pred_cache_manifest.json
  config/ + checkpoints/ + train_logs/
  code_snapshot/ + git_diff.patch
  figures/
  report/V0_7_FULL_REPORT.md
  report/V0_7_DECISION.md
  README_REPRODUCE.md
```

预测缓存不得使用只能pickle读取的object数组。保存sample_id、origin、target_ticks、model/seed/regime、完整U、输入索引、真值e/pose、预测e/pose、mask。文件过大就分片并提供sha256，不只交逐窗误差标量。
至少输出四张独立图：数据覆盖图、2×2数据/架构收益图、真实与预测轨迹图、逐组/逐时域误差图。PNG+PDF；所有数据CSV可复算。PDF导出图不需要文档排版工具。

### 最终决策格式

```text
CORRECTNESS: PASS / BLOCKED
COVERAGE_CONTRIBUTION: SUPPORTED / NOT_SUPPORTED / INCONCLUSIVE
ARCHITECTURE_INCREMENT: SUPPORTED / NOT_SUPPORTED / INCONCLUSIVE
TRAJECTORY_VALIDITY: PASS / INCONCLUSIVE
NOMINAL_COST: ACCEPTABLE / NOT_ACCEPTABLE / INCONCLUSIVE
STATISTICAL_SCOPE: PILOT / CONFIRMATORY（按实际设计，不自封）
NEXT_MAIN_ACTION: 只选一个，并列证据
```

允许的后续：
A. 数据覆盖有效而架构无增量：保留Direct，进入小规模视觉预测/动作后果验证；
B. 交互模型额外有效：扩大独立groups、再接同一个冻结视觉encoder对照；
C. 连Direct/R1也差：检查条件可观测性、时域和任务难度，不立刻上大模型；
D. 统计不够：只补新的独立groups，不用更多重叠窗口充样本。

停止于完整报告，不自动进入Risk、不自动训练V-JEPA。

---

## 11. 来源与边界

本轮动机源于用户提供的《V0_6_2_FULL_REPORT.md》§2–§6及《V0_6_2_DECISION.md》。本文中的数据规模、2×2设计、模型和阈值都是新方案，而非原报告实测事实。
实现以本机版本源码为准。参考入口（无需升级环境）：
- Isaac Lab v2.3 材质配置： https://isaac-sim.github.io/IsaacLab/v2.3.0/_modules/isaaclab/sim/spawners/materials/physics_materials_cfg.html
- Isaac Lab v2.3 sensors： https://isaac-sim.github.io/IsaacLab/v2.3.0/source/api/lab/isaaclab.sensors.html
- 后续视觉阶段官方代码： https://github.com/facebookresearch/vjepa2

不要把“结构性组合缺失已观察到”改写成“交互架构已证明能解决”；本轮就是检验这件事。
