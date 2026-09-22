# Claude Code 执行任务：V0.6.1 标签、Support 配对与评价协议修复

> 本轮只做 correctness patch + 有限公平重跑。不是另开研究方向。
> 先读《01_V06_独立验收与下一步.md》和《03_关键源码证据.md》。
> 依据：刚交付的 `v0.6.zip`。不要将 V0_6_DECISION 的 PASS/FAIL 当作无需验证的前提。

## 0. 任务目标与禁止事项
恢复以下定义的可执行一致性：
- input history：预测时真实可获得、严格过去/当前；
- support：同一窗口内成对的动作—响应观测；
- future execution：实际物理运动的明确时间采样目标；
- residual：与同一时间参考命令相减；
- evaluation：比较相同样本、相同单位、相同 prediction lead；
- statistics：按真实数据 lineage，而不是扩大后的 window 数推断。

本轮不新增 FiLM/mixture/uncertainty/JEPAv2 微调/控制器，不直接扩采千条数据。
不要删除旧结果，不重写历史，不为“过门”改指标。
任何新采集只限 policy-command 审计等少量 smoke；若确认旧 raw 不能复用，先记录影响再按预设小预算补采，超预算须停下说明。

## 1. 创建独立版本与源证据
输出根：
`results/v0_6_1_correctness/`
派生数据新目录，禁止覆盖旧 `.20hz.npz` 或原始 50Hz。
保存旧/新 git commit、输入文件 SHA256、配置 hash、依赖版本、checkpoint lineage。
公开仓库中不写任何 token；不用用户明文凭据。

结构：
```
audit/
unit_tests/
config/
manifests/
derived_data_summary/
predictions/
metrics/
figures/
code_snapshot/
report/
```

所有 Python/CLI 入口必须先检查文件和 schema；遇到关键字段缺失时给出阻塞原因，不伪造默认 metadata。

## 2. 必须首先验证并修复的确定性问题

### B1. SupportWindowBank.sample_batch 不成对
旧函数分别随机抽 hp 和 ha，可能属于不同 episode 或不同 origin。

正确抽样单位必须是：
```
SupportRecord = {
    support_episode_id,
    support_window_id,
    support_origin_tick,
    support_session_id,
    split,
    condition,
    history_proprio,
    history_command,
    timestamps,
}
```

每个 batch 样本只选一次 support_window_id，然后读取全部配套字段。
同 condition 配对允许通过 metadata 选择，但这是实验条件/可用参考库假设；主模型不得直接读 condition。
用 synthetic indexed data 单测：hp、ha、time、ID 每一帧一致。
修复后应覆盖 M2 train/val/test、M1 cross_support/diff_condition、bootstrap 的全部调用路径。

Self-exclusion：
- 若 query 本身是 support episode 的训练窗口，跨 episode 试验必须排除同 episode；
- 禁止抽 query 未来作为当前 context；
- 如果使用固定 train reference bank，明确叫 fixed-reference-bank evaluation；
- 若声称 unseen-support transfer，则另划 support-session train/val/test，不混用。

保持三种可辨别的实验模式：
A native causal history；
B paired same-condition cross-episode support；
C intentional mismatched/shuffled history（仅负对照，不能混成 B）。
本轮重训主 M2 用 B；C 可先只做离线诊断，不能为新增训练无限加预算。

### B2. 标签恒等式被不同重采样方式破坏
旧流程：
`lb_residual = resample_poly(raw_execution-raw_cmd)`
`cmd20 = previous_hold(raw_cmd)`
`gt_execution = cmd20 + lb_residual`
这不等于 `lb_execution=resample_poly(raw_execution)`。

修复规则：
1. 先建立 reference clock/ticks；
2. 由原始物理 execution 独立生成 `execution_label`；
3. 从真实 command 事件表生成 `command_reference`；
4. 最后 `residual_label = execution_label - command_reference`；
5. 训练可预测 residual，但 evaluate 必须独立读取 execution_label；
6. 不得通过“把新 gt 定义成 cmd+old_residual”让断言假通过。

离线非因果抗混叠可用于监督，但需要记录它预测的是平滑运动，而不是未经平滑的瞬时速度。
输入必须因果；保留未平滑真值以做对照。滤波与端点处理方式用配置和版本记录。
对所有 240 raw episode 重新派生，并输出 max identity error、boundary-specific error、单位与逐轴分布。

现有原始 50Hz `e-u-r` 正确的文件优先复用，不因为派生错误全部重采。

### B3. 浮点时间导致 hold 在边界取旧 command
旧 timestamp 累加 0.02，与理论 grid 比较会在 2.5s 等时刻错选前一 tick。
改用 `physics_tick/control_tick` 为时间身份，浮点秒只作展示。
50Hz→20Hz 为有理数关系，不用脆弱的浮点等式。对于真正异步测量，保留实际时间及 sample_age，不把观测来源时间强写成查询 grid。

单测：
- 在 1.5、2.0、2.5、3.0、3.5s 的 known command event，采样值与已定义的右/左连续约定一致；
- 2.5s 的 Q3 turn 应按所定义时刻读取 +0.8→−0.8 的正确分段；
- 不仅修一个样本：至少长时间和 ±微小浮点扰动测试；
- state 边界、future command、residual 同步采用相同约定。

### B4. Persistence 与 k0 轴选择错误
现有 proprio 前六项是 [vx,vy,vz,wx,wy,wz]。
`hp[:,-1,:3]` 不是 [vx,vy,wz]。

建立 `FeatureSchema`，记录 offsets、unit、frame、GT/deployable 角色。
统一函数 `extract_execution_from_proprio()`，本版根据 schema 对应 [0,1,5]。
用标识值 [11,12,13,21,22,23] 单测：输出必须 [11,12,23]。
修复 persistence、k0_continuity、所有使用 current execution 的代码。
k0 若实际预测的是第一未来步，就命名 first_step_error，并记录 lead；它不应被要求等于当前状态。

### B5. Split 实现与预注册不符
旧 held_anchor 筛选未排除 Q4，实际 Q4 占 96/384 窗。
构建显式独立 population：
- A: unseen_anchor_seen_family = AQ6/AQ7 × Q1–Q3
- B: seen_anchor_unseen_family = train AQ × Q4
- C: unseen_anchor_unseen_family = AQ6/AQ7 × Q4
- AQ5/Q4 单独记录为 evaluation-only supplementary 或明确排除，不隐式合并。
如需历史兼容 all-Q4 表，单列并说明 overlap，不作为独立确认。

为每个 split 断言：
family whitelist、anchor whitelist、episode/window 去重与交集；
support donor 来源真实ID完整保存；
训练/验证/测试的 query lineage 不混用。
旧数据已反复分析，修复版结果仍是开发/探索；不能追溯宣称新的独立 confirmatory test。

### B6. fixed-lead 与 prefix-average 分开
保留两个函数和字段：
```
MAE_lead_2s: abs(pred[:,39,j]-gt[:,39,j]).mean()
MAE_prefix_2s: abs(pred[:,:40,j]-gt[:,:40,j]).mean()
```
0.25/0.5/1/2s 对应索引必须通过 timestamp，不盲假固定长度。
构造 toy error: 仅最后一步非零，确保 two metrics 不相等。
主表用每轴 fixed lead；prefix 单列。旧 MAE_all 混合 m/s 与 rad/s 仅兼容，不作为唯一主门。
需要 scalar 选择模型时，用预先定义且只来自 train 的归一化/任务容差，并明确单位。

### B7. 统计单元与缓存一致性
模型预测只执行一次，保存后所有 summary 和 bootstrap 引用同一 pred cache，不在 bootstrap 内重抽 donor 或重新推理。
query anchor 是复用的高层单位；support 引用也构成依赖。保存 lineage 后再决定分层/two-way 处理。
held-anchor 只有2组，直接报告每组结果、training-seed 分布及探索性均值；不要靠48 episode宣称普遍新anchor显著。
若将来扩确认数据，独立 anchor 数量与 seed 需先冻结，再谈 confirmatory interval。
训练 seed 与数据 seed 分开，固定 window manifest，以免“3 seeds”同时改变数据样本后却声称排除了优化问题。

## 3. 需要运行时确认的风险：实际 policy 消费哪个 command？
`collect_support_query.run_wave` 当前写 cmd 后，policy 使用上次 env.step 返回的 obs。
不能仅靠源码认定最终行为，必须在当前 Isaac/RSL-RL 安装环境做≤6条短 smoke 验证。

记录每个控制 tick：
- requested command；
- command term buffer；
- policy observation 内实际 command slice；
- policy output joint action；
- control tick / capture tick / state；
- 是否重计算观测、是否存在 noise/history 更新副作用。

若命令滞后：
- 不要把日志中的 request 当作同一步已消费命令；
- 明确将其作为 command pipeline latency，或修正 observation refresh；
- 独立记录 u_requested、u_consumed，并决定研究输入；
- 不允许为了消除问题去“平移曲线到最高相关”。
如旧 raw 可以从已证实的固定时序规则准确恢复 consumed command，版本化恢复；否则只补必要数据。任何 restore 不能伪称记录到的实测。

检验正常 controller policy 的 checkpoint 是否冻结，physics step/decimation/action update 顺序不变。

## 4. 额外审计，不自动加入新研究模块
- 摩擦：读回 robot material 已有；再读/解析 floor 与两侧 combine mode。查不到就写 UNKNOWN，不把 hardcoded average 当实测。static/dynamic 分别报告。
- Quaternion：当前 resample_poly 逐分量滤波的 lb_base_orientation 范数在部分端点非1。用于 pose 前必须做合法姿态处理（符号连续、插值/归一化的合理方案）；不用不合法四元数计算姿态GT。原始四元数保留。
- 重复：rep1 改了 command duration，不能当作同一条件随机重复去估不可约误差。
- ARX：现有实现是 current-proprio + future-command 的 direct multi-output ridge。准确命名，不能暗称完成了标准多滞后 ARX 全搜索。
- Oracle：真实摩擦标量加到 c 的方式是弱诊断。输出无收益先标 INCONCLUSIVE，不得推导“同类数据更多无效”。
- Sources：提供缺失的模型类、registry、全部log。不再只导出normal预测示例。

## 5. 通过单测后才允许有限重跑

### R0: 修复回归单测
至少交付：
T01 support 共享 ID/时间/动作—状态一致性；
T02 privileged/标签改变不影响普通 inference input；
T03 `e_label = u_ref + r_label` 全量成立；
T04 命令边界 integer-tick 测试；
T05 persistence schema 正确；
T06 family/anchor split与donor lineage；
T07 fixed-lead vs prefix 分离；
T08 每个 prediction_origin 的 input source time ≤ origin，未来标签独立；
T09 model eval/summary/bootstrap 均复用同一缓存哈希；
T10 policy 消费时刻 smoke（无法执行则阻塞相应采集结论）。
可以另外检查改变未来计划后缀不应影响其之前的真实物理目标；若预测架构使用后缀产生前缀捷径，单独记录，不立即重构全部网络。

未过 T01/T03/T04/T05/T06，不允许正式训练。

### R1: 旧checkpoint离线重评（可选但有价值）
若能恢复相同窗口完整原始预测，用正确 execution 标签重评全部模型，标注：
`OLD_MODEL_NEW_EVALUATION_ONLY`。
这只能说明评价变化，不能代替正确标签重训。
不把旧误差标量相加就当新 MAE：需要完整逐元素 prediction。

### R2: 先2个smoke训练
固定 dataset/window manifest，M0与M1 seed42，各训练至预注册小预算。
只检查 loss可下降、评估正确、无NaN、无信息泄漏、时间/GPU·h合理。
不要求 Context 获胜。

### R3: 再运行主对照
模型结构尽量不变：
M0 Direct；
M1 native Context；
M2 paired-support Context（修复采样）；
M3 existing privileged摩擦诊断。

每类 seeds42/43/44，最多12个主run（含上一步可延续的run，不重复浪费）。
加入无需GPU的command-copy、正确persistence和现有ridge；需要一阶响应模型时只在train拟合、val定超参。
所有模型同目标、同train windows、同val、同future-action信息、同scaler和同调参预算。
M2保持support paired抽样，固定evaluation donor table，记录donor身份。

M2四个必要推理对照（先s42，其余按预算）：
1. same-condition paired donor；
2. wrong-condition paired donor；
3. native history；
4. zero / fixed context。
intentional unpaired donor可作诊断，但不要把它当作合法cross-support。
这样才可区分“学到condition”“不依赖c”“训练时噪声正则”等解释。

### R4: 评价表
每个真实 split × model × seed × condition × family：
- vx/vy/wz fixed-lead MAE 与 RMSE（物理单位）；
- prefix MAE单列；
- command transition后不同lead，steady和cold-start分开；
- 同command不同condition的真实execution差距（正确e，不是预测误差替代）；
- context variant 差值；
- 可合法获得时 relative pose/净yaw，缺失即NOT_RUN。
不只看aggregate，不比较m/s和rad/s数值比来宣称哪个物理量“更差”。

## 6. 本轮新报告必须回答
Q1 两个P0是否在实际代码/数据复现？影响哪些训练/评价？
Q2 修复标签后真实physical prediction排行有没有变？不知道就写数据缺失。
Q3 paired donor修复后，M1原生→cross的差距有多少，M2正确cross是否优于zero/wrong？
Q4 M1 vs M0、M2 vs ridge 的收益是否跨axis/horizon/独立anchor一致？
Q5 M3 oracle分支是否确实响应friction？无增益能否有替代解释？
Q6 支持池复用及ground-truth condition选donor使结论属于何种信息假设？
Q7 新anchor、新family、joint-shift三个结果分别如何？
Q8 后续最有价值的一个实验是什么？不同时开五条研究线。

判决字段：
```
LABEL_IDENTITY: PASS / FAIL
SUPPORT_PAIR_INTEGRITY: PASS / FAIL
POLICY_COMMAND_TIMELINE: VERIFIED / UNRESOLVED
FEATURE_SCHEMA: PASS / FAIL
SPLIT_AND_DONOR_LINEAGE: PASS / FAIL
FIXED_LEAD_EVALUATION: PASS / FAIL
NATIVE_PREDICTION_GAIN: SUPPORTED / UNSUPPORTED / INCONCLUSIVE
CONTEXT_TRANSFER: SUPPORTED / UNSUPPORTED / INCONCLUSIVE
ORACLE_DIAGNOSTIC: [description, not automatic upper bound]
STATISTICAL_SCOPE: EXPLORATORY
RECOMMENDED_NEXT_STEP: ...
```

## 7. 下一包必须能独立重算
不要只交报告或per-window MAE：
1. 全部修改源码与git diff；模型类/registry/schema。
2. 每个checkpoint的config、weights hash、train_log；可上传实际小checkpoint。
3. 数据清单包括所有240个raw与derived episode的hash、condition、seed、anchor、rep、family、termination。
4. 每个window的episode_id、origin_tick、origin_time、history source indices、future indices、split、donor_id。
5. 完整 `predictions.npz/parquet`：模型与seed索引、future command、execution GT、residual GT、每个模型prediction、donor context及ID。使用非object数组和JSON，避免不必要pickle。
6. 完整正确性单测输出；每条失败单测的最小复现。
7. 代表性 normal/mid/low × Q1–Q4，包含 raw50Hz/derived/metadata。
8. 新 summary、逐axis/lead表、误差按anchor图、修复前后例子。
9. `V0_6_1_DECISION.md` + `V0_6_1_FULL_REPORT.md` + `BUG_IMPACT_MATRIX.md`。
10. 报告区分用户输入路径与实际已完成结果；未执行项目不得写PASS。

完成后停止，不自动接V-JEPA大训练、Context Adaptation或真机在线纠错。
本轮的成功标准是实验可解释、可复核，不是保证M2/M1胜出。
