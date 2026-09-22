# Claude Code 执行任务：EA-WM V0.6 预测有效性与上下文可迁移性

请在现有 `execution-aware-world-model` 项目上继续工作。先完整读本文件，不要自行把任务改为“扩大3000条数据以修好Context Swap”。

本轮目标：区分数据/时间问题、评价依赖、简单基线遗漏、上下文与状态纠缠、命令分布变化、均值校准和数据量问题。允许结论为 INCONCLUSIVE；禁止把预想原因写成已验证根因。

## 0. 输入与已有证据

优先复用本机原始数据及仓库：

```text
/media/hdd1/yuhang/datasets/execution_wm/v0
/media/hdd1/yuhang/checkpoints/execution_wm/v0
results/context_swap_v2/
results/v0_5_diagnostics/
execution_wm/
```

实际路径由本机发现，不保证以上仍为最新。必须记录 resolved paths、commit、所有使用checkpoint的完整SHA256和训练配置。不要覆盖旧结果。

本次外部复核发现：

1. 原缓存 N→L P_target=-0.171499，L→N=+0.686717 可复现。
2. low source（L→N的5510行）Context native L2=1.349489，command-copy `e_hat=u` 为1.308615；low的逐轴MAE：command-copy=[0.070384,0.067221,0.090306]，Context=[0.068214,0.086826,0.129350]。这些是旧pair加权口径，不是全数据结论。
3. 25772行只有968种不同的actual future数组；N→L只有424种source future、334种target future。unique future不是独立episode，不能替代node-level统计。
4. episode-pair cluster仍共享source/target episode，不能据CI不变宽就说已排除依赖。
5. 用reverse/cache key恢复native target prediction后，N→L的swap-vs-native直接距离约0.370617，相当于target native误差的27.46%。到GT距离近似相等不等于两个prediction等效。
6. `calib_slope=Cov(pred,true)/Var(true)` 是pred-on-truth斜率，不足以证明均值失准。vlow/vx从Pearson推算true-on-pred斜率约0.993。
7. 全局affine后N→L P_target约+0.00857，但原CI=[-0.003,+0.021]跨0，不能称已通过。
8. 0.25/0.5秒transition标签完全相同，缓存每窗最小age只有0/1/inf；不是自动证明bug，但不能当成两个独立敏感性证据。
9. 原报告把N/L的“真实工况差距”写成1.22，但该值是D_before=norm(target_actual-source_prediction)；实际mean norm(target_actual-source_actual)=1.485315。target native error=1.349489，故“native error大于真实工况差距”的N/L论据不成立。必须修正变量名称与报告推论。
10. 原报告各轴norm差不可直接相加。可加平方误差分解N→L为vx +0.033757、vy +0.110085、wz -0.519081，总计-0.375238。

请逐项自行复算；不一致时输出 DIFFERENCE_REPORT，不要直接相信上述数字或原报告。

## 1. 工作范围、资源与停机规则

本轮分三阶段：A离线审计；B最多300条controlled pilot；C最多4种神经模型×3seed的小规模训练。

- 阶段A不训练、不采数。存在P0错误先最小修复并重算，标记旧结果影响范围。
- 只有数据因果性、日志和split正确，才能进入B/C；若无法确认，输出 BLOCKED 并停止，不能编造缺失metadata。
- B先12条smoke，确认注入/记录/restore有效后才能增量pilot。
- C所有新训练预计总预算先实测2个run，单轮建议不超过120 RTX5090 GPU·h；超过需人工批准。卡型不能互相按1:1折算。
- 不训练JEPA基模，不加入MDA/VoI/RL agent，不跑真机危险试验，不自动执行在线纠正。
- 不同时引入FiLM、mixture、ensemble、Neural ODE、linear attention等新架构。
- 不删负结果、不为过门调阈值、不以test挑checkpoint、不手选好看的pair。
- 保留V0、v2、V0.5原目录，新增`results/v0_6_validity_transfer/`。
- 只可本地git commit，不要求任何token，不读取或打印凭据，不自动push。

## 2. 阶段A：原始实现审计与独立重算

### A1. 建立可独立复核的manifest

导出完整表，而不仅仅pair_idx：

```text
pair_id
source_episode_uid / target_episode_uid
source_origin_index / target_origin_index
source_origin_time / target_origin_time
source_seed / target_seed（缺失明确null）
source_dataset_version / target_dataset_version
source_condition / target_condition（只作分析）
probe_or_command_family
shared_prefix_group / reset_anchor_group / support_query_group
command_hash / source_state_hash / target_state_hash
history_start / history_end / prediction_origin
future_start / future_end / sample_dt
train_val_test_role
source_reuse_count / target_reuse_count
accepted / reject_reason
```

episode_uid须包含数据版本，防止d0_ep001与d1_ep001碰撞。同源分支、重叠窗口和support/query lineage整体切分。报告真正的unique episodes、seeds、anchor groups，而不是只报告pair数。

### A2. 时间与causal input审计

查明collect→window slicing→model→plot代码链。写自动单元测试：

1. 输入的proprio/action/history timestamp全部≤prediction_origin。
2. 修改future_execution/future_residual不改变模型输出；修改privileged字段不改变主模型输出。
3. future command是预测时已知候选计划；若来自闭环日志，必须声明为回放条件，不叫在线已知计划。
4. 明确命令request、applied以及state pre/post-step的时间。第一标签是t+Δt还是t？代码与记录必须一致。
5. 50Hz control至20Hz记录不能用固定2或3步而仍标记0.05s；输出真实dt分布和时钟累计误差。使用正确时间调度/抗混叠方案，修改会改变数据时单独版本化。
6. 每个forecast记录完整[t,H,3]，不要把滚动一步预测拼成“2秒open-loop预测”。同一目标时刻不同lead要分开。
7. body/world frame用quaternion变换单测验证，不能靠corr(cmd,actual)=0.89证明。声明quaternion顺序；body wz在roll/pitch非零时不直接等于yaw导数。
8. GT状态、估计odom、IMU测量分开；GT可作标签，作为predictor输入则明确privileged版本并做deployable对照。

### A3. Reset/干预应用审计

检查base pose/velocity、joint q/dq、controller hidden、last action、command buffers、gait phase、外力和contact历史的reset。

记录PhysX中实际material和actuator参数readback、body/material索引、ground-foot组合规则和有效生效时刻。不得只凭YAML friction=0.3认定注入成功。摩擦变化不预设只能造成速度降低。

保留fall/termination标签并单独统计，不通过放松sanity标准把fall数据伪装成正常接触。外力条件本轮不作为friction主实验。

### A4. 复算旧metric并增加基线

最少输出：

- original pair-weighted metrics（保持旧定义）
- unique-window sensitivity（仅去重，不宣称独立）
- equal-episode / equal-anchor / equal-probe summary
- per-axis MAE/RMSE
- command-copy: e_hat[k]=future_u[k]
- persistence: e_hat[k]=last_available_execution_at_origin
- 原Direct、Context、Action-only，同input及同test。

如果原数据支持，在train/val拟合一个ridge ARX或一阶lag baseline；这是轻量拟合，归入阶段C预算，A中只准备接口。

不得用未来第一帧e作为persistence，禁止测试工况真值输入主baseline。

### A5. 重做依赖统计

旧edge=(source_episode,target_episode)不够。优先以真正独立reset_anchor_group resample并保留组内全部分支；旧数据只有episode时做node-aware/two-way分析和leave-one-episode-out。

- 独立source/target条件池：分别重采样source episode与target episode，并用出现次数乘积给pair权重。
- 同条件池/双向复用：同一episode必须共享同一重采样权重，不能当两份独立数据。
- 共享seed/common-prefix：外层按组重采样，不能破坏其结构。
- cross与same差值在同一次共享resample内计算，不各自抽完独立相减。
- 同时给equal-episode或equal-anchor估计量，避免高复用episode主导。
- 若无法恢复lineage，报告INSUFFICIENT_METADATA；可输出descriptive mean，但不可宣称严格CI。

先用一个已知episode random-effect的synthetic小测例检验：增加同episode重复window不能凭空提高独立证据数量。bootstrap 2000–5000次，固定seed。

### A6. 校准、geometry与transition诊断

校准分别拟合/报告pred-on-truth与truth-on-pred，标明哪个轴是预测、哪个是真值；按预测分箱报告mean truth、mean prediction、bias和episode CI。只在val拟合affine，test只评价。MSE条件均值收缩例子应写入文档，不能从slope<1归因Huber有害。

直接度量swap prediction与target-native prediction，报告绝对距离、相对native误差和per-axis距离；如果要做等效性检验，容差必须提前根据物理任务确定。不能用error ratio≈1做等效结论。

令d=prediction_swap-prediction_correct，a=target_actual-prediction_correct，输出：

```text
squared_target_gain = 2*dot(d,a) - norm(d)^2
parallel_shift
orthogonal_shift
overshoot_component
per_axis_squared_gain
```

旧S_dir仍保留，但不得当作充分成功条件。任何基于test求出的optimal-shrink只可标ORACLE_DIAGNOSTIC，不能变为部署参数。

transition从完整命令事件表计算；分target timestep，而不只按整窗any-change。origin到事件、target到事件、lead_time分别记录。未知道起步命令之前的状态标unknown，不默认inf=steady。0.25/0.5/1.0的分组如果完全一样，明确说明这不是额外稳健性证据。

### A阶段交付与门

输出`A_AUDIT_DECISION.md`：

```text
DATA_CAUSALITY: PASS / BLOCKED / INCONCLUSIVE
PAIR_INDEPENDENCE: PASS / BLOCKED / INCONCLUSIVE
SIMPLE_BASELINE_CHECK: model_better / baseline_better / mixed / inconclusive
CONTEXT_OUTPUT_SENSITIVITY: supported / unsupported
PHYSICAL_CONTEXT_TRANSFER: untested / supported / unsupported / inconclusive
NEXT_STAGE_ALLOWED: yes / no
```

存在影响训练标签/输入的P0错误，先修复并小样验证；不得直接扩采。过去PASS结论可能被降级，必须如实记录。

## 3. 阶段B：小规模support/query与查询起点试验

本轮核心不是制造更多相互比较的pair，而是增加独立物理问题。

### B1. 先冻结数据协议

写`PRE_REGISTRATION.md`和config，固定：工况范围、命令family、seed、split、预算、history/H、metrics、任何容差。旧v0_5所有数据视为development材料；新test group在训练与参数选择中不可见。

### B2. 数据结构

```text
context_session
    support episode S（提供过去action-response）
    query anchor x0（显式记录当前状态）
        query command Q1 -> future
        query command Q2 -> future
```

support与query同一稳定物理condition，但命令尽量不同。support不能包含query未来。

记录三类swap：

1. 同condition，不同support命令；
2. 不同condition，但support命令/phase相似；
3. 同condition、同family但独立episode。

Direct baseline必须获得相同support历史、query state、future commands，不能让Context独占更多数据。

### B3. 物理condition effect的query anchor

对同一个query anchor，在normal/.6/.3等已验证合法工况下执行同一未来command。尽可能恢复当前base/joints/controller/queue/gait phase，记录restoration differences。

同一个seed不等于同一物理状态；无法恢复完整求解器contact cache时必须注明approximate-paired，不使用“精确反事实”措辞。

`c_B`来自事先完成的独立support_B；`s_query`来自当前query anchor；把target-query后半段真值或target-query完整history偷作source context是禁止的。

这是一项support/query预测实验。若support后重置机器人但保留context记忆，明确这是跨episode但同工况的测试设置，不伪装成从未知新地面零样本识别。

### B4. 大小与取样

先12条smoke，包括三个工况、两种命令、正反/转向或侧向；确认日志正确。

pilot参考预算：4 query families×8 independent anchor groups×3 conditions×2 repetitions=192 query episodes；另约48 support episodes。总新增不超过300。

独立重复需说明随机源：同seed精确重放用于reproducibility；不同noise/gait/command seed用于分布统计，二者不混为样本数。

分别包含random dwell和fixed probes，预先规定命令幅值/驻留分布，保留reverse/accel/decel；不能只保留容易匹配的steady窗口。

训练/验证/测试按独立anchor/support-query group切分，不能仅按window。若旧0.15 OOD已用于许多设计决策，它只算development-OOD；新确认性split用独立seed/command/layout，明确哪些泛化维度真正未见。

### B5. 同步补高频日志和轻量RGB接口

保留50Hz原始proprio和control event；20Hz训练输入另行派生。明确logged torque是computed还是applied，contact阈值由配置给出。

不需要训练视觉模型，但可保存少量带时间戳的RGB/depth以验证未来接口。读取/加载V-JEPA2属于单独工程smoke，不计为本轮模型研究结果；不要让渲染阻塞核心审计。

## 4. 阶段C：公平模型对照（最多12个主要run）

### C1. 固定对照矩阵

无需神经训练：command-copy、last-execution。轻量拟合：train-only ARX/first-order lag。

神经模型4组×3seeds：

- M0 Direct recurrent predictor：全部合法support/query输入直接预测未来；
- M1 当前Context结构：复用已有定义，统一训练协议；
- M2 cross-support/query Context训练：同一工况跨命令/episode推断c，decoder尽量与M1相同；
- M3 privileged friction信息诊断：只此组可输入真实friction，明确非部署模型、非保证上界。

四组输入历史长度、future command、target scales、loss、early stop和训练预算相同；M3额外信息明确列出。M2不可隐藏使用更长history或更多样本。记录实际参数数和FLOPs/latency；无法严格配平时提供参数差并声明。

### C2. 输出与loss

先保留raw r=e-u标签；它是合法tracking error，不因nominal动态存在而删掉。

统一输出fixed-origin 0.25/0.5/1/2s未来execution。可同时监督relative pose，所有模型都获得相同target；权重只由train/val决定。

所有normalization只用train fit，记录每轴单位、scale、missing masks。不要用同一个未归一化overall混合m/s、rad/s选择checkpoint。报告raw逐轴+归一化总分，且解释总分权重。

暂不同时改Huber、FiLM、context dim、NLL。若要做nominal+delta或recurrent derivative decoder，写为下一轮候选，不自动添加为第五第六组。

### C3. 可迁移性与state/context纠缠检验

- matched条件下同condition跨命令support是否仍有效？
- held-out command family是否仍有效？
- 固定合法query state，换support condition；固定support，换合法query state。
- state-only / context-only / full预测作为path诊断，缺支路导致的OOD要标注，不能当作纯因果结论。
- prediction在k=0是否尊重query当前状态；near-term slope是否只是由c“复制”target轨迹。
- c的condition linear probe、command-phase probe均可做，但只是辅助相关性分析。

### C4. 高频与任务后果

比较raw wz、正确低通降采样wz、未来窗口平均wz、GT relative yaw/pose。目标平滑仅用于额外任务，不替换安全峰值记录；历史滤波只能因果使用过去样本。

当预测均值难拟合瞬时波纹时，先检查是否是采样/相位问题，再考虑噪声不可约。重复相同状态/条件/动作下的不同随机realization用于估计可预测性；没有重复证据不得称“wz无信号”。

## 5. 新的决策门（不强迫context胜出）

必须分别输出：

1. 数据与因果有效性。
2. 原生预测是否超过command-copy/persistence/ARX。
3. Context是否超过等输入Direct；若差异CI跨0写INCONCLUSIVE。
4. Context是否跨support动作/当前query状态迁移。
5. learned context vs privileged information gap。
6. 20Hz/50Hz和执行/位姿目标是否改变结论。
7. 初步学习曲线是否支持规模化；不得由总样本数直接推断data hunger。

建议后续选择：

```text
FIX_DATA_OR_TIMING
IMPROVE_QUERY_CONTEXT_DESIGN
SCALE_DATA_WITH_EVIDENCE
REDESIGN_STATE_CONTEXT_SEPARATION
ADD_PROBABILISTIC_MODEL_AFTER_MEAN_CHECK
PROCEED_TO_VISUAL_WORLD_MODEL_PILOT
KEEP_DIRECT_BASELINE_AND_REVISE_CLAIM
INCONCLUSIVE_REQUIRE_MORE_INDEPENDENT_GROUPS
```

可多选但必须按证据排序。不要自动进入Context Adaptation/真机修正；可以在建议中说明小规模adaptation的研究价值。

## 6. 最终交付目录

```text
results/v0_6_validity_transfer/
    PRE_REGISTRATION.md
    config/
    audit/
        source_inventory.json
        causal_time_frame_audit.md
        reset_and_injection_audit.md
        input_roles.csv
        metric_equivalence.csv
        bugs_and_impact.md
    manifests/
        episodes.csv
        full_pair_manifest.csv
        support_query_manifest.csv
        split_manifest.csv
        duplicate_and_reuse.csv
    metrics/
        native_baselines_per_episode.csv
        native_baselines_per_horizon.csv
        per_axis_and_pose_metrics.csv
        node_aware_bootstrap.csv
        leave_one_group_out.csv
        prediction_reliability.csv
        geometry_decomposition.csv
        transition_timestep_metrics.csv
        support_query_transfer.csv
        model_comparison_3seeds.csv
    raw/
        audit_prediction_cache.npz
        representative_full_episodes/   # 至少提供12条，可抽样但保留完整时间语义
        support_query_examples.npz
    figures/
    code_snapshot/                     # 本轮新增/修改相关.py和配置，或完整git diff
    report/
        A_AUDIT_DECISION.md
        V0_6_DECISION.md
        V0_6_FULL_REPORT.md
        NEXT_ACTIONS.md
```

缓存必须含完整manifest映射、history输入、future commands、GT labels、各模型predictions、source/target/support/anchor IDs、采样时间。敏感路径可相对化，但不能删掉用于科学复核的身份映射。不要只上传汇总表导致下一次无法验证bootstrap。

## 7. 终报告逐题回答

Q1 旧“机制PASS”和“40%过冲”哪些被支持、哪些需撤回？
Q2 为什么复杂模型在probe上不能稳定超过command-copy？是分布、目标、采样还是优化？
Q3 node-aware CI与旧CI差多少？独立episode/seed到底多少？
Q4 回归方向纠正后，还支持哪些校准问题？
Q5 matched pairing是否丢弃高失配状态？接受/拒绝分布如何？
Q6 state/context是否纠缠？same-condition跨命令support有无增益？
Q7 relative pose与raw speed评价是否一致？wz的难点如何定位？
Q8 同架构同协议3seed下Context相对Direct的效益是什么？
Q9 oracle信息对照指向context inference还是predictor/target瓶颈？
Q10 下一步是扩数据、改结构、接视觉还是缩减科学主张？给证据，不给乐观推测。

## 8. 首次开始时只需立即输出的内容

先完成代码/数据盘点与阶段A，不需等待为目录创建等小操作确认。完成A后若没有P0阻塞、预算与协议可满足，按本文件推进B/C；若有阻塞则交付已完成证据并停下。

每一阶段记录：完成内容、命令、耗时、GPU·h、输出路径、负结果、缺失数据、下一步。结束后打包分析材料与代码，不打包私钥、token、账号信息或全量大模型权重。
