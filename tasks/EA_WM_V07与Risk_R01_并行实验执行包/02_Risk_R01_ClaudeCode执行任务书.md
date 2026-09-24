# Risk 分支 R0.1：扰动有效性、可挽救性与提前预警
## Claude Code 可直接执行的独立任务书

版本：v1.0｜制定日期：2026-09-24  
研究目标：验证“正常时少干预，危险且仍可挽救时及时干预”的研究前提。  
本轮是 simulation-only pilot，不是救灾部署验证，也不是安全保证。

---

## 0. 为什么另开这一条线

CVPR V0.7 研究动作组合下的执行预测泛化；本文件独立研究物理干扰下的失败与干预。
二者可共享正确的数据契约、机器人和日志模块，但本轮不共享训练/测试样本，不混用结论。

用户提出的假设：旧摩擦设置可能只产生轻微误差，真正价值可能出现在强干扰/极端条件。
这只是待验证假设。旧实验已经报告不同摩擦的可测 execution 差异，不能写成“原来摩擦完全无效”。

本轮严格按三个问题推进：

1. **发生了吗？** 在数值有效、任务合理的条件下，nominal 系统是否出现可重复的失稳或任务失败？
2. **还能挽救吗？** 同样状态下，提前减速/限制转弯/短暂停稳等合法干预，是否能改善结果？
3. **来得及知道吗？** 仅用因果传感器历史，能否在有用提前量下识别这些事件？

只有“有失败 + 有干预价值 + 可提前发现”都成立，才值得进一步训练昂贵世界模型和研究路由器。
不要先训练一个 router，再通过降低摩擦直到 baseline 崩溃来证明它有用。

---

## 1. 范围、安全与代码隔离

- 新分支/worktree：`exp/risk-r01`；新代码：`execution_wm/risk_r01/`。
- 新数据目录 `.../datasets/execution_wm/risk_r01`；新结果 `results/risk_r01/`。
- 复用 V0.6.1 已修复的数据和时间接口；与 CVPR worktree 共享的底层修改须单独commit并先过回归测试。
- 固定 locomotion policy/checkpoint、正常控制增益、物理步长；不通过削弱 baseline 来制造优势。
- **严禁真机发送动作、扭矩、危险故障或低摩擦测试指令。** 默认硬编码安全开关 `allow_real_robot: false`，发现硬件连接端点就中止。
- 不装新模拟器，不升级 Isaac/PyTorch，不接 HumanoidVLN，不做电磁干扰、不声称刚体接触已经模拟了沙土/冰雪/灾害真实物理。
- 本轮不训练 V-JEPA、VLA、LLM agent，不研究跨具身泛化或MDA。
- 无法在允许动作集合内恢复的情形，应停机/拒绝任务或请求更高层重规划，不能让优化器“不断加动作直到成功”。

---

## 2. 系统与命名

固定一个可靠的目标点/路径跟踪器及低层 locomotion controller：

```text
目标/短路径 -> 冻结 nominal tracker -> u_nom
                      |                |
                因果观测历史       可选 gate/corrector
                                       |
                                    u_final
                                       |
                          低层 locomotion policy
                                       |
                                 实际 execution
```

没有现成导航栈时，先用已知地图上的短路径/航点跟踪器。不能把“走多久”直接当成“到达目标”；任务必须有真实目标和时间预算。
第一轮不需要VLN/VLA。所有比较使用同一个高层 tracker、同一个低层控制器。

命名必须分清：
- `u_nom`：原 tracker 的输出；
- `u_final`：经过干预/安全限制的命令；
- `u_consumed`：低层 policy 实际读到的命令；
- `e_actual`：实际速度/姿态等物理响应。

风险高不意味着所有修正都有效；router 不能只会“报警”，还必须和干预收益、响应时间一起评价。

---

## 3. Stage R0：最多24条 smoke，冻结仿真与事件定义

### 3.1 必要回归单测

1. 整数tick时间、requested/consumed滞后、未来标签不进入输入；
2. 真实 execution 标签独立生成，`u_ref+r=e_label` 恒等式；
3. vx/vy/wz 通过schema提取；body wz不直接当任意姿态下的world yaw rate；
4. reset包括控制器hidden state、动作队列、接触/IMU缓存和随机化状态；
5. 两接触体的摩擦值、静/动态系数、combine mode与材质绑定运行时读回；
6. 不把局部低摩擦patch引入的碰撞台阶当摩擦效应：地面连续高度、法线和collision overlap检查；
7. 50Hz原始日志完整保存；若要抓短接触事件，另外保存physics-rate事件/极值，不能只靠20Hz平均值。

所有实际API以安装版本为准。官方文档的combine mode默认值不是实际场景readback证据。

### 3.2 失败、滑移与未完成必须分开

在SMOKE数据上定义并冻结 `event_definition.yaml`：

- `fall`：非正常任务姿态下的高度塌陷/大倾角持续达到阈值，需视频与物理日志人工可解释地核验；
- `collision`：躯干与障碍的明确接触事件；不把正常足地接触计成碰撞；
- `large_slip`：足端处于接触时，相对地面的切向速度持续过大；它是事件，不必自动等价于任务失败；
- `tracking_failure`：轨迹偏离或终点/时间预算超界；
- `task_success`：目标误差与姿态/速度条件在固定阈值内且没有禁止事件；
- `safe_abort`：安全终止、未达到目标，不能算任务成功；
- `sim_invalid`：NaN、穿模、明显求解器爆炸等，独立统计，不当作机器人真实失败标签。

阈值按当前Go2标称站立高度、尺寸、正常roll/pitch和速度分布确定，全部在dev smoke后冻结。
可用相对高度/角度/持续时间形式，不从test outcomes调阈值。
足端滑移只在contact有效时计算，使用相对地面切向速度而非摆动足速度或原始net-force大小。

---

## 4. Stage R1：先绘制“失效地图”，而不是宣布极端场景成功

预算上限96条dev episodes，例如16个候选配置 × 6个独立reset seeds。
这批只用于范围标定和设计选择，不做最终测试宣称。

### 4.1 候选物理变量

先只用两个主轴：

A. Traction：局部低摩擦区域/已确认有效的全地面材质；示例候选 written friction 为 1.0/0.6/0.3/0.15/0.08。
B. Command latency：在高层命令队列添加已知延迟，例如额外0/1/2/4/8 control ticks；不冒充“电磁干扰”。

可选第三轴：0/5/10/15度的缓坡，只有flat smoke稳定、姿态指标不误用SE(2)时才启用。
这些数值是**仿真开发候选**，不是已校准地面物理参数，更不是允许真机执行范围。
优先单因素，再选少量低摩擦+延迟（或坡度）的联合条件；不要全笛卡尔穷举。
不在此轮加入非对称电机失效、巨大外力、碎石/可变形土壤等额外因素。

### 4.2 任务保持正常可执行

选直走到点、缓弯路径、已见的侧移/转弯路径。名义命令在正常地面应稳定成功。
本轮不要故意同时使用CVPR尚未解决的极端新动作组合，避免命令OOD和物理扰动混为一因。
记录正常任务可执行性。若normal大量失败，先修平台/任务，不训练风险模型。

### 4.3 严重程度如何判定

依据nominal控制器的真实事件率、任务代价和轨迹误差定义：nominal、mild、degraded、failure-heavy。
开发阶段可以寻找nominal失败率约20%–70%的候选可研究区间，但这是方便获得正负事件的采样目标，不是benchmark的固定物理真理。
必须保留所有扫过的参数及结果，包括“没影响”和“几乎必败”；不得只挑有利格子。

物理上几乎无可用牵引、路径本来不可达、仿真无效的区域单独标出。没有安全恢复办法时，最优结果可能是安全停机，不应宣称“导航恢复”。

---

## 5. Stage R2：先用少量分支检验“可挽救性”

预算上限96个分支rollout，例如24个dev query states × 4个干预。
仅在R1发现有效、非全不可控的失效后执行。

### 5.1 固定的四种短时干预

使用与原controller兼容的、幅度/加速度受限的高层命令：

0. nominal continuation；
1. 连续限速至nominal的0.75倍；
2. 连续限速至0.5倍，并限制转向速率；
3. 有限时长的平滑减速/站稳，再按冻结规则恢复跟踪。

具体倍率、转向限幅和恢复时长由dev烟测确认后冻结。不能瞬间跳变扭矩，不能把“停止=永远成功”。
所有干预使用相同目标、路径、时间上限和观测预算，最终同时记录安全与进度。

### 5.2 同源分支要求

从failure发生**以前**的query状态分叉，保持此前历史和控制器内部状态相同。
保存physical state、policy hidden state、动作队列、传感器buffer、随机源；恢复失败时以共同前缀重放替代。
PhysX求解器接触缓存未必能完整恢复：报告 `approximate_branch`、实际起点误差与相同动作重复差异，不称精确反事实。
如果近似分支状态差异大于要研究的效应，停止并修配对。

### 5.3 一个必须明确的上限诊断

对候选集合A，开发阶段可计算：

\[
B_{available}=J(a_{nom})-\min_{a\in A}J(a).
\]

该量使用各分支真实未来，只能称“有限菜单的best-observed branch诊断”，不是部署方法、不是全局最优控制上界。
选择成本J必须提前定义：fall/collision单独主报；辅助J组合进度、时间、路径偏差、安全abort。
不要因为减速减少摔倒却导致所有任务超时，就宣布恢复成功。

记录不同提前量下的干预效果：例如相对于事件前0.2/0.5/1.0秒的近似分支，实际可实现时间取决于日志/模拟重放。

**R2 gate：**若合法干预菜单根本不能带来可靠的安全/任务收益，不继续训练router；输出 NO_RECOVERY_WITH_CURRENT_ACTION_SET，建议后续研究低层控制/路线重规划，而不是不断降低摩擦重跑。

---

## 6. Stage R3：收集名义策略下的风险训练数据

R2支持有恢复空间后，新采最多192条nominal episodes：

`32 个独立 world/task groups × 3 个严重度类别 × 2 个随机实现 = 192`。

按group分16 train / 8 val / 8 test。整个group所有严重度、分支和重复都不跨split。
组内随机实现若改动命令时长，不能用于估计同一command的不可约方差；需要方差诊断时明确锁定完整command和reset状态。
所有额外分支及后续闭环再运行计入总预算，不把它们免费隐去。

严重度配置从dev阶段冻结，最终test groups使用新种子/起点，不再挑选“方法最好”的参数。
本轮主张只覆盖这些普通布局和扰动范围，不能称“救灾泛化”。

### 6.1 输入与标签

预测时：

\[
p_t=P(F^{nom}_{t:t+H}=1\mid h_{\le t},u^{nom}_t),\quad H\in\{0.5,1.0\}\text{s}.
\]

这里的事件是在冻结nominal策略继续运行时的未来事件。若nominal tracker是反馈策略，不读取其未来实际输出当作origin输入。
候选输入：因果IMU/gyro、可获得姿态估计、joint/contact历史、历史requested/consumed命令、当前nominal命令；记录每个通道的真实来源。

禁止输入：摩擦/延迟真值、hazard_ID、未来contact/姿态、未来风险标签、事件剩余时间、从全回合算出的统计量。
若实际只有仿真GT状态，必须标 `PRIVILEGED_DIAGNOSTIC`；不要通过字段改名假装deployable。
基于“过去真实execution”的tracking-error特征也必须有合法可用测量来源，否则仅作oracle输入对照。

标签只能从未干预nominal rollouts生成：事件尚未发生时，预测未来H内是否发生。
已摔倒后的样本不混入“提前预警”。回合提前结束且H内无完整观察时标censored/mask，不能把未观测未来当负样本。
失败发生时，之前满足horizon的窗口可以标正；终止样本不整体删除。

干预后“没有失败”不等于原nominal预测是假阳性：这是不同策略的未来，不能直接改写nominal风险标签。

---

## 7. Stage R4：轻量预警基线，最多6个训练run

### 7.1 方法

1. Rule：可部署的tilt/gyro/contact/跟踪误差阈值；阈值在val定。
2. Logistic：当前和过去统计特征，带L2正则，输出未来事件概率。
3. Small GRU：0.5–1s因果历史，输出未来事件概率。

Logistic/GRU各3个seed，共最多6个主要训练run。
不强制新增execution context，不强制使用V0.7模型，避免两条工作互相等待。
可额外只读评估旧WM风险特征，但必须标其训练分布范围；不能把未经risk数据验证的输出方差直接当安全概率。

概率阈值只用val选。若为类平衡而重采样，必须在原始prevalence的val上验证校准，不能把训练balanced prevalence当真实概率。

### 7.2 指标

- AUPRC/事件基准率；AUROC；Brier/reliability；
- 固定正常误报警预算下的事件召回；
- 每episode或每分钟的误报警数，不能只报frame accuracy；
- 首次正确报警距真实事件的时间；missed events包括在分母；
- 有用提前量：大于实测感知+推理+队列+执行响应时间，而非只看sim time；
- 不同严重度/任务类型分层；
- event/group-level CI，不能对重叠窗口当独立样本bootstrap。

不可见的突发扰动发生前没有任何观测线索时，不能要求模型预知随机发生时刻。区分pre-onset anticipation与post-onset early detection。

**R4 gate：**若只在事件之后识别或实测提前量不够，输出 INSUFFICIENT_ACTIONABLE_LEAD；不声称可部署risk router。

---

## 8. Stage R5：小规模闭环路由验证

只有R2有干预收益、R4有有用提前量时执行。
在test组上比较四种系统；建议同一test group×严重度×随机实现匹配运行：

| 系统 | 定义 |
|---|---|
| S0 nominal | 原tracker + 原低层controller |
| S1 always-on safeguard | 全程启用同一个冻结的限速/纠正规则 |
| S2 heuristic-gated | 简单Rule gate触发同一个corrector |
| S3 learned-risk-gated | val选定轻量预测器触发同一个corrector |

**重要：本轮corrector可以是R2选定的固定规则。若没有使用world model做候选后果预测，就只能宣称“风险路由pilot”，不能宣称“世界模型贡献已经验证”。**
世界模型预测式corrector的后续增量，应另比较相同gate下的规则corrector与WM selector；本轮不强行增加这项训练。

### 8.1 路由规则

\[
g_t=\begin{cases}
1 & p_t\ge\gamma_{on},\\
0 & p_t\le\gamma_{off}\ \text{且满足最短驻留时间},\\
g_{t-1} & \text{其它},
\end{cases}\quad\gamma_{on}>\gamma_{off}.
\]

阈值及最短on/off持续时间由val冻结。动作经相同幅度/变化率约束，不瞬时跳变；危险状态下优先安全abort，不宣称所有情况可恢复。

轻量gate必须可在昂贵WM未运行时获得输入。若gate依赖每步完整WM推理，不能宣称“平时关闭WM省算力”。统计传感器、encoder、gate、corrector全部开销。
关门时`u_final==u_nom`仅是实现检查，不保证整个系统正常性能无损；误触发、缓存状态、推理延迟仍可能造成损失。

### 8.2 主指标

分别报告正常与危险子集：
- task success；fall/collision；safe_abort；timeout；
- endpoint/path error；完成时间；
- intervention fraction、switch count、false activation；
- 每次推理wall latency（median/p95）、决策陈旧度、超时回退；
- tail cost可报P90/CVaR，但必须保留原始分布与样本数。

建议在预注册中设置正常成功率非劣容差，例如绝对2个百分点；这是投入判断阈值，不是安全保证。
有限样本下区间不能排除退化，就写 NORMAL_NONINFERIORITY=INCONCLUSIVE，不因均值相等写“无损”。
低摔倒率不能靠永远停住取得；到达目标与safe_abort同时报告。

比较重点：
1. S3是否减少危险事件同时保持任务进展？
2. S3是否比S1少干预/省时，且没有显著增加危险？
3. S3相对简单S2是否有额外收益？

如果S2已达到相同效果，保留这个负结果；它意味着目前未证明learned router的必要性，不是要求继续调test直到S3获胜。

---

## 9. 并行计算、预算与停机条件

建议上限：
- R0 smoke <=24 episodes；
- R1 severity-dev <=96；
- R2 recoverability <=96 branch rollouts；
- R3 main nominal <=192；
- R5新增closed-loop <=144（nominal已有48条test可复用时，总对照约192）；
- **总采集/分支/重跑 <=600**，全部计入，不删掉失败run后重新计数。

神经训练 <=6主run，<=20 GPU-hours；全流程<=120 GPU-hours为规划上限。实测超出先停并汇报，不强行跑完。
最多1张用户指定空闲GPU做采集、另1张用于训练/统计；不抢占CVPR任务，不根据显存空闲就结束别人进程。

automatic stop：
- 数值爆炸/材质写入无效/命令时间错位；
- normal任务本身大量失败；
- 根本无可挽救区间；
- 风险警报没有可用提前量；
- 预算触顶；
- 检测到真实硬件控制路径。

停机不是失败掩饰；保存所有数据、原因和下一步建议。

---

## 10. 交付目录与完整证据

```text
results/risk_r01/
  prereg/PROTOCOL_R01.md
  prereg/event_definition.yaml
  prereg/hazard_matrix.yaml
  prereg/action_menu.yaml
  audit/material_reset_time_tests.json
  audit/sensor_source_roles.json
  manifests/episodes.csv
  manifests/events.csv
  manifests/branch_groups.csv
  manifests/split_groups.json
  metrics/severity_map.csv
  metrics/recoverability_by_lead.csv
  metrics/risk_event_metrics.csv
  metrics/risk_calibration.csv
  metrics/closed_loop_by_group.csv
  metrics/normal_vs_hazard.csv
  metrics/latency_and_compute.csv
  predictions/risk_predictions.npz
  raw/representative_traces/ + hashes
  checkpoints/ + train_logs/ + configs/
  code_snapshot/ + git_diff.patch
  figures/
  report/RISK_R01_FULL_REPORT.md
  report/RISK_R01_DECISION.md
  README_REPRODUCE.md
```

每条prediction保存origin、source sample ticks、future有效区间、nominal标签定义、model/version、分数、gate状态；闭环保存nominal/final/consumed三个命令和实际execution。

至少图：严重度—失效率曲线；相同起点不同合法干预的轨迹；事件前风险曲线；正常/危险双表；gate激活率与任务进度。不同图各自独立保存PNG/PDF，附CSV。

代表性案例不可只挑成功：固定随机抽样+中位表现+所有failure类别各一个，说明选择规则。

---

## 11. 最终决策必须分层，不写一个万能GO

```text
SIMULATION_VALIDITY: PASS / BLOCKED
NOMINAL_TASK_VALIDITY: PASS / BLOCKED
MEASURABLE_FAILURE_REGIME: SUPPORTED / NOT_FOUND / INCONCLUSIVE
RECOVERABLE_WITH_ALLOWED_ACTIONS: SUPPORTED / NOT_SUPPORTED / INCONCLUSIVE
ACTIONABLE_EARLY_WARNING: SUPPORTED / NOT_SUPPORTED / INCONCLUSIVE
LEARNED_GATE_VS_RULE: BETTER / SIMILAR / WORSE / INCONCLUSIVE / NOT_RUN
NORMAL_NONINFERIORITY: SUPPORTED / NOT_SUPPORTED / INCONCLUSIVE / NOT_RUN
WORLD_MODEL_INCREMENT: NOT_TESTED（本轮默认）/ 单独证据说明
NEXT_MAIN_ACTION: 只推荐一条
```

允许结论：
A. 有可挽救事件和提前量：下一轮比较WM预测式修正与简单规则；
B. 有事件但不可挽救：需要改变行动集合/低层控制/重规划，不能继续单纯训练router；
C. 可挽救但无法及时预警：改进传感可观测性/延迟，先别训练大模型；
D. 简单Rule已足够：记录适用范围，尚不能声称learned gate新贡献；
E. 有潜力但独立样本不足：只扩独立组和预定边界案例。

报告写清：这是灾害相关物理因素的简化模拟，不是完成了救灾benchmark或真实安全验证。

---

## 12. 官方实现参考与来源边界

这是根据用户提出的独立Risk假设设计的新实验；没有把V0.6.2的普通动作预测结果当作风险修正已经成功。

实现参考，以本机版本源码/readback为准，不自动升级：
- 材质与combine mode： https://isaac-sim.github.io/IsaacLab/v2.3.0/_modules/isaaclab/sim/spawners/materials/physics_materials_cfg.html
- PhysX材质schema： https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.2/dev_guide/schemas/physxschema.html
- Contact/IMU sensor： https://isaac-sim.github.io/IsaacLab/v2.3.0/source/api/lab/isaaclab.sensors.html

完整执行到能支持的阶段后生成决策并停止。不要继续接V-JEPA、真机危险实验、MDA或HumanoidVLN。
