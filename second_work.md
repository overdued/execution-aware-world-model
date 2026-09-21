你现在继续处理这个公开仓库：

https://github.com/overdued/execution-aware-world-model.git

当前项目是 Execution-Aware World Model（EA-WM）。

你需要基于仓库中现有的：

- V0 数据
- Action-only / Direct / Context 三个模型
- Context Predictor checkpoint
- Context Swap v2 全部结果
- pair_level_metrics.csv
- GO_NO_GO.md
- CONTEXT_SWAP_REPORT.md
- diagnostics.json
- representative_pairs.npz
- pair manifest / quality / audit 文件

执行一个新的：

# V0.5 Diagnostic Stage

目标不是继续训练，而是彻底确认当前 Context Swap NO-GO 的真实原因到底是不是：

“Execution Context 已经被 predictor 使用，但 low-friction execution prediction 存在 calibration / amplitude bias。”

============================================================
0. 非常重要：本轮禁止事项
============================================================

本轮：

- 不重新训练任何模型
- 不采集新的 Isaac 数据
- 不扩展到 1000–3000 episodes
- 不修改 ContextEncoder
- 不修改 predictor architecture
- 不加入 FiLM
- 不加入 context dropout
- 不加入 uncertainty head
- 不加入 V-JEPA 2
- 不加入 RGB
- 不做 Context Adaptation
- 不做 online correction
- 不做 policy
- 不修改已有 V0 / Context Swap 原始结果

本轮只允许：

- 读取现有 checkpoint
- 读取现有 dataset / pair / npz
- 重新计算统计量
- 做诊断性 post-hoc calibration
- 输出新的分析报告和图

如果发现现有代码存在影响结论的统计 / metric bug：

不要静默修复然后继续。

必须：

1. 明确记录 bug
2. 给出影响范围
3. 保存 bug 前结果
4. 最小修复
5. 重算受影响指标
6. 在最终报告单独写“Metric / Evaluation Bug Audit”

============================================================
1. 当前已知背景
============================================================

V0 已验证：

1. execution mismatch 存在：

r_t = e_t - u_t

2. friction 越低，residual 越大

3. execution mismatch 可预测

4. context embedding PCA 与 friction condition 明显相关

Context Swap v2 当前严格判定：

NO-GO

但已有证据显示：

- S_dir 四方向约 +0.30 ~ +0.40
- 91% ~ 96% pair 的方向为正
- vx 上 S_dir 约 0.49 / 0.50
- cross-condition swap effect 明显大于 same-condition swap
- regime teleport ratio 约 1.016 ~ 1.029

所以当前核心假设是：

Context mechanism 可能已经成立，
但 execution decoder / predictor 对 low-friction regime 的幅值校准不够准确。

目前还报告过：

|r_hat|_low ≈ 0.107
|r|_low ≈ 0.076

即约 40% overshoot。

但是 V0 quick-look 中 friction_low 的 raw residual magnitude 又出现了约：

r_vx ≈ 0.320
r_vy ≈ 0.293
r_wz ≈ 0.291

这两个统计口径明显不同。

因此第一任务必须先把 metric definition 完全审清。

============================================================
2. 新增输出目录
============================================================

请创建：

results/v0_5_diagnostics/

建议结构：

results/v0_5_diagnostics/
    audit/
        metric_definition_audit.md
        metric_trace_examples.csv
        metric_bug_audit.md

    bootstrap/
        cluster_bootstrap_summary.csv
        pair_bootstrap_vs_cluster_bootstrap.csv
        cluster_bootstrap_raw.csv

    calibration/
        per_axis_calibration.csv
        per_condition_calibration.csv
        global_affine_params.json
        affine_recomputed_swap_metrics.csv

    transition/
        transition_vs_steady_metrics.csv
        transition_window_manifest.csv

    figures/
        CAL_*.png
        CAL_*.pdf
        BOOT_*.png
        BOOT_*.pdf
        TRANS_*.png
        TRANS_*.pdf

    report/
        V0_5_DIAGNOSTIC_REPORT.md
        V0_5_DECISION.md

不要覆盖：

results/v0/
results/context_swap_v2/

============================================================
3. Task A：Metric Definition Audit
============================================================

这是第一优先级。

必须查清下面三个数到底是什么：

A.
0.107

B.
0.076

C.
约 0.320 / 0.293 / 0.291

分别回答：

- 来自哪个文件
- 来自哪个函数
- 使用哪个 dataset subset
- 哪个 condition
- 哪个 axis
- raw 还是 normalized
- per-step / per-window / per-horizon
- mean absolute value / MAE / RMSE / vector norm / L1 / L2
- 是否 horizon flatten
- 是否经过 inverse normalization
- 单位是什么
- 是否是 true residual
- 是否是 predicted residual
- 是否是 prediction error
- 是否用了 matched pairs 子集

必须追踪代码调用链。

不要只靠猜。

------------------------------------------------------------
3.1 必须做人肉可复核例子
------------------------------------------------------------

随机选至少 10 个 representative windows。

保存：

metric_trace_examples.csv

每一行至少包含：

episode_id
pair_id
condition
probe
timestamp
u_vx
u_vy
u_wz
e_vx
e_vy
e_wz
r_vx
r_vy
r_wz
rhat_vx
rhat_vy
rhat_wz

以及根据当前 metric function 计算出来的：

reported_metric

然后独立使用最简单 numpy / torch 运算重新计算：

r = e - u

以及对应：

abs(r)
abs(rhat)
abs(rhat-r)

确认现有 metric 实现与数学定义一致。

------------------------------------------------------------
3.2 Audit 输出
------------------------------------------------------------

生成：

audit/metric_definition_audit.md

格式：

# Metric Definition Audit

## 0.107
Definition:
Source file:
Code path:
Dataset subset:
Normalization:
Units:

## 0.076
...

## 0.320 / 0.293 / 0.291
...

## Are these quantities directly comparable?
YES / NO

Reason:
...

如果不能直接比较：

必须明确说明为什么之前“40% overshoot”的表述是否仍然成立。

============================================================
4. Task B：重新做统计独立性分析
============================================================

目前 Context Swap 有约：

25772 pairs

但这些 pair 很可能来自同一 episode / 相邻 window，
不是独立统计样本。

所以现有 pair-level bootstrap 可能 CI 过窄。

本任务必须重新做：

# Cluster Bootstrap

------------------------------------------------------------
4.1 Cluster 定义
------------------------------------------------------------

优先级：

第一优先：
seed + source_episode + target_episode

如果 seed metadata 不完整：

使用：

source_episode + target_episode

作为 cluster。

同一 cluster 内所有 windows 视为相关数据。

------------------------------------------------------------
4.2 至少计算两套 bootstrap
------------------------------------------------------------

A. 原有 pair-level bootstrap

B. cluster-level bootstrap

建议：

n_bootstrap >= 5000

固定 random seed。

------------------------------------------------------------
4.3 重新统计指标
------------------------------------------------------------

至少：

Delta_E_self

P_target

S_dir

S_mag

M_cross

M_same

M_cross - M_same

分别报告：

normal -> friction_low
friction_low -> normal

如果已有 friction_vlow：

normal -> friction_vlow
friction_vlow -> normal

------------------------------------------------------------
4.4 必须比较
------------------------------------------------------------

输出：

pair_bootstrap_vs_cluster_bootstrap.csv

列：

metric
transition
pair_mean
pair_ci_low
pair_ci_high
cluster_mean
cluster_ci_low
cluster_ci_high
num_pairs
num_clusters

重点回答：

1. 原结论是否仍然成立？
2. CI 是否明显变宽？
3. S_dir 是否仍显著 > 0？
4. P_target 的正负号是否变化？
5. cross > same 是否仍成立？

============================================================
5. Task C：Per-axis Calibration Audit
============================================================

当前最大疑点是：

方向正确，但 amplitude calibration 不准。

必须分别分析：

vx
vy
wz

不能只看 overall。

------------------------------------------------------------
5.1 每个 axis 计算
------------------------------------------------------------

对每个 condition：

normal
friction_mid
friction_low
friction_vlow

如果有足够数据再加其他 condition。

计算：

true residual mean
pred residual mean

true |residual| mean
pred |residual| mean

bias：

b_j = mean(rhat_j - r_j)

MAE

RMSE

Pearson correlation

calibration slope：

a_j = Cov(rhat_j, r_j) / Var(r_j)

以及带 intercept 的线性拟合：

rhat_j = a * r_j + b

保存：

per_axis_calibration.csv
per_condition_calibration.csv

------------------------------------------------------------
5.2 Calibration plot
------------------------------------------------------------

每个主要 friction condition、每个 axis：

x = true residual
y = predicted residual

画：

- scatter / density
- y = x reference
- fitted linear regression

至少输出：

CAL_normal_vx
CAL_low_vx
CAL_vlow_vx

以及 vy / wz。

============================================================
6. Task D：Transition vs Steady State Analysis
============================================================

根据已有 Figure E，
low-friction 的主要差异可能集中在：

- command startup
- reversal
- acceleration
- overshoot
- lateral slip
- transient

所以现在的：

r_t = e_t - u_t

可能同时包含：

1. normal controller transient
2. condition-induced mismatch

必须拆开分析。

------------------------------------------------------------
6.1 Command change detection
------------------------------------------------------------

定义：

Delta_u_t = ||u_t - u_{t-1}||

根据当前 command 生成机制选择合理 threshold。

threshold 必须记录在 config / report。

检测：

command change point

------------------------------------------------------------
6.2 定义窗口
------------------------------------------------------------

例如：

Transition:
command change 后 0 ~ 0.5 s

Steady:
command 保持稳定至少 0.5 s 后

具体值根据 20 Hz sampling 合理设置。

请做一个小 sensitivity check，例如：

transition = 0.25 s
transition = 0.5 s
transition = 1.0 s

确认结论不是完全依赖单一 threshold。

------------------------------------------------------------
6.3 分别统计
------------------------------------------------------------

对 transition / steady：

E_correct
E_swap
Delta_E_self
D_before
D_after
P_target
S_dir
S_mag

以及：

MAE_vx
MAE_vy
MAE_wz

重点回答：

N -> L 方向的负 P_target：

主要来自 transition 还是 steady state？

============================================================
7. Task E：Global Affine Calibration Diagnostic
============================================================

这是诊断实验，不是正式方法。

禁止使用：

friction coefficient
condition ID
terrain label

作为 calibration 输入。

------------------------------------------------------------
7.1 Calibration function
------------------------------------------------------------

只允许每个 axis 一个全局 affine：

r_cal_j = alpha_j * rhat_j + beta_j

j ∈ {vx, vy,wz}

参数只能在 validation split 上拟合。

test / swap pair 不能用于拟合。

保存：

global_affine_params.json

------------------------------------------------------------
7.2 重新计算 Context Swap
------------------------------------------------------------

将：

rhat

替换为：

r_cal

重新计算：

E_correct
E_swap
Delta_E_self
D_before
D_after
P_target
S_dir
S_mag

注意：

Context embedding 不变。
Context Swap 不变。
只对 output 做简单 calibration。

------------------------------------------------------------
7.3 核心问题
------------------------------------------------------------

回答：

1. N -> L 的 P_target 是否从负变正？
2. L -> N 是否保持为正？
3. S_dir 是否保持？
4. cross > same 是否保持？
5. A/B/E gate 是否改善？
6. 一个极简单 global affine 是否能解释大部分 NO-GO？

如果答案是 YES：

支持：

“Context mechanism 有效，主要瓶颈是 output calibration。”

如果答案是 NO：

不要继续声称 calibration 是唯一根因。

============================================================
8. Task F：重新定义两个 Gate
============================================================

最终报告不要只给一个模糊的 NO-GO。

必须分别判定：

# Gate M — Mechanism Gate

问题：

c_t 是否真正改变 execution dynamics regime？

主要证据：

- S_dir
- positive pair ratio
- cross vs same swap
- regime teleport
- cluster bootstrap

输出：

PASS
或
FAIL

------------------------------------------------------------

# Gate C — Calibration Gate

问题：

切换到目标 context 后，execution prediction 是否足够准确？

主要证据：

- P_target 双方向
- target native prediction error
- per-axis bias
- calibration slope
- affine diagnostic

输出：

PASS
或
FAIL

============================================================
9. 一个重要的数学诊断问题
============================================================

请额外分析：

当前 residual：

r_t = e_t - u_t

是否把 normal-condition 下的 controller latency / gait dynamics / inertia
也大量算进了 “execution mismatch”。

不要修改模型。

只做诊断。

比较：

Normal condition 下：

r_t

在：

transition
vs
steady

的分布。

如果 normal transition residual 很大，而 steady residual 很小：

在报告中明确指出：

当前 r_t 同时包含：

normal controller dynamics
+
condition-induced execution deviation

并讨论下一阶段是否值得引入：

e_nom = F_nom(h,u)

delta_e = e - e_nom

但本轮禁止实现这个新模型。

============================================================
10. 最终必须回答的科研问题
============================================================

最终报告必须明确回答以下 8 个问题：

Q1.
0.107 / 0.076 / 0.320 这些数能否直接比较？

Q2.
“low friction 高估约 40%”这个结论是否成立？

Q3.
cluster bootstrap 后：

S_dir > 0

是否仍稳定成立？

Q4.
cross-condition swap > same-condition swap
是否仍稳定成立？

Q5.
N -> L 的 P_target < 0
主要由哪个 axis 导致？

Q6.
N -> L 失败
主要来自 transition 还是 steady segment？

Q7.
简单 global affine calibration
能否修复双向 P_target？

Q8.
当前 NO-GO 的最合理解释到底是：

A. Context 没被 predictor 使用

B. Context 被使用，但 calibration 不好

C. residual 定义混入太多 normal transient

D. state / pair confound

E. 其他

允许多因素，但必须按证据排序。

============================================================
11. V0.5_DECISION.md
============================================================

最终生成：

report/V0_5_DECISION.md

格式：

# V0.5 Diagnostic Decision

## Gate M — Context Mechanism
Decision: PASS / FAIL

Evidence:
1.
2.
3.

## Gate C — Calibration
Decision: PASS / FAIL

Evidence:
1.
2.
3.

## Metric Audit
Status:
...

## Cluster Bootstrap
Status:
...

## Transition vs Steady
Finding:
...

## Affine Calibration Diagnostic
Finding:
...

## Root Cause Ranking
1.
2.
3.

## Recommended Next Step

只能从以下候选中选：

A.
Proceed to V1-Friction data scaling

B.
Fix metric/evaluation bug first

C.
Redefine residual / nominal execution target first

D.
Modify context-conditioning architecture first

E.
Other（必须解释）

不要进入下一步执行，只给 recommendation。

============================================================
12. V0_5_DIAGNOSTIC_REPORT.md
============================================================

生成完整报告：

report/V0_5_DIAGNOSTIC_REPORT.md

需要包含：

1. Repo commit
2. Checkpoint
3. Dataset
4. Metric definitions
5. Cluster bootstrap
6. Per-axis calibration
7. Per-condition calibration
8. Transition vs steady
9. Affine diagnostic
10. Gate M
11. Gate C
12. Root-cause ranking
13. Limitations
14. Recommended next experiment

============================================================
13. Git 要求
============================================================

请保留清晰 git 历史。

建议：

commit 1:
add v0.5 diagnostic framework

commit 2:
add metric audit and cluster bootstrap

commit 3:
add calibration and transition diagnostics

commit 4:
add affine calibration diagnostic

commit 5:
add final v0.5 reports

不要重写已有 11 个 commit。

============================================================
14. 最终交付给我
============================================================

跑完后请给我：

1.
V0_5_DECISION.md 全文

2.
V0_5_DIAGNOSTIC_REPORT.md 全文

3.
下面几个最关键表格：

pair_bootstrap_vs_cluster_bootstrap.csv

per_axis_calibration.csv

per_condition_calibration.csv

affine_recomputed_swap_metrics.csv

transition_vs_steady_metrics.csv

4.
关键数字：

- cluster-bootstrap S_dir
- cluster-bootstrap P_target
- cluster-bootstrap cross-same difference
- low-friction vx calibration slope
- low-friction vx bias
- N→L transition P_target
- N→L steady P_target
- affine calibration 前后的 N→L / L→N P_target

5.
关键图：

- low-friction vx true-vs-pred calibration
- cluster bootstrap comparison
- transition vs steady
- affine calibration before/after

6.
最终只能推荐下一阶段，不要自行执行。

============================================================
15. 执行顺序
============================================================

严格：

Step 1
读取现有仓库和 Context Swap v2 全部结果。

Step 2
Metric Definition Audit。

Step 3
确认是否存在 metric bug。

如果存在影响结论的 bug：
修复后重新计算受影响结果，并在报告记录。

Step 4
Cluster Bootstrap。

Step 5
Per-axis / per-condition Calibration Audit。

Step 6
Transition vs Steady Analysis。

Step 7
Global Affine Calibration Diagnostic。

Step 8
Gate M / Gate C 判定。

Step 9
输出 V0_5_DIAGNOSTIC_REPORT.md。

Step 10
输出 V0_5_DECISION.md。

Step 11
停止。

不要训练。
不要采数据。
不要继续 Context Adaptation。
不要接 V-JEPA 2。