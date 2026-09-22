# A_AUDIT_DECISION — V0.6 阶段A 判定

> 日期：2026-09-22 · commit 见 source_inventory.json · 纯离线审计（未训练/未采数）
> 证据文件：audit/*, manifests/*, metrics/*（results/v0_6_validity_transfer/）

```text
DATA_CAUSALITY: PASS（带强制注记）
PAIR_INDEPENDENCE: PASS（旧 pair-level CI 作废；two-way/anchor 重采样 + LOEO 下主结论稳健）
SIMPLE_BASELINE_CHECK: mixed（分布依赖：random test 模型胜 / probe 基线胜或平 / OOD direct 胜 context）
CONTEXT_OUTPUT_SENSITIVITY: supported
PHYSICAL_CONTEXT_TRANSFER: inconclusive（方向性证据存在但双向不对称；等效性论据已撤回）
NEXT_STAGE_ALLOWED: yes（无 P0；阶段B smoke 必须含摩擦 readback 与 PRE_REGISTRATION）
```

## DATA_CAUSALITY: PASS（带强制注记）

证据（audit/causal_time_frame_audit.md，全部自动单测）：
- 输入因果：history ≤ origin；改 future 标签/改 L 之前历史，输出差 = 0.0；改 future command，
  输出变 0.77（确认其为有效输入）。
- frame：quaternion (w,x,y,z)，数值验证 max err 2.4e-07；body wz 与世界 yaw 速率 corr 0.87
  （roll/pitch 非零时不恒等，评价保持 body-frame 语义）。
- forecast：fixed-origin 完整 [40,3]，非滚动一步拼接。

强制注记（bugs_and_impact.md #1/#2/#5）：
1. e/u 存在一步滞后（定义性，全链一致）；
2. 50→20Hz 最近邻无抗混叠，真实 dt {0.04,0.06}s；
3. 模型输入含 GT 速度/姿态（privileged），无 deployable 对照；
4. 摩擦写入值未经运行时 readback，combine rule 未定（转 B smoke）。

## PAIR_INDEPENDENCE: PASS（经重做依赖统计）

- 旧 pair/cluster CI 作废（synthetic 验证：窗级 bootstrap 覆盖率 8%）。
- 新 two-way bootstrap（source×target 独立重采样，2000 次，seed=42；episode 与 anchor 两级）：
  - S_dir 四方向 +0.30~+0.39，CI 全部不含 0（如 N→L [+0.279,+0.332]）；
  - P_target 不对称维持：N→L −0.172 [−0.238,−0.099]、L→N +0.687 [+0.627,+0.750]；
  - M_shift cross−same 四方向 CI 全部为正（+0.37~+1.48）；
  - E_swap cross−same：VL→N CI 跨 0 → 该方向"swap 不劣于同条件 swap"结论降级。
- LOEO：去掉任一 source/target episode，P_target/S_dir 变化 <0.02，无单点支配。
- equal-episode / equal-anchor 估计与 pair-weighted 一致（无高复用 episode 主导）。

## SIMPLE_BASELINE_CHECK: mixed

同一批 probe 窗口、同一输入（metrics/baseline_summary.csv, unique-window L2）：

| model | unique_L2 | MAE_vx | MAE_vy | MAE_wz |
|---|---:|---:|---:|---:|
| action_only | **1.222** | 0.069 | 0.064 | 0.100 |
| command-copy | 1.337 | 0.067 | 0.067 | 0.102 |
| direct | 1.309 | 0.076 | 0.072 | 0.129 |
| context | 1.329 | 0.072 | 0.078 | 0.130 |
| persistence | 2.665 | 0.120 | 0.124 | 0.237 |

- **probe 分布：baseline_better**——action_only（零历史）最优；context 不胜 command-copy
  （two-way：N→L E_native−E_ccopy = −0.085 [−0.170,−0.012]，显著更差；L→N/VL→N CI 跨 0）。
- **random test：model_better**——context MAE 0.128 vs command-copy（=mean|r|≈0.3 量级）；
  但 context 仅微弱胜 direct（0.1278 vs 0.1291）。
- **OOD test：direct 胜 context**（0.1997 vs 0.2103）。
- 结论：context 的增量在 in-distribution 存在但微小；在 probe/OOD 分布为负。
  不能写"模型超过简单基线"，也不能写"模型全面不如基线"——按分布分开陈述。

## CONTEXT_OUTPUT_SENSITIVITY: supported

- c 改变输出：swap 与 native 预测直接距离 = native error 的 20.7%–27.5%（四方向）；
- 位移方向物理一致：S_dir two-way CI 不含 0；M_shift cross>same 全方向 CI 为正；
- 剂量效应（vlow>low 的 |M_shift|）维持。

## PHYSICAL_CONTEXT_TRANSFER: inconclusive

支持方：L→N / VL→N 方向 P_target 显著为正（two-way CI 不含 0），E_swap cross<same；
反对方：N→L / N→VL P_target 显著为负；geometry 分解显示 N→L 的 swap 位移 65% 在
target 误差的正交方向（parallel 0.67 vs orthogonal 1.22）；vy 的 per-axis squared gain
实际为正（旧"vy 无贡献"系不可加 norm 口径误读）；N/L "native error>工况差距"论据与
"teleport 统计不可分"论据均已撤回。→ 不能判 supported，也不能判 unsupported。

## NEXT_STAGE_ALLOWED: yes

无 P0（训练标签/输入因果链完整）。进入阶段B 的强制前置：
1. PRE_REGISTRATION.md 锁定协议（含等效容差、命令族、seed、split、预算）；
2. B smoke 12 条必须验证：摩擦/执行器 readback、reset 完整性、日志时间戳；
3. 新数据按 anchor group 切分 train/val/test；保留 50Hz 原始日志；
4. 阶段C 预算先实测 2 个 run。
