# V0.5 Diagnostic Decision

> 2026-09-21 · 纯后验诊断，未训练/未采数/未改模型。详见同目录 `V0_5_DIAGNOSTIC_REPORT.md`。

## Gate M — Context Mechanism
Decision: **PASS**

Evidence:
1. S_dir 四方向 +0.30 ~ +0.40，cluster bootstrap 95% CI 紧致且不贴 0（N→L [+0.298,+0.309]，L→N [+0.315,+0.326]，vlow 两方向更高）；91–96% pair S_dir>0
2. M_cross − M_same cluster CI 全部为正（N↔L +0.88 [+0.865,+0.897]，low 侧 +0.37，vlow 两组 +1.48 / +0.69）——swap 显著优于同条件对照
3. Regime teleport ratio 1.016–1.029（四方向）：c_B 注入后模型预测与 friction_B 原生预测统计上不可分，且呈剂量效应（vlow > low）；6 参数全局 affine 不改变以上任何方向性结论

## Gate C — Calibration
Decision: **FAIL**

Evidence:
1. 原生 P_target 双向不对称：N→L −0.172（cluster CI [−0.184,−0.159] 整体偏负）vs L→N +0.687；per-axis 分解显示 N→L 失败由 wz（−0.247）主导、vy（−0.076）次之、vx 为正（+0.032）
2. Per-axis calibration slope a=Cov(r̂,r)/Var(r) 系统性 <1（low vx 0.81、low wz 0.46、vlow vx 0.61、vlow wz 0.39）——大 residual 低估；同时 probe 温和窗口上小 residual 相对过冲 40%（native 性质）——校准误差是 scale-dependent 的
3. wz 通道全条件弱可预测（Pearson 0.55–0.64）；target 条件原生误差（low 1.35 / vlow 1.98）大于条件间真实差距（1.22 / 1.65），swap 增益被 native 噪声吞没

## Metric Audit
Status: **未发现影响结论的 metric bug。** 0.107 / 0.076 / 0.320 三个数字口径互不相同
（probe 窗口 pooled 混合单位 vs probe 同窗口真实残差 vs random episodes per-axis 全时段），
不可直接比较但各自口径内自洽；逐 timestep 复核 12 窗 × 4 步，r=e−u 恒等式与实现一致
（max diff 2.4e-07）。见 `audit/metric_definition_audit.md`、`audit/metric_bug_audit.md`。

## Cluster Bootstrap
Status: **v2 结论对 pair 相关性稳健。** cluster=(source_episode, target_episode) 5000 次重采样，
CI 相对 pair-level 几乎不变宽；S_dir 符号、P_target 不对称、cross>same 全部维持。
见 `bootstrap/pair_bootstrap_vs_cluster_bootstrap.csv`。

## Transition vs Steady
Finding: N→L 的负 P_target **主要来自 steady 窗口**（τ=0.5s：steady −0.269 vs transition −0.068；
vlow 更极端 −0.713 vs −0.141），τ∈{0.25,0.5,1.0}s 敏感性下排序不变；S_dir 在 transition 处
一致更高（0.38–0.46）。normal 条件 |r| 的 transition/steady 比为 vx 5.48× / vy 4.67× / wz 1.86×
——当前 r_t 定义把大量 normal controller transient 计入 "execution mismatch"，steady 段
才是更纯粹的 condition-induced 信号，而模型恰好在 steady 段校准最差。

## Affine Calibration Diagnostic
Finding: val split 拟合的 6 参数全局 affine（α=[0.867, 0.890, 0.603]，β≈0，无 condition 标签输入）
**基本修复主门**：N→L P_target −0.172 → +0.009（CI [−0.003,+0.021]），L→N +0.687 → +0.486，
ΔE_self 双向转正（+0.476 / +0.016）；vlow 方向部分改善（N→VL −0.377 → −0.083 仍未闭合）。
S_dir 不变。结论：context 方向信息有效，瓶颈集中在 output 幅值校准——支持"机制成立、校准不足"。

## Root Cause Ranking
1. **B. Context 被使用，但 calibration 不好**（主因：affine 近闭合主门、slope<1、probe 过冲 40%）
2. **C. residual 定义混入大量 normal transient**（次因：N→L 失败集中 steady；normal |r| transition 5.5×）
3. **E. wz 通道弱可预测性**（N→L 负 P_target 主由 wz 贡献；拉低 overall 向量指标）

（A. context 未被使用 / D. state-pair confound：均被证据排除）

## Recommended Next Step

**A. Proceed to V1-Friction data scaling**

理由：Gate M 已 PASS 且统计稳健，Gate C 失败形态（scale-dependent slope、数据量敏感的
regression-to-mean、probe/random 激励覆盖不足）正是数据规模与覆盖问题；V1 扩数据
（更多 episode、更多 friction 档位、probe 与 random 激励兼顾）预期直接改善 Gate C。
建议把 residual 重定义（选项 C 的 e_nom 名义模型分解 controller transient）作为 V1 的
设计讨论一并考虑，但本轮不执行。

（仅 recommendation，未进入下一步执行。）
