# V0.5 Diagnostic Report（second_work.md）

> 日期：2026-09-21 · 纯后验诊断（未训练/未采数/未改模型/未改 V0 与 Context Swap v2 原始结果）
> 输出目录：`results/v0_5_diagnostics/`（已入 git）

## 1. Repo commit

代码：`c9cbdf9`（v0.5 诊断框架 Task 0–E）；V0 数据/模型：`e4e1e6c`；Context Swap v2：`27f8972`

## 2. Checkpoint

`/media/hdd1/yuhang/checkpoints/execution_wm/v0/context/best.pt`
sha256 `670e6feb…3643db` · best val loss 0.016036 · context_dim=8 · L=20 · H=40（未改动）

## 3. Dataset

只读使用：V0 163 episodes + controlled probes 96 episodes + context_swap_v2 的
pair_manifest（25772 pairs）+ 新建预测缓存 `raw/pair_pred_cache.npz`
（每对 u/e_A/e_B/r̂_correct/r̂_swap/t_since_change；重算标量与 v2 CSV 对拍 max|diff|=4.8e-07）。

## 4. Metric definitions（Task A，`audit/metric_definition_audit.md`）

| 数字 | 口径 |
|---|---|
| 0.107 | `context_swap/diagnostics.py`：c_low 预测的 mean \|ê−u\|，probe 窗口，pooled 3 轴 × horizon，混合单位 |
| 0.076 | 同上窗口的真实 mean \|e−u\|（与 0.107 同口径，比值 1.40 在该口径内自洽） |
| 0.320/0.293/0.291 | `eval/quick_look.py`：friction_low **random** episodes 的 per-axis mean \|r\|（全部 timestep） |

**三者不可直接比较**（subset 不同：probe 温和命令 vs random 频繁跳变；聚合不同：pooled vs per-axis）。
逐 timestep 人肉复核（12 窗 × 4 步，`audit/metric_trace_examples.csv`）：r=e−u 恒等式与
metric 实现一致，max diff 2.4e-07。**Metric Bug Audit：未发现影响结论的 bug**（`audit/metric_bug_audit.md`）。

## 5. Cluster bootstrap（Task B，`bootstrap/`）

cluster=(source_episode, target_episode)（v0 probe 无 seed metadata，按 §4.1 回退），5000 次重采样：

| metric | direction | pair CI | **cluster CI** | pairs / clusters |
|---|---|---|---|---|
| S_dir | normal→low | [+0.298,+0.309] | **[+0.298,+0.309]** | 5510 / 951 |
| S_dir | low→normal | [+0.315,+0.326] | **[+0.315,+0.326]** | 5510 / 951 |
| P_target | normal→low | [−0.184,−0.159] | [−0.184,−0.159] | 同上 |
| P_target | low→normal | [+0.674,+0.699] | [+0.674,+0.700] | 同上 |
| M_cross−M_same | normal↔low | — | **[+0.865,+0.897]** | 12302 / 1503 |
| M_cross−M_same | low 侧 | — | [+0.346,+0.390] | 7554 / 1311 |

**CI 几乎不变宽**（同 episode 内窗口相关性不改变结论）：S_dir>0、P_target 符号、
cross>same 全部稳健。v2 结论不依赖"pair 独立"假设。

## 6. Per-axis calibration（Task C，`calibration/per_axis_calibration.csv`）

episode-level test split（无泄漏）原生预测，关键行：

| condition | axis | true \|r\| | pred \|r̂\| | bias | MAE | Pearson | slope a=Cov/Var(r) |
|---|---|---:|---:|---:|---:|---:|---:|
| normal | vx | 0.137 | 0.144 | +0.010 | 0.062 | 0.957 | 0.97 |
| friction_low | vx | 0.397 | 0.348 | +0.006 | 0.162 | 0.915 | **0.81** |
| friction_low | vy | 0.368 | 0.319 | −0.070 | 0.162 | 0.916 | 0.87 |
| friction_low | wz | 0.332 | 0.250 | +0.035 | 0.274 | 0.623 | **0.46** |
| friction_vlow | vx | 0.387 | 0.275 | −0.010 | 0.242 | 0.779 | **0.61** |
| friction_vlow | wz | 0.333 | 0.230 | −0.028 | 0.292 | 0.548 | **0.39** |

**发现**：原生校准是 **scale-dependent** 的 —— 大 residual 系统性低估（slope 0.4–0.8，
Huber 的 regression-to-mean）；小 residual 区间（probe 窗口）表现为相对过冲
（native low 预测 |r̂|=0.107 vs 真实 0.076，本轮分解证实过冲是 native 性质而非 swap 伪影）。
wz 全条件弱可预测（Pearson ≤0.64）。图：`figures/CAL_<cond>_<axis>.png/pdf`（12 张）。

## 7. Per-condition calibration

`calibration/per_condition_calibration.csv`（pooled）：normal MAE 0.10 → friction_low 0.20 →
friction_vlow 0.26（pooled RMSE 0.11→0.25→0.35），误差随扰动单调增长。

## 8. Transition vs steady（Task D，`transition/`）

定义：窗内 timestep 距命令变化 ≤τ 为 transition（τ=0.25/0.5/1.0s 敏感性），>0.5s 全窗为 steady。

τ=0.5s 主口径（mean）：

| direction | class | N | E_correct | E_swap | ΔE_self | P_target | S_dir |
|---|---|---:|---:|---:|---:|---:|---:|
| normal→low | transition | 2669 | 1.064 | 1.848 | +0.785 | **−0.068** | 0.381 |
| normal→low | steady | 2841 | 1.052 | 1.635 | +0.583 | **−0.269** | 0.231 |
| low→normal | transition | 2669 | 1.363 | 1.294 | −0.069 | **+0.792** | 0.389 |
| low→normal | steady | 2841 | 1.337 | 1.083 | −0.254 | **+0.588** | 0.256 |

- **Q6：N→L 的负 P_target 主要来自 steady**（−0.269 vs −0.068；vlow 更明显 −0.713 vs −0.141），
  τ 敏感性下排序不变（`transition_vs_steady_metrics.csv`）
- transition 处 S_dir 一致更高（0.38–0.46）：context 调制在命令瞬变最强
- **§9 数学诊断**：normal 条件 per-timestep |r| 的 transition/steady 比 =
  **vx 5.48×、vy 4.67×、wz 1.86×**（friction_low 为 2.85/2.99/2.07）——
  当前 r_t 定义确实把大量 **normal controller transient**（latency/gait/inertia）计入
  "execution mismatch"；steady 段的 residual 才更纯粹地反映 condition-induced 偏差

## 9. Affine diagnostic（Task E，`calibration/global_affine_params.json`）

val split 拟合（320 窗，无 condition 标签输入）：**α=[0.867, 0.890, 0.603]，β≈0**。
重算全部 25772 pairs（`calibration/affine_recomputed_swap_metrics.csv`）+ cluster bootstrap CI：

| direction | P_target before | **P_target after** [cluster CI] | ΔE_self before | **ΔE_self after** [cluster CI] | S_dir bef/aft |
|---|---:|---:|---:|---:|---:|
| normal→low | −0.172 | **+0.009** [−0.003,+0.021] | +0.680 | **+0.476** [+0.466,+0.485] | 0.30/0.32 |
| low→normal | +0.687 | **+0.486** [+0.476,+0.496] | −0.164 | **+0.016** [+0.004,+0.029] | 0.32/0.34 |
| normal→vlow | −0.377 | −0.083 [−0.119,−0.047] | +1.275 | +0.897 | 0.38/0.39 |
| vlow→normal | +1.270 | +0.897 | −0.359 | −0.066 | 0.39/0.41 |

**Q7：一个 6 参数全局 affine 基本修复主门**（N↔L 双向 P_target 非负、ΔE_self 双向转正），
vlow 方向部分改善（N→VL 仍小幅为负）。S_dir 不变（affine 不改方向）。
→ 支持"context 机制有效，主要瓶颈是 output 幅值校准"。

## 10. Gate M — Context Mechanism：**PASS**

1. S_dir 四方向 +0.30~+0.40，cluster bootstrap CI 紧致且不贴 0；vx 双向 ≈0.49/0.50
2. M_cross−M_same cluster CI 全正（+0.87/+0.37/+1.48/+0.69）
3. Regime teleport ratio 1.016–1.029（四方向）——swap 后预测与 target 原生预测统计不可分
4. 剂量效应：vlow > low（|ΔE|、|P_target|、S_dir 单调增强）
5. affine 校准前后上述结论不变（方向信息在 c_t，不在幅值缩放）

## 11. Gate C — Calibration：**FAIL**

1. 原生 P_target 双向不对称（N→L CI 整体偏负）
2. per-axis slope 0.39–0.81，scale-dependent 幅值误差（大低估/小过冲）
3. wz 弱可预测（Pearson 0.55–0.64，slope ≤0.50）
4. affine 只能闭合主门（CI 一端贴 0），vlow 方向未闭合
5. target 原生误差（low 1.35 / vlow 1.98）> 条件间真实差距（1.22/1.65）

## 12. Root-cause ranking（§10 Q8）

1. **B. Context 被使用，但 calibration 不好**（主因）——affine 近闭合主门；slope<1；probe 过冲 40%
2. **C. residual 定义混入大量 normal transient**（次因）——normal |r| transition 是 steady 的
   5.5×/4.7×；N→L 失败集中在 steady（说明 mismatch 的"有效信号"被 transient 稀释）
3. **E. wz 通道弱可预测性** —— 拉低 overall 向量指标（N→L 负 P_target 主由 wz −0.247 贡献）
4. ~~D. state/pair confound~~ 排除（corr(P_target,D_state)≈0；cluster bootstrap 稳定）
5. ~~A. context 未被使用~~ 排除（teleport / cross>same / S_dir / 剂量效应）

## 13. Limitations

- overall 指标混合 m/s 与 rad/s（v2 遗留口径）；本轮以 per-axis 复核为准
- probe 窗口激励温和，random 激励强 —— 两种 regime 校准方向相反，单一 affine 无法同时最优
- affine 拟合的 val split 只有 8 episodes / 320 窗，参数有不确定性
- same(friction_vlow) cluster 数较少（112），该组 CI 相对宽

## 14. Recommended next experiment

**A. Proceed to V1-Friction data scaling**（1000–3000 episodes，更多 friction 档位），
预期直接改善 Gate C（校准是数据/容量问题，机制已 PASS）。
V1 设计时建议并行讨论 §9 的 residual 重定义（e_nom 名义模型分离 controller transient，
本轮禁止实现）；若 V1 后 Gate C 仍 FAIL，再考虑 output 侧结构（per-condition 校准头 / FiLM）。
