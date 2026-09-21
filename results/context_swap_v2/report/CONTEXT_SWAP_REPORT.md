# Context Swap Test 实验报告（03.1 v2）

> 日期：2026-09-21 · 工作区 `~/cvpr_embed` · 输出目录
> `/media/hdd1/yuhang/checkpoints/execution_wm/v0/context_swap_v2/`
>
> **核心问题：只替换 \(c_t\)，execution prediction 是否朝 target 物理条件的真实 execution 移动？**

## A. Environment

- git commit：`27f8972`（代码）/ V0 数据与模型训练于 `e4e1e6c`
- Python 3.11.16 · PyTorch 2.7.0+cu126 · Isaac Sim 5.1.0 · Isaac Lab v2.3.2
- GPU：NVIDIA RTX 4090（驱动 535.309.01）

## B. Model

- checkpoint：`/media/hdd1/yuhang/checkpoints/execution_wm/v0/context/best.pt`
- sha256：`670e6febfd08b091a9277cba04c432e90bf87cff1fbc8c0a16849273d13643db`
- 原 V0 best val loss：0.016036（V0 唯一 context checkpoint，无挑选）
- context_dim=8 · history 1.0s（L=20 @20Hz）· horizon 2.0s（H=40）
- 接口拆分（`encode_state/encode_context/predict_execution`）regression：max|Δ|=0.0（`audit/model_interface_regression.json`）
- leakage/causality audit：PASS（`audit/leakage_check.json`，encoder 输入只有 40 维 proprio + 历史命令）

## C. Dataset

| 来源 | episodes | 用途 |
|---|---|---|
| V0 probes（normal / low_friction） | 52 | 主实验 pool |
| controlled probes 补采（§6） | 96 = 4 probes × 3 friction × 8 seeds | 补 friction_vlow + probe_2 缺口 |

- 补采原则：same reset pose / same seed set / same command，仅摩擦变化
  （`extra_data/controlled_probe_manifest.csv`；无 mid-probe 摔倒）
- 条件按 friction 归组：1.0→normal、0.3→friction_low、0.15→friction_vlow

## D. Pair quality

- 候选窗口 28336 → 接受 25772（拒绝 2564，全部因 D_state > 1.75 ≈ p90）
- command mismatch：**0.0000**（同 probe 同相对时刻，命令逐点一致）
- cross D_state 分布：p10=0.645 / p50=1.046 / p90=1.838
- 阈值依据：先出分布再定 1.75（§5.1），写入 `configs/context_swap.yaml`
- **混杂检查：corr(P_target, D_state) = −0.17 ~ +0.04 ≈ 0**（pair state 差异不构成混杂）

## E. Main metrics（overall 3H=120 维向量，mean [95% paired bootstrap CI]）

| direction | E_correct | E_swap | ΔE_self | D_before | D_after | P_target | S_dir | S_mag | N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| normal→friction_low | 1.058 | 1.738 | **+0.680** [+0.668,+0.693] | 1.217 | 1.388 | **−0.172** [−0.183,−0.160] | **+0.303** [+0.298,+0.309] | 0.992 | 5510 |
| friction_low→normal | 1.349 | 1.185 | **−0.164** [−0.177,−0.152] | 1.760 | 1.073 | **+0.687** [+0.674,+0.699] | **+0.321** [+0.315,+0.326] | 0.996 | 5510 |
| normal→friction_vlow | 1.048 | 2.323 | +1.275 [+1.246,+1.304] | 1.647 | 2.024 | −0.377 [−0.409,−0.346] | +0.379 [+0.371,+0.387] | 1.178 | 2675 |
| friction_vlow→normal | 1.976 | 1.617 | −0.359 [−0.390,−0.327] | 2.341 | 1.072 | +1.270 [+1.242,+1.298] | +0.395 [+0.387,+0.403] | 1.185 | 2675 |

### 分维度（§10，normal↔friction_low，mean）

| dim | ΔE_self (N→L / L→N) | P_target (N→L / L→N) | S_dir (N→L / L→N) | S_dir>0 比例 |
|---|---|---|---|---|
| **vx** | **+0.435 / +0.029** | **+0.032 / +0.450** | **+0.486 / +0.498** | 83% / 84% |
| vy | +0.367 / −0.074 | −0.076 / +0.367 | +0.019 / +0.039 | 43% / 45% |
| wz | +0.312 / −0.231 | −0.247 / +0.301 | +0.077 / +0.083 | 60% / 60% |

→ **摩擦信号通道 vx 上，ΔE_self、P_target、S_dir 双向全部为正**（S_dir≈0.5）；
vy/wz 无方向性信号（与 V0 结论一致），其 L2 噪声主导 overall 向量。

### friction_vlow 剂量效应（§15.2）

条件差更大时 swap effect 单调增强：|ΔE_self| 0.68→1.28、|P_target| 0.69→1.27、
S_dir 0.30→0.38、S_mag 0.99→1.18 ✓

## F. Controls

| control | 结果 | 解读 |
|---|---|---|
| shuffle context（5 seeds/pair） | E_shuffle=1.45（normal 源）介于 E_correct 与 E_swap 之间 | 错 context 比随机 context 伤害更大 → context 携带条件特异信息 |
| zero context | E_zero=1.49 ≈ E_shuffle | OOD 输入，仅辅助 |
| **same-condition swap（§8）** | M_same: normal 0.547 / low 1.067 / vlow 1.336 | 基线 episode-identity 效应 |
| **cross-condition swap** | M_cross: N↔L **1.43** / N↔VL **2.03** | **M_cross > M_same**，跨 probe/seed 稳定（CI 不重叠） |

## G. Transition vs steady（§11，τ_u=0.1）

| direction | window | N | E_correct | E_swap | ΔE_self | P_target | S_dir |
|---|---|---:|---:|---:|---:|---:|---:|
| normal→low | transition | 1346 | 1.089 | 1.865 | +0.777 | −0.070 | **0.389** |
| normal→low | steady | 4164 | 1.048 | 1.697 | +0.649 | −0.204 | 0.276 |
| low→normal | transition | 1346 | 1.346 | 1.288 | −0.058 | +0.765 | **0.394** |
| low→normal | steady | 4164 | 1.351 | 1.152 | −0.199 | +0.661 | 0.297 |

→ transition 窗口 S_dir 一致更高（0.28→0.39）：**命令瞬变处 context 调制更强**，
与 V0 Figure B 的 transient 观察一致。

## H. Failure cases

1. **最差 P_target**（−3.2，全部 normal→friction_vlow / probe_1 / steady / t=2s）：
   S_dir 仍 +0.15~0.32 —— 方向对、**幅值过冲**。根因见诊断 2（c_vlow 把 |r| 高估 40%+）。
2. **最差 S_dir**（−0.45，probe_3_turn、t=1s）：wz 主导窗口，噪声通道，方向信息不存在。
3. 失败与 state mismatch 无关（corr≈0），集中在 wz/稳态窗口 —— 非 contact 不稳定。

## 诊断（§16，不改模型）

1. **predictor 未绕过 c_t**：swap 引起 M_shift=1.43（≈E_correct 的 135%），c 是有效输入。
2. **校准性（核心发现）**：同一批窗口上，c_low 预测 |r̂|=**0.107** vs 真实 low |r|=**0.076**
   （高估 40%）；c_normal 预测 0.050 vs 真实 0.067（低估 25%）。
   → context 朝正确方向调制 residual 幅值，但 **low 方向系统性过冲**，
   这精确解释了 P_target 的方向不对称（N→L 过冲越过目标；L→N 回落反而接近目标）。
3. **Regime teleport（结构性证据）**：D_after(A→B) / 原生 E_correct(B) =
   **1.016 / 1.029 / 1.024 / 1.018**（四方向）—— swap 后的预测与 target 条件的
   原生预测统计不可分。**c_t 完整切换了预测 regime**；残差是模型在该条件的原生误差。
4. **z 已编码大部分 condition**：direct(无 c) test_id MAE 0.1291 ≈ context 0.1278，
   分布内 c 边际贡献 ~1% —— 这是 V0 数据量（68 训练 episodes）下的容量/数据限制。
5. P_target 不对称的算术根源：原生 low 误差（1.35）> 条件间真实差距 ||Δtarget||（1.22），
   N→L 方向 swap 无法取胜；L→N 方向原生 normal 误差（1.05）远小于差距（1.76），swap 大胜。

## I. Final conclusion

**NO-GO**（严格按 §15.1 overall 门：A/B 各在一个方向不成立，E 不成立）

证据（3–5 条）：
1. A 不成立：low→normal 方向 ΔE_self = −0.164 [−0.177,−0.152]，CI 整体偏**负**
   （swap 到 normal ctx 反而改善对 low 轨迹的预测 —— 校准过冲所致）。
2. B 不成立：normal→low 方向 P_target = −0.172 [−0.183,−0.160]，CI 整体偏负。
3. **但 C 强成立**：S_dir 四方向 +0.30~+0.40，CI 紧致，91–96% pair 为正；
   vx 通道双向 S_dir≈0.5、P_target 与 ΔE_self 双向全正。
4. **D 强成立**：M_cross（1.43/2.03）> M_same（0.55/1.07/1.34），CI 不重叠。
5. 诊断显示失败模式是 **residual 幅值校准**（low 方向 +40% 过冲）与
   **难条件原生误差 > 条件间差距**，而非 §16 的 "context 未被使用" 形态
   （regime teleport ratio≈1.02）。

候选后续（本轮不实施，§16）：FiLM conditioning / context-conditioned residual 校准项 /
context dropout 正则 / 扩数据后重跑本测试。**Recommended next step: 扩大数据规模后
重跑 Context Swap Test；若通过再进入 Context Adaptation Curve。**
