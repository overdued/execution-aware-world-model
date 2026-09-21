# Context Swap GO / NO-GO

Decision: **NO-GO**（严格按 §15.1 overall 门；但属于"校准失败"而非"context 未被使用"）

## Primary Evidence
1. **A（wrong context hurts）只在一个方向成立**：normal→low ΔE_self=+0.680
   [+0.668,+0.693] ✓；low→normal ΔE_self=−0.164 [−0.177,−0.152] ✗（CI 偏负）。
2. **B（target pull，最重要）只在一个方向成立**：low→normal P_target=+0.687
   [+0.674,+0.699] ✓；normal→low P_target=−0.172 [−0.183,−0.160] ✗（CI 偏负）。
3. **C（方向物理一致）强成立**：S_dir 四方向 +0.30/+0.32/+0.38/+0.40，
   91–96% pair 为正；vx 通道双向 S_dir≈0.49/0.50。
4. **D（cross > same）强成立**：M_cross=1.43（N↔L）/2.03（N↔VL）>>
   M_same=0.55（normal）/1.07（low）/1.34（vlow），CI 不重叠。
5. **E（双向存活）不成立**：A/B 各自只在 low→normal / normal→low 单方向成立。

## Weak / Negative Evidence
1. P_target 方向不对称的根因是**校准**而非 context 无效：c_low 预测 |r̂|=0.107
   vs 真实 0.076（+40% 过冲）；c_normal 预测 0.050 vs 真实 0.067（−25%）。
   过冲使 N→L 方向越过目标轨迹（方向对、落点远）。
2. 难条件原生误差（low: 1.35, vlow: 1.98）> 条件间真实差距（1.22/1.65），
   overall L2 被 vy/wz 无信号通道的噪声主导（vx 上 P_target 双向为正：
   +0.032/+0.450）。
3. 分布内 c 的边际贡献仅 ~1%（direct 0.1291 vs context 0.1278）——
   V0 数据量（68 训练 episodes）下 z/history 已含大部分可预测信息。

## Confounds Checked
1. **Pair state confound**：corr(P_target, D_state) = −0.17~+0.04 ≈ 0，
   且 command mismatch = 0.0000（逐点一致）。
2. **Episode identity / nuisance**：same-condition swap 基线 M_same 显著小于
   M_cross；shuffle/zero context 误差介于 correct 与 cross-swap 之间，
   说明 c 携带的是条件特异信息而非随机扰动。
3. **Regime teleport**：D_after(A→B)/原生E_correct(B)=1.016~1.029（四方向），
   swap 后预测与 target 条件原生预测统计不可分 —— c_t 确实完整切换预测 regime，
   残差是模型在该条件的原生误差而非 swap 机制失败。

## Recommendation
1. **不要改模型结构**（§16）。失败模式明确：low/vlow 方向 residual 幅值校准
   （+40% 过冲），源于训练数据中难条件样本少（68 训练 episodes）。
2. 下一步（需人工确认后执行，优先级从高到低）：
   a. 扩大执行数据集（1000–3000 episodes，覆盖更多 friction/actuator 档位）
      后**重跑本 Context Swap Test** —— 校准改善后门 A/B/E 有望双向通过；
   b. 训练时加 residual 幅值校准项或 per-condition 平衡采样；
   c. 候选结构：FiLM conditioning / context dropout 正则（仅当扩数据后仍 NO-GO）。
3. **Recommended next step: 扩数据 → 重跑 Context Swap Test；
   通过后再进入 Context Adaptation Curve。**
