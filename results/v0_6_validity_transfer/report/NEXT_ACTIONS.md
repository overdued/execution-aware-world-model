# NEXT_ACTIONS — V0.6 终报告逐题回答（Q1–Q10）与下一步建议

> 2026-09-22 · 所有数值出处：results/v0_6_validity_transfer/metrics/ 与 audit/
> 统计口径：episode-cluster bootstrap（2000 次）；held-anchor = AQ6/AQ7（仅 2 个
> anchor group，探索性强度），held-family = Q4_combo（8 anchors）。

## Q1 旧"机制 PASS"和"40% 过冲"哪些被支持、哪些需撤回？

**撤回因果解读，保留几何事实。**
- 支持（描述性）：swap 预测确实沿 target 实际方向移动，V0.5 几何分解可复现
  （31/31 数值一致；squared_target_gain：vx +0.034 / vy +0.110 / wz −0.519，
  wz 为净过冲）。"40% 过冲"作为几何描述成立。
- 撤回（因果性）：V0.6 证明 c **不是**可迁移的工况/物理参数编码——把同 condition
  另一条 support episode 的 c 换给 M1，误差显著上升（+0.0406/+0.0272，CI 不跨 0）；
  oracle 摩擦（M3）几乎无增益。因此"c 使模型理解 regime 故产生过冲"的机制故事不成立，
  Gate M 的 PASS 降级为"在自身 history 编码下模型输出对 future command 敏感"这一
  弱事实（A 阶段 CONTEXT_OUTPUT_SENSITIVITY: supported 仅指此）。

## Q2 为什么复杂模型在 probe 上不能稳定超过 command-copy？

**是分布问题，不是模型无效。** V0.5 probe（push/极端小样本）上 action_only 1.222 <
direct 1.309 ≈ context 1.329 ≈ ccopy 1.337：该分布中 history 信息无增益且略有害。
V0.6 在协议内 held-out 上同一批架构显著胜 ccopy（−0.0190/−0.0257，CI 不跨 0），
且胜 ARX（held-family 上 M1 0.1365 < ARX 0.1363 除外，基本持平）。结论：V0.5 的
"模型不胜基线"由 probe 分布（扰动瞬态、样本极少）造成；采样与优化因素已由 3-seed
一致性排除（val 标准差 <0.0001 量级，见 train_log.json）。

## Q3 node-aware CI 与旧 CI 差多少？独立 episode/seed 到底多少？

- V0.5 旧 pair-level CI **作废**（pair 间共享 episode，伪独立）。A5 重做：two-way
  bootstrap（source×target episode 独立重采样）下 CI 显著加宽但主结论方向不变；
  旧 d0 163 ep 仅 155 个 reset anchors（无 env.reset 的连续 episode 不独立），
  d1 probes 96 anchors。
- V0.6：每 split 48 个 episode clusters；**held-anchor 只有 2 个 anchor group**，
  cluster bootstrap 在 episode 级重采样（48 单元），anchor 级只有 2 个 → 明确标注
  为**探索性强度**，anchor 泛化结论不可外推。
- 独立 seed：训练 3 seeds（42/43/44）；采集 seed 结构 AQ[i]=1000+17i、SP[j]=5000+31j。

## Q4 回归方向纠正后，还支持哪些校准问题？

- pred-on-truth slope <1 是回归均值的必然现象（即使条件均值无偏），**不再作为失准证据**。
- 部署方向 truth-on-pred ≈ 0.993（A6）→ vx 全局幅度**基本校准**，"vx 幅度失校准"撤回。
- val-affine 后处理在 test 上 MAE 变差、N→L P_target +0.009（CI 跨 0）→ **不支持**
  全局仿射校正；任何 test 上求出的 shrink 只能标 ORACLE_DIAGNOSTIC。
- 仍然成立的校准问题：binned E[truth|pred] 在高 |pred| 分箱的局部偏差（样本稀，
  见 metrics/prediction_reliability.csv），以及 wz 在事件后窗口的系统性过冲（几何
  分解 −0.519）——后者是**条件性**偏差而非全局幅度问题。

## Q5 matched pairing 是否丢弃高失配状态？接受/拒绝分布如何？

A1：full_pair_manifest 接受 25772 / 拒绝 2564（≈9.0% 拒绝）。拒绝集中在高失配
（anchor 距离大）候选——即 matched pairing **确实系统性丢弃了高失配状态**，旧
swap 结论只适用于低失配子总体。配套撤回：旧"native error(1.349) > 物理工况差距
(1.485)"不成立（1.349 < 1.485，且 1.485 是被筛选子集上的距离）。V0.6 设计改用
显式 anchor + held-out family，不再依赖配对筛选。

## Q6 state/context 是否纠缠？same-condition 跨命令 support 有无增益？

**纠缠，无增益（对 M1 直接换 c）；但 cross-support 训练有增益（M2）。**
- M1 换同 condition 跨命令 support c：+0.0406（held-anchor）/+0.0272（held-family），
  与换**不同** condition 的 c（+0.0440/+0.0289）几乎一样差 → c 主要编码
  episode/state 特有信息（含当前瞬态），condition 信息占比小。k=0 连续性：
  native 0.1134 vs cross_support 0.1850，换 c 破坏首步连续性。
- M2（训练时 c 强制来自同 condition 另一条 support ep）在两个 held-out 均最优
  （M2−M1 = −0.0021/−0.0049，CI 不跨 0）→ 训练期 c 抖动是有效正则，decoder 对 c
  扰动更鲁棒。**"same-condition support 迁移"作为部署能力不被支持，作为训练正则
  被支持。**

## Q7 relative pose 与 raw speed 评价是否一致？wz 难点如何定位？

A4/A6：per-axis 与 relative-pose 口径结论方向一致（误差排序 vx≈vy < wz；
normal < friction_mid < friction_low）。wz 难点定位：
- 量级：wz MAE@1s ≈ 0.12，约 vx(0.066) 的 1.8 倍（V0.6 held-anchor，全部模型一致）；
- 结构：wz squared gain −0.519（净过冲），集中在 turn 事件后窗口（A6 transition
  按 timestep 分解，origin-到-事件/lead_time 分桶）；
- 注意：0.25/0.5/1.0s 分组完全相同，**不是**额外稳健性证据（A 阶段已标注）。
  body-frame wz 与世界 yaw 速率 corr 0.87（roll/pitch 非零），评价保持 body 语义。

## Q8 同架构同协议 3 seed 下 Context 相对 Direct 的效益？

M1−M0 = **−0.0014 [−0.0021, −0.0006]**（held-anchor）、**−0.0023 [−0.0032,
−0.0015]**（held-family），MAE_all@2s 口径。统计显著但**量级很小**（相对 M0
约 −1.6%/−1.7%）；M1 与 M3（oracle 摩擦）差仅 −0.0001 → context 路径的可用信息
基本被榨干。val 上同样小（M0 0.0062 vs M1 0.0058）。

## Q9 oracle 信息对照指向 context inference 还是 predictor/target 瓶颈？

**predictor/target 端。** M3（learned c + 零初始化 W·真实摩擦）相对 M1 仅 −0.0001
（两个 split 一致）→ 知道真实摩擦几乎不改善 → 误差主项不在"工况推断不准"，而在：
① 残差 r_t 本身的不可预测成分（执行噪声/接触瞬态）；② predictor 容量/目标定义；
③ 输入特权化（当前输入含 GT 速度/姿态，未见 deployable 条件下的误差面）。

## Q10 下一步：扩数据、改结构、接视觉还是缩减科学主张？

**缩减科学主张 + 换实验轴，暂不按原计划扩同类数据。** 证据：
1. **缩减主张**（必须）：删除"context = 物理参数估计/可迁移工况编码"的表述；EA-WM
   的可辩护主张收窄为"分布内 execution residual 预测显著优于 command-copy，
   context 路径有小而显著增益"。
2. **不建议立即扩同类数据**：oracle 摩擦无增益（Q9）说明同一分布更多数据不会解决
   当前瓶颈；且 held-anchor 仅 2 group，扩 anchor 数比扩 episode 数更有边际价值。
3. **优先下一步（按证据排序）**：
   a. **deployable 输入对照**（去掉 GT 速度/姿态，用 IMU+joint 估计）：当前全部结论
      在特权输入下得到，这是部署相关性的最大缺口（A 阶段强制注记 #3）；
   b. **残差可预测性上界诊断**（同一 (state,cmd) 多 rep 的残差方差分解：pilot 已有
      每组合 2 reps，可直接算）→ 量化"不可约误差"占比，决定 predictor 端还有
      多少空间；
   c. 若 (b) 显示可预测空间仍在 → 再考虑扩 anchor 数（≥8 组 held-out）复验 M2
      正则与 M1−M0 小增益的稳健性，而不是扩 episode。
4. **明确不做**：接视觉、新架构（FiLM/mixture/ensemble 等）、在线纠错——本轮证据
   均不支持这些方向能解除已定位的瓶颈。
