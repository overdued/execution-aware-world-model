# NEXT_ACTIONS_V061 — 任务书 §6 逐题回答（Q1–Q8）

> 所有数值出处：`results/v0_6_1_correctness/metrics/` 与 `audit/`。统计口径为
> episode-cluster bootstrap（2000 次），**全部标注 EXPLORATORY**（held-anchor 仅 2 组）。

## Q1 两个 P0 是否在实际代码/数据复现？影响哪些训练/评价？

**都复现，且逐位吻合验收报告。**

- **P0-1（support 不成对）**：原函数玩具重放 10000 次中 9859 次错配（98.59%）。
  影响 M2 的 train/val/test、M1 的 cross_support/diff_condition、bootstrap variants，
  以及由它们推出的"状态/context 纠缠""cross-support 迁移不可行""M2 是有效正则"。
  旧 donor ID 未保存 → 无法追溯每条旧样本抽到了谁（不伪造，只记影响面）。
- **P0-2（标签恒等式）**：全 240 ep **0/240** 成立，最大差 **1.120208**
  （`ep_00162` t=2.5s wz：1.885307 vs 0.765099）。影响**全部模型的监督目标与评估真值**
  ——即 V0.6 的所有模型排行榜与差值表。
  偏差可精确分解为 B3 浮点 hold（max 1.600）+ B2 命令采样 vs 滤波（max 0.480），残差 7.9e-08。

附带的 P1-1/P1-2/P1-3/P1-4 也都复现（8751 点时间错位；axis 错取 vz；Q4 混入 held-anchor
96/384 窗且与 held_family 重叠 12 ep；`mae_rows` 实为 prefix）。

## Q2 修复标签后真实 physical prediction 排行有没有变？

**变了，而且是方向性改变。**（新真值 = 独立的 `lb_execution`）

| 对比 | V0.6 旧口径 | V0.6.1 修复后 |
|---|---|---|
| M1 − command-copy (A) | −0.0190 胜 | −0.0244 胜（一致） |
| M1 − command-copy (B) | −0.0257 胜 | −0.0002 **跨 0**（不再胜） |
| M1 − M0 (B/C) | −0.0023 判 M1 胜 | +0.0043 / +0.0041 **判 M0 胜** |
| M2 − M1 (A) | −0.0021 判 M2 胜 | +0.0038 **判 M1 胜** |

归一化主标量：A 上模型 0.70–0.75 < ccopy 0.98（模型明确胜）；B/C 上模型 1.80–2.07
vs ccopy 1.81/1.83（**相当或更差**）。逐轴 27 单元胜出次数：M0 13、M2 9、M3 5、**M1 0**。
具体机制：模型在 vy 通道于 B/C 上明显劣于 command-copy（0.06–0.09 vs 0.0285–0.0305 m/s）。

## Q3 paired donor 修复后，M1 原生→cross 的差距有多少，M2 正确 cross 是否优于 zero/wrong？

用**修复后的成对抽样**（固定 donor 表）：

- **M1 native − M1 cross_support（同 condition 成对 donor）**：−0.0185 (A) / −0.0043 (B)
  / −0.0029 (C)，均不跨 0 → native 更好。**但注意**：M1 的 encoder 训练于自身 history，
  support 窗口对它是分布外 → 该差距混合了"信息量"与"分布匹配"，不能单独归因于不可迁移。
- **M2 四个对照**（负=前者更好）：

  | diff | A | B | C |
  |---|---|---|---|
  | same_cond − wrong_cond | −0.0215 | −0.0097 | −0.0088 |
  | native − wrong_cond | −0.0162 | −0.0076 | −0.0070 |
  | native − same_cond | +0.0054 | +0.0021 | +0.0018（跨0） |
  | native − zero | −0.0029 | **+0.0041** | **+0.0038** |

  即：**same-condition 成对 donor 一致优于 wrong-condition donor**（这是干净的组内对照，
  两个都是 support 窗口，只差 condition）；wrong-condition 一致最差。
  但 M2 的 native/zero 对它是分布外，故 "same > native > zero" 的排序**不能**读成
  "迁移优于自历史"。

## Q4 M1 vs M0、M2 vs ridge 的收益是否跨 axis/horizon/独立 anchor 一致？

**不一致。**

- M1−M0：A 上 −0.0011（M1 略优，CI 不跨 0），B +0.0043、C +0.0041（M0 优，CI 不跨 0）
  → **符号随 split 反转**。逐轴 27 单元里 M0 胜 17、M1 胜 10，且 M1 在 0.25s 短 lead 上
  一致更差（B/C +0.013~+0.018）。
- M2−ridge：A +0.0015（ridge 优）、B −0.0020（M2 优）、C −0.0018（CI 跨 0）→ 不一致。
- 逐 anchor：held-anchor 只有 2 组，两组之间已有可见差异；不构成"跨独立 anchor 一致"的宣称。

## Q5 M3 oracle 分支是否确实响应 friction？无增益能否有替代解释？

**确实响应，但增益为零。**（`audit/oracle_branch_response.json`）

零初始化 `Linear(1,8)` 训练后 ‖W‖ = 0.117–0.153，把 friction 从 1.0 改到 0.3 使 r̂ 变化
max 0.008–0.014、mean 0.0019–0.0026，约为 r̂ 量级（0.072–0.074）的 **3%**。
同时 M3 − M1 ≈ **−0.0001**（三个 split 一致）。

替代解释（**均不得**推导出"真实工况信息无用"或"扩数据无效"）：
1. 注入方式弱：零初始化线性加到 8 维 c 上，容量与表达力都极小；
2. 真实摩擦对足-地接触的作用被控制器扭矩补偿大量抵消（本任务测的是**执行残差**不是接触力）；
3. 残差主导成分对 friction 不敏感（控制器瞬态、步态相位）；
4. 预算内该分支未被有效使用（≤100 epoch / patience 15）。
→ 记 **INCONCLUSIVE**，oracle 是诊断基线、不是上界。

## Q6 支持池复用及用 ground-truth condition 选 donor 使结论属于何种信息假设？

- 支持池 = 全部 48 条 support（train 侧）→ **固定 reference bank evaluation**，
  **不是** unseen-support 迁移；本轮未声称跨 support session 泛化。
- 每个 query 的 donor 用 **真值 condition** 选取（`same_cond` / `wrong_cond` 变体）→
  **oracle-condition-selection 假设**，部署时不可得（需先推断工况）。
- 每个窗口的 donor 身份已逐条落盘（`manifests/donor_tables.json`，含 episode_id /
  window_id / origin_tick / session），可独立复现每次交换。
- 因此可辩护的表述是：**在给定真值工况的条件下，成对同工况 support context 优于错配工况
  support context**；"context 可迁移"这一部署主张仍需条件推断环节才能成立。
- 另注：support 池的 16 条/工况同时用于 M2 训练 → 该评估属**分布内 donor**，
  对 same_cond 有利；这也是 native/zero 对 M2 分布外的原因。

## Q7 新 anchor、新 family、joint-shift 三个结果分别如何？

用 A/B/C 三个透明 split：

| split | 含义 | 结果 |
|---|---|---|
| A | 新 anchor × 已见 family（288 窗 / 36 ep / 2 anchor） | 模型 0.70–0.75 明确优于 ccopy 0.98；M0≈M1≈M3≈ridge |
| B | 已见 anchor × 新 family（240 窗 / 30 ep / 5 anchor） | 模型 1.83–2.07 vs ccopy 1.81 → 无优势；vy 通道明显更差 |
| C | 新 anchor × 新 family（96 窗 / 12 ep / 2 anchor） | 模型 1.86–2.06 vs ccopy 1.83 → 无优势 |

**joint-shift（C）没有比单一 shift（B）更差**——B 与 C 的模型表现接近，说明瓶颈主要是
**新命令族**而非 anchor 数量。
另：`supp_AQ5xQ4`（AQ5×Q4，evaluation-only 单列）与 B/C 同型，未并入主表。
净 yaw（相对位姿合法子集）排序与瞬时 wz 排序**不一致**：A 上全部 variant ≈1.55 rad，
ccopy 1.554 最好；B/C 上 ccopy（0.306/2.648）领先所有模型。

## Q8 后续最有价值的一个实验是什么？（不同时开五条研究线）

**单条：deployable 输入 + 条件推断的最小对照。**

理由是三条独立证据的交汇：

1. **瓶颈已定位在"未见命令族"而非 anchor**（Q7：B≈C；A 上模型明确胜出）；
2. **当前所有输入都是 privileged**（`schema.py` 标注 `base_linear_velocity_body` /
   `base_angular_velocity` 为 `privileged_gt`）——真实机械狗拿不到这些量，
   因此现有优势的可部署性未知；
3. **oracle-condition 选择带来的增益真实存在但依赖真值标签**（Q3/Q6：same−wrong 一致显著），
   而部署时条件未知 → 需要把"选 donor"换成"推断工况"并联测。

具体设计（一轮内可完成）：在现有 240 ep 上做同一窗口、同一协议的两组对照——
(1) privileged 输入 vs 仅 deployable（projected_gravity / IMU / joint / contact）输入；
(2) donor 选择由"真值 condition"换成"从 support 推断 condition"，
报告端到端（含推断误差）的 context 增益。

**明确不做**（本轮证据均不支持）：扩同类数据（oracle 无增益 + A 上已饱和）、
新结构（FiLM/mixture/ensemble）、接视觉、真机在线纠错、V-JEPA 大训练。

## 附：与"最多 12 主 run"预算的对照

12 run（M0–M3 × 3 seed）全部完成，总 **0.0058 GPU·h**；R1 旧模型重评 <0.01；
无额外训练用于 context variant —— 全部为同一 checkpoint 的推理变体，共用一次推理缓存。

- M2 donor 四对照（same_cond / wrong_cond / native / zero）：**3 个 seed 全部执行**
  （缓存中 4 split × 3 seed × 4 variant = 48 条）；bootstrap 差值表使用 s42，
  因 donor 表固定，seed 间差异仅来自模型参数。
- M1 context variant（cross_support / diff_condition / zero / shuffle）：3 个 seed 全部执行。
