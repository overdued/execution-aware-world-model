# V0.8 FULL REPORT — 冻结 V-JEPA 2 视觉预测 pilot

> 完成日期：2026-09-27 · 分支 `exp/cvpr-v08-visual-pilot`（worktree `~/cvpr_embed-v08`）
> 冻结协议：`report/V0_8_PRE_REGISTRATION.md`（采集前 commit `0b6feb7`，
> split plan sha256 见 `prereg/split_plan_v08.json`）
> 偏离记录：`report/V0_8_DEVIATIONS.md`（D1: PROPRIO_DIM 48→40；D2: G 头形状不匹配
> 按规则随机初始化、R-GRU 按 seed 匹配从 V0.7 D_R1 初始化）
> 前置收尾：V0.7 报告复核与统计口径收尾见 `V0_7_CLOSURE.md` / `V0_7_ERRATA.md`
> （commit `a049fb5`），smoke gate 见 `VISUAL_SMOKE_REPORT.md`（commit `fd53bc9`，6/6 PASS）
> 本报告所有数值可由 `predictions/*_{val,test}_preds.npz`（18 个文件）+
> `metrics/*.csv` 复算；复现步骤见 `README_REPRODUCE.md`。

---

## 0. 一句话结论

**动作依赖的视觉信号存在（候选匹配 0.73–0.83，chance=1/3，噪声底 ~0.31），但显式
执行条件（V-EXEC）相对辅助监督（V-AUX）在主终点上无增量（+1.10 / −0.94 / +0.15%，
方向不一致、量级 ≪ 5% 门）→ EXECUTION_CONDITIONING_INCREMENT: NOT_SUPPORTED；
辅助监督相对直接预测反而一致劣化 2.2–3.9%（3/3 seed CI 不含零）→
AUXILIARY_SUPERVISION_EFFECT: NOT_SUPPORTED；物理预测保持在 5% 容差内（PASS）。**
下一步（任务书选项 C）：视觉表示的可利用动作后果信号不足——定位相机/时域/表示
瓶颈，不堆大模型。

---

## 1. 正确性（全部 PASS）

| 检查 | 结果 |
|---|---|
| Smoke gate（12 条同步 RGB 采集） | **6/6 PASS**（`VISUAL_SMOKE_REPORT.md`） |
| 物理 A/B 对照：开相机 vs 不开相机轨迹 | **逐位一致** |
| V-JEPA 2 冻结核验（参数 sha、eval mode、无梯度） | PASS（`audit/encoder_and_checkpoint.json`） |
| 因果单测 T1–T4 + T2b（origin 帧不含未来、特征时间对齐） | PASS（`audit/time_camera_causality.json`） |
| 分支 pre-action 状态（tick 75，48 组 × U0/U1/U2） | **逐位一致**（`audit/branch_prestate_quality.csv`） |
| U0 重复分支轨迹 | **逐位一致** |
| 20Hz 派生恒等式 | max err **1.11e-16** |
| checkpoint 选择 | 仅用 **val 2s visual loss**，未看测试成绩 |
| 特征归一化统计 | 仅 train 布局拟合（`manifests/e_norm.json`） |
| 已知负结果（如实保留） | 渲染跨 episode 非逐位确定（displayColor 强光洗白属正常视觉现象） |

## 2. 数据现状（Stage C，192/192 全部校验）

- **192 episodes** = 6 布局（train L0–L2 / val L3 / test L4–L5）× 8 group ×
  4 分支（U0/U1/U2 + U0 重复），全部 schedule_end，各 227 帧 RGB（间隔恒 0.06s）。
- 首 segment 钉 c422 公共前缀；匹配原点 50Hz tick 75；预算 240 内实际用 204
  （smoke 12 + pilot 192）。
- 窗口：每 episode 7 个（6 回归 + 1 匹配原点），共 1344 窗
  （`manifests/windows.csv`），特征缓存 `predictions/feature_cache.npz`
  （2.5GB，sha256 `0f3b2496…`，与 `feature_cache_manifest.json` 一致），
  V-JEPA 2 冻结特征 4×4 空间 pool。
- 数据集位置：`/media/hdd1/yuhang/datasets/execution_wm/v0_8_pilot`（交付时锁定只读）。

## 3. 训练（9 run 全部完成，无失败）

| variant | 参数量 | best_val_vis2s（s42/s43/s44） | 单 run 耗时 |
|---|---|---|---|
| V-DIRECT | 26.46M | 0.5963 / 0.5957 / 0.5958 | ~31 min |
| V-AUX | 26.51M | 0.6111 / 0.6108 / 0.6097 | ~30 min |
| V-EXEC | 26.53M | 0.6068 / 0.6096 / 0.6081 | ~30 min |

- λ_E=1.0、AdamW 3e-4、30k 步、batch 64、fp32；峰值显存 1.19GB；
  推理延迟 batch1 ≈ 0.22ms、batch64 ≈ 0.65–0.87ms（`metrics/latency_memory_cost.csv`）。
- **重要现象（如实报告）**：全部 9 个 run 的 val 最优点都在 **step 1000**
  （第一个评价点），之后 val loss 单调恶化——模型迅速过拟合 3 个 train 布局，
  泛化到 val 布局（L3）的能力随训练下降。ckpt 规则（val 最优）按设计工作，
  实际选出的都是早期 checkpoint。这本身指向表示/数据多样性瓶颈而非训练不足。

## 4. 主终点（P1 err_2s，test 16 group，paired group bootstrap 10,000 次）

### 4.1 V-EXEC vs V-AUX（EXECUTION_CONDITIONING_INCREMENT）

| seed | 相对改善 | abs diff 95% CI | 方向 |
|---|---|---|---|
| 42 | **+1.10%** | [+0.0041, +0.0107] | V-EXEC 略优 |
| 43 | **−0.94%** | [−0.0098, −0.0023] | V-EXEC 略劣 |
| 44 | +0.15% | [−0.0010, +0.0030] | 零 |

量级全部 ≪ 5% 门；方向不一致（1 正显著 / 1 负显著 / 1 零）。
**判定：NOT_SUPPORTED**——CI 足够紧（±0.3–0.5%），有充分精度排除 ≥5% 改善，
不是统计不足。

### 4.2 V-AUX vs V-DIRECT（AUXILIARY_SUPERVISION_EFFECT）

| seed | 相对改善 | abs diff 95% CI |
|---|---|---|
| 42 | **−2.59%** | [−0.0272, −0.0069] |
| 43 | **−3.86%** | [−0.0353, −0.0148] |
| 44 | **−2.21%** | [−0.0229, −0.0062] |

3/3 seed 同向、CI 均不含零：**辅助监督在主终点上一致劣化 2.2–3.9%**。
P0（−1.7 ~ −2.0%）与 P2（−5.3 ~ −6.5%）同向。**判定：NOT_SUPPORTED（明显劣化）**。
解读：E 预测辅助任务挤占了视觉回归容量；与 §6 的弱 E 依赖一致——
E 头学到的东西对视觉预测没有可迁移的增量。

### 4.3 候选匹配（test 16 group，chance = 1/3）

| variant | matching acc（s42/s43/s44） |
|---|---|
| V-DIRECT | 0.771 / 0.771 / 0.729 |
| V-AUX | 0.813 / 0.771 / 0.833 |
| V-EXEC | 0.833 / 0.771 / 0.792 |

- 噪声底（U0 重复分支窗间距离）≈ 0.30–0.34，候选间距离 0.60–0.93，
  **near-indistinguishable 组 = 0**（`metrics/candidate_noise_floor.csv`）。
- V-EXEC vs V-AUX 匹配退化 ≤ 4.2pp（门 ≤5pp ✓）。
- **ACTION_DEPENDENT_VISUAL_SIGNAL: SUPPORTED**——三个 variant 都远超 chance，
  且候选可分性高于噪声底 2 倍以上；信号存在于冻结表示+浅头中。

### 4.4 V-EXEC E 条件敏感性（test，mean err_2s）

| seed | E=U | E_hat | oracle E（诊断） | shuffled E | zero E |
|---|---|---|---|---|---|
| 42 | 0.6227 | 0.6262 | 0.6238 | 0.6390 | 0.6650 |
| 43 | 0.6290 | 0.6319 | 0.6299 | 0.6448 | 0.6503 |
| 44 | 0.6307 | 0.6323 | 0.6315 | 0.6383 | 0.6559 |

排序 E≈oracle<E_hat<shuffled<zero 在 3/3 seed 成立（组级有少数反转），
**但梯度很小**：zero E 仅比 E=U 差 3.4–6.8%，shuffled 差 1.2–2.6%，
oracle 相对 E=U 无收益（≤0.2%）。模型对 E 条件的**依赖弱**——
它主要从首帧+上下文推断动态，E 只是弱调制。这是 §4.1 零增量的机制性解释。

### 4.5 物理预测保持（PHYSICAL_PREDICTION_RETENTION）

| level | 指标 | s42 | s43 | s44 | 容差 |
|---|---|---|---|---|---|
| P0 | latent err_2s 相对变化 | +0.53% | +0.15% | −0.04% | ≤5% ✓ |
| P0 | FDE_xy(2s) 相对变化 | +1.4% | +0.7% | +0.4% | ≤5% ✓ |
| P1 | FDE_xy(2s) 相对变化 | −1.9% | −2.3% | +2.0% | ≤5% ✓ |

V-EXEC vs V-AUX 全部在容差内 → **ACCEPTABLE**。
（V-DIRECT 无 E 输出头，物理口径不涉及；脚本按设计跳过。）

## 5. 负结果与局限（如实陈述）

1. **E 条件依赖弱**（§4.4）：显式执行条件在当前表示下几乎不提供增量。
2. **辅助监督一致有害**（§4.2）：2.2–3.9% 劣化，3/3 seed 显著。
3. **早停于 step 1000**（§3）：30k 步训练中 97% 是浪费；布局泛化是瓶颈。
4. **渲染跨 episode 非逐位确定**（smoke 期发现）：不影响因果与统计结论，
   已记录。
5. 统计范围 PILOT：test 仅 2 场景 16 group；不外推广泛场景总体。
6. V-EXEC 相对 V-AUX 的 s42/s43 两个方向相反的小效应（±1%）提示存在
   seed 级噪声底 ≈1%，低于该量级的差异不可解释。

## 6. 预算与资源

| 项 | 预算 | 实际 |
|---|---|---|
| 新增 episode | ≤240 | 204（smoke 12 + pilot 192） |
| 主训练 run | ≤9 | 9 |
| GPU·h | ≤100 | ≈ 8（训练 4.6 + 采集/特征/评价 ≈ 3.4） |
| 缓存 | ≤50GB | ≈ 5GB（特征 2.5GB + 预测 ~0.5GB + ckpt ~2GB） |

## 7. 可追溯性

每个预测键为 `window_id/{pred_1s,pred_2s,e_hat}`，`window_id = <episode_id>_k<origin_k20>`，
经 `manifests/windows.csv` 追到 origin/future 帧时间、candidate U、group、seed；
ckpt sha256 在各 run `checkpoints/<run>/summary.json`；
18 个预测文件与 1344 窗一一对应（val/test 拆分见 `manifests/splits.json`）。
