# V0_6_DECISION — Support/Query 有效性迁移试验判定

> 日期：2026-09-22 · 阶段A（离线审计）+ 阶段B（240 ep pilot）+ 阶段C（M0–M3 × 3 seeds）
> 证据：results/v0_6_validity_transfer/{audit,manifests,metrics,raw,figures,code_snapshot}
> 数据：/media/hdd1/yuhang/datasets/execution_wm/v0_6_sq（240 ep，50Hz 原始 + 20Hz 派生）
> 模型：/media/hdd1/yuhang/checkpoints/execution_wm/v0_6/{M0..M3}_s{42,43,44}/best.pt

```text
DATA_CAUSALITY:            PASS（带 4 条强制注记，见 A_AUDIT_DECISION）
PAIR_INDEPENDENCE:         PASS（episode-cluster / two-way bootstrap；旧 pair CI 作废）
SMOKE_AND_INJECTION:       PASS（write=readback 精确；anchor 复原 bit 级 0.0；ground=1.0 average combine）
MODELS_BEAT_COMMAND_COPY:  SUPPORTED_ON_DISTRIBUTION（held-anchor −0.0190 [−0.0271,−0.0118]；held-family −0.0257 [−0.0361,−0.0148]）
CONTEXT_VS_DIRECT_3SEED:   SMALL_BUT_SIGNIFICANT（M1−M0 = −0.0014 [−0.0021,−0.0006] / −0.0023 [−0.0032,−0.0015]）
CONTEXT_TRANSFER_ACROSS_EPISODES: UNSUPPORTED（cross-support c 显著变差 +0.0406/+0.0272，CI 不跨 0）
ORACLE_FRICTION_GAP:       NEGLIGIBLE（M3−M1 ≈ −0.0001；瓶颈不在 context inference）
CROSS_SUPPORT_TRAINING:    SUPPORTED_AS_REGULARIZER（M2 两个 held-out 均最优，M2−M1 = −0.0021 / −0.0049）
STATISTICAL_STRENGTH:      EXPLORATORY（held-anchor 仅 2 个 anchor group；48 ep clusters/split）
```

## 一句话结论

在冻结协议、held-out anchor 与 held-out command family 上，EA-WM 模型**显著优于
command-copy**（V0.5 probe 上的"不胜基线"是分布问题而非模型无效）；但 context 向量
**不是可跨 episode 迁移的"物理参数编码"**（cross-support c 显著降性能，oracle 摩擦几乎
无增益），旧"机制 PASS"的因果解释须撤回，仅保留其几何描述性事实。

## 关键数字（3 seeds 均值，MAE_all@2s）

| split | ccopy | ARX | M0 | M1 | M2 | M3 |
|---|---|---|---|---|---|---|
| held-anchor (AQ6/7, Q1–Q3) | 0.1054 | 0.0897 | 0.0878 | 0.0864 | **0.0843** | 0.0863 |
| held-family (Q4_combo) | 0.1622 | 0.1363 | 0.1388 | 0.1365 | **0.1317** | 0.1365 |

episode-cluster bootstrap（2000 次，48 clusters/split；负值=前者更优）全部 CI 不跨 0：
M1−M0 −0.0014/−0.0023 · M2−M1 −0.0021/−0.0049 · M3−M1 −0.0001/−0.0001 ·
M1−ccopy −0.0190/−0.0257 · native−cross_support −0.0406/−0.0272 ·
native−zero −0.0113/−0.0034 · native−shuffle −0.0189/−0.0097。

## 资源消耗

- 采集：240 ep（=预算 300 的 80%），wall ~16 min，GPU·h ≈ 0.27
- 训练：12 run 合计 GPU·h = 0.0059（协议要求先实测 2 run：M0/M1 s42 各 ≈0.0005，确认后放开）
- 评估+bootstrap：<5 min

## 撤回与支持清单（详见 NEXT_ACTIONS.md Q1–Q10）

- **撤回**：①"context 编码可迁移物理参数/工况"（cross-support c 显著变差）；②"机制 PASS"
  的因果解读；③"40% 过冲证明模型理解 regime"（几何事实保留、机制解读撤回）；④ V0.5
  "native error > 真实工况差距"；⑤ "vx 幅度失校准"（部署方向 truth-on-pred ≈ 0.993）。
- **支持**：① 模型在分布内 held-out 显著优于 command-copy；② native c 携带真实信息
  （优于 zero/shuffle）；③ cross-support 训练作为正则有效（M2 最优）；④ V0.5 外部复核
  31/31 数值全部可复现。
