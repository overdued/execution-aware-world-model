# Execution-Aware World Model (EA-WM)

Execution-Aware World Model 第一阶段实验：在 Isaac Lab / Unitree Go2 上验证
**执行失配（execution mismatch, \(r_t = e_t - u_t\)）可测、可学**，
并通过 **Context Swap Test** 检验 execution context \(c_t \in \mathbb{R}^8\) 是否真正参与
execution dynamics prediction。

## 结论速览

| 阶段 | 状态 | 结论 |
|---|---|---|
| V0 sanity（第一阶段） | ✅ 完成 | mismatch 可测（friction 下 residual +124%）、可学（ID −11%）、context PCA 按摩擦可分 |
| Context Swap Test（第二阶段） | ✅ 完成，**NO-GO（严格门）** | c_t 确实切换预测 regime（teleport ratio ≈1.02）、方向物理一致（S_dir 0.3–0.5）、cross>same；但 low-friction residual 幅值高估 40%（校准失败，数据量不足）→ P_target 方向不对称 |

**当前判定：NO-GO（校准失败型，非"context 未被使用"型）。**
下一步建议：扩数据（1000–3000 episodes）后重跑 Context Swap Test。

## 仓库结构

```
first_work.md                                    # 第一阶段实验规范
03.1_Context_Swap_ClaudeCode_Execution_v2.md     # 第二阶段（Context Swap）执行规范
execution_wm/                                    # 全部代码
    data/         collector / controlled probes / dataset / sanity check
    models/       action_only / direct / context 三模型
    train/        训练脚本
    eval/         评估 + Figures A-E + quick_look
    context_swap/ audit / pairs / run_swap / metrics / figures / diagnostics
    configs/      全部实验配置
docs/             streaming_fix.md / experiment_v0_summary.md
results/
    v0/           第一阶段产物：Figures A-E、quick_look、三模型 checkpoint+summary、对比评估
    context_swap_v2/
        report/   CONTEXT_SWAP_REPORT.md + GO_NO_GO.md  ← 验收入口
        metrics/  pair_level / aggregate / bootstrap CI / per-probe / per-dim / transition-steady
        figures/  CS1–CS6（PNG+PDF）
        audit/    leakage / causality / interface regression（全 PASS）
        pairs/    pair manifest / quality / rejected
        raw/      representative_pairs.npz（人工复核用）
        extra_data/  controlled_probe_manifest.csv（96 个 controlled probe episodes）
```

## 验收入口（给评审 AI / 人工）

1. 先读 `results/context_swap_v2/report/GO_NO_GO.md`（判决 + 证据 + 混杂检查）
2. 再读 `results/context_swap_v2/report/CONTEXT_SWAP_REPORT.md`（完整 A–I 节）
3. 关键图：`results/context_swap_v2/figures/CS4_direction_alignment.png`（方向一致性）、
   `CS5_cross_vs_same.png`（cross vs same control）、`CS1_*.png`（代表时间序列）
4. 原始数据复核：`results/context_swap_v2/metrics/pair_level_metrics.csv`（25772 pairs）、
   `results/context_swap_v2/raw/representative_pairs.npz`

## 环境

- Isaac Sim 5.1.0（pip）+ Isaac Lab v2.3.2 + torch 2.7.0（PyPI cu126）+ Python 3.11（conda env `isaaclab`）
- 数据采集：RTX 4090，50Hz 控制 / 20Hz 输出
- 复现细节与环境坑见 `CLAUDE.md` 与 `docs/streaming_fix.md`

## 数据与 checkpoint

仓库内含三模型 checkpoint（`results/v0/models/*/best.pt`，各 <1MB）与全部汇总指标。
原始 163 episode 数据集（npz）体积较大未上传；controlled probe manifest 与全部 pair-level
指标已包含，足以复核全部结论。
