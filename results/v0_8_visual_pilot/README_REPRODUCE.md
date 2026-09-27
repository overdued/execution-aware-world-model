# V0.8 复现指南 — 冻结 V-JEPA 2 视觉预测 pilot

> worktree：`~/cvpr_embed-v08`（分支 `exp/cvpr-v08-visual-pilot`，只本地 commit）
> 冻结协议：`report/V0_8_PRE_REGISTRATION.md`（commit `0b6feb7`）
> 结果判定：`report/V0_8_DECISION.md`；完整证据：`report/V0_8_FULL_REPORT.md`

## 0. 环境

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
cd ~/cvpr_embed-v08
# 采集/仿真才需要：cd ~/IsaacLab && TERM=xterm OMNI_KIT_ACCEPT_EULA=YES \
#   PYTHONUNBUFFERED=1 ./isaaclab.sh -p <script> ... --enable_cameras
# 约束：单 GPU，不要并行开两个仿真会话。
```

## 1. 数据（已锁定只读）

- 位置：`/media/hdd1/yuhang/datasets/execution_wm/v0_8_pilot`（2.7GB，204 episodes
  = smoke 12 + pilot 192），交付时已 `chmod -R a-w`。
- split：train L0–L2 / val L3 / test L4–L5，各 8 group × 4 分支；
  血缘与哈希见 `manifests/{episodes.csv,windows.csv,splits.json,hashes.json}`。

## 2. 快速复算（不训练、不采集，约 20s）

```bash
python results/v0_8_visual_pilot/recompute_v08.py
```

只读 `predictions/pred_cache.npz`（sha256 见 `pred_cache_manifest.json`）+
`predictions/feature_cache.npz`，重算主终点统计并与 `metrics/stats_primary.json`
对拍，输出 6/6 MATCH + `RECOMPUTE PASS`（退出码 0）。

## 3. 完整流水线（重跑全部，供审计）

```bash
# (a) 派生 20Hz + 特征缓存（需先激活 isaaclab）
python -m execution_wm.v08_visual.derive_v08 --raw-root /media/hdd1/yuhang/datasets/execution_wm/v0_8_pilot
python -m execution_wm.v08_visual.build_cache_v08 --results results/v0_8_visual_pilot \
  --raw-root /media/hdd1/yuhang/datasets/execution_wm/v0_8_pilot

# (b) 训练（9 run，每 run ~31min，单 GPU 顺序执行）
for v in V-DIRECT V-AUX V-EXEC; do for s in 42 43 44; do
  python -m execution_wm.v08_visual.train_v08 --variant $v --seed $s \
    --results results/v0_8_visual_pilot \
    --raw-root /media/hdd1/yuhang/datasets/execution_wm/v0_8_pilot
done; done

# (c) 预测（18 个文件）
for run in VDIRECT_s42 VDIRECT_s43 VDIRECT_s44 VAUX_s42 VAUX_s43 VAUX_s44 \
           VEXEC_s42 VEXEC_s43 VEXEC_s44; do
  for split in val test; do
    python -m execution_wm.v08_visual.eval_v08 --results results/v0_8_visual_pilot \
      --raw-root /media/hdd1/yuhang/datasets/execution_wm/v0_8_pilot \
      --predict $run --split $split
  done
done

# (d) 统计 / 匹配 / 物理 / 图
python -m execution_wm.v08_visual.eval_v08 --results results/v0_8_visual_pilot \
  --raw-root /media/hdd1/yuhang/datasets/execution_wm/v0_8_pilot --metrics
python -m execution_wm.v08_visual.matching_v08 --results results/v0_8_visual_pilot \
  --raw-root /media/hdd1/yuhang/datasets/execution_wm/v0_8_pilot --matching --sensitivity
python -m execution_wm.v08_visual.physical_metrics_v08 \
  --results results/v0_8_visual_pilot \
  --raw-root /media/hdd1/yuhang/datasets/execution_wm/v0_8_pilot
python -m execution_wm.v08_visual.figures_v08 --results results/v0_8_visual_pilot
```

采集本身（需 Isaac Sim）：`execution_wm/v08_visual/collect_v08.py` 由
`prereg/split_plan_v08.json` 驱动，从 `~/IsaacLab` 用 isaaclab.sh 启动（见 §0）。

## 4. 交付清单对照（任务书 §6）

| 交付物 | 位置 |
|---|---|
| 完整报告 / 判定 | `report/V0_8_FULL_REPORT.md`、`report/V0_8_DECISION.md` |
| V0.7 收尾 | `report/V0_7_CLOSURE.md`、`report/V0_7_ERRATA.md`、`metrics/v07_*` |
| 审计 | `audit/`（ENV_AND_SCOPE、因果、encoder 校验、target_support、branch_prestate_quality） |
| manifests | `manifests/`（scene_assets、episodes、windows、splits、hashes、e_norm） |
| 指标 | `metrics/`（v07 复核、visual_error_by_group_seed、candidate_matching、physical_metrics、latency_memory_cost、stats_primary、predictions_per_window_*） |
| 预测缓存 | `predictions/pred_cache.npz` + manifest（sha256 校验）；另保留 18 个 per-run 文件与 feature_cache |
| 示例 | `examples/context_and_target_frames/`（12 PNG + index.json）、`representative_physics_and_commands.npz` |
| 配置 / 源码快照 | `config/`（split_plan、train_config、e_norm）、`code_snapshot/`（v08_visual 全部 17 个模块） |
| checkpoints / 训练日志 | `checkpoints/<run>/{best.pt,summary.json,train_log.csv}`、`train_logs/<run>/` |
| 图 | `figures/`（primary_by_scene、candidate_matching、exec_sensitivity、train_curves） |

## 5. 可追溯性

预测键 `<run>/<split>/<window_id>/<field>` → `manifests/windows.csv` 给出
origin/future 帧时间、candidate U、group、level、seed；ckpt sha256 在
`checkpoints/<run>/summary.json` 与 `metrics/latency_memory_cost.csv`。
