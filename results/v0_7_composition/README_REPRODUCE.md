# README_REPRODUCE — V0.7 结构化命令覆盖 × 动作交互建模

> 分支/worktree：`exp/cvpr-v07-composition`（独立 worktree，与 main 及 Risk 线隔离）
> 冻结协议：`prereg/PRE_REGISTRATION_V07.md`（采集前 commit）
> 偏离记录：`prereg/PRE_REGISTRATION_DEVIATIONS.md`

## 0. 环境

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
# Isaac Sim 5.1 + Isaac Lab 2.3.2 (checkout tag) + torch 2.7.0 (PyPI cu126)
export OMNI_KIT_ACCEPT_EULA=YES
cd ~/IsaacLab && ./isaaclab.sh -p <脚本>      # 所有 Isaac 脚本
python -m execution_wm.composition_v07.<模块>  # 所有离线脚本（仓库根目录）
```

## 1. 复现步骤（严格顺序）

```bash
cd ~/cvpr_embed-v07

# --- Stage 0：运行时审计（3 条 smoke wave，≤12 条预算内） ---
cd ~/IsaacLab && ./isaaclab.sh -p ~/cvpr_embed-v07/execution_wm/composition_v07/stage0_runtime_audit.py \
    --out ~/cvpr_embed-v07/results/v0_7_composition/audit/time_label_feature_material_tests.json
cd ~/cvpr_embed-v07 && python -m execution_wm.composition_v07.audit_offline

# --- Stage 1：预注册（采集前必须已 commit） ---
python -m execution_wm.composition_v07.prereg_build

# --- Stage 2：采集（smoke 12 条 -> 全量 864 条） ---
cd ~/IsaacLab && ./isaaclab.sh -p ~/cvpr_embed-v07/execution_wm/composition_v07/collect_v07.py \
    --plan ~/cvpr_embed-v07/results/v0_7_composition/prereg/split_plan.json \
    --out-root /media/hdd1/yuhang/datasets/execution_wm/v0_7_smoke --smoke
# 全量（去掉 --smoke；数据根默认 /media/hdd1/yuhang/datasets/execution_wm/v0_7）
cd ~/cvpr_embed-v07 && python -m execution_wm.composition_v07.derive_v07 \
    --root /media/hdd1/yuhang/datasets/execution_wm/v0_7

# --- 单测 ---
python -m execution_wm.composition_v07.tests_v07
python -m execution_wm.validity_v061.tests_v061        # 继承的 T01–T08

# --- Stage 3：训练（先 D/R0 与 D/R1 各 seed42 测吞吐，再放开其余 10 个） ---
python -m execution_wm.composition_v07.train_v07 --data-root /media/hdd1/yuhang/datasets/execution_wm/v0_7 \
    --model D --regime R0 --seed 42
# … D/R1、I/R0、I/R1 × seeds 42/43/44，共 12 run

# --- Stage 4：评价与统计 ---
python -m execution_wm.composition_v07.eval_v07 --data-root /media/hdd1/yuhang/datasets/execution_wm/v0_7
python -m execution_wm.composition_v07.stats_v07
#   stats 输出的主比较含两版口径：test_P1（全部窗口）与 test_P1_purity08（purity≥0.8）
python -m execution_wm.composition_v07.manifests_v07
python -m execution_wm.composition_v07.figures_v07
python -m execution_wm.composition_v07.postprocess_v07
```

## 2. 关键约定（冻结，勿改）

| 项 | 值 |
|---|---|
| 值集 | `A=0.35, B=0.65`；`VALUES=[-0.65,-0.35,0,0.35,0.65]`，索引 2 == 0 |
| cell id | `c{ix}{iy}{iw}`（ix/iy/iw ∈ 0..4） |
| R0 数据集 | 仅 zero + single（共激活 = 0） |
| R1 数据集 | R0 + 30 个 train double（共激活 = 0.2083）；每轴占空比与 R0 精确相等 |
| P1 | 每类 double 的 3 个 held-out cell（XY/XW/YW 各 3） |
| P2 | 5 val / 9 test 个 triple，三个二元投影均在 R1 train |
| 窗口 | L=20、H=40、每 episode 确定性取 6 个 origin；**无 RNG**，全模型共用 |
| 时间 | 整数 control tick；20Hz `hold=(5k)//2` |
| 输入 | simulated deployable-candidate（GT body 速度/角速度通道置零） |
| 损失 | `Huber((ê−e)/s_E)`，`ê = u_ref + r̂`；s_E 取自 R0 训练池，四格共用 |
| checkpoint 选择 | val FDE_xy（共同指标） |
| 轨迹积分 | 中点法则（`traj.py`），预测侧 yaw 由预测 wz 自回归，不用未来 GT yaw |

## 3. 目录

```
prereg/      PRE_REGISTRATION_V07.md, command_cells.json, split_plan.json,
             r0_vs_r1_marginals.json, PRE_REGISTRATION_DEVIATIONS.md
audit/       time_label_feature_material_tests.json, information_roles.json,
             traj_integration_tests.json
manifests/   episodes.csv, windows.csv, command_coverage.csv, data_manifest.json
metrics/     main_axis_lead.csv, trajectory_metrics.csv, data_vs_arch_effect.csv,
             per_anchor_seed.csv, termination_and_masks.csv, param_check.json
predictions/ pred_cache.npz (+ pred_cache_manifest.json，含 sha256)
config/ checkpoints/ train_logs/  code_snapshot/  figures/
report/      V0_7_FULL_REPORT.md, V0_7_DECISION.md
```

## 4. 复算校验点

- `predictions/pred_cache.npz` 的 sha256 必须与 `pred_cache_manifest.json` 一致；
  所有汇总/bootstrap 只读该缓存，不重新推理。
- `derive_identity_audit.csv`：全量 `u_ref + r_label == e_label`。
- `manifests/data_manifest.json`：`index_sha256` / `plan_hash` / `controller_hash`。
- `metrics/param_check.json`：D/I 参数量比与延迟（公平性披露）。
