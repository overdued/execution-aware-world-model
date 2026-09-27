# V0.8 Review Package Manifest

> 打包日期：2026-09-27
> 来源 worktree：`~/cvpr_embed-v08` · 分支 `exp/cvpr-v08-visual-pilot` · commit `d0fc311`
> （GitHub 公开镜像分支：`exp/cvpr-v08-visual-pilot-slim`，内容一致但不含大二进制）
> 冻结协议：commit `0b6feb7`（采集前）；偏离登记 D1–D2 见 `report/V0_8_DEVIATIONS.md`

## 包含内容

| 路径 | 内容 | 来源 |
|---|---|---|
| `report/V0_8_DECISION.md` | 最终判定（10 字段） | git 跟踪，d0fc311 |
| `report/V0_8_FULL_REPORT.md` | 完整报告 | git 跟踪，d0fc311 |
| `report/V0_8_PRE_REGISTRATION.md` / `V0_8_DEVIATIONS.md` / `VISUAL_SMOKE_REPORT.md` / `V0_7_CLOSURE.md` / `V0_7_ERRATA.md` | 预注册、偏离、smoke gate、V0.7 收尾 | git 跟踪 |
| `README_REPRODUCE.md` | 复现步骤 + 交付清单对照 | git 跟踪 |
| `recompute_v08.py` | 独立复算脚本（本机验证 6/6 MATCH, PASS） | git 跟踪 |
| `metrics/` | 全部指标：组级误差、stats_primary、候选匹配、噪声底、E 敏感性、物理指标、latency/memory、逐窗预测 CSV、V0.7 复核表 | git 跟踪 |
| `audit/` | 环境范围、时间/相机因果、encoder 冻结校验、target support、分支 pre-state 逐位校验 | git 跟踪 |
| `prereg/split_plan_v08.json` | 冻结 split/cell 分配（内含 plan_sha256） | git 跟踪 |
| `code_snapshot/` | 模型/数据/训练/评价/采集/派生全部 17 个模块（`execution_wm/v08_visual/` 快照） | git 跟踪 |
| `config/` | split_plan、train_config、e_norm（train-fit 归一化统计） | git 跟踪 |
| `train_logs/<run>/` | 9 run 的 train_log.csv + summary.json（含 ckpt sha256、耗时、峰值显存、延迟） | git 跟踪 |
| `predictions/pred_cache.npz` + `pred_cache_manifest.json` | **完整预测缓存**（9 run × val/test × 896 窗 × {pred_1s, pred_2s, e_hat}，共 16128 keys），manifest 含 sha256 | git 跟踪（完整分支 d0fc311） |
| `predictions/feature_cache_manifest.json` | 特征缓存的 manifest + sha256（npz 本体未纳入，见下） | git 跟踪 |
| `figures/` | primary_by_scene / candidate_matching / exec_sensitivity / train_curves | git 跟踪 |
| `examples/` | 代表性 context/target 帧（12 PNG + index.json）+ representative_physics_and_commands.npz | git 跟踪 |
| `BIG_FILES_LOCAL.md` | 未上传 GitHub 的大文件清单（本地路径 + sha256） | slim 分支 |

## 未纳入的文件（及原因）

| 文件 | 大小 | 原因 | 校验方式 |
|---|---|---|---|
| `predictions/feature_cache.npz` | 2.6GB | 冻结 V-JEPA 2 特征（属数据缓存，非预测结果）；`recompute_v08.py` 全保真复算需要它 | sha256 见 `feature_cache_manifest.json`；本地完整分支可取 |
| `predictions/<run>_<split>_preds.npz`（18 个） | 各 162–325MB | 与 `pred_cache.npz` 内容完全重复（pred_cache 即其合并） | sha256 见 `BIG_FILES_LOCAL.md` |
| `checkpoints/<run>/best.pt`（9 个） | 各 ~106MB | 训练产物权重；验收不需重训，逐窗预测与 ckpt sha256 已足够 | `train_logs/<run>/summary.json` 的 `ckpt_sha` |
| 原始 RGB 视频/episode 数据 | 2.7GB | 任务约定不含全部原始数据；已在采集机器锁定只读（`/media/hdd1/yuhang/datasets/execution_wm/v0_8_pilot`，`chmod -R a-w`） | `manifests/hashes.json` |
| V-JEPA 2 预训练权重 | ~GB 级 | 任务约定不含大型预训练权重 | `audit/encoder_and_checkpoint.json` |

## 验收建议路径

1. 先读 `report/V0_8_DECISION.md`（判定）→ `report/V0_8_FULL_REPORT.md`（证据）。
2. 复核统计：`metrics/stats_primary.json` 对拍 `metrics/visual_error_by_group_seed.csv`；
   或用 `metrics/predictions_per_window_*.csv`（逐窗误差）独立重聚合——无需大缓存。
3. 全保真复算 `recompute_v08.py` 需 `pred_cache.npz`（本包已含）+ `feature_cache.npz`
   （未含，sha256 可校验）。
4. 血缘：任一预测键 `<run>/<split>/<window_id>/<field>` 经 `manifests/windows.csv`
   追到 origin/帧时间/candidate U/seed，ckpt sha256 在 `train_logs/<run>/summary.json`。

## 合规声明

本包不含任何 token、.env、SSH key 或其他凭据；未重新训练、未修改任何结果；
所有数值与 git commit `d0fc311` 一致。
