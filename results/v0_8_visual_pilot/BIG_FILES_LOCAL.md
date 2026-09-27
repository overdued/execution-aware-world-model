# 大二进制文件（仅存本地，未上传 GitHub）

GitHub 单文件上限 100MB；以下文件合计 ~12.7GB，保留在本机，sha256 用于完整性校验。
完整分支（含全部二进制）在本地 `exp/cvpr-v08-visual-pilot`（commit d0fc311）。

| 本地路径 | 大小 MB | sha256（前16位） |
|---|---|---|
| `results/v0_8_visual_pilot/predictions/VAUX_s42_test_preds.npz` | 324.9 | `a8b0c8035bf70a57` |
| `results/v0_8_visual_pilot/predictions/VAUX_s42_val_preds.npz` | 162.5 | `92992fc2390e442e` |
| `results/v0_8_visual_pilot/predictions/VAUX_s43_test_preds.npz` | 324.9 | `a3d4f9046cac4aab` |
| `results/v0_8_visual_pilot/predictions/VAUX_s43_val_preds.npz` | 162.5 | `5190cbae6cab925e` |
| `results/v0_8_visual_pilot/predictions/VAUX_s44_test_preds.npz` | 324.9 | `74844752e403a776` |
| `results/v0_8_visual_pilot/predictions/VAUX_s44_val_preds.npz` | 162.5 | `8570653246045fae` |
| `results/v0_8_visual_pilot/predictions/VDIRECT_s42_test_preds.npz` | 324.7 | `4348a58cdeebe67e` |
| `results/v0_8_visual_pilot/predictions/VDIRECT_s42_val_preds.npz` | 162.4 | `47290b38cf57d70e` |
| `results/v0_8_visual_pilot/predictions/VDIRECT_s43_test_preds.npz` | 324.7 | `b5f8b0f90bac8449` |
| `results/v0_8_visual_pilot/predictions/VDIRECT_s43_val_preds.npz` | 162.4 | `7bf4ed3e9bb72a23` |
| `results/v0_8_visual_pilot/predictions/VDIRECT_s44_test_preds.npz` | 324.7 | `ac1f06f6229b6310` |
| `results/v0_8_visual_pilot/predictions/VDIRECT_s44_val_preds.npz` | 162.4 | `ce8f70d130fe08ae` |
| `results/v0_8_visual_pilot/predictions/VEXEC_s42_test_preds.npz` | 324.9 | `688acc2289554af2` |
| `results/v0_8_visual_pilot/predictions/VEXEC_s42_val_preds.npz` | 162.5 | `768a6bce18eff92c` |
| `results/v0_8_visual_pilot/predictions/VEXEC_s43_test_preds.npz` | 324.9 | `5b25cba4c45f1c0a` |
| `results/v0_8_visual_pilot/predictions/VEXEC_s43_val_preds.npz` | 162.5 | `497a6dccfc1ce783` |
| `results/v0_8_visual_pilot/predictions/VEXEC_s44_test_preds.npz` | 325.0 | `75f530b663563439` |
| `results/v0_8_visual_pilot/predictions/VEXEC_s44_val_preds.npz` | 162.5 | `a20d115fdac977d6` |
| `results/v0_8_visual_pilot/predictions/feature_cache.npz` | 2629.1 | `0f3b24967451a9f9` |
| `results/v0_8_visual_pilot/predictions/pred_cache.npz` | 4762.3 | `d830da2a39e67783` |
| `results/v0_8_visual_pilot/predictions/smoke_features.npz` | 0.7 | `92f55126da405397` |
| `results/v0_8_visual_pilot/checkpoints/VAUX_s42/best.pt` | 106.1 | `840bb1cc82d2e38f` |
| `results/v0_8_visual_pilot/checkpoints/VAUX_s43/best.pt` | 106.1 | `6fe1cdc715ffc687` |
| `results/v0_8_visual_pilot/checkpoints/VAUX_s44/best.pt` | 106.1 | `39b2d5ef3f9b5a39` |
| `results/v0_8_visual_pilot/checkpoints/VDIRECT_s42/best.pt` | 105.9 | `01388556168bc09a` |
| `results/v0_8_visual_pilot/checkpoints/VDIRECT_s43/best.pt` | 105.9 | `71c880e17464b8e6` |
| `results/v0_8_visual_pilot/checkpoints/VDIRECT_s44/best.pt` | 105.9 | `63b6542e5207d0c2` |
| `results/v0_8_visual_pilot/checkpoints/VEXEC_s42/best.pt` | 106.1 | `8e464a6e410018ef` |
| `results/v0_8_visual_pilot/checkpoints/VEXEC_s43/best.pt` | 106.1 | `9e6ec3e1f761e8b1` |
| `results/v0_8_visual_pilot/checkpoints/VEXEC_s44/best.pt` | 106.1 | `2a1e698fcbec92dc` |

ckpt 完整 sha256 见各 `checkpoints/<run>/summary.json` 的 `ckpt_sha` 字段；
pred_cache 完整 sha256 见 `predictions/pred_cache_manifest.json`。