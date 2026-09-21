# 工作区说明（cvpr_embed）

这是 yuhang 的 Execution-Aware World Model 研究工作区。当前任务见 `first_work.md`（第一阶段实验规范）。
串流修复记录见 `docs/streaming_fix.md`。

## 环境（服务器 10.20.30.50，RTX 4090，驱动 535.309.01）

- conda 环境：`isaaclab`（Python 3.11，`~/miniconda3`）；激活：`source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab`
- Isaac Sim **5.1.0**（pip 装）+ Isaac Lab **v2.3.2**（`~/IsaacLab`，checkout tag，**不要用 main 分支**，main 是 3.0 要 Isaac Sim 6.1 + torch 2.12/cu130，驱动带不动）
- torch **2.7.0**（PyPI cu126 版）。**不要用 download.pytorch.org 的 cuXXX 源**——`download-r2.pytorch.org` 在此网络不可达（Errno 101）
- 跑任何东西前加 `OMNI_KIT_ACCEPT_EULA=YES`
- pip 给该环境装包时锁 torch：`PIP_CONSTRAINT` 指到含 `torch==2.7.0 / torchvision==0.22.0 / torchaudio==2.7.0` 的文件

## 已打补丁（重装会被覆盖，需重打）

1. `~/IsaacLab/isaaclab.sh`：`ensure_cuda_torch()` 的精确匹配 `2.7.0+cu128` 放宽为 `2.7.0*`
2. kit 文件加 `rtx.verifyDriverVersion.enabled = false`（驱动版本被 Vulkan 误读的官方 bug）：
   `~/IsaacLab/apps/isaaclab.python*.kit`（4 个）+ site-packages 里 `isaacsim/apps/isaacsim.exp.full*.kit`（3 个）

## 数据盘布局

- 大数据全部放 HDD：`/media/hdd1/yuhang`（700 权限）
- 软链接：`~/datasets`、`~/checkpoints`、`~/IsaacLab/logs` → HDD
- 本项目的：`/media/hdd1/yuhang/datasets/execution_wm` 和 `/media/hdd1/yuhang/checkpoints/execution_wm`

## 看渲染画面（WebRTC 串流）

```bash
cd ~/IsaacLab && OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh -p \
    scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Flat-Unitree-Go2-Play-v0 \
    --num_envs 8 --livestream 2 --experience go2_full_streaming.kit
```

Mac 客户端连 `10.20.30.50`。黑屏排查见 `docs/streaming_fix.md`；保险方案是前面套 `xvfb-run -a`。

## 常见坑

- `~/IsaacLab/scripts/tutorials/` 的示例脚本是无限循环，headless 下不会自己退出
- headless 脚本结束时 `simulation_app.close()` 可能挂起（已知 bug），工作已完成直接杀进程或 `os._exit(0)`
- locomotion controller checkpoint：`~/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-09-20_22-52-26/model_299.pt`
- **这台机器不要同时开两个 AI 会话干活**（会互相 kill 进程）

## 项目结构

`execution_wm/`：collector / dataset / 3 个模型 / train / eval，配置全在 `execution_wm/configs/*.yaml`。
执行顺序按 `first_work.md` §19：先数据采集验证，再训 action_only → direct → context 对比。
