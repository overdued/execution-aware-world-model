# ENV_AND_SCOPE — V0.7 收尾 / V0.8 视觉 pilot 环境与范围审计

> 2026-09-27 · 执行者：Kimi Code · 任务书：`01_KimiCode_V07收尾与V08视觉预测执行任务书.md` v1.0

## 1. 工作区实地确认

| 项 | 实测值 |
|---|---|
| V0.7 worktree | `/home/yuhang/cvpr_embed-v07`，分支 `exp/cvpr-v07-composition`，HEAD `b3b8278`，**工作区干净**（0 dirty） |
| V0.8 worktree（新建） | `/home/yuhang/cvpr_embed-v08`，分支 `exp/cvpr-v08-visual-pilot`，基于 `b3b8278` |
| 旧数据 | `/media/hdd1/yuhang/datasets/execution_wm/v0_7`（864 eps + 864 个 .20hz.npz，只读） |
| 旧结果 | `~/cvpr_embed-v07/results/v0_7_composition/`（只读；收尾产物写入 v08 树） |

## 2. 完整 SHA256（不截断）

| 文件 | SHA256 |
|---|---|
| `v0_7/index.json` | `05096525f8a097884acc7a502e792cc58370b3af8b4d1c26934a7e00c3379c65` |
| `prereg/split_plan.json` | `6957649b0c74da06a77771e15d97014f1ea8e175a28fa8ab21fcf27acb111750` |
| `prereg/command_cells.json` | `315028853e14c14c6450f0ea838374d9fab71973f4dd5f4fbd414e97a6156b48` |
| `predictions/pred_cache.npz` | `839a3dfedb6b81bf91341116c04d0c433c3f57ebe85a1cd7c930d0c7cc6eccdd` |

与 V0.7 manifest/报告中的截断前缀一致（839a3dfedb6b81bf… / 6957649b0c74da06…）。

## 3. GPU 实测（nvidia-smi，不照抄）

- 本机 **1 张** GPU：index 0，**RTX 4090，49140 MiB（≈48 GB）**，驱动 **535.309.01**。
- **审计时发现一个已运行 6 天的残留进程**：PID 1149699，`play.py --task Isaac-Velocity-Flat-Unitree-Go2-Play-v0 --livestream 2`（Isaac 串流会话，6.4 GB 显存，25% util）。
  非本任务进程，**未 kill**（任务书 §1.6）；已向用户报告。本任务与其串行/错峰使用，
  小模型训练显存需求 <1 GB，不冲突；Isaac 采集前需与用户确认其去留。

## 4. 环境（沿用已验证组合，不升级）

- conda env `isaaclab`（Python 3.11）：Isaac Sim 5.1.0 + Isaac Lab v2.3.2（tag checkout）+ torch 2.7.0（PyPI cu126）。
- 已打补丁：`isaaclab.sh` 的 torch 检查放宽；7 个 .kit 文件关闭 `rtx.verifyDriverVersion`（驱动 535 Vulkan 版本号溢出 bug）。
- 已知网络限制：`download-r2.pytorch.org` 不可达 → V-JEPA 2 权重若走 PyTorch 官方源会失败，
  需在 Stage B 实测下载通道（HF hub / GitHub release），失败则写 BLOCKED，不随机初始化冒充。
- EULA：`OMNI_KIT_ACCEPT_EULA=YES`；后台 Isaac 脚本必须 `PYTHONUNBUFFERED=1`。

## 5. 范围声明

- V0.7 **冻结**：不重跑采集、不重训 12 run、不改 split/门槛/纯度阈值/窗口、不给 I 打补丁。
  收尾修订全部写入 v08 树的 errata/重算产物，旧结果只读。
- Direct/R1 保留为物理预测参考；Interaction I 冻结为 negative/inconclusive baseline。
- 不做：Risk、极端摩擦扫描、路由器、MDA、语言任务、HumanoidVLN、真机、在线纠正、
  locomotion 重训、V-JEPA 全参微调、RGB 生成模型、强制 c_t 瓶颈。
- git 只本地 commit，不 push；不打印任何凭据。
