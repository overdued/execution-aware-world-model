# Isaac Sim 5.1 WebRTC 串流黑屏修复记录

> 2026-09-20/21，两个会话（Claude Code yuhang-44 + VS Code Copilot ik-60）联合排查。
> 协调记录见 `~/STREAMING_COORDINATION.md`。

## 症状

Mac Omniverse Streaming Client 2.0.0 连接 `10.20.30.50` 成功，但画面全黑；
客户端日志反复出现 `No offer is received from remote`，随后断开。

## 原因链（两条，叠加）

1. **kit 体验文件不对**：`play.py --livestream 2` 默认走 Isaac Lab 的 headless 精简 kit，
   缺少完整 streaming 组件，NVST 服务从不发送 offer（IsaacLab GitHub issue #7616 同类）。
   - 验证：裸 `isaacsim.exp.full.streaming` 实例在协议层能成功（"Stream started success"）。
2. **服务器无可用显示**：SSH 会话无 DISPLAY（X0 是 gdm 的物理会话，X99 是 yanbo 的）。
   完整 GUI kit 无窗口会在 `omni.physx.ui` 的 `get_keyboard` 处 SIGSEGV (139)，
   或客户端调整分辨率时触发 `GLFW initialization failed` 后崩溃。
   - 解法：`xvfb-run -a --server-args="-screen 0 1920x1080x24"` 提供虚拟显示
     （窗口走 Xvfb，渲染仍走 Vulkan/RTX 4090，不受影响）。

## 最终可用方案

`~/IsaacLab/apps/go2_full_streaming.kit`（ik-60 创建）：

```kit
[dependencies]
"isaacsim.exp.full.streaming" = {}

[settings]
persistent.isaac.asset_root.default = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"
persistent.isaac.asset_root.cloud = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"
persistent.isaac.asset_root.nvidia = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"
```

启动命令（当前验证可用，无需 Xvfb）：

```bash
cd ~/IsaacLab && source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh -p \
    scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Flat-Unitree-Go2-Play-v0 \
    --num_envs 8 --livestream 2 --experience go2_full_streaming.kit
```

**保险方案**（若上述再次黑屏/崩溃）：用 `xvfb-run -a --server-args="-screen 0 1920x1080x24"`
包裹同一条命令。

## 端口与客户端

- TCP 8011（信令）、TCP 49100（媒体）、UDP 47998（媒体），服务器防火墙关闭（ufw inactive）
- 客户端：Mac Omniverse Streaming Client 2.0.0，地址 `10.20.30.50`，同局域网
- 客户端黑屏时可先试 View → Reload（官方文档建议）
- 同一时刻只允许一个客户端连接

## 其他已排除项

- 驱动 535.309.01 被 Vulkan 误读为 535.53 导致 RTX 检查失败 → 已在 kit 文件加
  `rtx.verifyDriverVersion.enabled = false`（见 CLAUDE.md 环境节）
- ROS2 Bridge 启动失败 → 与串流无关，忽略
- `allowDynamicResize` 补丁 → 非必需，且其 resize 路径在无显示时会触发 GLFW 崩溃，可不加

## 教训

- **同一台机器不要跑两个 AI 会话干同一任务**（互相 kill 进程 3 次）。
  协调文件：`~/STREAMING_COORDINATION.md`。
