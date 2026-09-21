# Pre-Swap Audit（03.1 v2 §4）

## 4.4 Checkpoint
- path: `/media/hdd1/yuhang/checkpoints/execution_wm/v0/context/best.pt`
- sha256: `670e6febfd08b091a9277cba04c432e90bf87cff1fbc8c0a16849273d13643db`
- size: 869839 bytes
- original V0 best val loss: 0.016036120502071247
- context_dim: 8, gru_hidden: 128, hidden_dims: [256, 256]
- 说明：V0 只训练过一个 context checkpoint，无挑选空间（§4.4）。

## 4.1 Privileged information leakage
- 结果: PASS
- encoder 输入: base_linear_velocity_body, base_angular_velocity, projected_gravity ... 共 7 组 proprio + 历史命令
- 禁止字段在 encoder 源码中出现: 无

## 4.2 Causality
- 结果: PASS — c_t = GRU(proprio[t0-L+1..t0], cmd[t0-L+1..t0])，GRU 因果；不读取 t+1..t+H 任何数据。

## 4.3 接口可拆分 + regression
- 结果: PASS
- max|Δr_hat| = 0.000e+00, max|Δe_hat| = 0.000e+00, max|Δz| = 0.000e+00
- encode_state / encode_context / predict_execution 三个接口已加入 ExecutionContextModel，forward 改调它们，参数未动。

## Environment
- git commit: d2ea8f5a04f49de3387900312f5c66aca610732f
- python 3.11.16, torch 2.7.0+cu126 (cuda 12.6), GPU: NVIDIA GeForce RTX 4090
