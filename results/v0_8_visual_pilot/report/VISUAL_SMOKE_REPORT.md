# V0.8 Visual Smoke Report（Stage B 门禁）

日期：2026-09-27 ｜ 工作区：`~/cvpr_embed-v08`（exp/cvpr-v08-visual-pilot）
数据：`/media/hdd1/yuhang/datasets/execution_wm/v0_8_smoke`（12 eps，161MB，只读归档）
审计：`results/v0_8_visual_pilot/audit/{encoder_and_checkpoint,physics_invariance_ab,time_camera_causality,feature_usability}.json`
测试：`python -m execution_wm.v08_visual.tests_v08`（T1–T4 ALL PASS）

## 采集与同步事实（实测，非名义值）

- 12/12 episode（3 group × nominal 摩擦 × 4 脚本），phys=680/681、rgb=**227 帧**、term=schedule_end。
- 帧间隔恒为 **0.06s**（渲染网格 0.02s 对相机名义周期 0.05s 的量化结果），首帧 0.02s；
  frame_counter 1..227 连续单调；ctrl_tick 间隔恒为 3。
- 时间戳量化：记录值 = 实际捕获 + 0.02s（first_seen 滞后 −1 tick，方向保守）。
  Stage C 建议改用 `rgb_ctrl_tick_first_seen` 的 tick 比较划定 context/target 边界。
- 物理通道 50Hz 与 V0.7 同口径；相机内外参逐 episode 落盘；label_ 前缀通道仅作标签。

## 六字段门禁

| 字段 | 判定 | 依据 |
|---|---|---|
| RGB_SYNCHRONIZATION | **PASS** | 12/12 eps、227 帧/ep、间隔恒 0.06s、counter 无洞；帧非黑非恒定（std≈62，帧间 absdiff 1.6–5.5）；抽帧目视确认为前视透视场景 |
| ENCODER_LOADED_AND_FROZEN | **PASS** | snapshot b3c1679 固定、model.safetensors sha256 校验、全参数 requires_grad=False、重复编码逐位一致（T3）、165ms/clip、峰值 1.25GiB |
| CONTEXT_CAUSALITY | **PASS** | T1：60 窗口全部 context 帧 ≤ origin；T3：篡改 clip 外所有帧后 context 特征逐位不变；时间戳方向保守（实际捕获早于记录值） |
| TARGET_HORIZON_SEMANTICS | **PASS** | T1：target 帧严格 ∈ (origin, origin+lead]（lead=1s/2s 均验）；边界模糊 ≤0.02s 已记录；horizon 用真实捕获时间戳而非名义网格 |
| ACTION_DEPENDENT_VISUAL_SIGNAL | **PASS（有保留）** | 异命令 target 两两差异（mean 5.64/6.01/6.99，G01/G02/G00）> 同命令渲染噪声基线（1.18/1.98/5.65）逐组成立；但 G00 裕量小，见"发现的负结果" |
| RESOURCE_WITHIN_BUDGET | **PASS** | 采集 3.1 min/12 eps（≈15.5 s/ep）；单编码 165ms/clip、1.25GiB；按 pilot 上限 240 eps 外推：采集 ≈1.0 h，全量特征预计算 ≈分钟级，远低于预算 |

## B3 物理不变性 A/B（相机副作用）

v0_8_smoke（有相机）vs v0_7_smoke（无相机，同 plan/seed）：**26 个物理+锚点通道
× 12 episodes 全部逐位一致（max_abs_diff=0.0）**。相机传感器对物理零影响。

## B4 因果性单测结论

- **关键审计发现**：V-JEPA2 encoder 在 clip 内是双向 attention，**非时间因果**
  （改 clip 后半段，前半 token max|diff|≈10–44；T4 回归护栏已固化此事实）。
- 因此因果性在 **clip 级**强制：context/target 各自独立 encoder 调用，
  任何前向路径不得混合编码。模型签名契约层面不含 target 参数（T2）。

## B5 特征可用性结论

- 通道方差：192 token × 1024 维，死通道（var<1e-8）= **0**；min/median/max = 0.012/0.19/6.95。
- 动作依赖视觉信号存在（见门禁表），但幅度受渲染噪声压缩。

## 发现的负结果（必须随报告保留）

1. **渲染跨 episode 非确定**：物理逐位相同、帧时间戳逐位相同，但 RGB 像素从
   第 0 帧起即不同（max 像素差最高 193，疑似 TAA/累积缓冲非确定）。
   后果：同命令特征基线非零（1.2–5.7），压缩动作依赖信号裕量（G00 尤其小）。
   Stage C 预注册应（a）固定渲染配置并记录该噪声底，或（b）评估关闭 TAA 后
   重采 smoke（若改配置须写入预注册与 DEVIATIONS）。
2. 帧间隔 0.06s ≠ 名义 0.05s：20Hz 派生网格与视频帧网格**不重合**，
   视觉 clip 选取必须基于真实时间戳/tick（windows_v08 已如此实现）。
3. 首帧内容跳变大（absdiff 47.7）：episode 初始渲染未稳定，Stage C 窗口
   origin 应避开首帧邻域（当前 fixed_origins 取 phase==1 段，天然规避）。

## Stage B 总结论

**VISUAL_SMOKE_GATE: PASS（6/6）** — 允许进入 Stage C（预注册 → 受限规模 pilot）。
预注册必须先声明 V0.8 自己的最小有用效应、固定渲染配置口径、tick 级边界定义。
