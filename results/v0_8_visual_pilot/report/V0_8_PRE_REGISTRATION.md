# V0.8 PRE-REGISTRATION：受限规模视觉预测 pilot（采集前冻结）

日期：2026-09-27 ｜ 工作区：`~/cvpr_embed-v08`（exp/cvpr-v08-visual-pilot）
机器可读计划：`prereg/split_plan_v08.json`（由 `execution_wm/v08_visual/prereg_build_v08.py`
确定性生成，plan_sha256 见文件内；**任何改动视为偏离，须记入 V0_8_DEVIATIONS.md**）
前置状态：V0.7 CLOSURE（commit a049fb5）、Stage B smoke gate 6/6 PASS（commit fd53bc9）。

本文件在**正式采集之前**冻结以下各项：值集/场景/分组/分支/时序模板/输入目标口径/
模型结构/损失/训练控制/评价口径/统计方法/最小有用效应。任务书建议的 pilot 门
**原样接受，不作修改**。

---

## 1. 数据预算与隔离（§C1）

```
6 场景布局 × 8 独立 group × 4 分支 = 192 episodes
  train L0-L2: 96 eps    val L3: 32 eps    test L4-L5: 64 eps
+ smoke 12 eps（已完成）= 204 ≤ 硬上限 240（失败/重试计入余额 36，全部登记）
```

- **场景布局**（几何真不同，非换皮）：半径 12m 八方位 slot 周界地标。
  L0：面板 slot 0/2/4/6；L1：slot 1/3/5/7；L2：8 slot 全布、高 2/4m 交替；
  L3（val）：slot 0/3/6；L4（test）：slot 1/5 面板 + 半径 10m 圆柱 ×4；
  L5（test）：slot 0/1/2/4/5/6 面板（2.5m）。几何参数冻结在 plan `layouts`。
- **group**：每布局 8 个；`reset_seed = 9000 + 37·(8·L + g)`（v08 命名空间）。
  摩擦按组循环 [nominal 1.0, mid 0.6, low 0.3]（V0.7 正常到低摩擦范围，不注风险）；
  外观 palette_id = (3g+L) mod 8，与摩擦周期不相关（生成器断言：每个摩擦值 ≥3 种外观）。
  **不把颜色编码成摩擦标签。**
- **分支（4/group）**：U0 / U1 / U2 三个预先固定的不同合法候选命令序列 + U0 重复试验
  （同初态/动作/工况，给出同命令重复噪声底）。分支除命令外物理工况与场景全固定。
- **命令脚本**：复用 V0.7 `split_plan.json` 已冻结的 cell 值集（±0.35/±0.65，三轴）、
  dwell 多重集（0.6/0.9/1.2/1.5 各 ×3）与模板 T0–T5。
  - train 分支：U0/U1/U2 = v07 train group 的 R1 脚本 sc06/07/08（R1 命令原则）；
  - val/test 分支：U0 = P0 脚本 sc00，U1 = P1 脚本 sc04，U2 = P2 脚本 sc08
    （P0/P1 为主，P2 单列 EXPLORATORY）；
  - 脚本源映射：train→v07 G[(L·8+g)%12]，val→v07 G[12+g%6]，
    test L4→v07 G[18+g%6]、L5→v07 G[18+(g+3)%6]；逐 episode 落盘血缘。
- **公共前缀钉定**：每个脚本首个 segment 的 cell 统一钉为 `c422`（vx=+0.65），
  dwell 不变。效果：同 group 4 分支在 settle(1.0s)+首段（最短 0.6s）内命令逐位相同，
  给候选匹配提供完全相同 context；首段 cell 为单轴训练 cell，不破坏 P1/P2 的
  held-out 性质（仅作用于其余 11 段）。
- **时序层级**（沿用 V0.7 定义，互不混同）：P0 seen-cell / P1 held-out pair /
  P2 triple（EXPLORATORY）。主终点只用 test P1。
- **pre-action 状态**：同 group 各分支同 reset_seed、同前缀 → 物理确定性
  （Stage B A/B 已证相机不影响物理）。采集时逐分支保存匹配原点（tick 75）处的
  实际 pre-action 状态距离（base pos/quat、关节、速度）；>1e-4（m 或 rad）的分支对
  标记 approximate-matched，仅从候选匹配分析排除、单列不删，不宣称精确反事实。
  PhysX 求解器内部状态无法完全快照恢复，统一标注 approximate-matched 口径。
- 6 场景只支持 pilot；结论不外推跨场景总体；血缘（布局/group/分支/重复/源脚本）
  整体切分，同 group 全部内容属于同一 split。

## 2. 统一输入/目标（§C2）

**时间基准**：物理通道用与 V0.7 完全相同的派生管线（50Hz→20Hz，同代码同参数）。
视觉帧用真实捕获记录（实际间隔 0.06s，首帧 0.02s，时间戳 = 捕获+0.02s，方向保守）。
**边界用 tick 级判定**（`rgb_ctrl_tick_first_seen`），不用名义网格：

- **Z_t（输入）**：context clip = `first_seen_tick ≤ origin_tick` 的最后 16 帧 →
  冻结 V-JEPA2（snapshot b3c1679，sha256 已审计）→ 8 时域 token × 4×4 空间
  mean-pool × 1024。context-only，单独 encoder 调用（clip 级因果分离，B4 已固化）。
- **H_t^p（输入）**：过去 L=20 tick（20Hz，1s）deployable-candidate 本体观测
  （privileged 通道 base_linear_velocity_body / base_angular_velocity 屏蔽，同 V0.7
  PRIV_FIELDS 口径）+ 过去命令。
- **U（输入）**：未来 H=40 tick（2s）公开候选命令序列。不是未来执行结果。
- **目标 Z\*_{t+h}**：target clip = `origin_tick < first_seen_tick ≤ origin_tick + h·50`
  的前 16 帧（h=1s）/ 前 32 帧（h=2s）→ 同一冻结 encoder → 8×16×1024（1s）/
  16×16×1024（2s）。单独 encoder 调用。
- **目标 E\***：未来 40 tick physical execution（20Hz，3 轴，同 V0.7 派生口径）；
  相对姿态/位置标签沿用 V0.7 派生（lb_yaw / lb_base_position 等）作物理副终点。
- **归一化**：Z\* 逐通道（1024 维，跨时域/空间共享）与 E\* 逐轴的 mean/std **仅由
  train 窗口拟合**，冻结后用于 val/test。对称预处理。future label 无权进入 forward
  （B4 T2 单测契约，pilot manifest 上重跑）。

**窗口**：与 V0.7 同规则（phase==1 内确定性等距 ≤6 origin/episode，无 RNG）+
 钉定匹配原点 tick 75（1.5s，所有模板首段内；与回归原点重叠则去重，
 `eligible_for_matching=true` 仅标此原点）。origin 处 phase 断言在 manifest 构建时
 校验（确定性规则：取 ≤110 的最大 phase==1 tick，记入 manifest）。

## 3. 三个小模型（§C3），最多 9 个主 run

三变体共享冻结 encoder、输入、split/window manifest、训练步数预算，seed ∈ {42,43,44}，
各 run 独立训练。结构（`execution_wm/v08_visual/model_v08.py`，commit 时冻结）：

```
h = R(H^p, U)：GRU(51→128)[过去] + MLP(120→128→128)[未来命令] + MLP(256→256→256)
Z_t 分支：pool → MLP(1024→256→128)
G（V-AUX/V-EXEC 完全相同）：MLP(256→128→120)  -> Ê（未来 40×3 execution）
P：trunk MLP(384[+16]→512→512) + factored head ×2（1s: 128 tok, 2s: 256 tok；
   每头 = Linear(512→n_tok×128) + 共享 Linear(128→1024)）
V-DIRECT：Ẑ = P(Z_t, h)
V-AUX：   Ê = G(h)；Ẑ = P(Z_t, h)                    （L_exec 只经 R 反传）
V-EXEC：  Ê = G(h)；Ẑ = P(Z_t, h, stop_gradient(Ê)→MLP(120→32→16))
```

- **参数量 / FLOPs（实测，batch=1 前向）**：

  | 变体 | params | FLOPs/fwd |
  |---|---:|---:|
  | V-DIRECT | 26,467,328 | ≈155.6 M |
  | V-AUX | 26,515,704 | ≈155.7 M |
  | V-EXEC | 26,528,296 | ≈155.7 M |

  V-AUX vs V-EXEC 参数差 12,592（0.05%）≪ ±10%。仅 factored token 投影，无复杂交互网络。
- **初始化规则**：R 的 GRU 与 G（若 V0.7 D/R1 checkpoint 张量形状严格一致）按
  seed 匹配（42/43/44→42/43/44）从 V0.7 D/R1 checkpoint 初始化；**三变体用同一
  规则**（V-DIRECT 同样复用 R，不给 V-EXEC 预训练优势）。形状不一致则三组统一
  随机初始化并记入 DEVIATIONS，冻结 Direct/R1 仅作物理参照。P/Z 分支一律随机初始化。
  不允许选测试最优 seed。
- **stop_gradient(Ê)**：本轮简化设计，显式声明——防止视觉梯度让物理头编码
  不可解释信息。
- **单测（训练前必须全过）**：B4 T1–T4（窗口因果、前向对 target 不变、context
  缓存不变、encoder clip 非因果护栏）+ T2b（L_exec 对 V-AUX/V-EXEC 的 R 非零梯度、
  V-EXEC stop_gradient 完整）。两头独立则不称辅助监督对照，须改实现而非事后解释。
- 诊断基线（context-feature 重复、零运动）只作低成本对照报告，不声称复现
  DINO-WM / V-JEPA2-AC。

## 4. 损失与训练控制（§C4）

- **L_vis** = train-fit z 归一化目标上的 **patch L1**：|Ẑ−Z\*| 对（时域 token ×
  空间 token × 通道）取 mean；逐 token reduction 如实记录。1s/2s 两头等权相加。
  不搜索 cosine/MSE 取测试最优。
- **L_exec** = train-fit 归一化 execution 上 L1（tick×轴 mean）。
- **总损失**：V-DIRECT: L_vis；V-AUX/V-EXEC: L_vis + **λ_E = 1.0** · L_exec
  （两组相同，预注册冻结，不调）。本轮无只对 V-EXEC 生效的额外 loss。
- **优化**：AdamW lr 3e-4、wd 1e-4、cosine 至 0、warmup 500 步、batch 64、
  **30,000 步固定预算**（三组相同）、fp32（保确定性；特征已离线预计算）。
- **checkpoint**：统一取 **validation（L3 布局）2s visual loss** 最优；不用 test
  调权重；物理性能仅为预登记回归检查。
- **特征缓存**：冻结 encoder 对全部窗口 clip 只编码一次，npz + sha256 +
  source-clip 时间/split 血缘（`predictions/`）。
- **放行顺序**：先跑 V-AUX seed42 + V-EXEC seed42，实测吞吐/显存/过拟合 smoke
  正常后再放开其余 7 run。报告 warm-up、同步 CUDA 计时、batch1 与 batch64 延迟；
  不把摊销时间叫真机单步时延。

## 5. 评价（§C5）

- **主终点**：**test P1 的 2s spatial latent error** = train-fit z 归一化后
  |Ẑ−Z\*| 对（16 时域 token × 16 空间 token × 1024 通道）mean；逐窗 → 组内均值 →
  **group-equal**（16 个 test group 等权）汇总；同时按场景（L4/L5）分别给出。
- **副终点**：1s latent error；P0 代价（同口径）；P2（EXPLORATORY）；
  physical FDE/ADE/net-yaw（Ê，V-AUX/V-EXEC）与 fixed-lead 逐轴误差。
- **同组动作区分**（test group，匹配原点 tick 75，完全相同 context）：
  - 对 U0/U1/U2 分别预测 future，与各自真实 future latent 比对：
    candidate-future matching accuracy（argmin 命中率）与 paired difference
    prediction error；
  - 三候选间真实 feature 距离 vs U0-repeat 同命令重复噪声底；近乎不可分的组
    单列、不事后删除；
  - 候选真实 future 只用于离线打分，不输入预测。
- **motion head 敏感性（V-EXEC）**：固定全部输入，Ê / nominal E=U / zero E /
  shuffled E 四档对照；true future E 仅作 ORACLE_DIAGNOSTIC，oracle 好不算部署结果；
  wrong E 有害只证明依赖，不宣称因果发现。
- latent matching 不称为导航/恢复成功率；本轮不执行动作选择控制。

## 6. 统计与门（§C6）

- 每 seed 分别报告；组内候选/重复/重叠窗口聚合后按独立 group 汇总；按场景分列；
  test 仅 2 场景，不推断广泛场景总体（STATISTICAL_SCOPE: PILOT）。
- 相同样本上 paired group bootstrap（10,000 次重抽 16 个 test group，percentile
  95% CI）；CI 解释限于当前测试设计。相对与绝对改善同时报告。
- **V0.8 自己的最小有用效应（本轮声明，不继承 V0.7 的 5% 门之外的任何门）**：
  1. **EXECUTION_CONDITIONING_INCREMENT SUPPORTED** ⟺ V-EXEC 相对 V-AUX 主终点
     相对改善 ≥5%，paired group CI 下界 >0，≥2/3 seed 同向，且候选 matching
     accuracy 退化 ≤5pp；
  2. **AUXILIARY_SUPERVISION_EFFECT SUPPORTED** ⟺ V-AUX 相对 V-DIRECT 同口径成立；
  3. **回归容差**：V-EXEC vs V-AUX 在 P0 主 latent 误差与 physical FDE 上相对恶化
     ≤5%（PHYSICAL_PREDICTION_RETENTION）；
  4. CI 不足写 INCONCLUSIVE；明显劣化写 NOT_SUPPORTED。不用 test 选门；
     该门只用于资源/下一步决策，不是录用标准。

## 7. 预算与中止（§5）

- ≤240 新增 episode（smoke 12 + pilot 192 = 204 已计划，重试计入余额 36）；
  ≤9 主训练 run；特征缓存只算一次。
- 总预算上限 100 GPU·h、数据缓存 50GB。smoke 实测外推：采集 ≈15.5 s/ep →
  192 eps ≈ 0.9 h；特征 ≈0.5 h；9 run × ≈0.3 h ≈ 3 h；合计 ≈5–8 GPU·h，余量充足。
  超预算即停并报告，不换分辨率/帧数偷降成本。
- 阻断条件：不稳定渲染、目标退化、因果泄漏、无法恢复 pre-action 状态——
  先阻断，不用"结果不好"触发重设计。GPU·h = 独占卡数 × 墙钟；kernel 时间另记。

## 8. 已知限制与负结果承接（自 smoke 报告）

1. 渲染跨 episode 非确定（物理逐位相同下 RGB 像素不同）：本轮以 U0-repeat 分支
   实测同命令噪声底并如实报告；渲染配置冻结为 smoke 同版（collect_v08 @ fd53bc9，
   render_interval=10，256×256 前视相机），不在 pilot 中途更改。
2. 帧间隔 0.06s ≠ 名义 0.05s：一律使用真实捕获 tick/时间戳（§2）。
3. 采集首帧渲染未稳定：origin 规则天然避开（phase==1 段）。
4. 时间戳 = 捕获+0.02s（方向保守）：context 边界安全；target 下界模糊 ≤0.02s 已声明。

## 9. 不做（违反即作废）

不改本文件任何冻结项（偏离必须记入 V0_8_DEVIATIONS.md 并标注影响）；不删失败/终止
样本；不为过门改指标；不按 test 成绩选 checkpoint/seed/门；不做 Risk 场景、极端摩擦
扫描、真机动作、在线纠正、动作选择控制、V-JEPA 全参微调、RGB 生成模型、策略重训；
不升级 Isaac/PyTorch/CUDA；不改 locomotion controller 权重；只本地 commit 不 push；
不打印任何凭据；单 GPU 不并行开两个仿真会话。
