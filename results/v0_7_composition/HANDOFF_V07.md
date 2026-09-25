# HANDOFF_V07 — 交接文档（给下一个执行者 / 另一个 AI 编码工具）

> 写于 2026-09-25 · 交接时状态：**Stages 0–2 完成，Stage 3–4 未开始**
> 本文是自包含的：读完这一份 + `prereg/PRE_REGISTRATION_V07.md` 就能接着干。

---

## 0. 一分钟版

CVPR 主线 V0.7（结构化命令覆盖 × 动作交互建模）的**数据已按冻结协议采集完毕**
（864 episodes，预算内）。**训练与评价尚未开始**。下一步是按 `README_REPRODUCE.md`
的顺序跑：派生 → 单测 → 训练 12 个 run → 评价/统计/图 → 三份报告。
代码已用部分数据做过端到端 dry-run，已知集成 bug 全部修复。

---

## 1. 工作区与仓库

| 项 | 路径 |
|---|---|
| **主仓库** | `/home/yuhang/cvpr_embed`（分支 `main`，**不要在这里做 V0.7**） |
| **V0.7 worktree** | `/home/yuhang/cvpr_embed-v07`（分支 `exp/cvpr-v07-composition`） |
| 交接时 commit | `07739b1` |
| 数据（新，未覆盖旧） | `/media/hdd1/yuhang/datasets/execution_wm/v0_7`（208 MB，864 eps） |
| 派生数据 | 同上目录的 `*.20hz.npz`（**尚未生成**） |
| checkpoints | `/media/hdd1/yuhang/checkpoints/execution_wm/v0_7`（**空**，等训练写入） |
| 结果树 | `/home/yuhang/cvpr_embed-v07/results/v0_7_composition/` |

**所有命令都在 `~/cvpr_embed-v07` 目录下执行**（Isaac 脚本除外，需在 `~/IsaacLab` 下）。

旧数据目录 `v0` / `v0_6_sq` / `v0_6_1` / `v0_7_smoke` 一律**只读参考，不要改**。

---

## 2. 环境（每次开新 shell 都要做）

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
cd ~/IsaacLab && OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 ./isaaclab.sh -p <isaac脚本>
cd ~/cvpr_embed-v07 && python -m execution_wm.composition_v07.<模块>   # 离线脚本
```

- Isaac Sim **5.1.0** + Isaac Lab **v2.3.2**（`~/IsaacLab`，tag checkout，**不要用 main**）
  + torch **2.7.0**（PyPI cu126）
- **不要升级** Isaac / PyTorch / CUDA；**不要改** locomotion controller 权重
- 本机 **只有 1 张 RTX 4090（49 GB）**，采集与训练都串行；`nvidia-smi` 里显示为 index 0
- 常见坑：
  - `os._exit(0)` 不 flush stdout → 后台跑 Isaac 脚本必须 `PYTHONUNBUFFERED=1`
  - headless 下 `simulation_app.close()` 可能挂起 → 脚本末尾已用 `os._exit(0)`
  - **不要 import `collect_v07`**（它在模块级解析 argparse，会吃掉你的 argv）
  - `--resume`：采集器支持断点续采（扫描已有 episode、跳过已完成三元组、续号、重建 index）

---

## 3. 本轮研究问题与设计（细节见预注册）

**两个问题**：Q-data（R0 单轴 vs R1 单轴+部分二元组合，同预算的差别）与
Q-model（同 R1 数据下 Interaction I 是否优于 Direct D）。

**2×2 = {R0, R1} × {D, I}，每格 3 seed，共 12 主 run。**
`Δ_data = E(D,R0) − E(D,R1)`；`Δ_arch = E(D,R1) − E(I,R1)`，**分别报告，不许混算**。

关键冻结量（**不要改**，改了就不是本轮实验）：

- 值集 `A=0.35, B=0.65`；`VALUES=[-0.65,-0.35,0,0.35,0.65]`，索引 2 == 0
- cell id `c{ix}{iy}{iw}`；R0 = zero+single（共激活 0）；R1 = R0 + 30 train double
- **每轴占空比 R0 = R1 = 0.2778（精确相等）**；差异只在共激活（0 vs 0.2083）
- 泛化层级 P0（seen cell）/ P1（held-out pair cells，**主终点**）/ P2（triple，次要、EXPLORATORY）
- 窗口 L=20、H=40、每 episode 确定性取 6 个 origin（**无 RNG**，全模型共用）
- 输入 = simulated deployable-candidate（GT body 速度/角速度通道置零）
- 损失 `Huber((ê−e)/s_E)`，`ê = u_ref + r̂`；**s_E 取自 R0 训练池，四格共用**
- checkpoint 只用 **val FDE_xy** 选择，**不许看测试成绩**
- 轨迹积分用**中点法则**（`traj.py`），预测侧 yaw 由预测 wz 自回归，**不读未来 GT yaw**
- 改善门（预注册 §7）：主 FDE 相对改善 ≥5% **且** group CI 方向为改善 **且** ≥2/3 seed 同向；
  seen-cell FDE 恶化 ≤5%；CI 跨零 → 写 INCONCLUSIVE，**不许为过门改指标**

---

## 4. 数据现状（已校验）

```
864 episodes = 24 groups × 3 摩擦 × 12 scripts        （预算上限 900，含 smoke ≤12）
  train 216 R0 + 216 R1  |  val 72 P0 + 72 P1 + 72 P2  |  test 72 P0 + 72 P1 + 72 P2
termination: schedule_end 863 / terminated 1（原地保留，未删）
friction readback: 写入=读回精确（static 1.0/0.6/0.3，dynamic 0.70/0.42/0.21）
plan_hash 6957649b0c74da06 · controller_hash 98a0ed5bb1bb5a7a
```

**两个必须在报告中单列的事实**（dry-run 时发现）：

1. **`future_cell_purity` 中位数仅约 0.43** —— 2 s 预测窗常跨越命令切换。
   主比较需给**两版**：全部 P1 窗口 + 高纯度窗口（purity ≥ 0.8）。
   纯度定义在 `data_v07.py`，逐窗记录在 `manifests/windows.csv`。
2. `terminated` 的那 1 条**保留**，用 valid-prefix mask 处理，不做样本筛除。

---

## 5. 下一步怎么做（严格按序）

```bash
cd ~/cvpr_embed-v07
export OMNI_KIT_ACCEPT_EULA=YES

# 1) 派生 20Hz（会写 *.20hz.npz 与 derive_identity_audit.csv）
python -m execution_wm.composition_v07.derive_v07 \
    --root /media/hdd1/yuhang/datasets/execution_wm/v0_7
#    期望：identity max err ≈ 1e-16；ts grid max err ≈ 1e-15

# 2) 单测（全部必须 PASS 才继续）
python -m execution_wm.composition_v07.tests_v07          # 轨迹积分 U1–U8
python -m execution_wm.validity_v061.tests_v061           # 继承的 T01–T08

# 3) 先实测 2 个 run 的吞吐（预注册 §9.2 要求）
python -m execution_wm.composition_v07.train_v07 --data-root /media/hdd1/yuhang/datasets/execution_wm/v0_7 \
    --model D --regime R0 --seed 42
python -m execution_wm.composition_v07.train_v07 --data-root ... --model D --regime R1 --seed 42
#    记录 wall time / 峰值显存 / GPU·h；与 40 GPU·h（神经训练）和 150 GPU·h（全程）上限核算

# 4) 放开其余 10 个 run：{D,I} × {R0,R1} × seeds {42,43,44}

# 5) 评价与产出（顺序不可乱，后一步依赖前一步）
python -m execution_wm.composition_v07.eval_v07 --data-root /media/hdd1/yuhang/datasets/execution_wm/v0_7
python -m execution_wm.composition_v07.stats_v07        # Δ_data / Δ_arch + group bootstrap
python -m execution_wm.composition_v07.manifests_v07
python -m execution_wm.composition_v07.figures_v07      # 4 张图 PNG+PDF
python -m execution_wm.composition_v07.postprocess_v07  # param_check / main_axis_lead / code_snapshot
```

隔离运行（可选，用于测试）：`V07_OUT` / `V07_DATA` / `V07_CKPT` / `V07_PREREG`
四个环境变量可覆盖输出、数据、checkpoint、预注册目录。

### 必须交付（任务书 §10）

```
results/v0_7_composition/
  prereg/     PRE_REGISTRATION_V07.md, command_cells.json, split_plan.json,
              r0_vs_r1_marginals.json, PRE_REGISTRATION_DEVIATIONS.md
  audit/      time_label_feature_material_tests.json, information_roles.json,
              traj_integration_tests.json
  manifests/  episodes.csv, windows.csv, command_coverage.csv, data_manifest.json
  metrics/    main_axis_lead.csv, trajectory_metrics.csv, data_vs_arch_effect.csv,
              per_anchor_seed.csv, termination_and_masks.csv, param_check.json
  predictions/ pred_cache.npz + pred_cache_manifest.json（含 sha256）
  config/ checkpoints/ train_logs/ code_snapshot/ figures/
  report/     V0_7_FULL_REPORT.md, V0_7_DECISION.md
  README_REPRODUCE.md
```

### 最终决策格式（任务书 §10，逐字照抄这些字段名）

```text
CORRECTNESS: PASS / BLOCKED
COVERAGE_CONTRIBUTION: SUPPORTED / NOT_SUPPORTED / INCONCLUSIVE
ARCHITECTURE_INCREMENT: SUPPORTED / NOT_SUPPORTED / INCONCLUSIVE
TRAJECTORY_VALIDITY: PASS / INCONCLUSIVE
NOMINAL_COST: ACCEPTABLE / NOT_ACCEPTABLE / INCONCLUSIVE
STATISTICAL_SCOPE: PILOT / CONFIRMATORY
NEXT_MAIN_ACTION: 只选一个（A 数据覆盖有效架构无增量 / B 交互模型额外有效 /
                  C 连 Direct+R1 也差 / D 统计不够只补新独立 groups），并列证据
```

**统计口径**：单元 = 独立 anchor group（6 个 test group），先 group 内平均再跨 group
配对 bootstrap，3 个 seed 分别报告。**绝不把窗口或多摩擦变体当独立 n。
6 个 test group 撑不起强外推主张，CI 宽就写 INCONCLUSIVE。**

---

## 6. 禁止事项（任务书 §0 + 用户约束）

- 不做：Risk 场景、极端摩擦扫描、路由器、真机动作、MDA、LLM agent、HumanoidVLN、
  V-JEPA 大训练、策略重训、真机 correction
- **不加**：显式 context 作为强制结构、oracle 摩擦、概率头、语言模型；
  第一轮不加只对 I 生效的额外损失
- **不改**：预注册冻结的值集/cell 分配/group 切分/时序模板/R0-R1 构造/窗口规则/
  损失/评价口径/改善门。任何偏离写进 `PRE_REGISTRATION_DEVIATIONS.md`
- **不删**：失败与终止样本、负结果
- **不许**：为了过门改指标；按测试成绩选 checkpoint；把 R0/R1 的收益归因给架构
- **git**：只本地 commit，**不 push**（当前协议）；不打印任何 token/凭据
- **机器**：这台机器**不要同时开两个 AI 会话**干活（会互相 kill 进程）

---

## 7. 历史背景（为什么这么设计）

- V0.6.2 发现：train 的 119 个不同 future-command 向量里**多轴同时非零 = 0**，
  而 B/C 测试窗口 **100%** 含多轴 → 失效是**结构性命令组合缺失**，不是数值越界
- V0.6.2 的 additive baseline 是**轴受限**形式（每个 g_a 只输出 a 轴），
  其负结果**不能**推广为"交互模型不可能有效" → 本轮 I 用完整 `[H,3]` 输出
- V0.6.1 修好的东西要继承：整数 tick 时间、标签恒等式 `u_ref+r_label==e_label`、
  FeatureSchema（vx/vy/wz = 索引 [0,1,5]）、合法化四元数、A/B/C 式透明 split

**不要把"结构性组合缺失已观察到"改写成"交互架构已证明能解决"。本轮就是检验这件事。**

---

## 8. 已完成 vs 未完成

| 阶段 | 状态 | 证据 |
|---|---|---|
| Stage 0 运行时审计 | ✅ 完成 | `audit/time_label_feature_material_tests.json`（3 条 smoke；1-tick 滞后复核 199/199；combine=multiply） |
| Stage 1 预注册 | ✅ 完成并 commit（采集前） | `prereg/*`，commit `2ecdd01` |
| Stage 2 采集 | ✅ 完成 864 eps | `/media/hdd1/yuhang/datasets/execution_wm/v0_7` + `index.json` |
| Stage 2b 派生 20Hz | ❌ **未做** | 下一步第 1 条 |
| Stage 3 训练 12 run | ❌ **未做** | checkpoints 目录为空 |
| Stage 4 评价/统计/图 | ❌ **未做** | — |
| 报告 | ❌ **未做** | `report/` 为空 |

代码已通过端到端 dry-run（用部分真实数据 + 重标 split），发现并修复了 7 个集成 bug，
dry-run 产物全部隔离在 `/tmp/`，**未污染真实数据与 checkpoint 目录**（已复核清理）。
