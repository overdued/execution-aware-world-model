# §3 运行时 policy command 消费审计（T10）

> 状态：**VERIFIED**（在安装环境直接插桩，非源码推断）
> 脚本：`execution_wm/data/policy_command_audit.py` · 产物：`audit/policy_command_timeline.json`
> 环境：Isaac Sim 5.1 + Isaac Lab 2.3.2 + rsl_rl；policy = `unitree_go2_flat/2026-09-20_22-52-26/model_299.pt`
> 规模：3 条 wave × 200 tick（≤6 条限制内）；Q1_step_vx / Q3_turn / Q2_step_vy

## 1. 记录内容（每个控制 tick）

- `u_requested`：本 tick 写入 `cmd_term.vel_command_b` 的值；
- `term_after_write`：写入后立即读回 command term buffer；
- `policy_obs_cmd`：**policy 调用之前**从 `obs_td["policy"]` 复制的 command slice
  （obs 布局 48 维，command 位于 `[9:12]`，见 `obs_terms`）；
- `policy_obs_poststep_cmd`：`env.step` 返回后的同一 slice（对照）；
- `action_l2` / `action_first3` / `base_lin_vel_body` / `phase` / `t`。

**插桩自身的教训**：第一版记录的是 `env.step` 之后的 obs（被覆盖），得到"零滞后"的
假象。修正为在 policy 调用前 clone 才是 policy 真正消费的张量。这正是"不能仅靠
源码认定最终行为"的实例。

## 2. 实测结果

| wave | tick 数 | `term_after_write == u_requested` | **消费 obs == u[t−1]** | 消费 obs == u[t] | 命令事件处的消费滞后 |
|---|---:|---|---:|---:|---|
| Q1_step_vx | 200 | 全等 | **199/199** | 196/200 | 3 个事件全部 = 1 tick |
| Q3_turn | 200 | 全等 | **199/199** | 196/200 | 3 个事件全部 = 1 tick |
| Q2_step_vy | 200 | 全等 | **199/199** | 196/200 | 3 个事件全部 = 1 tick |

- `u_consumed[i] = u_requested[i−1]` 在 3×199 个 tick 上**精确成立**；
- 每个命令事件的消费侧都**精确滞后 1 control tick = 0.02 s**（不是"相关性更高"，是逐 tick 相等）；
- `env.step` 返回的 obs 携带本 tick 写入的命令（`poststep == u[t]` 200/200）。

## 3. 物理含义与本研究输入的决定

因果链（混合了脚本结构与真实系统特性）：

```
tick i-2: 写 u[i-2] -> env.step 后 obs 携带 u[i-2]
tick i-1: policy 消费 u[i-2] -> a[i-1] --作用于--> [t_{i-1}, t_i)
tick i  : 记录状态 e[i]（t_i 时刻速度）
```

因此 **记录在 tick i 的 execution 由 u[i−2] 驱动**（执行滞后 2 tick = 0.04 s）。

**决定**：
- 主口径 `cmd_ref[k] = u_requested`（同一 tick），即"部署忠实的计划命令"；
  residual 中含有该 2-tick 执行滞后，作为系统的物理/流水线属性被显式记录，
  不作为 bug 处理；
- `cmd_consumed` 作为诊断键一并保存（= `cmd_ref` 平移 1 tick），恒等式仍定义在
  `cmd_ref` 上，不受影响；
- **旧 raw 可复用**：滞后是固定可验证规则，240 条 50Hz 日志逐 tick 记录了
  `u_requested`，consumed 命令可版本化精确恢复 → 不重采任何数据。

## 4. 与旧结论的关系

V0.6 记录的"e/u 一步滞后（corr(e_t,u_{t−1})=0.887 > corr(e_t,u_t)=0.835）"现被
**直接测量**替代：滞后是 2 tick 而非 1 tick，且是逐 tick 精确相等而非相关性推断。
V0.6 报告里"corr 更高所以滞后一步"属于用相关性替代因果测量，本版不再使用该论证。
