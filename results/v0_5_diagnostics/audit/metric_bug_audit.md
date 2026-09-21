# Metric / Evaluation Bug Audit（second_work §0）

## 对拍 1：pred_cache 重算 vs pair_level_metrics.csv

见 `/tmp/v05_cache.log` 末尾或重跑 `execution_wm.diagnostics_v05.pred_cache`：
E_correct / E_swap / D_before / D_after 逐 pair max|diff|。

## 对拍 2：trace examples 逐 timestep 独立重算

12 个随机窗口 × 4 timestep：`metric_trace_examples.csv`。
- r = e − u 恒等式逐点复核
- E_correct_overall 独立重算 vs CSV reported：max abs diff = **2.38e-07**
- 结论：**无 metric bug**

## 已知口径问题（非 bug，需读者注意）

1. v2 的 pooled 指标混合 m/s 与 rad/s（overall 3H 向量 L2 同样混合）——
   本轮全部结论以 per-axis 为准复核（Task C/D/E）。
2. v2 `window_type` 的 transition 定义为"horizon 内存在命令跳变"，
   本轮 Task D 用 τ∈{0.25,0.5,1.0}s 的 time-since-change 定义做敏感性复核。

## 处理

无影响结论的统计/metric bug 被发现（若对拍 1 失败此处会改为完整 bug 记录：
影响范围 / 修复 / 重算结果）。
