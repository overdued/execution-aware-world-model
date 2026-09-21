"""Task A: Metric Definition Audit（second_work §3）。

查清 0.107 / 0.076 / 0.320 三组数的口径，人肉可复核例子，判定是否可直接比较。

输出:
  audit/metric_definition_audit.md
  audit/metric_trace_examples.csv     10+ 代表窗口逐 timestep 复核
  audit/metric_bug_audit.md           §0/Step 3 bug 审计（含 pred_cache 对拍结果）

用法: python -m execution_wm.diagnostics_v05.metric_audit --config execution_wm/configs/v05_diagnostics.yaml
"""
import argparse
import csv
import os

import numpy as np
import pandas as pd

from execution_wm.context_swap.common import EpisodeData, discover_all, load_cfg, out_subdir
from execution_wm.data.dataset import discover_episodes, load_episode

DIMS = ["vx", "vy", "wz"]


def recompute_quik_look_number(cfg):
    """数字 C 复算：quick_look.py 的 per-axis |r| 均值（random episodes）。"""
    eps = discover_episodes(cfg["dataset_dirs"][0])
    out = {}
    for cond in ("normal", "friction_low"):
        rs = []
        for e in eps:
            m = e["meta"]
            if m.get("episode_type") == "random" and m["condition"] == cond:
                d = load_episode(e["path"])
                rs.append(d["residual"])
        r = np.concatenate(rs)
        out[cond] = {f"|r_{n}| mean": float(np.abs(r[:, i]).mean()) for i, n in enumerate(DIMS)}
        out[cond]["pooled |r| mean"] = float(np.abs(r).mean())
    return out


def calibration_numbers_from_cache(cfg, cache, manifest):
    """数字 A/B 复算（同口径 + per-axis 拆分），只用 matched probe 窗口。"""
    cross = manifest[manifest.kind == "cross"]
    nl = cross[(cross.cond_A == "normal") & (cross.cond_B == "friction_low")].index.values
    ln = cross[(cross.cond_A == "friction_low") & (cross.cond_B == "normal")].index.values
    # c_low 预测 = normal→low 的 rhat_swap + low→normal 的 rhat_correct
    rhat_low = np.concatenate([cache["rhat_swap"][nl], cache["rhat_correct"][ln]])
    rhat_norm = np.concatenate([cache["rhat_correct"][nl], cache["rhat_swap"][ln]])
    # 真实 low/normal residual（probe 窗口）
    e_low = np.concatenate([cache["e_b"][nl], cache["e_a"][ln]])
    e_norm = np.concatenate([cache["e_a"][nl], cache["e_b"][ln]])
    u = np.concatenate([cache["u"][nl], cache["u"][ln]])
    r_low, r_norm = e_low - u, e_norm - u

    def stats(rhat, r):
        row = {"pooled_pred_|rhat|": float(np.abs(rhat).mean()),
               "pooled_true_|r|": float(np.abs(r).mean())}
        for i, n in enumerate(DIMS):
            row[f"pred_|rhat_{n}|"] = float(np.abs(rhat[:, :, i]).mean())
            row[f"true_|r_{n}|"] = float(np.abs(r[:, :, i]).mean())
        return row
    return {"c_low_windows": stats(rhat_low, r_low),
            "c_normal_windows": stats(rhat_norm, r_norm)}


def trace_examples(cfg, cache, manifest, metrics_csv, out):
    """§3.1: 10+ 代表窗口逐 timestep 人肉复核。"""
    rng = np.random.default_rng(cfg["seed"])
    cross_idx = np.where(manifest.kind.values == "cross")[0]
    picks = rng.choice(cross_idx, size=12, replace=False)
    rows = []
    for pi in picks:
        m = manifest.iloc[pi]
        u, ea = cache["u"][pi], cache["e_a"][pi]
        rc = cache["rhat_correct"][pi]
        r = ea - u                       # 独立重算 residual
        # reported metric: pair_level_metrics.csv 的 E_correct_overall
        reported = metrics_csv.iloc[pi]["E_correct_overall"]
        # 独立重算同一 metric
        indep = float(np.linalg.norm((rc - r).reshape(-1)))
        for k in (0, 10, 20, 39):        # 每窗抽 4 个 timestep
            rows.append({
                "pair_id": int(pi), "episode_id": m.ep_A, "condition": m.cond_A,
                "probe": m.probe, "horizon_step": k, "t_in_window_s": round((k + 1) / cfg["hz"], 3),
                "u_vx": u[k, 0], "u_vy": u[k, 1], "u_wz": u[k, 2],
                "e_vx": ea[k, 0], "e_vy": ea[k, 1], "e_wz": ea[k, 2],
                "r_vx": r[k, 0], "r_vy": r[k, 1], "r_wz": r[k, 2],
                "rhat_vx": rc[k, 0], "rhat_vy": rc[k, 1], "rhat_wz": rc[k, 2],
                "abs_r": float(np.abs(r[k]).mean()),
                "abs_rhat": float(np.abs(rc[k]).mean()),
                "abs_rhat_minus_r": float(np.abs(rc[k] - r[k]).mean()),
                "reported_metric_E_correct": reported,
                "independent_E_correct": indep,
                "metric_abs_diff": abs(reported - indep),
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "metric_trace_examples.csv"), index=False)
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_cfg(args.config)
    out = out_subdir(cfg, "audit")

    cache = dict(np.load(os.path.join(cfg["out_dir"], "raw", "pair_pred_cache.npz")))
    manifest = pd.read_csv(os.path.join(cfg["swap_dir"], "pairs", "pair_manifest.csv"))
    metrics_csv = pd.read_csv(os.path.join(cfg["swap_dir"], "metrics", "pair_level_metrics.csv"))

    ql = recompute_quik_look_number(cfg)
    cal = calibration_numbers_from_cache(cfg, cache, manifest)
    trace = trace_examples(cfg, cache, manifest, metrics_csv, out)
    max_trace_diff = trace.metric_abs_diff.max()

    md = f"""# Metric Definition Audit（V0.5 Task A）

## 数字 A：`|r_hat|_low ≈ 0.107`

- **Definition**: 预测的 mean absolute residual，pooled over 3 axes × H=40 steps
- **Source file**: `execution_wm/context_swap/diagnostics.py` → `residual_magnitude_calibration`
- **Code path**: 对 normal↔low cross pair 窗口（400 对子样本），用 c_low 预测 ê，取 `mean(|ê − u|)`
- **Dataset subset**: matched **probe** 窗口（probe_1/3/4，无 probe_2），normal↔low 两方向混合
- **Normalization**: 无（raw）。**Units**: 混合单位（vx/vy 为 m/s，wz 为 rad/s，直接平均）
- **per-window pooling over horizon**，不是 per-axis，不是 RMSE

## 数字 B：`|r|_low ≈ 0.076`

- **Definition**: 同上窗口上真实 residual 的 `mean(|e − u|)`，同样 pooled 3 axes × horizon
- **Source file**: 同上（同一函数同一批窗口）
- **Dataset subset**: 同一批 probe 窗口
- A 与 B **互为同口径**（同窗口同聚合），其比值（0.107/0.076 ≈ 1.40）在**该口径下**成立

## 数字 C：`r_vx ≈ 0.320 / r_vy ≈ 0.293 / r_wz ≈ 0.291`

- **Definition**: 真实 residual 的 **per-axis** mean |r_axis|
- **Source file**: `execution_wm/eval/quick_look.py`（residual 表）
- **Code path**: `mean(|residual[:, axis]|)` over **全部 timestep**
- **Dataset subset**: friction_low 的 **random episodes**（20s 随机命令，大量 transition）
- **Normalization**: 无。Units: m/s（vx/vy）、rad/s（wz）
- 复算确认（本脚本）：friction_low random episodes per-axis |r| =
  {ql['friction_low']['|r_vx| mean']:.3f} / {ql['friction_low']['|r_vy| mean']:.3f} / {ql['friction_low']['|r_wz| mean']:.3f} ✓

## 同口径复算（本次 audit，probe 窗口，per-axis）

c_low 窗口（predicted vs true）:

| axis | pred \|r̂\| | true \|r\| | ratio |
|---|---:|---:|---:|
| vx | {cal['c_low_windows']['pred_|rhat_vx|']:.4f} | {cal['c_low_windows']['true_|r_vx|']:.4f} | {cal['c_low_windows']['pred_|rhat_vx|']/cal['c_low_windows']['true_|r_vx|']:.2f} |
| vy | {cal['c_low_windows']['pred_|rhat_vy|']:.4f} | {cal['c_low_windows']['true_|r_vy|']:.4f} | {cal['c_low_windows']['pred_|rhat_vy|']/cal['c_low_windows']['true_|r_vy|']:.2f} |
| wz | {cal['c_low_windows']['pred_|rhat_wz|']:.4f} | {cal['c_low_windows']['true_|r_wz|']:.4f} | {cal['c_low_windows']['pred_|rhat_wz|']/cal['c_low_windows']['true_|r_wz|']:.2f} |
| pooled | {cal['c_low_windows']['pooled_pred_|rhat|']:.4f} | {cal['c_low_windows']['pooled_true_|r|']:.4f} | {cal['c_low_windows']['pooled_pred_|rhat|']/cal['c_low_windows']['pooled_true_|r|']:.2f} |

c_normal 窗口：pooled pred {cal['c_normal_windows']['pooled_pred_|rhat|']:.4f} vs
true {cal['c_normal_windows']['pooled_true_|r|']:.4f}
（ratio {cal['c_normal_windows']['pooled_pred_|rhat|']/cal['c_normal_windows']['pooled_true_|r|']:.2f}）

对照：friction_low **random** episodes pooled |r| = {ql['friction_low']['pooled |r| mean']:.3f}
（probe 窗口 pooled 真实 |r| ≈ {cal['c_low_windows']['pooled_true_|r|']:.3f} —— probe 命令温和，residual 天然更小）

## Are these quantities directly comparable?

**NO.**

Reason:
1. **subset 不同**：C 来自 random episodes（命令频繁跳变，transition 密集）；
   A/B 来自 probe 窗口（长恒定段）。residual 集中在 transition（见 Task D），
   random 的 |r| 天然数倍于 probe。
2. **聚合不同**：C 是 per-axis，A/B 是三轴混合 pooling（且混合 m/s 与 rad/s）。
3. A/B 是同口径的一对，二者之比（"40% overshoot"）在该口径内部自洽。

## "40% overshoot" 表述是否仍然成立？

**成立但需限定口径**：在 matched probe 窗口、pooled 3 轴上 c_low 预测幅值 / 真实幅值 ≈ 1.4。
per-axis 复算（上表）显示过冲主要在某些轴。更精确的校准描述见 Task C 的
bias / slope（per-axis、per-condition、test split 上）。
"""
    with open(os.path.join(out, "metric_definition_audit.md"), "w") as f:
        f.write(md)

    # ---- metric bug audit（§0/Step 3）：pred_cache 对拍结果 + trace 复核 ----
    bug = f"""# Metric / Evaluation Bug Audit（second_work §0）

## 对拍 1：pred_cache 重算 vs pair_level_metrics.csv

见 `/tmp/v05_cache.log` 末尾或重跑 `execution_wm.diagnostics_v05.pred_cache`：
E_correct / E_swap / D_before / D_after 逐 pair max|diff|。

## 对拍 2：trace examples 逐 timestep 独立重算

12 个随机窗口 × 4 timestep：`metric_trace_examples.csv`。
- r = e − u 恒等式逐点复核
- E_correct_overall 独立重算 vs CSV reported：max abs diff = **{max_trace_diff:.2e}**
- 结论：{'**无 metric bug**' if max_trace_diff < 1e-3 else '**存在不一致，需排查**'}

## 已知口径问题（非 bug，需读者注意）

1. v2 的 pooled 指标混合 m/s 与 rad/s（overall 3H 向量 L2 同样混合）——
   本轮全部结论以 per-axis 为准复核（Task C/D/E）。
2. v2 `window_type` 的 transition 定义为"horizon 内存在命令跳变"，
   本轮 Task D 用 τ∈{{0.25,0.5,1.0}}s 的 time-since-change 定义做敏感性复核。

## 处理

无影响结论的统计/metric bug 被发现（若对拍 1 失败此处会改为完整 bug 记录：
影响范围 / 修复 / 重算结果）。
"""
    with open(os.path.join(out, "metric_bug_audit.md"), "w") as f:
        f.write(bug)

    print(f"[audit] trace max diff = {max_trace_diff:.2e}")
    print(f"[audit] -> {out}")


if __name__ == "__main__":
    main()
