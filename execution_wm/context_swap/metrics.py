"""Step 11-13: 聚合指标 + paired bootstrap CI（§9/§10/§11）。

读 metrics/pair_level_metrics.csv，输出:
    aggregate_metrics.csv      mean/median/std，按 cond pair × direction × 维度
    bootstrap_ci.csv           dE_self / P_target / S_dir 的 95% paired bootstrap CI
    per_probe_metrics.csv
    per_dimension_metrics.csv
    transition_vs_steady.csv

用法: python -m execution_wm.context_swap.metrics --config execution_wm/configs/context_swap.yaml
"""
import argparse
import os

import numpy as np
import pandas as pd

from execution_wm.context_swap.common import load_cfg, out_subdir

DIMS = ["overall", "vx", "vy", "wz"]
MAIN_METRICS = ["E_correct", "E_swap", "dE_self", "D_before", "D_after",
                "P_target", "S_dir", "S_mag", "M_shift_cross"]


def bootstrap_ci(values, n=10000, ci=0.95, seed=42):
    """paired bootstrap：对 pair 重采样取均值分布的分位数。"""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n, len(v)))
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return (float(lo), float(hi))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_cfg(args.config)
    out = out_subdir(cfg, "metrics")
    df = pd.read_csv(os.path.join(out, "pair_level_metrics.csv"))
    cross = df[df.kind == "cross"].copy()
    same = df[df.kind == "same"].copy()
    cross["direction"] = cross.cond_A + "->" + cross.cond_B

    # ---- aggregate：cond pair × direction × 维度 ----
    rows = []
    for direction, g in cross.groupby("direction"):
        for dim in DIMS:
            row = {"direction": direction, "dim": dim, "N_pairs": len(g)}
            for m in MAIN_METRICS:
                col = f"{m}_{dim}"
                if col not in g:
                    continue
                row[f"{m}_mean"] = g[col].mean()
                row[f"{m}_median"] = g[col].median()
                row[f"{m}_std"] = g[col].std()
            rows.append(row)
    agg = pd.DataFrame(rows)
    agg.to_csv(os.path.join(out, "aggregate_metrics.csv"), index=False)

    # ---- bootstrap CI（关键指标）----
    n_boot, ci_level = cfg["bootstrap"]["n_resamples"], cfg["bootstrap"]["ci"]
    rows = []
    for direction, g in list(cross.groupby("direction")) + [("ALL", cross)]:
        for metric in ("dE_self", "P_target", "S_dir", "S_mag"):
            for dim in DIMS:
                col = f"{metric}_{dim}"
                if col not in g:
                    continue
                lo, hi = bootstrap_ci(g[col].values, n_boot, ci_level, cfg["seed"])
                rows.append({"direction": direction, "metric": metric, "dim": dim,
                             "mean": g[col].mean(), "ci_lo": lo, "ci_hi": hi,
                             "frac_positive": (g[col] > 0).mean(), "N": len(g)})
    # cross vs same（§Metric 5）
    for direction in cross.direction.unique():
        cond = direction.split("->")[0]
        g_same = same[(same.cond_A == cond) & (same.cond_B == cond)]
        lo_c, hi_c = bootstrap_ci(cross[cross.direction == direction]["M_shift_cross_overall"].values,
                                  n_boot, ci_level, cfg["seed"])
        lo_s, hi_s = bootstrap_ci(g_same["M_shift_same"].values, n_boot, ci_level, cfg["seed"])
        rows.append({"direction": direction, "metric": "M_shift", "dim": "overall",
                     "mean": cross[cross.direction == direction]["M_shift_cross_overall"].mean(),
                     "ci_lo": lo_c, "ci_hi": hi_c,
                     "frac_positive": np.nan, "N": len(cross[cross.direction == direction]),
                     })
        rows.append({"direction": f"same({cond})", "metric": "M_shift", "dim": "overall",
                     "mean": g_same["M_shift_same"].mean(),
                     "ci_lo": lo_s, "ci_hi": hi_s,
                     "frac_positive": np.nan, "N": len(g_same)})
    boot = pd.DataFrame(rows)
    boot.to_csv(os.path.join(out, "bootstrap_ci.csv"), index=False)

    # ---- per probe ----
    rows = []
    for (direction, probe), g in cross.groupby(["direction", "probe"]):
        rows.append({"direction": direction, "probe": probe, "N_pairs": len(g),
                     "dE_self_mean": g.dE_self_overall.mean(),
                     "P_target_mean": g.P_target_overall.mean(),
                     "S_dir_mean": g.S_dir_overall.mean(),
                     "S_mag_mean": g.S_mag_overall.mean()})
    pd.DataFrame(rows).to_csv(os.path.join(out, "per_probe_metrics.csv"), index=False)

    # ---- per dimension（§10）----
    rows = []
    for dim in DIMS[1:]:
        for direction, g in list(cross.groupby("direction")):
            rows.append({"dim": dim, "direction": direction, "N_pairs": len(g),
                         "dE_self_mean": g[f"dE_self_{dim}"].mean(),
                         "P_target_mean": g[f"P_target_{dim}"].mean(),
                         "S_dir_mean": g[f"S_dir_{dim}"].mean(),
                         "S_dir_frac_pos": (g[f"S_dir_{dim}"] > 0).mean()})
    pd.DataFrame(rows).to_csv(os.path.join(out, "per_dimension_metrics.csv"), index=False)

    # ---- transition vs steady（§11）----
    rows = []
    for (direction, wtype), g in cross.groupby(["direction", "window_type"]):
        rows.append({"direction": direction, "window_type": wtype, "N_pairs": len(g),
                     "E_correct_mean": g.E_correct_overall.mean(),
                     "E_swap_mean": g.E_swap_overall.mean(),
                     "dE_self_mean": g.dE_self_overall.mean(),
                     "P_target_mean": g.P_target_overall.mean(),
                     "S_dir_mean": g.S_dir_overall.mean()})
    pd.DataFrame(rows).to_csv(os.path.join(out, "transition_vs_steady.csv"), index=False)

    # ---- 控制组对比打印 ----
    print("[metrics] === 主指标（overall, mean [95% CI]）===")
    key = boot[(boot.dim == "overall") & boot.metric.isin(["dE_self", "P_target", "S_dir", "S_mag"])]
    for _, r in key.iterrows():
        print(f"  {r.direction:32s} {r.metric:9s} {r['mean']:+.4f} [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}] "
              f"pos={r.frac_positive:.2f} N={r.N}")
    print("[metrics] === M_shift cross vs same ===")
    for _, r in boot[boot.metric == "M_shift"].iterrows():
        print(f"  {r.direction:32s} {r['mean']:.4f} [{r.ci_lo:.4f}, {r.ci_hi:.4f}] N={r.N}")
    print(f"[metrics] -> {out}")


if __name__ == "__main__":
    main()
