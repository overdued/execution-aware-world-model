"""Task B: Cluster Bootstrap（second_work §4）。

pair 不独立（同 episode pair 的相邻窗口高度相关），pair-level bootstrap CI 过窄。
cluster = (source_episode, target_episode)，cluster 内所有窗口视为相关。

输出:
  bootstrap/cluster_bootstrap_raw.csv        每次 resample 的均值（复核用）
  bootstrap/cluster_bootstrap_summary.csv
  bootstrap/pair_bootstrap_vs_cluster_bootstrap.csv
  figures/BOOT_pair_vs_cluster.png/.pdf

用法: python -m execution_wm.diagnostics_v05.cluster_bootstrap --config execution_wm/configs/v05_diagnostics.yaml
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from execution_wm.context_swap.common import load_cfg, out_subdir

METRICS_CROSS = ["dE_self_overall", "P_target_overall", "S_dir_overall", "S_mag_overall",
                 "M_shift_cross_overall"]
DIRECTIONS = ["normal->friction_low", "friction_low->normal",
              "normal->friction_vlow", "friction_vlow->normal"]


def boot(values, n, ci, rng):
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    idx = rng.integers(0, len(v), size=(n, len(v)))
    means = v[idx].mean(axis=1)
    return means, np.percentile(means, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])


def cluster_boot(df, metric, n, ci, rng):
    """按 cluster 重采样，pool cluster 内全部 pair 取均值。"""
    clusters = [g[metric].values for _, g in df.groupby("cluster")]
    sizes = np.array([len(c) for c in clusters])
    idx = rng.integers(0, len(clusters), size=(n, len(clusters)))
    means = np.empty(n)
    for i in range(n):
        means[i] = np.concatenate([clusters[j] for j in idx[i]]).mean()
    return means, np.percentile(means, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_cfg(args.config)
    n_boot, ci = cfg["cluster_bootstrap"]["n_bootstrap"], cfg["cluster_bootstrap"]["ci"]
    rng = np.random.default_rng(cfg["seed"])
    out = out_subdir(cfg, "bootstrap")
    fig_dir = out_subdir(cfg, "figures")

    df = pd.read_csv(os.path.join(cfg["swap_dir"], "metrics", "pair_level_metrics.csv"))
    df["cluster"] = df.ep_A.astype(str) + "|" + df.ep_B.astype(str)
    cross = df[df.kind == "cross"].copy()
    same = df[df.kind == "same"].copy()
    cross["direction"] = cross.cond_A + "->" + cross.cond_B

    rows, raw_rows = [], []
    for direction in DIRECTIONS:
        g = cross[cross.direction == direction]
        for metric in METRICS_CROSS:
            p_means, p_ci = boot(g[metric].values, n_boot, ci, rng)
            c_means, c_ci = cluster_boot(g, metric, n_boot, ci, rng)
            for i in range(n_boot):
                raw_rows.append({"direction": direction, "metric": metric,
                                 "iter": i, "pair_mean": p_means[i], "cluster_mean": c_means[i]})
            rows.append({"metric": metric.replace("_overall", ""), "transition": direction,
                         "pair_mean": g[metric].mean(),
                         "pair_ci_low": p_ci[0], "pair_ci_high": p_ci[1],
                         "cluster_mean": float(c_means.mean()),
                         "cluster_ci_low": c_ci[0], "cluster_ci_high": c_ci[1],
                         "num_pairs": len(g), "num_clusters": g.cluster.nunique()})
        # same-condition：cond_A==cond_B==该方向 source
        cond = direction.split("->")[0]
        gs = same[(same.cond_A == cond) & (same.cond_B == cond)]
        p_means, p_ci = boot(gs.M_shift_same.values, n_boot, ci, rng)
        c_means, c_ci = cluster_boot(gs, "M_shift_same", n_boot, ci, rng)
        rows.append({"metric": "M_same", "transition": f"same({cond})",
                     "pair_mean": gs.M_shift_same.mean(),
                     "pair_ci_low": p_ci[0], "pair_ci_high": p_ci[1],
                     "cluster_mean": float(c_means.mean()),
                     "cluster_ci_low": c_ci[0], "cluster_ci_high": c_ci[1],
                     "num_pairs": len(gs), "num_clusters": gs.cluster.nunique()})
        # M_cross - M_same（两套 cluster 独立重采样取差）
        gx = g.dropna(subset=["M_shift_cross_overall"])
        c_means_x, _ = cluster_boot(gx, "M_shift_cross_overall", n_boot, ci, rng)
        diff = c_means_x - c_means
        dci = np.percentile(diff, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
        rows.append({"metric": "M_cross_minus_M_same", "transition": direction,
                     "pair_mean": gx.M_shift_cross_overall.mean() - gs.M_shift_same.mean(),
                     "pair_ci_low": np.nan, "pair_ci_high": np.nan,
                     "cluster_mean": float(diff.mean()),
                     "cluster_ci_low": dci[0], "cluster_ci_high": dci[1],
                     "num_pairs": len(gx) + len(gs),
                     "num_clusters": gx.cluster.nunique() + gs.cluster.nunique()})

    comp = pd.DataFrame(rows)
    comp.to_csv(os.path.join(out, "pair_bootstrap_vs_cluster_bootstrap.csv"), index=False)
    pd.DataFrame(raw_rows).to_csv(os.path.join(out, "cluster_bootstrap_raw.csv"), index=False)
    summary = comp[["transition", "metric", "cluster_mean", "cluster_ci_low",
                    "cluster_ci_high", "num_pairs", "num_clusters"]]
    summary.to_csv(os.path.join(out, "cluster_bootstrap_summary.csv"), index=False)

    # ---- figure: pair vs cluster CI ----
    key_metrics = ["dE_self", "P_target", "S_dir", "S_mag"]
    fig, axes = plt.subplots(1, len(key_metrics), figsize=(4.5 * len(key_metrics), 5))
    for ax, m in zip(axes, key_metrics):
        sub = comp[comp.metric == m]
        x = np.arange(len(sub))
        for k, (_, r) in enumerate(sub.iterrows()):
            ax.plot([x[k] - 0.12, x[k] - 0.12], [r.pair_ci_low, r.pair_ci_high], "b-", lw=2)
            ax.plot([x[k] + 0.12, x[k] + 0.12], [r.cluster_ci_low, r.cluster_ci_high], "r-", lw=2)
            ax.plot(x[k] - 0.12, r.pair_mean, "bo", ms=4)
            ax.plot(x[k] + 0.12, r.cluster_mean, "rs", ms=4)
        ax.axhline(0, color="k", ls="--", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([t.replace("friction_", "f_").replace("normal", "N")
                            for t in sub.transition], rotation=20, fontsize=8)
        ax.set_title(m)
    axes[0].plot([], [], "b-", label="pair bootstrap")
    axes[0].plot([], [], "r-", label="cluster bootstrap")
    axes[0].legend(fontsize=8)
    fig.suptitle("BOOT: pair-level vs cluster-level bootstrap 95% CI")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "BOOT_pair_vs_cluster.png"), dpi=150)
    fig.savefig(os.path.join(fig_dir, "BOOT_pair_vs_cluster.pdf"))
    plt.close(fig)

    pd.set_option("display.width", 220)
    print(comp.round(4).to_string(index=False))
    print(f"[boot] -> {out}")


if __name__ == "__main__":
    main()
