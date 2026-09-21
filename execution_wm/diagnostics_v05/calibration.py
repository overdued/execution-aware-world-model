"""Task C: Per-axis / Per-condition Calibration Audit（second_work §5）。

在 episode-level test split（test_id + test_ood，无训练泄漏）上用 correct context 做原生
预测，统计 r̂ vs r 的 bias / MAE / RMSE / Pearson / calibration slope / 线性拟合。

输出:
  calibration/per_axis_calibration.csv       condition × axis
  calibration/per_condition_calibration.csv  condition pooled
  figures/CAL_<cond>_<axis>.png/.pdf         true-vs-pred scatter + y=x + 回归线

用法: python -m execution_wm.diagnostics_v05.calibration --config execution_wm/configs/v05_diagnostics.yaml
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from execution_wm.context_swap.common import (  # noqa: E402
    EpisodeData, VEL_NAMES, load_cfg, load_context_model, out_subdir, predict,
)
from execution_wm.data.dataset import discover_episodes, split_episodes  # noqa: E402


def fit_affine(x, y):
    """y ≈ a*x + b（最小二乘）。返回 a, b。"""
    A = np.stack([x, np.ones_like(x)], axis=1)
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(a), float(b)


def stats_block(rhat, r):
    out = {}
    out["true_mean"] = float(r.mean())
    out["pred_mean"] = float(rhat.mean())
    out["true_abs_mean"] = float(np.abs(r).mean())
    out["pred_abs_mean"] = float(np.abs(rhat).mean())
    out["bias"] = float((rhat - r).mean())
    out["MAE"] = float(np.abs(rhat - r).mean())
    out["RMSE"] = float(np.sqrt(((rhat - r) ** 2).mean()))
    out["pearson"] = float(np.corrcoef(rhat.reshape(-1), r.reshape(-1))[0, 1])
    out["calib_slope"] = float(np.cov(rhat.reshape(-1), r.reshape(-1))[0, 1]
                               / (np.var(r.reshape(-1)) + 1e-12))
    a, b = fit_affine(r.reshape(-1), rhat.reshape(-1))   # rhat ≈ a*r + b
    out["fit_a"], out["fit_b"] = a, b
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_cfg(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    L = int(round(cfg["history_s"] * cfg["hz"]))
    H = int(round(cfg["horizon_s"] * cfg["hz"]))
    max_w = cfg["calibration"]["max_windows_per_episode"]
    out = out_subdir(cfg, "calibration")
    fig_dir = out_subdir(cfg, "figures")

    # episode-level test split（与训练一致的无泄漏划分；只用 random episodes）
    dc = __import__("yaml").safe_load(open(cfg["train_config"]))["dataset"]
    eps_all = discover_episodes(cfg["dataset_dirs"][0])
    eps_all = [e for e in eps_all if e["meta"].get("episode_type") == "random"]
    splits = split_episodes(eps_all, dc["val_fraction"], dc["test_id_fraction"],
                            dc["ood_conditions"], True, 42)
    test_eps = splits["test_id"] + splits["test_ood"]

    model = load_context_model(cfg, device)
    rng = np.random.default_rng(cfg["seed"])

    conds = cfg["calibration"]["conditions"]
    data = {c: {"rhat": [], "r": []} for c in conds}
    for e in test_eps:
        cond = e["meta"]["condition"]
        if cond not in data:
            continue
        ep = EpisodeData(e)
        t0s = np.arange(L - 1, ep.T - H - 1)
        if len(t0s) == 0:
            continue
        pick = rng.choice(t0s, size=min(max_w, len(t0s)), replace=False)
        for t0 in pick:
            w = ep.window(int(t0), L, H)
            e_hat, _ = predict(model, ep, int(t0), L, H, device)
            data[cond]["rhat"].append(e_hat - w["future_action"])
            data[cond]["r"].append(w["future_residual"])

    axis_rows, cond_rows = [], []
    for cond in conds:
        if not data[cond]["r"]:
            print(f"[cal] WARNING: {cond} 无 test episode，跳过")
            continue
        rhat = np.stack(data[cond]["rhat"])   # [N,H,3]
        r = np.stack(data[cond]["r"])
        n_win = len(rhat)
        blk = stats_block(rhat, r)
        cond_rows.append({"condition": cond, "n_windows": n_win, **blk})
        for i, ax in enumerate(VEL_NAMES):
            blk_ax = stats_block(rhat[:, :, i], r[:, :, i])
            axis_rows.append({"condition": cond, "axis": ax, "n_windows": n_win, **blk_ax})
            # ---- CAL figure ----
            fig, axx = plt.subplots(figsize=(6, 6))
            x, y = r[:, :, i].reshape(-1), rhat[:, :, i].reshape(-1)
            sub = rng.choice(len(x), size=min(20000, len(x)), replace=False)
            axx.scatter(x[sub], y[sub], s=1, alpha=.1)
            lim = np.percentile(np.abs(np.concatenate([x, y])), 99.5)
            axx.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="y=x")
            xs = np.array([-lim, lim])
            axx.plot(xs, blk_ax["fit_a"] * xs + blk_ax["fit_b"], "r-", lw=1.5,
                     label=f"fit: rhat={blk_ax['fit_a']:.2f}·r+{blk_ax['fit_b']:.3f}")
            axx.set_xlim(-lim, lim)
            axx.set_ylim(-lim, lim)
            axx.set_xlabel("true residual")
            axx.set_ylabel("predicted residual")
            axx.set_title(f"CAL {cond} {ax}: bias={blk_ax['bias']:+.3f} "
                          f"slope={blk_ax['calib_slope']:.2f} r={blk_ax['pearson']:.2f}")
            axx.legend(fontsize=9)
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, f"CAL_{cond}_{ax}.png"), dpi=150)
            fig.savefig(os.path.join(fig_dir, f"CAL_{cond}_{ax}.pdf"))
            plt.close(fig)

    pd.DataFrame(axis_rows).to_csv(os.path.join(out, "per_axis_calibration.csv"), index=False)
    pd.DataFrame(cond_rows).to_csv(os.path.join(out, "per_condition_calibration.csv"), index=False)
    pd.set_option("display.width", 220)
    print(pd.DataFrame(axis_rows).round(4).to_string(index=False))
    print(f"[cal] -> {out}")


if __name__ == "__main__":
    main()
