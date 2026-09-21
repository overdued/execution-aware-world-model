"""Task D: Transition vs Steady 分析（second_work §6）+ §9 normal transient 诊断。

窗口分类（time-since-last-command-change，来自 pred_cache 的 t_since_change）:
  transition: 窗内存在 timestep 距命令变化 <= τ（τ ∈ {0.25, 0.5, 1.0}s 敏感性）
  steady:     窗内全部 timestep 距命令变化 > 0.5s

§9: normal 条件下 per-timestep |r| 在 transition/steady 的分布（r_t 是否混入大量
    controller transient）。

输出:
  transition/transition_vs_steady_metrics.csv
  transition/transition_window_manifest.csv
  transition/normal_transient_diagnostic.json
  figures/TRANS_normal_residual.png/.pdf
  figures/TRANS_p_target_by_window.png/.pdf

用法: python -m execution_wm.diagnostics_v05.transition --config execution_wm/configs/v05_diagnostics.yaml
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from execution_wm.context_swap.common import VEL_NAMES, load_cfg, out_subdir
from execution_wm.data.dataset import discover_episodes, load_episode

EPS = 1e-8


def cosine_rows(a, b):
    fa = a.reshape(len(a), -1)
    fb = b.reshape(len(b), -1)
    return (fa * fb).sum(1) / (np.linalg.norm(fa, axis=1) * np.linalg.norm(fb, axis=1) + EPS)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_cfg(args.config)
    out = out_subdir(cfg, "transition")
    fig_dir = out_subdir(cfg, "figures")
    hz = cfg["hz"]
    taus = cfg["transition"]["windows_s"]
    steady_min_s = 0.5

    cache = dict(np.load(os.path.join(cfg["out_dir"], "raw", "pair_pred_cache.npz")))
    manifest = pd.read_csv(os.path.join(cfg["swap_dir"], "pairs", "pair_manifest.csv"))
    cross_mask = (manifest.kind == "cross").values
    N = len(manifest)

    u, ea, eb = cache["u"], cache["e_a"], cache["e_b"]
    e_c = u + cache["rhat_correct"]
    e_s = u + cache["rhat_swap"]
    tsc = cache["t_since_change"]                    # [N,H]
    d_swap = e_s - e_c
    d_tgt = eb - ea
    s_dir_all = cosine_rows(d_swap, d_tgt)
    n_swap = np.linalg.norm(d_swap.reshape(N, -1), axis=1)
    n_tgt = np.linalg.norm(d_tgt.reshape(N, -1), axis=1)

    E_correct = np.linalg.norm((e_c - ea).reshape(N, -1), axis=1)
    E_swap = np.linalg.norm((e_s - ea).reshape(N, -1), axis=1)
    D_before = np.linalg.norm((e_c - eb).reshape(N, -1), axis=1)
    D_after = np.linalg.norm((e_s - eb).reshape(N, -1), axis=1)
    mae_ax = {n: {"c": np.abs(e_c[:, :, i] - ea[:, :, i]).mean(1),
                  "s": np.abs(e_s[:, :, i] - ea[:, :, i]).mean(1)}
              for i, n in enumerate(VEL_NAMES)}

    directions = (manifest.cond_A + "->" + manifest.cond_B).values

    # ---- 窗口 manifest + 逐 τ 分类 ----
    wm = pd.DataFrame({"pair_idx": np.arange(N), "direction": directions,
                       "kind": manifest.kind.values, "probe": manifest.probe.values})
    for tau in taus:
        wm[f"class_tau{tau}"] = np.where(
            np.nanmin(tsc, axis=1) <= tau, "transition",
            np.where(np.nanmin(tsc, axis=1) > steady_min_s, "steady", "gap"))
    wm.to_csv(os.path.join(out, "transition_window_manifest.csv"), index=False)

    # ---- 逐 τ × direction × class 指标 ----
    rows = []
    for tau in taus:
        cls = wm[f"class_tau{tau}"].values
        for direction in np.unique(directions[cross_mask]):
            for c in ("transition", "steady"):
                sel = cross_mask & (directions == direction) & (cls == c)
                if sel.sum() < 10:
                    continue
                row = {"tau_s": tau, "direction": direction, "window_class": c,
                       "N": int(sel.sum()),
                       "E_correct": E_correct[sel].mean(), "E_swap": E_swap[sel].mean(),
                       "dE_self": (E_swap - E_correct)[sel].mean(),
                       "D_before": D_before[sel].mean(), "D_after": D_after[sel].mean(),
                       "P_target": (D_before - D_after)[sel].mean(),
                       "S_dir": s_dir_all[sel].mean(),
                       "S_mag": (n_swap / (n_tgt + EPS))[sel].mean()}
                for n in VEL_NAMES:
                    row[f"MAE_{n}_correct"] = mae_ax[n]["c"][sel].mean()
                    row[f"MAE_{n}_swap"] = mae_ax[n]["s"][sel].mean()
                rows.append(row)
    tsm = pd.DataFrame(rows)
    tsm.to_csv(os.path.join(out, "transition_vs_steady_metrics.csv"), index=False)
    pd.set_option("display.width", 220)
    print(tsm[tsm.tau_s == 0.5].round(3).to_string(index=False))

    # ---- §9: normal 条件 per-timestep residual transition vs steady ----
    tau_u = cfg["transition"]["tau_u"]
    diag = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, cond in zip(axes, ("normal", "friction_low")):
        trans_r, steady_r = {n: [] for n in VEL_NAMES}, {n: [] for n in VEL_NAMES}
        for e in discover_episodes(cfg["dataset_dirs"][0]):
            m = e["meta"]
            if m.get("episode_type") != "random" or m["condition"] != cond:
                continue
            d = load_episode(e["path"])
            cmd, r = d["cmd_vel"], d["residual"]
            du = np.zeros(len(cmd))
            du[1:] = np.linalg.norm(np.diff(cmd, axis=0), axis=1)
            change_idx = np.where(du > tau_u)[0]
            t_since = np.array([t - change_idx[change_idx <= t][-1] if (change_idx <= t).any()
                                else np.inf for t in range(len(cmd))]) / hz
            for i, n in enumerate(VEL_NAMES):
                trans_r[n].append(np.abs(r[t_since <= 0.5, i]))
                steady_r[n].append(np.abs(r[t_since > 0.5, i]))
        stats = {}
        for i, n in enumerate(VEL_NAMES):
            tr = np.concatenate(trans_r[n])
            st = np.concatenate(steady_r[n])
            stats[n] = {"transition_mean": float(tr.mean()), "steady_mean": float(st.mean()),
                        "ratio": float(tr.mean() / st.mean())}
            ax.hist(st, bins=80, density=True, histtype="step", label=f"{n} steady", lw=1.2)
            ax.hist(tr, bins=80, density=True, histtype="step", ls="--", label=f"{n} transition", lw=1.2)
        ax.set_xlim(0, 1.2)
        ax.set_title(f"TRANS: |r| distribution — {cond}")
        ax.set_xlabel("|r| per timestep")
        ax.legend(fontsize=7)
        diag[cond] = stats
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "TRANS_normal_residual.png"), dpi=150)
    fig.savefig(os.path.join(fig_dir, "TRANS_normal_residual.pdf"))
    plt.close(fig)
    with open(os.path.join(out, "normal_transient_diagnostic.json"), "w") as f:
        json.dump(diag, f, indent=2)
    print("\n[normal transient] |r| transition/steady ratio:",
          {n: round(diag["normal"][n]["ratio"], 2) for n in VEL_NAMES})
    print("[friction_low] ratio:",
          {n: round(diag["friction_low"][n]["ratio"], 2) for n in VEL_NAMES})

    # ---- figure: P_target by class × direction（τ=0.5 主口径 + 敏感性） ----
    fig, ax = plt.subplots(figsize=(9, 5))
    dirs4 = ["normal->friction_low", "friction_low->normal",
             "normal->friction_vlow", "friction_vlow->normal"]
    width = 0.25
    x = np.arange(len(dirs4))
    for k, tau in enumerate(taus):
        vals_t, vals_s = [], []
        for d in dirs4:
            for c, acc in (("transition", vals_t), ("steady", vals_s)):
                row = tsm[(tsm.tau_s == tau) & (tsm.direction == d) & (tsm.window_class == c)]
                acc.append(float(row.P_target.iloc[0]) if len(row) else np.nan)
        ax.bar(x + (k - 1) * width - 0.06, vals_t, width * 0.48,
               label=f"transition τ={tau}s", alpha=.85)
        ax.bar(x + (k - 1) * width + 0.06, vals_s, width * 0.48,
               label=f"steady τ={tau}s", alpha=.6, hatch="//")
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("friction_", "f_") for d in dirs4], fontsize=9)
    ax.set_ylabel("P_target (mean)")
    ax.set_title("TRANS: P_target by window class × direction (sensitivity over τ)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "TRANS_p_target_by_window.png"), dpi=150)
    fig.savefig(os.path.join(fig_dir, "TRANS_p_target_by_window.pdf"))
    plt.close(fig)
    print(f"[trans] -> {out}")


if __name__ == "__main__":
    main()
