"""Step 14: Figures CS1-CS6（§13）。PNG + PDF + representative_pairs.npz（§19 Deliverable 5）。

用法: python -m execution_wm.context_swap.figures --config execution_wm/configs/context_swap.yaml
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
    EpisodeData, VEL_NAMES, discover_all, load_cfg, load_context_model, out_subdir,
    predict,
)


def save(fig, fig_dir, name):
    fig.savefig(os.path.join(fig_dir, f"{name}.png"), dpi=150)
    fig.savefig(os.path.join(fig_dir, f"{name}.pdf"))
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_cfg(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    L = int(round(cfg["history_s"] * cfg["hz"]))
    H = int(round(cfg["horizon_s"] * cfg["hz"]))
    hz = cfg["hz"]

    model = load_context_model(cfg, device)
    eps = {uid: EpisodeData(e) for uid, e in discover_all(cfg).items()}
    df = pd.read_csv(os.path.join(cfg["out_dir"], "metrics", "pair_level_metrics.csv"))
    cross = df[df.kind == "cross"].copy()
    same = df[df.kind == "same"].copy()
    cross["direction"] = cross.cond_A + "->" + cross.cond_B
    fig_dir = out_subdir(cfg, "figures")
    rng = np.random.default_rng(cfg["seed"])

    # ---------- CS1: 每个主 probe 的代表性 pair（median P_target，非 cherry-pick） ----------
    rep_raw = {}
    main_dirs = ["normal->friction_low", "friction_low->normal"]
    for direction in main_dirs:
        g = cross[cross.direction == direction]
        for probe in sorted(g.probe.unique()):
            gp = g[g.probe == probe]
            if len(gp) == 0:
                continue
            med = gp.P_target_overall.median()
            r = gp.iloc[(gp.P_target_overall - med).abs().argsort().iloc[0]]
            ea, eb = eps[r.ep_A], eps[r.ep_B]
            t0 = int(r.t0)
            wa, wb = ea.window(t0, L, H), eb.window(t0, L, H)
            e_correct, c_a = predict(model, ea, t0, L, H, device)
            c_b = torch.from_numpy(predict(model, eb, t0, L, H, device)[1])[None].to(device)
            e_swap, _ = predict(model, ea, t0, L, H, device, context=c_b)
            t = np.arange(H) / hz
            fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            for i, n in enumerate(VEL_NAMES):
                ax = axes[i]
                ax.plot(t, wa["future_action"][:, i], "k--", alpha=.6, label="commanded")
                ax.plot(t, wa["future_execution"][:, i], "b", label=f"actual {r.cond_A}")
                ax.plot(t, wb["future_execution"][:, i], "g", label=f"actual {r.cond_B}")
                ax.plot(t, e_correct[:, i], "r", label="pred correct ctx")
                ax.plot(t, e_swap[:, i], "m--", label="pred swapped ctx")
                ax.set_ylabel(n)
                ax.legend(loc="best", fontsize=8)
            axes[0].set_title(f"CS1: {probe} | {direction} (median-P_target pair, "
                              f"P={r.P_target_overall:.3f})")
            axes[-1].set_xlabel("t [s]")
            fig.tight_layout()
            save(fig, fig_dir, f"CS1_{probe}_{direction.replace('->', '_to_')}")
            key = f"{probe}|{direction}"
            rep_raw[key] = dict(command=wa["future_action"], source_actual=wa["future_execution"],
                                target_actual=wb["future_execution"], pred_correct=e_correct,
                                pred_swapped=e_swap, c_a=c_a, c_b=c_b[0].cpu().numpy(),
                                t0=np.array([t0]),
                                meta=np.array([probe, r.cond_A, r.cond_B, str(r.ep_A), str(r.ep_B)]))
    np.savez(os.path.join(cfg["out_dir"], "raw", "representative_pairs.npz"),
             **{k: {kk: vv for kk, vv in v.items()} for k, v in rep_raw.items()})

    # ---------- CS2: Target Pull 分布 ----------
    fig, ax = plt.subplots(figsize=(8, 5))
    for direction in sorted(cross.direction.unique()):
        ax.hist(cross[cross.direction == direction].P_target_overall, bins=60,
                alpha=.5, label=f"{direction} (n={(cross.direction == direction).sum()})")
    ax.axvline(0, color="k", lw=1.5, ls="--")
    ax.set_xlabel("P_target = D_before - D_after")
    ax.set_ylabel("count")
    ax.set_title("CS2: Target Pull distribution")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, fig_dir, "CS2_target_pull")

    # ---------- CS3: Error controls ----------
    fig, axes = plt.subplots(1, len(main_dirs), figsize=(6 * len(main_dirs), 5), sharey=True)
    if len(main_dirs) == 1:
        axes = [axes]
    for ax, direction in zip(axes, main_dirs):
        g = cross[cross.direction == direction]
        data = [g.E_correct_overall, g.E_swap_overall, g.E_shuffle_mean, g.E_zero_overall]
        parts = ax.violinplot(data, showmeans=True)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(["correct", "swap", "shuffle", "zero"], fontsize=9)
        ax.set_title(f"CS3: {direction}")
        ax.set_ylabel("prediction error (L2 over horizon)")
    fig.tight_layout()
    save(fig, fig_dir, "CS3_error_controls")

    # ---------- CS4: Direction Alignment ----------
    order = ["normal->friction_low", "friction_low->normal",
             "normal->friction_vlow", "friction_vlow->normal"]
    order = [d for d in order if d in set(cross.direction)]
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [cross[cross.direction == d].S_dir_overall for d in order]
    ax.violinplot(data, showmeans=True)
    ax.axhline(0, color="k", lw=1.5, ls="--")
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=15, fontsize=9)
    ax.set_ylabel("S_dir (horizon-level cosine)")
    ax.set_title("CS4: Swap Direction Alignment")
    fig.tight_layout()
    save(fig, fig_dir, "CS4_direction_alignment")

    # ---------- CS5: Cross vs Same-condition swap ----------
    fig, ax = plt.subplots(figsize=(8, 5))
    labels, data = [], []
    for direction in main_dirs:
        data.append(cross[cross.direction == direction].M_shift_cross_overall)
        labels.append(f"cross\n{direction}")
    for cond in ("normal", "friction_low", "friction_vlow"):
        g = same[(same.cond_A == cond) & (same.cond_B == cond)]
        if len(g):
            data.append(g.M_shift_same)
            labels.append(f"same\n{cond}")
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_ylabel("M_shift = ||pred_swapped - pred_correct||")
    ax.set_title("CS5: Cross-condition vs Same-condition swap effect")
    fig.tight_layout()
    save(fig, fig_dir, "CS5_cross_vs_same")

    # ---------- CS6: Δ_target vs Δ_swap scatter（重算子样本的向量） ----------
    sub = []
    for direction in main_dirs:
        g = cross[cross.direction == direction]
        sub.append(g.iloc[rng.choice(len(g), size=min(300, len(g)), replace=False)])
    sub = pd.concat(sub)
    d_swap_all, d_tgt_all = [], []
    for r in sub.itertuples():
        ea, eb = eps[r.ep_A], eps[r.ep_B]
        t0 = int(r.t0)
        wa, wb = ea.window(t0, L, H), eb.window(t0, L, H)
        e_correct, _ = predict(model, ea, t0, L, H, device)
        c_b = torch.from_numpy(predict(model, eb, t0, L, H, device)[1])[None].to(device)
        e_swap, _ = predict(model, ea, t0, L, H, device, context=c_b)
        d_swap_all.append(e_swap - e_correct)
        d_tgt_all.append(wb["future_execution"] - wa["future_execution"])
    d_swap = np.stack(d_swap_all)   # [N, H, 3]
    d_tgt = np.stack(d_tgt_all)
    for i, n in enumerate(VEL_NAMES):
        fig, ax = plt.subplots(figsize=(6, 6))
        x, y = d_tgt[:, :, i].reshape(-1), d_swap[:, :, i].reshape(-1)
        ax.scatter(x, y, s=2, alpha=.15)
        lim = np.percentile(np.abs(np.concatenate([x, y])), 99)
        ax.plot([-lim, lim], [-lim, lim], "r--", lw=1, label="identity")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        corr = np.corrcoef(x, y)[0, 1]
        ax.set_xlabel("true physical delta (e_B - e_A)")
        ax.set_ylabel("swap-induced delta")
        ax.set_title(f"CS6: {n}  (corr={corr:.3f}, {len(x)} points)")
        ax.legend()
        fig.tight_layout()
        save(fig, fig_dir, f"CS6_delta_scatter_{n}")

    print(f"[figures] -> {fig_dir}")


if __name__ == "__main__":
    main()
