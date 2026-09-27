"""V0.8 Stage C：图（交付 figures/）。
运行：python -m execution_wm.v08_visual.figures_v08 --results R
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RUN_ORDER = [f"{v}_s{s}" for v in ("VDIRECT", "VAUX", "VEXEC") for s in (42, 43, 44)]
COLOR = {"VDIRECT": "#888888", "VAUX": "#1f77b4", "VEXEC": "#d62728"}


def _variant(run):
    return run.rsplit("_s", 1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    args = ap.parse_args()
    rr = Path(args.results)
    fig = rr / "figures"
    fig.mkdir(exist_ok=True)

    # F1: 训练曲线（val 2s visual loss）
    f, ax = plt.subplots(figsize=(7, 4))
    for run in RUN_ORDER:
        p = rr / "checkpoints" / run / "train_log.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p).dropna(subset=["val_vis2s"])
        ax.plot(df.step, df.val_vis2s, color=COLOR[_variant(run)],
                alpha=0.35 + 0.25 * (run.endswith("42")), lw=1,
                label=run if run.endswith("42") else None)
    ax.set_xlabel("step"); ax.set_ylabel("val 2s visual loss")
    ax.set_title("Validation 2s visual loss (checkpoint selection metric)")
    ax.legend(fontsize=8)
    f.tight_layout(); f.savefig(fig / "train_curves.png", dpi=150); plt.close(f)

    # F2: 主终点（test P1 2s spatial latent error，按场景 × run）
    g = pd.read_csv(rr / "metrics" / "visual_error_by_group_seed.csv")
    p1 = g[(g.split == "test") & (g.level == "P1")]
    f, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, lay in zip(axes, ("L4", "L5")):
        sub = p1[p1.layout == lay]
        for run in RUN_ORDER:
            v = sub[sub.run_id == run].err_2s
            if len(v):
                ax.bar(run, v.mean(), color=COLOR[_variant(run)], alpha=0.8)
                ax.errorbar([run], [v.mean()], yerr=[v.std() / max(len(v), 1) ** .5],
                            color="k", capsize=3, lw=1)
        ax.set_title(f"test layout {lay}"); ax.tick_params(axis="x", rotation=60)
    axes[0].set_ylabel("P1 2s spatial latent error (group mean)")
    f.suptitle("Primary endpoint by scene (bar=group-equal mean, err=SE over groups)")
    f.tight_layout(); f.savefig(fig / "primary_by_scene.png", dpi=150); plt.close(f)

    # F3: 候选匹配
    m = pd.read_csv(rr / "metrics" / "candidate_matching.csv")
    s = m.groupby("run_id")[["matching_acc", "paired_diff_rel_err"]].mean()
    f, ax = plt.subplots(figsize=(7, 4))
    s["matching_acc"].plot(kind="bar", ax=ax,
                           color=[COLOR[_variant(r)] for r in s.index], alpha=0.85)
    ax.axhline(1 / 3, color="k", ls="--", lw=1, label="chance=1/3")
    ax.set_ylabel("candidate-future matching accuracy")
    ax.set_title("Same-context action discrimination (test groups, matching origin)")
    ax.tick_params(axis="x", rotation=60); ax.legend()
    f.tight_layout(); f.savefig(fig / "candidate_matching.png", dpi=150); plt.close(f)

    # F4: V-EXEC 敏感性
    sp = rr / "metrics" / "exec_sensitivity.csv"
    if sp.exists():
        d = pd.read_csv(sp)
        s2 = d.groupby(["run_id", "condition"])["err_2s"].mean().unstack()
        cols = ["E_hat", "E_eq_U", "zero_E", "shuffled_E", "oracle_E_diagnostic"]
        s2 = s2[[c for c in cols if c in s2.columns]]
        f, ax = plt.subplots(figsize=(8, 4))
        s2.T.plot(kind="bar", ax=ax, alpha=0.85)
        ax.set_ylabel("2s latent error (matching windows)")
        ax.set_title("V-EXEC motion-head sensitivity (oracle = diagnostic only)")
        ax.tick_params(axis="x", rotation=30)
        f.tight_layout(); f.savefig(fig / "exec_sensitivity.png", dpi=150); plt.close(f)
    print("figures ->", fig)


if __name__ == "__main__":
    main()
