"""V0.6.2 图（figures/）—— Task 1-5 汇总。"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from execution_wm.validity_v062.common import OUT
FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 8})
S = ["A_unseen_anchor_seen_family", "B_seen_anchor_unseen_family", "C_unseen_anchor_unseen_family"]
SL = ["A: unseen anchor", "B: unseen family", "C: both unseen"]

# F2 structure vs error-level
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
st = pd.read_csv(f"{OUT}/metrics/command_space_structure.csv", index_col=0)
ax[0].bar(range(len(st)), st.frac_windows_with_multi_axis.astype(float), color="C3")
ax[0].set_xticks(range(len(st))); ax[0].set_xticklabels(st.index, rotation=30, fontsize=6)
ax[0].set_ylabel("frac windows with >=2 axes active")
ax[0].set_title("Structural novelty: multi-axis co-activation", fontsize=8); ax[0].grid(alpha=.3, axis="y")
ax[1].bar(range(len(st)), st.mahalanobis_mean.astype(float), color="C0")
ax[1].set_xticks(range(len(st))); ax[1].set_xticklabels(st.index, rotation=30, fontsize=6)
ax[1].set_ylabel("Mahalanobis (train-fitted, regularized)")
ax[1].set_title("Metric novelty", fontsize=8); ax[1].grid(alpha=.3, axis="y")
fig.tight_layout(); fig.savefig(f"{FIG}/F2_novelty_structure.png"); plt.close(fig)

# F3 composition
c = pd.read_csv(f"{OUT}/metrics/composition_baseline.csv")
fig, ax = plt.subplots(figsize=(8, 3))
for k, sp in enumerate(S):
    d = c[c.split == sp].set_index("model")
    models = ["command-copy", "ridge_multioutput", "M0", "M1", "M2", "composition_additive"]
    vals = [np.mean([d.loc[m, f"MAE_{a}"] for a in ("vx", "vy", "wz")]) for m in models]
    ax.bar(np.arange(len(models)) + k * 0.26, vals, 0.26, label=SL[k],
           color=["C0", "C1", "C2"][k])
ax.set_xticks(np.arange(6) + 0.26); ax.set_xticklabels(["ccopy", "ridge", "M0", "M1", "M2", "composit"], fontsize=7)
ax.set_ylabel("mean per-axis MAE (mixed units)"); ax.legend(fontsize=7); ax.grid(alpha=.3, axis="y")
ax.set_title("Task 2 additive primitive baseline vs models", fontsize=8)
fig.tight_layout(); fig.savefig(f"{FIG}/F3_composition.png"); plt.close(fig)

# F4 probes heatmap
p = pd.read_csv(f"{OUT}/metrics/context_probe_results.csv")
for probe, fname, ttl in (("A_friction_condition", "F4_probe_condition.png",
                           "Probe A friction condition (chance 0.333)"),
                          ("B_current_vxvywz", "F4b_probe_velocity.png",
                           "Probe B current velocity R2 (identity ref = 1.0)")):
    d = p[p.probe == probe].set_index("representation")
    cols = [c for c in ["train", "val"] + S if c in d.columns]
    v = d[cols].astype(float)
    fig, ax = plt.subplots(figsize=(6, 2.6))
    im = ax.imshow(v.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([c.split("_")[0] for c in cols], fontsize=7)
    ax.set_yticks(range(len(v))); ax.set_yticklabels(v.index, fontsize=7)
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            ax.text(j, i, f"{v.values[i, j]:.2f}", ha="center", va="center", fontsize=6, color="w")
    plt.colorbar(im); ax.set_title(ttl, fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIG}/{fname}"); plt.close(fig)

# F5 instantaneous vs trajectory
t = pd.read_csv(f"{OUT}/metrics/trajectory_consequence.csv")
t2 = t[t.horizon == "2.0s"]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
for k, sp in enumerate(S):
    d = t2[t2.split == sp].set_index("model")
    ms = ["command-copy", "ridge_multioutput", "M0", "M1", "M2"]
    ax[0].scatter([d.loc[m, "inst_MAE_wz"] for m in ms], [d.loc[m, "net_yaw_err_rad"] for m in ms],
                  s=30, label=SL[k], color=["C0", "C1", "C2"][k])
    ax[1].scatter([d.loc[m, "inst_MAE_vx"] for m in ms], [d.loc[m, "traj_dxy_err_m"] for m in ms],
                  s=30, label=SL[k], color=["C0", "C1", "C2"][k])
ax[0].set_xlabel("instantaneous MAE wz @2s"); ax[0].set_ylabel("net yaw error [rad]")
ax[1].set_xlabel("instantaneous MAE vx @2s"); ax[1].set_ylabel("trajectory dxy error [m]")
for a in ax: a.grid(alpha=.3); a.legend(fontsize=6)
ax[0].set_title("Instantaneous vs integrated yaw", fontsize=8)
ax[1].set_title("Instantaneous vs integrated position", fontsize=8)
fig.tight_layout(); fig.savefig(f"{FIG}/F5_inst_vs_traj.png"); plt.close(fig)

# F6 deployable gap
b = pd.read_csv(f"{OUT}/metrics/deployable_gap_bootstrap.csv")
fig, ax = plt.subplots(figsize=(6, 2.8))
x = np.arange(len(b))
ax.bar(x, b.gap_mean, yerr=[b.gap_mean - b.ci_low, b.ci_high - b.gap_mean], capsize=3, color="C0")
ax.axhline(0, color="k", lw=.6)
ax.set_xticks(x); ax.set_xticklabels([f"{r.split.split('_')[0]}/{r.model}" for r in b.itertuples()], fontsize=6)
ax.set_ylabel("gap = deployable - privileged (mixed MAE)")
ax.set_title("Task 5 deployable gap (all tiny; significance != practical size)", fontsize=8)
ax.grid(alpha=.3, axis="y")
fig.tight_layout(); fig.savefig(f"{FIG}/F6_deployable_gap.png"); plt.close(fig)
print("figures:", sorted(os.listdir(FIG)))
