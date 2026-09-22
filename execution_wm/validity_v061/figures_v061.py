"""V0.6.1 图（figures/）：修复前后对照 + 逐轴/逐 anchor 误差 + 时间线。"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "results/v0_6_1_correctness"
FIG = os.path.join(OUT, "figures")
SQ = "/media/hdd1/yuhang/datasets/execution_wm/v0_6_sq"
DER = "/media/hdd1/yuhang/datasets/execution_wm/v0_6_1"
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 8})


def fig1_label_fix():
    """修复前后：ep_00162 t=2.5s 附近 wz 的三种量。"""
    old = np.load(f"{SQ}/friction_low/ep_00162.20hz.npz")
    new = np.load(f"{DER}/friction_low/ep_00162.20hz.npz")
    t = new["timestamp"]
    fig, ax = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    ax[0].plot(t, old["cmd_vel"][:, 2], "k--", lw=1, label="cmd (OLD, float hold)")
    ax[0].plot(t, new["cmd_ref"][:, 2], "k-", lw=1.6, label="cmd_ref (NEW, integer tick)")
    ax[0].set_ylabel("wz cmd [rad/s]"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
    ax[0].axvline(2.5, color="r", ls=":", lw=1)
    ax[1].plot(t, old["cmd_vel"][:, 2] + old["lb_residual"][:, 2], "C3-",
               label="OLD eval truth = cmd+lb_residual")
    ax[1].plot(t, new["lb_execution"][:, 2], "C0-", lw=1.6,
               label="NEW truth lb_execution = F(e)")
    ax[1].plot(t, old["lb_execution"][:, 2], "C3:", lw=1, label="F(e) stored in OLD file")
    ax[1].axvline(2.5, color="r", ls=":", lw=1)
    ax[1].annotate("diff 1.120 rad/s", xy=(2.5, 1.885), xytext=(2.75, 1.5),
                   arrowprops=dict(arrowstyle="->", lw=.8), fontsize=7)
    ax[1].set_xlabel("t [s]"); ax[1].set_ylabel("wz [rad/s]")
    ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
    fig.suptitle("P0-2 label identity: OLD eval truth != stored resampled physical execution (ep_00162)")
    fig.tight_layout(); fig.savefig(f"{FIG}/F1_label_identity_fix.png"); plt.close(fig)


def fig2_timeline():
    """policy 消费命令 vs 请求命令（Q3_turn）。"""
    d = json.load(open(f"{OUT}/audit/policy_command_timeline.json"))
    log = d["waves"]["Q3_turn"]
    # 使用修正后记录：policy_obs_cmd = 消费时刻的 command slice
    u = np.array(log["u_requested"]); po = np.array(log["policy_obs_cmd"])
    t = np.array(log["t"])
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.step(t, u[:, 2], "k-", where="post", lw=1.4, label="u_requested (written)")
    ax.step(t, po[:, 2], "C0--", where="post", lw=1.4, label="u_consumed (actually read by policy)")
    ax.set_xlabel("t [s]"); ax.set_ylabel("wz cmd [rad/s]")
    ax.legend(fontsize=7); ax.grid(alpha=.3)
    ax.set_title("S3 runtime evidence: consumption lags exactly 1 control tick (0.02 s), 199/199 ticks")
    fig.tight_layout(); fig.savefig(f"{FIG}/F2_command_pipeline_latency.png"); plt.close(fig)


def fig3_axis_lead():
    m = pd.read_csv(f"{OUT}/metrics/per_axis_lead_ranking.csv")
    splits = m.split.unique()
    fig, axes = plt.subplots(1, len(splits), figsize=(12, 3.2), sharey=False)
    for ax, sp in zip(axes, splits):
        s = m[m.split == sp]
        labels, vals = [], []
        for lead in ["0.25s", "1.0s", "2.0s"]:
            for axis in ["vx", "vy", "wz"]:
                r = s[(s.lead == lead) & (s.axis == axis)].iloc[0]
                labels.append(f"{axis}@{lead}")
                vals.append([r["M0"], r["M1"], r["M2"], r["M3"], r["ccopy"]])
        vals = np.array(vals)
        x = np.arange(len(labels)); w = 0.16
        for i, (nm, c) in enumerate(zip(["M0", "M1", "M2", "M3", "ccopy"],
                                        ["C0", "C1", "C2", "C3", "k"])):
            ax.bar(x + (i - 2) * w, vals[:, i], w, label=nm, color=c, alpha=.85)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=60, fontsize=6)
        ax.set_title(sp.replace("_", "\n"), fontsize=7)
        ax.grid(alpha=.3, axis="y")
    axes[0].set_ylabel("fixed-lead MAE (m/s or rad/s)")
    axes[0].legend(fontsize=6, ncol=2)
    fig.suptitle("V0.6.1 per-axis fixed-lead MAE (3-seed mean) -- no model wins all cells")
    fig.tight_layout(); fig.savefig(f"{FIG}/F3_per_axis_lead_mae.png"); plt.close(fig)


def fig4_anchor():
    a = pd.read_csv(f"{OUT}/metrics/per_anchor_group.csv")
    a = a[a.variant.str.match(r"M[0-3]_s4[234]$|command-copy")]
    a["fam"] = np.where(a.variant.str.startswith("M"), a.variant.str[:2], a.variant)
    p = a.groupby(["split", "anchor_group", "fam"])["mean_err"].mean().unstack()
    fig, ax = plt.subplots(figsize=(8, 3.2))
    p.plot(kind="bar", ax=ax, width=.8)
    ax.set_ylabel("mean |err| (mixed units, descriptive only)"); ax.grid(alpha=.3, axis="y")
    ax.set_title("Error by anchor group (held-anchor = 2 groups only -> exploratory)")
    ax.legend(fontsize=6, ncol=5); plt.setp(ax.get_xticklabels(), rotation=30, fontsize=6)
    fig.tight_layout(); fig.savefig(f"{FIG}/F4_error_by_anchor_group.png"); plt.close(fig)


def fig5_label_delta():
    df = pd.read_csv(f"{OUT}/derived_data_summary/label_identity_per_episode.csv")
    fig, ax = plt.subplots(1, 2, figsize=(8, 2.8))
    ax[0].hist(df.max_abs_label_diff, bins=40, color="C3")
    ax[0].set_xlabel("max |OLD - NEW truth| [mixed units]"); ax[0].set_ylabel("#episodes")
    ax[0].set_title("Per-episode max label change", fontsize=8); ax[0].grid(alpha=.3)
    ax[1].hist(df.old_hold_vs_new_mismatch, bins=40, color="C0")
    ax[1].set_xlabel("grid points where OLD float hold != NEW integer tick")
    ax[1].set_title("B3 time-hold mismatch per episode", fontsize=8); ax[1].grid(alpha=.3)
    fig.suptitle("P0-2 / P1-1 impact distribution (240 episodes, all retained)")
    fig.tight_layout(); fig.savefig(f"{FIG}/F5_label_delta_distribution.png"); plt.close(fig)


def fig6_scalar():
    t = pd.read_csv(f"{OUT}/metrics/train_normalized_scalar.csv")
    p = t.pivot(index="variant", columns="split", values="normalized_scalar")
    order = [f"M{m}_s{s}" for m in range(4) for s in (42, 43, 44)] + \
            ["command-copy", "persistence", "ridge_multioutput"]
    p = p.reindex([o for o in order if o in p.index])
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    im = ax.imshow(p.values, cmap="RdYlGn_r", aspect="auto", vmin=0.6, vmax=2.2)
    ax.set_xticks(range(len(p.columns)))
    ax.set_xticklabels([c.split("_")[0] for c in p.columns])
    ax.set_yticks(range(len(p.index))); ax.set_yticklabels(p.index, fontsize=6)
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            ax.text(j, i, f"{p.values[i, j]:.2f}", ha="center", va="center", fontsize=6)
    plt.colorbar(im, label="train-only normalized scalar (<1 beats command-copy)")
    ax.set_title("V0.6.1 normalized scalar: A clearly beats ccopy; B/C comparable or worse", fontsize=7)
    fig.tight_layout(); fig.savefig(f"{FIG}/F6_normalized_scalar.png"); plt.close(fig)


if __name__ == "__main__":
    fig1_label_fix(); fig2_timeline(); fig3_axis_lead(); fig4_anchor()
    fig5_label_delta(); fig6_scalar()
    print("figures ->", sorted(os.listdir(FIG)))
