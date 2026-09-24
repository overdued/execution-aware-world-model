"""V0.7 必需四图（PNG + PDF）:
  F1 数据覆盖图        coverage_map.{png,pdf}
  F2 2×2 数据/架构收益  data_vs_arch.{png,pdf}
  F3 真实与预测轨迹     trajectory_examples.{png,pdf}
  F4 逐组 / 逐时域误差  per_group_horizon.{png,pdf}
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from execution_wm.composition_v07 import cells as C
from execution_wm.composition_v07.data_v07 import build_datasets
from execution_wm.composition_v07.eval_v07 import CELLS, CKPT, SEEDS
from execution_wm.composition_v07.models_v07 import build
from execution_wm.composition_v07.traj import integrate_xy

OUT = "results/v0_7_composition"
FIG = f"{OUT}/figures"
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 8})


def save(fig, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{FIG}/{name}.{ext}")
    plt.close(fig)


def fig1_coverage():
    cov = pd.read_csv(f"{OUT}/manifests/command_coverage.csv")
    ds_names = ["train_R0", "train_R1", "val", "test_all"]
    cells_all = sorted(cov.cell_id.unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    # 左：cell 覆盖率（0/1 矩阵）
    M = np.zeros((len(ds_names), len(cells_all)))
    for i, d in enumerate(ds_names):
        s = cov[cov.dataset == d]
        for _, r in s.iterrows():
            M[i, cells_all.index(r.cell_id)] = r.n_windows
    ax = axes[0]
    im = ax.imshow(np.log1p(M), aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(ds_names))); ax.set_yticklabels(ds_names, fontsize=7)
    ax.set_xlabel(f"command cell ({len(cells_all)} with windows)")
    ax.set_title("Coverage: log1p(#windows) per cell", fontsize=8)
    plt.colorbar(im, ax=ax, shrink=.8)
    # 右：每轴 duty + 共激活（R0 vs R1）
    rr = json.load(open(f"{OUT}/prereg/r0_vs_r1_marginals.json"))
    x = np.arange(4); w = .35
    labels = ["duty vx", "duty vy", "duty wz", "co-activation"]
    v0 = [rr["R0"]["axis_duty_vx"], rr["R0"]["axis_duty_vy"], rr["R0"]["axis_duty_wz"],
          rr["R0"]["coactivation_slot_frac"]]
    v1 = [rr["R1"]["axis_duty_vx"], rr["R1"]["axis_duty_vy"], rr["R1"]["axis_duty_wz"],
          rr["R1"]["coactivation_slot_frac"]]
    ax = axes[1]
    ax.bar(x - w / 2, v0, w, label="R0", color="C0")
    ax.bar(x + w / 2, v1, w, label="R1", color="C3")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("fraction"); ax.legend(fontsize=7); ax.grid(alpha=.3, axis="y")
    ax.set_title("R0 vs R1 marginals: duty matched, co-activation differs by design",
                 fontsize=8)
    save(fig, "coverage_map")


def fig2_data_vs_arch():
    e = pd.read_csv(f"{OUT}/metrics/data_vs_arch_effect.csv")
    e = e[e.metric == "FDE_xy_2s_m"]
    splits = ["test_P0", "test_P1", "test_P2"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=False)
    for ax, col, ttl in ((axes[0], "Delta_data", "Delta_data = E(D,R0) - E(D,R1)"),
                         (axes[1], "Delta_arch", "Delta_arch = E(D,R1) - E(I,R1)")):
        x = np.arange(len(splits)); w = .25
        for j, seed in enumerate(SEEDS):
            s = e[(e.seed == seed)].set_index("split")
            vals = [s.loc[k, col] if k in s.index else np.nan for k in splits]
            ax.bar(x + (j - 1) * w, vals, w, label=f"seed {seed}")
        lo = [e[(e.split == k)][f"{col}_ci_low"].mean() for k in splits]
        hi = [e[(e.split == k)][f"{col}_ci_high"].mean() for k in splits]
        mean = [e[(e.split == k)][col].mean() for k in splits]
        ax.errorbar(x, mean, yerr=[np.array(mean) - np.array(lo), np.array(hi) - np.array(mean)],
                    fmt="k_", capsize=4, label="mean +- group CI")
        ax.axhline(0, color="k", lw=.6)
        ax.set_xticks(x); ax.set_xticklabels(splits, fontsize=7)
        ax.set_ylabel("Δ FDE_xy (m)   (>0 = 前者更差)")
        ax.set_title(ttl, fontsize=8); ax.legend(fontsize=6); ax.grid(alpha=.3, axis="y")
    save(fig, "data_vs_arch")


def fig3_trajectories(n_examples=6):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = build_datasets("/media/hdd1/yuhang/datasets/execution_wm/v0_7")
    d = ds["test_P1"]
    if len(d) == 0:
        d = ds["test_all"]
    hp, ha, fa, fr, conds, eps, groups, levels = d.batch(np.arange(len(d)), device)
    gt, yaw0, yawf = d.fe, d.yaw0, d.yawf
    variants = {"command-copy": fa.cpu().numpy()}
    for (mn, rg) in CELLS:
        ps = []
        for seed in SEEDS:
            p = os.path.join(CKPT, f"{mn}_{rg}_s{seed}", "best.pt")
            if not os.path.exists(p):
                continue
            ck = torch.load(p, weights_only=False, map_location=device)
            m = build(mn).to(device); m.load_state_dict(ck["model_state"]); m.eval()
            with torch.no_grad():
                ps.append((fa + m(hp, ha, fa)).cpu().numpy())
        if ps:
            variants[f"{mn}/{rg}"] = np.mean(ps, axis=0)
    idx = np.linspace(0, len(d) - 1, n_examples).astype(int)
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for ax, i in zip(axes.ravel(), idx):
        t_xy = integrate_xy(gt[i, :, :2], yaw0[i], yaw_seq=yawf[i, :, 0])
        ax.plot(t_xy[:, 0], t_xy[:, 1], "k-", lw=2, label="truth")
        for name, pr in variants.items():
            p_xy = integrate_xy(pr[i, :, :2], yaw0[i], wz=pr[i, :, 2])
            ax.plot(p_xy[:, 0], p_xy[:, 1], lw=1.1, label=name,
                    ls="--" if name == "command-copy" else "-")
        ax.scatter([0], [0], c="k", s=12, zorder=5)
        ax.set_title(f"ep{eps[i]} {conds[i]} {levels[i]}", fontsize=7)
        ax.grid(alpha=.3); ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x local [m]"); ax.set_ylabel("y local [m]")
    axes[0, 0].legend(fontsize=6)
    save(fig, "trajectory_examples")


def fig4_per_group_horizon():
    pa = pd.read_csv(f"{OUT}/metrics/per_anchor_seed.csv")
    t = pd.read_csv(f"{OUT}/metrics/trajectory_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    # 左：逐 group Δ_arch
    p = pa[pa.metric == "FDE_xy_2s_m"]
    for split, grp in p.groupby("split"):
        g = grp.groupby("group_id")["Delta_arch"].mean()
        axes[0].scatter([split] * len(g), g.values, alpha=.8)
    axes[0].axhline(0, color="k", lw=.6)
    axes[0].set_ylabel("Δ_arch FDE per group (m)")
    axes[0].set_title("Per-group Δ_arch (each dot = one independent test group)", fontsize=8)
    axes[0].grid(alpha=.3, axis="y")
    # 右：逐时域误差
    for key, ls in (("command-copy", "--"), ("D_R1_seedmean", "-"), ("I_R1_seedmean", "-.")):
        s = t[(t.split == "test_P1") & (t.variant == key)]
        if len(s) == 0:
            continue
        leads = ["0.25s", "0.5s", "1.0s", "2.0s"]
        vals = [s.iloc[0][f"lead_MAE_vx@{l}"] for l in leads]
        axes[1].plot(range(4), vals, ls, marker="o", ms=3, label=key)
    axes[1].set_xticks(range(4)); axes[1].set_xticklabels(["0.25", "0.5", "1.0", "2.0"])
    axes[1].set_xlabel("lead [s]"); axes[1].set_ylabel("MAE vx (m/s)")
    axes[1].set_title("Per-horizon error on P1 (held-out pair cells)", fontsize=8)
    axes[1].legend(fontsize=7); axes[1].grid(alpha=.3)
    save(fig, "per_group_horizon")


if __name__ == "__main__":
    fig1_coverage()
    fig2_data_vs_arch()
    fig3_trajectories()
    fig4_per_group_horizon()
    print("figures:", sorted(os.listdir(FIG)))
