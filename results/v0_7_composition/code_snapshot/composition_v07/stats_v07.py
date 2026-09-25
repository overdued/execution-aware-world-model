"""V0.7 Stage 4b：统计 —— Δ_data / Δ_arch（group 级配对 bootstrap）+ 主指标表。

统计单元 = 独立 anchor/session group：先 group 内平均，再跨 group 配对 bootstrap；
3 个模型 seed 分别报告，不伪装成额外物理环境。
输出: metrics/data_vs_arch_effect.csv, main_axis_lead.csv, per_anchor_seed.csv,
      termination_and_masks.csv, command_coverage.csv
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd
import torch

from execution_wm.composition_v07.data_v07 import build_datasets
from execution_wm.composition_v07.eval_v07 import summarize
from execution_wm.composition_v07.models_v07 import build

OUT = os.environ.get("V07_OUT", "results/v0_7_composition")
CKPT = os.environ.get("V07_CKPT", "/media/hdd1/yuhang/checkpoints/execution_wm/v0_7")
CELLS = [("D", "R0"), ("D", "R1"), ("I", "R0"), ("I", "R1")]
SEEDS = (42, 43, 44)
N_BOOT = 2000
METRICS = ["FDE_xy_2s_m", "ADE_xy_2s_m", "net_yaw_err_rad"]


def group_errors(pred, gt, groups, yaw0, yawf, n=40):
    """每窗 FDE（m）与逐轴 horizon MAE，按 group 聚合。"""
    T = pred.shape[1]
    from execution_wm.composition_v07.traj import integrate_xy, wrap_pi
    nn = len(pred)
    pred_xy = np.stack([integrate_xy(pred[i, :, :2], yaw0[i], wz=pred[i, :, 2])
                        for i in range(nn)])
    true_xy = np.stack([integrate_xy(gt[i, :, :2], yaw0[i], yaw_seq=yawf[i])
                        for i in range(nn)])
    fde = np.linalg.norm(pred_xy[:, -1] - true_xy[:, -1], axis=1)
    ade = np.linalg.norm(pred_xy - true_xy, axis=2).mean(axis=1)
    dyaw_pred = np.cumsum(pred[:, :, 2], axis=1)[:, -1] * 0.05
    dyaw_true = wrap_pi(yawf[:, -1] - yaw0)
    yaw_err = np.abs(wrap_pi(dyaw_pred - dyaw_true))
    per_axis = {ax: np.abs(pred[:, -1, i] - gt[:, -1, i])
                for i, ax in enumerate(("vx", "vy", "wz"))}
    g = np.array(groups)
    ug = np.unique(g)
    raw = {"FDE_xy_2s_m": fde, "ADE_xy_2s_m": ade, "net_yaw_err_rad": yaw_err}
    for ax, v in per_axis.items():
        raw[f"lead_MAE_{ax}@2.0s"] = v
    return ug, {m: np.array([v[g == k].mean() for k in ug]) for m, v in raw.items()}


def paired_boot(d, n_boot=N_BOOT, rng=None):
    rng = rng or np.random.default_rng(20260925)
    n = len(d)
    out = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = build_datasets(os.environ.get("V07_DATA", "/media/hdd1/yuhang/datasets/execution_wm/v0_7"))
    splits = ["test_all", "test_P0", "test_P1", "test_P2"]
    preds = {}
    rng = np.random.default_rng(20260925)
    for split in splits:
        dset = ds[split]
        if len(dset) == 0:
            continue
        n = len(dset)
        hp, ha, fa, fr, conds, eps, groups, levels = dset.batch(np.arange(n), device)
        gt, yaw0, yawf = dset.fe, dset.yaw0, dset.yawf
        for (mname, regime) in CELLS:
            for seed in SEEDS:
                p = os.path.join(CKPT, f"{mname}_{regime}_s{seed}", "best.pt")
                if not os.path.exists(p):
                    continue
                ck = torch.load(p, weights_only=False, map_location=device)
                m = build(mname).to(device); m.load_state_dict(ck["model_state"]); m.eval()
                with torch.no_grad():
                    pr = (fa + m(hp, ha, fa)).cpu().numpy()
                ug, agg = group_errors(pr, gt, groups, yaw0, yawf)
                preds[(split, mname, regime, seed)] = {"groups": ug, "agg": agg,
                                                       "cond": np.array(conds),
                                                       "level": np.array(levels)}
        print(f"[stats] {split}: {n} windows, {len(np.unique(groups))} groups")

    rows, per_anchor = [], []
    for split in splits:
        if not any(k[0] == split for k in preds):
            continue
        groups = None
        for (s, mname, regime, seed), v in preds.items():
            if s == split:
                groups = v["groups"]; break
        for metric in METRICS + ["lead_MAE_vx@2.0s", "lead_MAE_vy@2.0s", "lead_MAE_wz@2.0s"]:
            for seed in SEEDS:
                def gv(mn, rg):
                    k = (split, mn, rg, seed)
                    return preds[k]["agg"][metric] if k in preds else None
                d_r0 = gv("D", "R0"); d_r1 = gv("D", "R1")
                i_r1 = gv("I", "R1"); i_r0 = gv("I", "R0")
                if d_r0 is None or d_r1 is None or i_r1 is None:
                    continue
                d_data = d_r0 - d_r1
                d_arch = d_r1 - i_r1
                lo_d, hi_d = paired_boot(d_data, rng=rng)
                lo_a, hi_a = paired_boot(d_arch, rng=rng)
                rows.append({
                    "split": split, "metric": metric, "seed": seed,
                    "n_groups": len(groups),
                    "E_D_R0": float(d_r0.mean()), "E_D_R1": float(d_r1.mean()),
                    "E_I_R1": float(i_r1.mean()),
                    "E_I_R0": float(i_r0.mean()) if i_r0 is not None else np.nan,
                    "Delta_data": float(d_data.mean()),
                    "Delta_data_ci_low": lo_d, "Delta_data_ci_high": hi_d,
                    "Delta_data_crosses_zero": bool(lo_d < 0 < hi_d),
                    "Delta_data_rel_pct": float(100 * d_data.mean() / d_r0.mean()),
                    "Delta_arch": float(d_arch.mean()),
                    "Delta_arch_ci_low": lo_a, "Delta_arch_ci_high": hi_a,
                    "Delta_arch_crosses_zero": bool(lo_a < 0 < hi_a),
                    "Delta_arch_rel_pct": float(100 * d_arch.mean() / d_r1.mean()),
                })
                for gi, gname in enumerate(groups):
                    per_anchor.append({"split": split, "metric": metric, "seed": seed,
                                       "group_id": gname,
                                       "E_D_R0": float(d_r0[gi]), "E_D_R1": float(d_r1[gi]),
                                       "E_I_R1": float(i_r1[gi]),
                                       "Delta_data": float(d_data[gi]),
                                       "Delta_arch": float(d_arch[gi])})
    os.makedirs(f"{OUT}/metrics", exist_ok=True)
    e = pd.DataFrame(rows)
    e.to_csv(f"{OUT}/metrics/data_vs_arch_effect.csv", index=False)
    pd.DataFrame(per_anchor).to_csv(f"{OUT}/metrics/per_anchor_seed.csv", index=False)
    pd.set_option("display.width", 240)
    print("\n=== Δ_data / Δ_arch（按 seed，group 级配对 bootstrap）===")
    print(e[e.metric.isin(["FDE_xy_2s_m", "net_yaw_err_rad"])]
          [["split", "metric", "seed", "n_groups", "E_D_R0", "E_D_R1", "E_I_R1",
            "Delta_data", "Delta_data_ci_low", "Delta_data_ci_high",
            "Delta_arch", "Delta_arch_ci_low", "Delta_arch_ci_high"]]
          .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
