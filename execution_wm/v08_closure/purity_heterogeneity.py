"""V0.8 Stage A3：高/低纯度窗口的效应异质性（只读 pred_cache，不重推理）。

对每个 group、每个 seed：
  d_high = mean(FDE_D_R0 - FDE_D_R1 | purity >= 0.8)
  d_low  = mean(FDE_D_R0 - FDE_D_R1 | purity <  0.8)
  d_het  = d_low - d_high
并逐组报告子集样本量、cell 构成、命令幅值、转向占比、当前速度分布；
d_het 的区间 = 跨 6 group 配对 bootstrap。d_het > 0 只说明**该划分下**的效应异质性，
不是因果证明；子集缺失的 group 显式记录为 NaN，不填 0。

运行：python -m execution_wm.v08_closure.purity_heterogeneity
输出：metrics/v07_purity_heterogeneity.csv（逐 group×seed）+ 汇总打印
"""
import os
import sys

import numpy as np
import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ)
from execution_wm.composition_v07.data_v07 import build_datasets  # noqa: E402
from execution_wm.composition_v07.traj import integrate_xy  # noqa: E402

V07 = os.environ.get("V07_RESULTS", "/home/yuhang/cvpr_embed-v07/results/v0_7_composition")
DATA = os.environ.get("V07_DATA", "/media/hdd1/yuhang/datasets/execution_wm/v0_7")
OUT = os.environ.get("V08_OUT", "results/v0_8_visual_pilot")
SPLIT = "test_P1"
SEEDS = (42, 43, 44)
PURITY_TH = 0.8
N_BOOT = 2000


def per_window_fde(pred, gt, yaw0, yawf):
    n = len(pred)
    pred_xy = np.stack([integrate_xy(pred[i, :, :2], yaw0[i], wz=pred[i, :, 2])
                        for i in range(n)])
    true_xy = np.stack([integrate_xy(gt[i, :, :2], yaw0[i], yaw_seq=yawf[i])
                        for i in range(n)])
    return np.linalg.norm(pred_xy[:, -1] - true_xy[:, -1], axis=1)


def main():
    cache = np.load(f"{V07}/predictions/pred_cache.npz")
    dset = build_datasets(DATA)[SPLIT]
    gt, yaw0, yawf = dset.fe, dset.yaw0, dset.yawf
    W = dset.windows
    groups = np.array([w["group_id"] for w in W])
    purity = np.array([w["future_cell_purity"] for w in W])
    ug = np.unique(groups)
    hi = purity >= PURITY_TH

    fde = {}
    for rg in ("R0", "R1"):
        for s in SEEDS:
            fde[(rg, s)] = per_window_fde(cache[f"{SPLIT}__D_{rg}_s{s}"], gt, yaw0, yawf)

    rows = []
    for s in SEEDS:
        diff = fde[("R0", s)] - fde[("R1", s)]          # 逐窗配对差（同窗口两模型）
        for g in ug:
            m_g = groups == g
            out = {"seed": s, "group_id": g}
            for name, msk in (("high", m_g & hi), ("low", m_g & ~hi)):
                idx = np.where(msk)[0]
                out[f"n_{name}"] = len(idx)
                if len(idx):
                    out[f"d_{name}"] = float(diff[idx].mean())
                    cells = [W[i]["dominant_cell"] for i in idx]
                    out[f"cells_{name}"] = "|".join(sorted(set(cells)))
                    fa = dset.fa[idx]                     # 未来命令 [n,H,3]
                    amp = np.abs(fa).max(axis=(1, 2))
                    out[f"cmd_amp_max_{name}"] = float(amp.mean())
                    out[f"turn_frac_{name}"] = float((np.abs(fa[:, :, 2]) > 0).any(axis=1).mean())
                    cur = np.linalg.norm(dset.hp[idx, -1, :][:, [0, 1]], axis=1)  # schema vx,vy
                    out[f"cur_speed_mean_{name}"] = float(cur.mean())
                else:
                    out[f"d_{name}"] = np.nan
                    out[f"cells_{name}"] = "MISSING_SUBSET"
                    out[f"cmd_amp_max_{name}"] = np.nan
                    out[f"turn_frac_{name}"] = np.nan
                    out[f"cur_speed_mean_{name}"] = np.nan
            out["d_het"] = out["d_low"] - out["d_high"] if not (
                np.isnan(out.get("d_low", np.nan)) or np.isnan(out.get("d_high", np.nan))) else np.nan
            rows.append(out)
    df = pd.DataFrame(rows)

    # d_het 的 group bootstrap（逐 seed；跳过含 NaN 的 seed 时如实记录）
    rng = np.random.default_rng(20260927)
    summ = []
    for s in SEEDS:
        sub = df[df.seed == s]
        dhet = sub["d_het"].to_numpy()
        ok = ~np.isnan(dhet)
        rec = {"seed": s, "n_groups_total": len(sub), "n_groups_with_both_subsets": int(ok.sum()),
               "groups_missing_subset": "|".join(sub.group_id[~ok]) or "none"}
        if ok.sum() >= 2:
            v = dhet[ok]
            boot = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(N_BOOT)])
            rec.update(d_het_mean=float(v.mean()),
                       d_het_ci_low=float(np.percentile(boot, 2.5)),
                       d_het_ci_high=float(np.percentile(boot, 97.5)),
                       d_het_ci_crosses_zero=bool(np.percentile(boot, 2.5) < 0
                                                  < np.percentile(boot, 97.5)))
        else:
            rec.update(d_het_mean=np.nan, d_het_ci_low=np.nan, d_het_ci_high=np.nan,
                       d_het_ci_crosses_zero=None)
        summ.append(rec)
    sm = pd.DataFrame(summ)

    os.makedirs(f"{OUT}/metrics", exist_ok=True)
    df.to_csv(f"{OUT}/metrics/v07_purity_heterogeneity.csv", index=False)
    sm.to_csv(f"{OUT}/metrics/v07_purity_heterogeneity_summary.csv", index=False)
    pd.set_option("display.width", 240)
    print(f"test_P1: n={len(dset)}, purity>=0.8: {int(hi.sum())} ({hi.mean():.1%})")
    print(sm.round(4).to_string(index=False))
    print()
    print(df.round(4)[["seed", "group_id", "n_high", "n_low", "d_high", "d_low", "d_het",
                       "turn_frac_high", "turn_frac_low"]].to_string(index=False))


if __name__ == "__main__":
    main()
