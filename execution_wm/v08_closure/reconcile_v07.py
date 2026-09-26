"""V0.8 Stage A1：V0.7 报告 §4.1 vs §4.4 的 P1 数值对账（只读旧缓存，不重推理）。

两个来源：
  §4.1  stats_v07.py  -> data_vs_arch_effect.csv：每 seed、group-equal（组内窗口均值的组间均值）
  §4.4  eval_v07.py   -> trajectory_metrics.csv 的 *_seedmean：3 seed 预测算术平均后、window-equal

本脚本从同一个 pred_cache.npz 独立复算 2x2 分解：
  {window-equal, group-equal} x {每 seed 分别算 FDE, 三 seed 预测先平均再算 FDE}
并验证与两处报告值的精确一致性，输出 metrics/v07_table_reconciliation.csv。

运行（v08 worktree 根目录，conda isaaclab）：
  python -m execution_wm.v08_closure.reconcile_v07
"""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ)
from execution_wm.composition_v07.data_v07 import build_datasets  # noqa: E402
from execution_wm.composition_v07.traj import integrate_xy, wrap_pi  # noqa: E402

V07 = os.environ.get("V07_RESULTS", "/home/yuhang/cvpr_embed-v07/results/v0_7_composition")
DATA = os.environ.get("V07_DATA", "/media/hdd1/yuhang/datasets/execution_wm/v0_7")
OUT = os.environ.get("V08_OUT", "results/v0_8_visual_pilot")
SPLIT = "test_P1"
CELLS = [("D", "R0"), ("D", "R1"), ("I", "R0"), ("I", "R1")]
SEEDS = (42, 43, 44)


def per_window_fde(pred, gt, yaw0, yawf):
    """与 V0.7 完全同口径：中点积分，预测侧 wz 自回归，真值侧真 yaw 序列。"""
    n = len(pred)
    pred_xy = np.stack([integrate_xy(pred[i, :, :2], yaw0[i], wz=pred[i, :, 2])
                        for i in range(n)])
    true_xy = np.stack([integrate_xy(gt[i, :, :2], yaw0[i], yaw_seq=yawf[i])
                        for i in range(n)])
    return np.linalg.norm(pred_xy[:, -1] - true_xy[:, -1], axis=1)


def main():
    cache_p = f"{V07}/predictions/pred_cache.npz"
    sha = hashlib.sha256(open(cache_p, "rb").read()).hexdigest()
    cache = np.load(cache_p)
    ds = build_datasets(DATA)
    dset = ds[SPLIT]
    gt, yaw0, yawf = dset.fe, dset.yaw0, dset.yawf
    groups = np.array([w["group_id"] for w in dset.windows])
    ug = np.unique(groups)
    win_ids = [f"{w['episode_id']}@{w['origin_grid']}" for w in dset.windows]

    # 旧报告的两处数值（从旧 CSV 读，不手抄）
    old_stats = pd.read_csv(f"{V07}/metrics/data_vs_arch_effect.csv")
    old_traj = pd.read_csv(f"{V07}/metrics/trajectory_metrics.csv")

    rows = []
    for mname, regime in CELLS:
        cell = f"{mname}/{regime}"
        fde_seed = {}
        for s in SEEDS:
            pr = cache[f"{SPLIT}__{mname}_{regime}_s{s}"]
            fde_seed[s] = per_window_fde(pr, gt, yaw0, yawf)
        pr_mean = np.mean([cache[f"{SPLIT}__{mname}_{regime}_s{s}"] for s in SEEDS], axis=0)
        fde_ens = per_window_fde(pr_mean, gt, yaw0, yawf)

        # 口径 a：window-equal，每 seed
        for s in SEEDS:
            rows.append({"cell": cell, "seed": s, "aggregation": "window-equal",
                         "pred_pooling": "per-seed", "FDE_xy_2s_m": fde_seed[s].mean()})
        # 口径 b：group-equal，每 seed（= §4.1 stats 表口径）
        for s in SEEDS:
            gm = np.array([fde_seed[s][groups == g].mean() for g in ug])
            rows.append({"cell": cell, "seed": s, "aggregation": "group-equal",
                         "pred_pooling": "per-seed", "FDE_xy_2s_m": gm.mean()})
        # 口径 c：window-equal，seed 预测先平均（= §4.4 trajectory_metrics 的 seedmean 行）
        rows.append({"cell": cell, "seed": "seedmean", "aggregation": "window-equal",
                     "pred_pooling": "mean-of-predictions", "FDE_xy_2s_m": fde_ens.mean()})
        # 口径 d：group-equal，seed 预测先平均
        gm = np.array([fde_ens[groups == g].mean() for g in ug])
        rows.append({"cell": cell, "seed": "seedmean", "aggregation": "group-equal",
                     "pred_pooling": "mean-of-predictions", "FDE_xy_2s_m": gm.mean()})

    df = pd.DataFrame(rows)

    # ---- 与旧报告两处数值核对 ----
    checks = []
    for mname, regime in CELLS:
        for s in SEEDS:
            ref = old_stats[(old_stats.split == SPLIT) & (old_stats.metric == "FDE_xy_2s_m")
                            & (old_stats.seed == s)][f"E_{mname}_{regime}"].iloc[0]
            got = df[(df.cell == f"{mname}/{regime}") & (df.seed == s)
                     & (df.aggregation == "group-equal")
                     & (df.pred_pooling == "per-seed")]["FDE_xy_2s_m"].iloc[0]
            checks.append({"source": "§4.1 data_vs_arch_effect.csv", "cell": f"{mname}/{regime}",
                           "seed": s, "reported": ref, "recomputed": got,
                           "max_abs_diff": abs(ref - got)})
        ref = old_traj[(old_traj.split == SPLIT)
                       & (old_traj.variant == f"{mname}_{regime}_seedmean")]["FDE_xy_2s_m"].iloc[0]
        got = df[(df.cell == f"{mname}/{regime}") & (df.seed == "seedmean")
                 & (df.aggregation == "window-equal")
                 & (df.pred_pooling == "mean-of-predictions")]["FDE_xy_2s_m"].iloc[0]
        checks.append({"source": "§4.4 trajectory_metrics.csv seedmean", "cell": f"{mname}/{regime}",
                       "seed": "seedmean", "reported": ref, "recomputed": got,
                       "max_abs_diff": abs(ref - got)})
    ck = pd.DataFrame(checks)

    # ---- 分解：差异 = 集成效应 + 聚合口径效应（以 D/R1 为例逐 seed 记录）----
    decomp = []
    for mname, regime in CELLS:
        cell = f"{mname}/{regime}"
        perseed_we = df[(df.cell == cell) & (df.aggregation == "window-equal")
                        & (df.pred_pooling == "per-seed")]["FDE_xy_2s_m"].mean()
        perseed_ge = df[(df.cell == cell) & (df.aggregation == "group-equal")
                        & (df.pred_pooling == "per-seed")]["FDE_xy_2s_m"].mean()
        ens_we = df[(df.cell == cell) & (df.seed == "seedmean")
                    & (df.aggregation == "window-equal")]["FDE_xy_2s_m"].iloc[0]
        ens_ge = df[(df.cell == cell) & (df.seed == "seedmean")
                    & (df.aggregation == "group-equal")]["FDE_xy_2s_m"].iloc[0]
        decomp.append({"cell": cell,
                       "perseed_group_equal_mean(§4.1口径)": perseed_ge,
                       "seedmeanpred_window_equal(§4.4口径)": ens_we,
                       "perseed_window_equal_mean": perseed_we,
                       "seedmeanpred_group_equal": ens_ge,
                       "delta_ensemble(window-equal)": perseed_we - ens_we,
                       "delta_group_vs_window(per-seed)": perseed_ge - perseed_we,
                       "total_gap": perseed_ge - ens_we})
    dc = pd.DataFrame(decomp)

    os.makedirs(f"{OUT}/metrics", exist_ok=True)
    df.to_csv(f"{OUT}/metrics/v07_table_reconciliation.csv", index=False)
    ck.to_csv(f"{OUT}/metrics/v07_table_reconciliation_checks.csv", index=False)
    dc.to_csv(f"{OUT}/metrics/v07_table_reconciliation_decomp.csv", index=False)
    meta = {
        "metric": "FDE_xy_2s_m（中点积分、预测侧 wz 自回归、终点 t=+2.0s；非 ADE/prefix/fixed-lead 速度）",
        "population": f"{SPLIT} 全部 432 窗（非高纯度子集）",
        "window_id_rule": "episode_id@origin_grid（确定性 6 origins/episode，无 RNG）",
        "n_windows": len(dset), "n_groups": int(len(ug)), "groups": list(ug),
        "aggregation_axes": ["window-equal", "group-equal"],
        "pred_pooling_axes": ["per-seed", "mean-of-predictions(3 seeds)"],
        "termination": "test_P1 432 窗全部 full-H 有效（termination_and_masks.csv），无截断终点",
        "checkpoints": "/media/hdd1/yuhang/checkpoints/execution_wm/v0_7/<M>_<R>_s<seed>/best.pt（val FDE_xy 选择）",
        "data_index_sha256": hashlib.sha256(open(f"{DATA}/index.json", "rb").read()).hexdigest(),
        "pred_cache_sha256": sha,
        "note": "pred_cache 只含预测数组（100 个），GT/姿态未入缓存；GT 由冻结数据+确定性"
                "窗口 manifest 重建（build_datasets 无 RNG），窗口数与顺序经 manifest hash 校验。",
        "window_ids_preview": win_ids[:6],
    }
    json.dump(meta, open(f"{OUT}/metrics/v07_table_reconciliation_meta.json", "w"),
              ensure_ascii=False, indent=1)

    pd.set_option("display.width", 240)
    print("=== 2x2 复算 ===")
    print(df.round(5).to_string(index=False))
    print("\n=== 与旧报告核对（max_abs_diff 应 ~0）===")
    print(ck.to_string(index=False))
    print("\n=== 差异分解 ===")
    print(dc.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
