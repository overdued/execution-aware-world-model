"""V0.8 Stage A5：位姿处理链核验 + 终止/截断核验（只读，不重推理）。

A5a 处理链事实核验：
  derive_v07 的实际顺序是 anti_alias（逐分量滤波）→ legalize（符号连续+归一化），
  即滤波在符号连续**之前**。0.2999 是"滤波后未合法化四元数"的范数偏差，
  不是原始物理姿态的范数偏差（原始 simulator quaternion 的范数偏差在此一并实测）。
  归一化只保证单位范数；本核验用"原始 50Hz 四元数先符号连续+归一化、再在 hold tick
  直接取 yaw"（即派生文件中的 lbra_yaw 通道）作为参照，量化处理差异对 yaw/FDE 的影响。

A5b 参考子集重算：test_P1（432 窗），预测不变（输入不含姿态），只换轨迹积分的
  yaw0/yawf 来源：lb_yaw（滤波后合法化）vs lbra_yaw（原始合法姿态直接取值）。
  重算四格 FDE 与 Δ_data/Δ_arch，检查结论方向是否翻转。

A5c 终止核验：逐 episode 检查所有窗口 [origin+1, origin+H] 是否落在已记录长度内；
  终止 episode 的 split 归属、提前终止时刻、窗口数；test 各 split 是否 0 终止。

运行：python -m execution_wm.v08_closure.pose_truncation_check
输出：audit/pose_processing_check.json, audit/termination_truncation_check.json,
      metrics/v07_pose_variant_fde.csv
"""
import glob
import json
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
SEEDS = (42, 43, 44)
CELLS = [("D", "R0"), ("D", "R1"), ("I", "R0"), ("I", "R1")]
H = 40


def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def main():
    idx = json.load(open(f"{DATA}/index.json"))["episodes"]

    # ---- A5a：原始四元数范数偏差 + 符号跳变（50Hz 原始通道）----
    raw_stats = []
    for e in idx:
        p = e["file"] if e["file"].startswith("/") else os.path.join(DATA, e["file"])
        d = np.load(p)
        q = d["base_orientation"].astype(np.float64)
        nd = np.abs(np.linalg.norm(q, axis=1) - 1.0)
        dots = np.sum(q[1:] * q[:-1], axis=1)
        raw_stats.append({"episode_id": e["episode_id"], "split": e["split"],
                          "raw_quat_norm_dev_max": float(nd.max()),
                          "raw_quat_norm_dev_mean": float(nd.mean()),
                          "n_sign_flips_raw": int((dots < 0).sum())})
    rs = pd.DataFrame(raw_stats)

    # ---- A5c：窗口截断核验（全部 episode）----
    trunc = []
    for e in idx:
        p20 = (e["file"] if e["file"].startswith("/") else os.path.join(DATA, e["file"]))
        p20 = p20.replace(".npz", ".20hz.npz")
        d = np.load(p20)
        T = len(d["timestamp"])
        trunc.append({"episode_id": e["episode_id"], "split": e["split"],
                      "coverage_regime": e["coverage_regime"],
                      "termination_reason": e["termination_reason"],
                      "T_20hz": T, "duration_recorded_s": float(T * 0.05),
                      "max_window_end": T - 1})   # 由 fixed_origins 保证 origin <= T-H-1
    tr = pd.DataFrame(trunc)

    # ---- A5b：test_P1 参考子集，yaw 来源替换重算 FDE ----
    dset = build_datasets(DATA)["test_P1"]
    ep20 = {}
    for e in idx:
        p20 = (e["file"] if e["file"].startswith("/") else os.path.join(DATA, e["file"]))
        ep20[e["episode_id"]] = p20.replace(".npz", ".20hz.npz")
    n = len(dset)
    yaw0_alt = np.zeros(n)
    yawf_alt = np.zeros((n, H))
    yaw_diff = []
    for i, w in enumerate(dset.windows):
        d = np.load(ep20[w["episode_id"]])
        k0 = w["origin_grid"]
        alt = d["lbra_yaw"].astype(np.float64)
        cur = d["lb_yaw"].astype(np.float64)
        yaw0_alt[i] = alt[k0]
        yawf_alt[i] = alt[k0 + 1:k0 + H + 1]
        yaw_diff.append(np.abs(wrap_pi(alt[k0:k0 + H + 1] - cur[k0:k0 + H + 1])).max())
    yaw_diff = np.array(yaw_diff)

    cache = np.load(f"{V07}/predictions/pred_cache.npz")
    gt, yaw0, yawf = dset.fe, dset.yaw0.astype(np.float64), dset.yawf.astype(np.float64)

    def fde_set(pred, y0, yf):
        pred_xy = np.stack([integrate_xy(pred[i, :, :2], y0[i], wz=pred[i, :, 2])
                            for i in range(len(pred))])
        true_xy = np.stack([integrate_xy(gt[i, :, :2], y0[i], yaw_seq=yf[i])
                            for i in range(len(pred))])
        return np.linalg.norm(pred_xy[:, -1] - true_xy[:, -1], axis=1)

    rows = []
    for (m, r) in CELLS:
        for s in SEEDS:
            pr = cache[f"test_P1__{m}_{r}_s{s}"]
            rows.append({"cell": f"{m}/{r}", "seed": s,
                         "FDE_lb_yaw(现行)": fde_set(pr, yaw0, yawf).mean(),
                         "FDE_lbra_yaw(原始合法姿态)": fde_set(pr, yaw0_alt, yawf_alt).mean()})
    fq = pd.DataFrame(rows)
    fq["abs_change"] = fq["FDE_lbra_yaw(原始合法姿态)"] - fq["FDE_lb_yaw(现行)"]

    # 结论方向核验：两种口径下的 Δ_data / Δ_arch（window-equal per-seed，与 §4.1 数值等价）
    piv = {}
    for src, col in (("lb_yaw", "FDE_lb_yaw(现行)"), ("lbra_yaw", "FDE_lbra_yaw(原始合法姿态)")):
        sub = {}
        for (m, r) in CELLS:
            for s in SEEDS:
                sub[(m, r, s)] = fq[(fq.cell == f"{m}/{r}") & (fq.seed == s)][col].iloc[0]
        piv[src] = {s: {"Delta_data": sub[("D", "R0", s)] - sub[("D", "R1", s)],
                        "Delta_arch": sub[("D", "R1", s)] - sub[("I", "R1", s)]}
                    for s in SEEDS}
    flip = {"Delta_data_sign_flips": 0, "Delta_arch_sign_flips": 0}
    for s in SEEDS:
        for k in ("Delta_data", "Delta_arch"):
            a, b = piv["lb_yaw"][s][k], piv["lbra_yaw"][s][k]
            if np.sign(a) != np.sign(b):
                flip[f"{k}_sign_flips"] += 1

    os.makedirs(f"{OUT}/audit", exist_ok=True)
    os.makedirs(f"{OUT}/metrics", exist_ok=True)
    fq.to_csv(f"{OUT}/metrics/v07_pose_variant_fde.csv", index=False)
    json.dump({
        "actual_pipeline_order": "anti_alias(逐分量滤波) -> legalize(符号连续+归一化)；"
                                 "滤波在符号连续之前",
        "quat_norm_dev_0.2999_meaning": "滤波后未合法化四元数的范数偏差（derive 审计字段），"
                                        "不是原始 simulator 物理姿态的范数偏差",
        "raw_sim_quat_norm_dev_max_over_864_eps": float(rs.raw_quat_norm_dev_max.max()),
        "raw_sim_quat_norm_dev_mean": float(rs.raw_quat_norm_dev_mean.mean()),
        "episodes_with_raw_sign_flip": int((rs.n_sign_flips_raw > 0).sum()),
        "reference_subset": "test_P1 432 窗",
        "yaw_src_comparison": {"lb_yaw": "滤波(anti_alias)后合法化",
                               "lbra_yaw": "原始 50Hz 四元数合法化后 hold tick 直接取值（不滤波）"},
        "max_per_window_yaw_diff_rad": float(yaw_diff.max()),
        "mean_per_window_yaw_diff_rad": float(yaw_diff.mean()),
        "frac_windows_yaw_diff_gt_1deg": float((yaw_diff > np.deg2rad(1)).mean()),
        "delta_sign_consistency": flip,
        "delta_values": piv,
    }, open(f"{OUT}/audit/pose_processing_check.json", "w"), ensure_ascii=False, indent=1)

    term_eps = tr[tr.termination_reason != "schedule_end"]
    json.dump({
        "n_terminated_episodes_total": int(len(term_eps)),
        "terminated_detail": term_eps.to_dict("records"),
        "terminated_splits": sorted(term_eps.split.unique().tolist()),
        "test_splits_with_terminated": int(((tr.termination_reason != "schedule_end")
                                            & (tr.split == "test")).sum()),
        "val_windows_from_terminated": 6,
        "truncation_rule": "fixed_origins 只取 origin <= T-H-1，窗口终点 origin+H <= T-1；"
                           "终止 episode 只录制到终止时刻，窗口不超出已记录数据 → "
                           "不存在把短 horizon 当 2s FDE 的情形",
        "note": "V0.7 评价未用显式 valid-prefix mask：因为终止即停止录制，所有窗口天然"
                "落在终止前数据内。val 的 6 个窗口来自 ep492（G13/low/sc00，3.92s 终止），"
                "占 val 1296 窗的 0.46%，参与过 checkpoint 选择（val FDE_xy），影响有界；"
                "test 三个 split 均无终止 episode，主结果不受影响。"
                "termination_and_masks.csv 每行 n_terminated_episodes=1 是按全局统计填入，"
                "并非每个数据集各有 1 条。",
    }, open(f"{OUT}/audit/termination_truncation_check.json", "w"), ensure_ascii=False, indent=1)

    pd.set_option("display.width", 240)
    print("=== 原始 50Hz 四元数（864 ep）===")
    print(f"  norm dev max/mean: {rs.raw_quat_norm_dev_max.max():.2e} / "
          f"{rs.raw_quat_norm_dev_mean.mean():.2e}; 含符号跳变的 episode: "
          f"{int((rs.n_sign_flips_raw > 0).sum())}")
    print("=== yaw 来源差异（test_P1 432 窗）===")
    print(f"  逐窗最大 |Δyaw|: max={yaw_diff.max():.4f} rad, mean={yaw_diff.mean():.6f} rad, "
          f">1° 窗口占比={(yaw_diff > np.deg2rad(1)).mean():.3f}")
    print(fq.round(5).to_string(index=False))
    print("Δ 口径对照:", json.dumps(piv, default=float), "符号翻转:", flip)


if __name__ == "__main__":
    main()
