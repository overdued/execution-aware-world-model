"""V0.6.1 R4：relative pose / 净 yaw 指标（合法可得的子集）+ 图。

合法可得：净 yaw = ∫wz dt（预测） vs 合法化四元数给出的真实 Δyaw。
完整相对位姿轨迹（体速度 + 姿态积分）未预注册 -> NOT_RUN（如实标注）。
"""
import glob
import json
import os

import numpy as np
import pandas as pd

OUT = "results/v0_6_1_correctness"
DER = "/media/hdd1/yuhang/datasets/execution_wm/v0_6_1"
DT = 0.05
H = 40


def yaw_from_episode(path, origins):
    d = np.load(path)
    yaw = d["lb_yaw"].astype(np.float64)
    return np.array([yaw[o + H] - yaw[o] for o in origins])


def main():
    lin = pd.read_csv(os.path.join(OUT, "manifests", "window_lineage.csv"))
    cache = np.load(os.path.join(OUT, "predictions", "pred_cache.npz"))
    rows = []
    # 真值 Δyaw（按 episode 缓存）
    truth_cache = {}
    for split, g in lin.groupby("split_label"):
        for _, r in g.iterrows():
            key = (r.condition, r.episode_id)
            if key not in truth_cache:
                path = os.path.join(DER, r.condition, f"ep_{r.episode_id:05d}.20hz.npz")
                d = np.load(path)
                truth_cache[key] = d["lb_yaw"].astype(np.float64)
            o = int(r.origin_grid)
            dyaw_true = float(truth_cache[key][o + H] - truth_cache[key][o])
            # 模型净 yaw：对 wz 预测积分
            rec = {"split": split, "episode_id": r.episode_id, "origin_grid": o,
                   "anchor_group": r.anchor_group, "delta_yaw_true": dyaw_true}
            for k in cache.files:
                if not k.startswith(split + "__"):
                    continue
                name = k.split("__", 1)[1]
                if name.endswith("OLD_wrong_axis"):
                    continue
                wz = cache[k][int(r.row_index), :, 2]
                rec[f"delta_yaw_pred::{name}"] = float(np.sum(wz) * DT)
            rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "metrics", "relative_yaw_per_window.csv"), index=False)

    pred_cols = [c for c in df.columns if c.startswith("delta_yaw_pred::")]
    agg = []
    for split, g in df.groupby("split"):
        for c in pred_cols:
            name = c.split("::", 1)[1]
            err = np.abs(g[c] - g["delta_yaw_true"])
            agg.append({"split": split, "variant": name, "n_windows": len(g),
                        "mean_abs_delta_yaw_err_rad": float(err.mean()),
                        "delta_yaw_true_std": float(g["delta_yaw_true"].std()),
                        "delta_yaw_true_mean_abs": float(g["delta_yaw_true"].abs().mean())})
    a = pd.DataFrame(agg)
    a.to_csv(os.path.join(OUT, "metrics", "relative_yaw_summary.csv"), index=False)
    print(a[a.variant.str.match(r"(M[0-3]_s4[234]|command-copy|persistence)$")]
          .pivot(index="variant", columns="split", values="mean_abs_delta_yaw_err_rad")
          .round(4).to_string())
    print("\n注: 完整相对位姿轨迹 NOT_RUN（体速度+姿态积分未预注册）。"
          "净 yaw 由 ∫wz dt 得到；真值来自合法化四元数。")
    json.dump({"relative_pose_full_trajectory": "NOT_RUN",
               "reason": "需要体速度与姿态联合积分，未在 PRE_REGISTRATION 中定义",
               "net_yaw": "RAN",
               "net_yaw_truth_source": "lb_yaw（合法化四元数：符号连续 + 归一化）",
               "raw_quaternion_illegal_max_norm_dev": 0.5000427537490804,
               "raw_quaternion_illegal_points": 4541},
              open(os.path.join(OUT, "metrics", "relative_pose_status.json"), "w"),
              indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
