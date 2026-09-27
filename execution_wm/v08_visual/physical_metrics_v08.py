"""V0.8 Stage C：物理副终点（预注册 §5）：physical FDE/ADE/net-yaw 与
fixed-lead 逐轴误差（Ê，V-AUX/V-EXEC；V-DIRECT 无物理头，列 N/A）。

口径：Ê（train-fit 归一化空间）反归一化 -> body 速度序列 [40,3]；
以原点 lb_yaw 为初始航向积分得平面位移/yaw 轨迹；GT 用 lb_execution 同法积分
（label-based，与 V0.7 评价口径一致）。lead=1s=20 步、2s=40 步（20Hz）。

运行：python -m execution_wm.v08_visual.physical_metrics_v08 \
  --results R --raw-root D
"""
import argparse
import json
from pathlib import Path

import numpy as np

DT = 0.05


def integrate(e, yaw0):
    """e [H,3] (vx,vy,wz body) + 初始 yaw -> (xy 轨迹 [H,2], yaw 轨迹 [H])。"""
    yaw = np.zeros(len(e))
    xy = np.zeros((len(e), 2))
    y = yaw0
    p = np.zeros(2)
    for t in range(len(e)):
        y = y + e[t, 2] * DT
        c, s = np.cos(y), np.sin(y)
        p = p + np.array([c * e[t, 0] - s * e[t, 1],
                          s * e[t, 0] + c * e[t, 1]]) * DT
        yaw[t], xy[t] = y, p
    return xy, yaw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--raw-root", required=True)
    args = ap.parse_args()
    import pandas as pd
    rr = Path(args.results)
    df = pd.read_csv(rr / "manifests/windows.csv")
    norm = json.loads((rr / "manifests/e_norm.json").read_text())
    e_mean, e_std = np.array(norm["mean"]), np.array(norm["std"])
    rows = []
    for pred_npz in sorted((rr / "predictions").glob("*_test_preds.npz")):
        run_id = pred_npz.name.replace("_test_preds.npz", "")
        if run_id.startswith("VDIRECT"):
            continue
        preds = np.load(pred_npz)
        test_df = df[df.split == "test"]
        for _, r in test_df.iterrows():
            wid = r["window_id"]
            if f"{wid}/e_hat" not in preds:
                continue
            d20 = np.load(Path(args.raw_root) / r["split"] / r["group_id"] /
                          f"b{r['branch']}" / f"{r['episode_id']}.20hz.npz")
            k = int(r["origin_k20"])
            e_pred = preds[f"{wid}/e_hat"].astype(np.float32) * e_std + e_mean
            e_pred = e_pred.reshape(40, 3)
            e_gt = d20["lb_execution"][k + 1:k + 41].astype(np.float32)
            yaw0 = float(d20["lb_yaw"][k])
            xy_p, yaw_p = integrate(e_pred, yaw0)
            xy_g, yaw_g = integrate(e_gt, yaw0)
            pos_err = np.linalg.norm(xy_p - xy_g, axis=1)
            rows.append({
                "run_id": run_id, "window_id": wid, "group_id": r["group_id"],
                "level": r["level"], "layout": r["layout"],
                "fde_xy_1s": float(pos_err[19]), "fde_xy_2s": float(pos_err[39]),
                "ade_xy_2s": float(pos_err.mean()),
                "net_yaw_err_1s": float(abs(yaw_p[19] - yaw_g[19])),
                "net_yaw_err_2s": float(abs(yaw_p[39] - yaw_g[39])),
                "e_err_vx_lead1s": float(abs(e_pred[19, 0] - e_gt[19, 0])),
                "e_err_vy_lead1s": float(abs(e_pred[19, 1] - e_gt[19, 1])),
                "e_err_wz_lead1s": float(abs(e_pred[19, 2] - e_gt[19, 2])),
                "e_err_vx_lead2s": float(abs(e_pred[39, 0] - e_gt[39, 0])),
                "e_err_vy_lead2s": float(abs(e_pred[39, 1] - e_gt[39, 1])),
                "e_err_wz_lead2s": float(abs(e_pred[39, 2] - e_gt[39, 2])),
            })
    out = pd.DataFrame(rows)
    out.to_csv(rr / "metrics" / "physical_metrics.csv", index=False)
    g = (out.groupby(["run_id", "level"])[
        ["fde_xy_2s", "ade_xy_2s", "net_yaw_err_2s"]].mean().round(4))
    print(g)


if __name__ == "__main__":
    main()
