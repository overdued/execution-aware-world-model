"""V0.6.2 Task 4 — Full Trajectory Consequence（SE(2) 口径）。

用 V0.6.1 正确物理标签 + 合法化四元数，计算 task-relevant horizon metric：
    Δx, Δy（世界系位移，m）  Δyaw（净偏航，rad）
在 0.5s / 1.0s / 2.0s 三个 horizon 报告。

明确区分：
  (i)  instantaneous velocity error —— 单点 |ê(t+h) − e(t+h)|
  (ii) integrated trajectory consequence —— 积分后的位移/净偏航误差

平面近似与合法性：
  - **不**把 body wz 直接当 world yaw rate。采用两条口径：
    (a) velocity-only rollout：用 **真值 yaw** 积分预测体速度 -> 隔离速度误差
    (b) full SE(2) rollout：用预测体速度 + ∫wz_pred dt 的预测 yaw（从真值初值起）
  - 另定义 **valid-planar subset**：horizon 内最大倾角 < 10°（由 projected_gravity 计算），
    并在该子集上核验 |∫wz dt − Δyaw_quat| 的平面近似误差。
  - 真值侧一律用合法化四元数的 yaw（lb_yaw），不用 ∫wz。

输出: metrics/trajectory_consequence.csv, metrics/planar_validity.json
"""
import json
import os

import numpy as np
import pandas as pd
import torch

from execution_wm.validity_v062.common import (DERIVED, OUT, SPLITS_EVAL, get_manifests,
                                                load_cache, pred_from_cache, seed_mean_pred)

DT = 0.05
HORIZONS = {"0.5s": 10, "1.0s": 20, "2.0s": 40}
TILT_LIMIT_DEG = 10.0



def main():
    man = get_manifests()
    cache = load_cache()
    # yaw 真值表：按 (condition, episode_id) 缓存
    yaw_cache = {}
    rows, planar = [], {}

    def get_yaw(cond, ep_id):
        key = (cond, ep_id)
        if key not in yaw_cache:
            p = os.path.join(DERIVED, cond, f"ep_{ep_id:05d}.20hz.npz")
            d = np.load(p)
            yaw_cache[key] = d["lb_yaw"].astype(np.float64)
        return yaw_cache[key]

    def tilt_deg(man_split):
        """每窗 horizon 内最大倾角（projected_gravity 在 proprio 的 [6:9]）。"""
        g = man_split.hp[:, :, 6:9].astype(np.float64)            # [N,L,3] body frame
        gz = g[:, :, 2]
        return np.degrees(np.arccos(np.clip(-gz, -1, 1))).max(axis=1)

    def rollout(body_vel, yaw0, yaw_seq, n_steps):
        """body 速度 + yaw 序列 -> 世界系位移。yaw_seq 为 rollout 用 yaw 序列（含起点）。"""
        p = np.zeros(2)
        for k in range(n_steps):
            c, s = np.cos(yaw_seq[k]), np.sin(yaw_seq[k])
            vx, vy = body_vel[k, 0], body_vel[k, 1]
            p += np.array([c * vx - s * vy, s * vx + c * vy]) * DT
        return p

    for split in SPLITS_EVAL:
        m = man[split]
        n = len(m)
        tilt = tilt_deg(m)
        valid = tilt < TILT_LIMIT_DEG
        planar[split] = {"n_windows": int(n), "n_valid_planar": int(valid.sum()),
                         "frac_valid_planar": float(valid.mean()),
                         "tilt_deg_median": float(np.median(tilt)),
                         "tilt_deg_p95": float(np.quantile(tilt, 0.95)),
                         "tilt_limit_deg": TILT_LIMIT_DEG}
        variants = {"command-copy": pred_from_cache(cache, split, "command-copy"),
                    "ridge_multioutput": pred_from_cache(cache, split, "ridge_multioutput")}
        for name in ("M0", "M1", "M2"):
            variants[name] = seed_mean_pred(cache, split, name)

        # 真值轨迹参数
        yaws0 = np.array([get_yaw(w["condition"], w["episode_id"])[w["origin_grid"]]
                          for w in m.windows])
        yaws_true = np.stack([get_yaw(w["condition"], w["episode_id"])[
            w["origin_grid"] + 1:w["origin_grid"] + 1 + 40] for w in m.windows])

        # 平面近似核验：∫wz_true dt vs Δyaw_quat
        dyaw_quat = yaws_true[:, -1] - yaws0
        # 处理 wrap（±π）
        dyaw_quat = (dyaw_quat + np.pi) % (2 * np.pi) - np.pi
        dyaw_int = np.cumsum(m.fe[:, :, 2], axis=1)[:, -1] * DT
        approx_err = np.abs(dyaw_int - dyaw_quat)
        planar[split]["planar_approx_err_rad_mean"] = float(approx_err.mean())
        planar[split]["planar_approx_err_rad_p95"] = float(np.quantile(approx_err, 0.95))
        planar[split]["planar_approx_err_rad_mean_valid_subset"] = float(
            approx_err[valid].mean()) if valid.any() else None

        for vname, pr in variants.items():
            for hname, hs in HORIZONS.items():
                # (i) instantaneous
                inst_vx = float(np.abs(pr[:, hs - 1, 0] - m.fe[:, hs - 1, 0]).mean())
                inst_vy = float(np.abs(pr[:, hs - 1, 1] - m.fe[:, hs - 1, 1]).mean())
                inst_wz = float(np.abs(pr[:, hs - 1, 2] - m.fe[:, hs - 1, 2]).mean())
                # (ii-a) velocity-only rollout（真值 yaw）
                preds_a, trues = [], []
                for i in range(n):
                    yaws_step = np.concatenate([[yaws0[i]], yaws_true[i, :hs - 1]])
                    preds_a.append(rollout(pr[i, :hs, :], yaws0[i], yaws_step, hs))
                    trues.append(rollout(m.fe[i, :hs, :], yaws0[i], yaws_step, hs))
                preds_a, trues = np.array(preds_a), np.array(trues)
                ex = float(np.abs(preds_a[:, 0] - trues[:, 0]).mean())
                ey = float(np.abs(preds_a[:, 1] - trues[:, 1]).mean())
                erad = float(np.linalg.norm(preds_a - trues, axis=1).mean())
                # (ii-b) full SE(2) rollout（预测 yaw = yaw0 + Σwz_pred dt）
                preds_b = []
                for i in range(n):
                    dy = np.cumsum(pr[i, :hs, 2]) * DT
                    yaw_pred = yaws0[i] + np.concatenate([[0.0], dy[:-1]])
                    preds_b.append(rollout(pr[i, :hs, :], yaws0[i], yaw_pred, hs))
                preds_b = np.array(preds_b)
                ex_b = float(np.abs(preds_b[:, 0] - trues[:, 0]).mean())
                ey_b = float(np.abs(preds_b[:, 1] - trues[:, 1]).mean())
                # 净 yaw 误差（预测 ∫wz vs 合法化四元数真值）
                dyaw_pred = np.cumsum(pr[:, :hs, 2], axis=1)[:, -1] * DT
                dyaw_true_h = (yaws_true[:, hs - 1] - yaws0 + np.pi) % (2 * np.pi) - np.pi
                eyaw = float(np.abs(dyaw_pred - dyaw_true_h).mean())
                rows.append({
                    "split": split, "model": vname, "horizon": hname, "n_windows": n,
                    "n_valid_planar": int(valid.sum()),
                    "inst_MAE_vx": inst_vx, "inst_MAE_vy": inst_vy, "inst_MAE_wz": inst_wz,
                    "traj_dx_err_m": ex, "traj_dy_err_m": ey, "traj_dxy_err_m": erad,
                    "traj_dx_err_m_se2": ex_b, "traj_dy_err_m_se2": ey_b,
                    "net_yaw_err_rad": eyaw,
                    "traj_dxy_err_m_valid_planar": float(
                        np.linalg.norm(preds_a - trues, axis=1)[valid].mean()) if valid.any() else None,
                    "net_yaw_err_rad_valid_planar": float(
                        np.abs(dyaw_pred - dyaw_true_h)[valid].mean()) if valid.any() else None,
                })
        print(f"[task4] {split} done (valid-planar {valid.sum()}/{n})")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "metrics", "trajectory_consequence.csv"), index=False)
    json.dump(planar, open(os.path.join(OUT, "metrics", "planar_validity.json"), "w"),
              indent=1, ensure_ascii=False)
    pd.set_option("display.width", 260)
    print("\n=== 平面合法性 ===")
    print(pd.DataFrame(planar).T[["n_windows", "n_valid_planar", "frac_valid_planar",
                                  "tilt_deg_p95", "planar_approx_err_rad_mean",
                                  "planar_approx_err_rad_p95"]].round(4).to_string())
    print("\n=== 瞬时 vs 轨迹（@2.0s）===")
    d2 = df[df["horizon"] == "2.0s"].copy()
    cols = [c for c in ["split", "model", "inst_MAE_vx", "inst_MAE_wz", "traj_dxy_err_m",
                        "traj_dxy_err_m_se2", "net_yaw_err_rad"] if c in d2.columns]
    print(d2[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
