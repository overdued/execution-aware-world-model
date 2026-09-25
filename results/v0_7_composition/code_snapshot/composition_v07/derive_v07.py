"""V0.7 Stage 2b：50Hz -> 20Hz 目标派生（继承 V0.6.1 口径，新增 u_consumed/joint_command/tick）。

规则（PRE_REGISTRATION_V07 §4）:
  e_label = F(e_raw)                        抗混叠，独立于 command
  u_ref   = S(u_requested)                  事件表，整数 tick
  r_label = e_label - u_ref                 断言 u_ref + r_label == e_label
  输入    = 因果低通 + 整数 tick hold（严格 <= origin）
  姿态    = 合法化四元数（符号连续 + 归一化），另存 yaw
输出: <out>/<split>/<group>/<friction>/ep_XXXXX.20hz.npz + derive_identity_audit.csv
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys  # noqa: E402
sys.path.insert(0, PROJ)
from execution_wm.validity_v061 import timebase as tb  # noqa: E402
from execution_wm.validity_v061.derive_targets import (anti_alias, causal_lowpass,  # noqa: E402
                                                       legalize_quaternion, quat_to_yaw)
# 注意：不要 import collect_v07 —— 它会在模块级解析 argparse（需 --plan/--out-root）

INPUT_KEYS = ["base_linear_velocity_body", "base_angular_velocity", "projected_gravity",
              "imu_linear_acceleration", "joint_position", "joint_velocity", "feet_contact"]
LABEL_KEYS = ["execution", "base_linear_velocity_world", "base_position",
              "base_orientation", "applied_torque", "foot_velocity"]
FC = 8.0
DERIVE_VERSION = "v07.1"


def derive(path):
    d = np.load(path)
    ts = d["sim_time"].astype(np.float64)
    n_ticks, ts_err = tb.assert_tick_grid_consistent(ts, label=os.path.basename(path))
    n_grid = tb.n_grid_points(n_ticks)
    hold = tb.grid_hold_ticks(n_grid)
    if hold[-1] >= n_ticks:
        raise AssertionError(f"hold 越界 {hold[-1]} >= {n_ticks}")
    grid_t = np.arange(n_grid, dtype=np.float64) * tb.GRID_DT_S

    e_label = anti_alias(d["execution"].astype(np.float64), n_grid)
    u_ref = d["cmd_vel"].astype(np.float64)[hold]
    r_label = e_label - u_ref
    u_cons = d["u_consumed"].astype(np.float64)[hold]
    jcmd = anti_alias(d["joint_command"].astype(np.float64), n_grid) if \
        d["joint_command"].ndim > 1 else d["joint_command"][hold]
    jcmd = d["joint_command"].astype(np.float64)[hold]          # 输入口径：hold（因果）

    out = {"timestamp": grid_t, "episode_time": grid_t,
           "grid_index": np.arange(n_grid, dtype=np.int64),
           "hold_ctrl_tick": hold,
           "tick_time": hold.astype(np.float64) * tb.CTRL_DT_S,
           "phase": d["phase"][hold],
           "cmd_ref": u_ref.astype(np.float32),
           "cmd_vel": u_ref.astype(np.float32),
           "cmd_consumed": u_cons.astype(np.float32),
           "joint_command": jcmd.astype(np.float32),
           "lb_execution": e_label.astype(np.float32),
           "lb_residual": r_label.astype(np.float32)}
    for k in ("execution", "base_linear_velocity_world", "base_position",
              "base_orientation", "applied_torque", "foot_velocity"):
        out[f"lbra_{k}"] = d[k][hold].astype(np.float32)
    for k in LABEL_KEYS:
        if k == "execution":
            continue
        out[f"lb_{k}"] = anti_alias(d[k].astype(np.float64), n_grid).astype(np.float32)
    for k in INPUT_KEYS:
        out[f"in_{k}"] = causal_lowpass(d[k].astype(np.float64), tb.CTRL_DT_S, FC)[hold].astype(np.float32)
    out["in_execution"] = causal_lowpass(d["execution"].astype(np.float64), tb.CTRL_DT_S, FC)[hold].astype(np.float32)
    out["in_residual"] = (out["in_execution"] - u_ref).astype(np.float32)
    for k in INPUT_KEYS:
        out[f"rawin_{k}"] = d[k][hold].astype(np.float32)
    out["rawin_execution"] = d["execution"][hold].astype(np.float32)
    # 合法姿态 + yaw
    q = out["lb_base_orientation"].astype(np.float64)
    nd = np.abs(np.linalg.norm(q, axis=1) - 1.0)
    ql = legalize_quaternion(q)
    out["lb_base_orientation_legal"] = ql.astype(np.float32)
    out["lb_yaw"] = quat_to_yaw(ql).astype(np.float32)
    qr = legalize_quaternion(d["base_orientation"].astype(np.float64)[hold])
    out["lbra_yaw"] = quat_to_yaw(qr).astype(np.float32)
    out["quat_norm_max_dev_raw"] = np.float64(nd.max())
    for k in d.files:
        if k.startswith("anchor_"):
            out[k] = d[k]
    err = float(np.abs(u_ref + r_label - e_label).max())
    return out, {"file": os.path.basename(path), "n_ticks": n_ticks, "n_grid": n_grid,
                 "timestamp_grid_max_err": ts_err, "identity_max_err": err,
                 "quat_norm_max_dev_raw": float(nd.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    files = sorted(f for f in glob.glob(os.path.join(args.root, "*", "*", "*", "ep_*.npz"))
                   if not f.endswith(".20hz.npz"))
    rows = []
    for f in files:
        out, meta = derive(f)
        np.savez_compressed(f.replace(".npz", ".20hz.npz"), **out)
        rows.append(meta)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.root, "derive_identity_audit.csv"), index=False)
    print(f"[derive] {len(files)} episodes; identity max err = {df.identity_max_err.max():.2e}; "
          f"ts grid max err = {df.timestamp_grid_max_err.max():.2e}; "
          f"quat raw norm dev max = {df.quat_norm_max_dev_raw.max():.4f}")


if __name__ == "__main__":
    main()
