"""V0.6.1 B2+B3：正确的 50Hz -> 20Hz 目标派生（不覆盖旧 .20hz.npz）。

修复的旧错误（P0-2 / P1-1）:
  旧: r20 = F(e - u);  u20 = S(u);  e_used = u20 + r20 = F(e) + [S(u) - F(u)] != F(e)
  新: 先独立生成 execution_label = F(e)（只依赖物理 execution，不含 command）；
      再从命令事件表按整数 tick 生成 command_reference = S_event(u)；
      最后 residual_label = execution_label - command_reference（恒等式就此定义）。

F = 离线非因果抗混叠重采样（resample_poly 2/5，仅用于监督标签 = 平滑物理运动）
S = 保持/事件采样（严格因果，用于输入与 command 参考）
另外保留未平滑真值（lbra_* / rawin_*）做对照。

输出: <out_dir>/<cond>/ep_XXXXX.20hz.npz（新目录，键名向后兼容 + 新增参考/审计键）
用法: python -m execution_wm.validity_v061.derive_targets --sq-dir ... --out-dir ...
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy.signal import resample_poly

from execution_wm.validity_v061 import timebase as tb

INPUT_KEYS = ["base_linear_velocity_body", "base_angular_velocity", "projected_gravity",
              "imu_linear_acceleration", "joint_position", "joint_velocity", "feet_contact"]
LABEL_KEYS = ["execution", "base_linear_velocity_world", "base_position",
              "base_orientation", "applied_torque", "foot_velocity"]
FC = 8.0
DERIVE_VERSION = "v061.3"
PIPELINE_LATENCY_TICKS = 1   # §3 运行时实证：policy 在 tick i 消费 u[i-1]（execution lag = 2 tick）


def causal_lowpass(x, dt, fc):
    """一阶因果低通（只向前传播）。输入用，不作为标签。"""
    rc = 1.0 / (2 * np.pi * fc)
    a = dt / (rc + dt)
    y = np.empty_like(x)
    y[0] = x[0]
    for k in range(1, len(x)):
        y[k] = y[k - 1] + a * (x[k] - y[k - 1])
    return y


def anti_alias(x, n_grid):
    """F: resample_poly(2,5)，输出 j 对应 t = j*0.05 s；对齐到 grid 长度。"""
    y = resample_poly(x, 2, 5, axis=0)
    if len(y) >= n_grid:
        return y[:n_grid]
    return np.concatenate([y, np.repeat(y[-1:], n_grid - len(y), axis=0)], axis=0)


def legalize_quaternion(q):
    """§4: 逐分量滤波会破坏单位范数 -> 符号连续 + 归一化。返回合法姿态与合法性统计。"""
    q = np.asarray(q, dtype=np.float64)
    qs = q.copy()
    for k in range(1, len(qs)):
        if np.dot(qs[k], qs[k - 1]) < 0:
            qs[k] = -qs[k]
    n = np.linalg.norm(qs, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return qs / n


def quat_to_yaw(q):
    """(w,x,y,z) -> yaw（仅用于相对净 yaw 指标，姿态本身保留）。"""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def derive(path, cmd_lead_ticks=0):
    d = np.load(path)
    ts = d["timestamp"].astype(np.float64)
    n_ticks, ts_err = tb.assert_tick_grid_consistent(ts, label=os.path.basename(path))

    n_grid = tb.n_grid_points(n_ticks)
    hold = tb.grid_hold_ticks(n_grid)                      # 整数 tick，无浮点比较
    if hold[-1] >= n_ticks:
        raise AssertionError(f"hold 索引越界: {hold[-1]} >= {n_ticks}")
    grid_t = np.arange(n_grid, dtype=np.float64) * tb.GRID_DT_S
    tick_time = hold.astype(np.float64) * tb.CTRL_DT_S      # 输入的真实采样时刻（展示用）

    # ---- 1. execution_label：只由原始物理 execution 生成（不含 command）----
    lb_execution = anti_alias(d["execution"].astype(np.float64), n_grid)
    e_label = lb_execution.copy()

    # ---- 2. command_reference：按整数 tick 从记录命令序列取（右连续）----
    #      cmd_lead_ticks: 0 = 区间起点语义（tick 处生效的命令）
    #      +1 = 若运行时审计证实 policy 实际消费的是上一步命令则置 1（见 policy_timeline 报告）
    ci = np.clip(hold + cmd_lead_ticks, 0, n_ticks - 1)
    u_ref = d["cmd_vel"].astype(np.float64)[ci]

    # ---- 3. residual_label = execution_label - command_reference ----
    r_label = e_label - u_ref

    # ---- 3b. command pipeline latency（§3 运行时验证：consumption 滞后 1 tick） ----
    #   policy 在 tick i 实际消费的 obs 携带 u[i-1]（policy_command_timeline.json 实证
    #   199/199 tick）。execution 在 tick i 由 a[i-1] 产生 -> 由 u[i-2] 驱动。
    #   主口径仍用 u_ref = 同一 tick 的命令（部署忠实：输入是"计划"），
    #   本键仅供诊断，不改变恒等式（恒等式定义在 cmd_ref 上）。
    ci_cons = np.clip(hold - PIPELINE_LATENCY_TICKS, 0, n_ticks - 1)
    cmd_consumed = d["cmd_vel"].astype(np.float64)[ci_cons]

    out = {
        "timestamp": grid_t, "episode_time": grid_t,
        "grid_index": np.arange(n_grid, dtype=np.int64),
        "hold_ctrl_tick": hold,                       # 整数 tick 身份
        "tick_time": tick_time,
        "phase": d["phase"][hold],
        "cmd_ref": u_ref.astype(np.float32),          # B2: 命令参考（事件采样）
        "cmd_vel": u_ref.astype(np.float32),          # 向后兼容别名 = cmd_ref
        "cmd_consumed": cmd_consumed.astype(np.float32),   # 诊断：policy 实际消费的命令
        "pipeline_latency_ticks": np.int64(PIPELINE_LATENCY_TICKS),
        "lb_execution": e_label.astype(np.float32),   # 独立物理真值（评估唯一真值）
        "lb_residual": r_label.astype(np.float32),    # = e_label - u_ref（恒等式定义）
    }
    # 未平滑真值（对照）：raw 在 hold tick 的采样
    for k in ("execution", "base_linear_velocity_world", "base_position",
              "base_orientation", "applied_torque", "foot_velocity"):
        out[f"lbra_{k}"] = d[k][hold].astype(np.float32)
    # 其余标签：抗混叠（离线监督用）
    for k in LABEL_KEYS:
        if k == "execution":
            continue
        out[f"lb_{k}"] = anti_alias(d[k].astype(np.float64), n_grid).astype(np.float32)
    # 因果输入（低通 + hold）
    for k in INPUT_KEYS:
        out[f"in_{k}"] = causal_lowpass(d[k].astype(np.float64), tb.CTRL_DT_S, FC)[hold].astype(np.float32)
    out["in_execution"] = causal_lowpass(d["execution"].astype(np.float64), tb.CTRL_DT_S, FC)[hold].astype(np.float32)
    out["in_residual"] = (out["in_execution"] - u_ref).astype(np.float32)
    # 未平滑输入（对照）
    for k in INPUT_KEYS:
        out[f"rawin_{k}"] = d[k][hold].astype(np.float32)
    out["rawin_execution"] = d["execution"][hold].astype(np.float32)
    # §4 姿态合法性：非法四元数不用于姿态 GT；合法化后另存
    q_illegal = out["lb_base_orientation"].astype(np.float64)
    norm_dev = np.abs(np.linalg.norm(q_illegal, axis=1) - 1.0)
    q_legal = legalize_quaternion(q_illegal)
    out["lb_base_orientation_legal"] = q_legal.astype(np.float32)
    out["lb_yaw"] = quat_to_yaw(q_legal).astype(np.float32)
    out["quat_norm_max_dev_raw"] = np.float64(norm_dev.max())
    out["quat_n_illegal_points"] = np.int64(int((norm_dev > 1e-3).sum()))
    q_raw_hold = d["base_orientation"].astype(np.float64)[hold]
    out["lbra_base_orientation_legal"] = legalize_quaternion(q_raw_hold).astype(np.float32)
    out["lbra_yaw"] = quat_to_yaw(legalize_quaternion(q_raw_hold)).astype(np.float32)
    for k in d.files:
        if k.startswith("anchor_"):
            out[k] = d[k]

    # ---- 5. 恒等式断言 ----
    err = float(np.abs(u_ref + r_label - e_label).max())
    return out, {"file": os.path.basename(path), "n_ticks": n_ticks, "n_grid": n_grid,
                 "timestamp_grid_max_err": ts_err, "identity_max_err": err,
                 "identity_err_vx": float(np.abs(u_ref[:, 0] + r_label[:, 0] - e_label[:, 0]).max()),
                 "identity_err_vy": float(np.abs(u_ref[:, 1] + r_label[:, 1] - e_label[:, 1]).max()),
                 "identity_err_wz": float(np.abs(u_ref[:, 2] + r_label[:, 2] - e_label[:, 2]).max()),
                 "boundary_err_first": float(np.abs(u_ref[0] + r_label[0] - e_label[0]).max()),
                 "boundary_err_last": float(np.abs(u_ref[-1] + r_label[-1] - e_label[-1]).max()),
                 "n_tick_aligned_grid": int(sum(tb.grid_is_tick_aligned(k) for k in range(n_grid))),
                 "quat_norm_max_dev_raw": float(norm_dev.max()),
                 "quat_n_illegal_points": int((norm_dev > 1e-3).sum()),
                 "quat_norm_max_dev_legal": float(np.abs(np.linalg.norm(q_legal, axis=1) - 1).max())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sq-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cmd-lead-ticks", type=int, default=0)
    args = p.parse_args()
    files = sorted(glob.glob(os.path.join(args.sq_dir, "*", "ep_*.npz")))
    files = [f for f in files if not f.endswith(".20hz.npz")]
    rows = []
    for f in files:
        cond = os.path.basename(os.path.dirname(f))
        out, meta = derive(f, args.cmd_lead_ticks)
        os.makedirs(os.path.join(args.out_dir, cond), exist_ok=True)
        np.savez_compressed(os.path.join(args.out_dir, cond, os.path.basename(f).replace(".npz", ".20hz.npz")), **out)
        meta["condition"] = cond
        meta["cmd_lead_ticks"] = args.cmd_lead_ticks
        meta["derive_version"] = DERIVE_VERSION
        rows.append(meta)
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "derive_identity_audit.csv"), index=False)
    print(f"[derive] {len(files)} episodes -> {args.out_dir}")
    print(f"[derive] identity max err = {df.identity_max_err.max():.2e} "
          f"(阈值 1e-6，全部通过={bool((df.identity_max_err < 1e-6).all())})")
    print(f"[derive] timestamp grid max err = {df.timestamp_grid_max_err.max():.2e}")


if __name__ == "__main__":
    main()
