"""V0.6 阶段B5：50Hz -> 20Hz 双版本派生（PRE_REGISTRATION §7）。

- **inputs 版（因果）**：一阶低通（fc=8Hz）只向前传播（y[k]=y[k-1]+α(x[k]-y[k-1])），
  然后在 0.05s 精确网格上取"不超过 t 的最后一个样本"（previous-sample hold）。
  无未来泄漏；真实时间戳 = 精确网格。
- **labels 版（抗混叠，非因果）**：scipy resample_poly(2/5)（FIR 抗混叠），
  仅用于标签与离线分析，不作为模型输入。
- cmd_vel：piecewise-constant，取网格时刻精确值（applied cmd，两版相同）。

输出: 每条 ep_XXXXX.npz 旁生成 ep_XXXXX.20hz.npz（inputs 版 + labels 版同文件不同键前缀）
用法: python -m execution_wm.validity_v06.derive_20hz --sq-dir /media/hdd1/yuhang/datasets/execution_wm/v0_6_sq
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy.signal import resample_poly

INPUT_KEYS = ["base_linear_velocity_body", "base_angular_velocity", "projected_gravity",
              "imu_linear_acceleration", "joint_position", "joint_velocity", "feet_contact"]
LABEL_KEYS = ["execution", "residual", "base_linear_velocity_world", "base_position",
              "base_orientation", "applied_torque", "foot_velocity"]
FC = 8.0


def causal_lowpass(x, dt, fc):
    rc = 1.0 / (2 * np.pi * fc)
    a = dt / (rc + dt)
    y = np.empty_like(x)
    y[0] = x[0]
    for k in range(1, len(x)):
        y[k] = y[k - 1] + a * (x[k] - y[k - 1])
    return y


def derive(path):
    d = np.load(path)
    t = d["timestamp"] - d["timestamp"][0]
    dt = float(np.median(np.diff(t)))
    dur = t[-1] + dt
    grid = np.arange(0, dur - 1e-9, 0.05)
    # previous-sample hold 索引（严格因果）
    idx_hold = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, len(t) - 1)

    out = {"timestamp": grid.astype(np.float64),
           "episode_time": grid.astype(np.float64),
           "phase": d["phase"][idx_hold],
           "cmd_vel": d["cmd_vel"][idx_hold].astype(np.float32)}
    for k in INPUT_KEYS:
        out[f"in_{k}"] = causal_lowpass(d[k].astype(np.float64), dt, FC)[idx_hold].astype(np.float32)
    T20 = int(round(dur * 20))
    for k in LABEL_KEYS:
        out[f"lb_{k}"] = resample_poly(d[k], 2, 5, axis=0)[:T20].astype(np.float32)
    # execution/residual 同时给因果版（训练标签一致性可选）
    for k in ("execution", "residual"):
        out[f"in_{k}"] = causal_lowpass(d[k].astype(np.float64), dt, FC)[idx_hold].astype(np.float32)
    for k in d.files:
        if k.startswith("anchor_"):
            out[k] = d[k]
    return out, len(grid), T20


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sq-dir", required=True)
    args = p.parse_args()
    files = sorted(glob.glob(os.path.join(args.sq_dir, "*", "ep_*.npz")))
    files = [f for f in files if not f.endswith(".20hz.npz")]
    n_len_mismatch = 0
    for f in files:
        out, n_grid, t20 = derive(f)
        if n_grid != t20:
            n_len_mismatch += 1
        np.savez_compressed(f.replace(".npz", ".20hz.npz"), **out)
    print(f"[derive] {len(files)} episodes derived; grid/T20 mismatch count={n_len_mismatch}")


if __name__ == "__main__":
    main()
