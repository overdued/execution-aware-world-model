"""B3: physics-invariance A/B — camera vs no-camera collector, same plan/seeds.

Compares v0_8_smoke (with RGB camera) against v0_7_smoke (no camera) episode by
episode on all physical channels. Expectation: near-bitwise agreement, since the
camera sensor prim has no collision/mass and must not perturb physics.
"""
import argparse
import json
from pathlib import Path

import numpy as np

PHYS_CHANNELS = [
    "base_position", "base_orientation", "base_linear_velocity_world",
    "base_linear_velocity_body", "base_angular_velocity", "execution",
    "residual", "u_consumed", "joint_position", "joint_velocity",
    "joint_command", "applied_torque", "feet_contact", "foot_velocity",
    "imu_angular_velocity", "imu_linear_acceleration", "projected_gravity",
    "cmd_vel", "phase", "sim_time",
]
ANCHOR_CHANNELS = [k for k in [
    "anchor_base_position", "anchor_base_orientation", "anchor_base_lin_vel_w",
    "anchor_base_ang_vel_w", "anchor_joint_position", "anchor_joint_velocity",
]]


def compare_episode(pa: Path, pb: Path) -> dict:
    da, db = np.load(pa), np.load(pb)
    out = {"episode": str(pa.relative_to(pa.parents[3])), "channels": {}}
    n = min(da["base_position"].shape[0], db["base_position"].shape[0])
    out["len_a"] = int(da["base_position"].shape[0])
    out["len_b"] = int(db["base_position"].shape[0])
    for ch in PHYS_CHANNELS + ANCHOR_CHANNELS:
        if ch not in da.files or ch not in db.files:
            out["channels"][ch] = {"status": "missing"}
            continue
        a, b = da[ch][:n].astype(np.float64), db[ch][:n].astype(np.float64)
        if a.shape != b.shape:
            out["channels"][ch] = {"status": "shape_mismatch",
                                   "shape_a": list(a.shape), "shape_b": list(b.shape)}
            continue
        diff = np.abs(a - b)
        out["channels"][ch] = {
            "status": "ok",
            "bitwise_equal": bool(np.array_equal(da[ch][:n], db[ch][:n])),
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
            "ref_scale": float(np.abs(b).max()) if b.size else 0.0,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir-a", default="/media/hdd1/yuhang/datasets/execution_wm/v0_8_smoke")
    ap.add_argument("--dir-b", default="/media/hdd1/yuhang/datasets/execution_wm/v0_7_smoke")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dir_a, dir_b = Path(args.dir_a), Path(args.dir_b)
    eps_a = sorted(dir_a.rglob("ep_*.npz"))
    results, worst = [], {}
    for pa in eps_a:
        rel = pa.relative_to(dir_a)
        pb = dir_b / rel
        if not pb.exists():
            results.append({"episode": str(rel), "error": "missing counterpart"})
            continue
        r = compare_episode(pa, pb)
        results.append(r)
        for ch, c in r["channels"].items():
            if c.get("status") == "ok":
                w = worst.setdefault(ch, {"max_abs_diff": 0.0, "bitwise_equal": True})
                w["max_abs_diff"] = max(w["max_abs_diff"], c["max_abs_diff"])
                w["bitwise_equal"] = w["bitwise_equal"] and c["bitwise_equal"]

    summary = {
        "dir_a_camera": str(dir_a),
        "dir_b_no_camera": str(dir_b),
        "n_episodes_compared": len(results),
        "worst_per_channel": worst,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "episodes": results},
                                         indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
