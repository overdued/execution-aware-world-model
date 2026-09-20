"""数据 sanity checks（first_work.md §17）。

采完数据后、训练前必须运行：
    python -m execution_wm.data.sanity_check --config execution_wm/configs/collect_v0.yaml

任一关键项失败 -> 不要继续训练，先排查（§17）。
"""
import argparse
import sys

import numpy as np
import yaml

from .dataset import discover_episodes, load_episode


def check(condition, msg, failures):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {msg}")
    if not condition:
        failures.append(msg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    eps = discover_episodes(cfg["output"]["dataset_dir"])
    print(f"[sanity] 共 {len(eps)} episodes")
    assert eps, "没有找到 episode，先运行 collector"
    failures = []

    n_check = min(50, len(eps))
    rng = np.random.default_rng(0)
    for e in rng.choice(eps, n_check, replace=False):
        d = load_episode(e["path"])
        m = e["meta"]
        tag = f"ep{m['episode_id']}({m['condition']})"
        # 2. timestamp 单调
        check(np.all(np.diff(d["timestamp"]) >= 0), f"{tag}: timestamp 单调", failures)
        # 3. residual == execution - command
        r = d["execution"] - d["cmd_vel"]
        check(np.allclose(r, d["residual"], atol=1e-4), f"{tag}: residual == actual - cmd", failures)
        # 7. NaN / Inf
        ok = all(np.isfinite(d[k]).all() for k in d)
        check(ok, f"{tag}: 无 NaN/Inf", failures)
        # 1. cmd 与 actual 同 frame：|actual| 应在命令量级的合理倍数内（粗略检查）
        spd = np.linalg.norm(d["execution"][:, :2], axis=-1).max()
        check(spd < 5.0, f"{tag}: velocity 单位合理 (max {spd:.2f} m/s)", failures)
        # 4. contact 合理：大部分时间在站立/行走，4 足接触不全为 0
        frac_contact = (d["feet_contact"].sum(axis=-1) > 0).mean()
        check(frac_contact > 0.7, f"{tag}: contact 合理 ({frac_contact:.2f})", failures)
        # 6. episode boundary：时长与 metadata 一致
        dur = d["timestamp"][-1] - d["timestamp"][0]
        check(abs(dur - m["duration_s"]) < 1.0, f"{tag}: episode 边界一致", failures)
        # 8. metadata 正确
        check(m["friction"] is not None and m["actuator_scale"] is not None,
              f"{tag}: perturbation metadata 存在", failures)

    # 1(frame) 语义检查：normal 条件下 vx_actual 与 vx_cmd 应强相关（同 frame 才会高）
    normal_exec, normal_cmd = [], []
    for e in eps:
        if e["meta"]["condition"] == "normal" and e["meta"]["episode_type"] == "random":
            d = load_episode(e["path"])
            normal_exec.append(d["execution"][:, 0])
            normal_cmd.append(d["cmd_vel"][:, 0])
    if normal_exec:
        ex = np.concatenate(normal_exec)
        cm = np.concatenate(normal_cmd)
        corr = np.corrcoef(ex, cm)[0, 1]
        check(corr > 0.7, f"cmd/actual 同 frame (vx corr={corr:.2f})", failures)

    # 9. friction 扰动必须造成 measurable mismatch（§17.9）
    def mean_abs_resid(cond):
        vals = [np.abs(load_episode(e["path"])["residual"][:, 0]).mean()
                for e in eps if e["meta"]["condition"] == cond and e["meta"]["episode_type"] == "random"]
        return float(np.mean(vals)) if vals else None

    base = mean_abs_resid("normal")
    low = mean_abs_resid("friction_low")
    if base and low:
        inc = (low - base) / base
        thr = cfg.get("sanity", {}).get("min_residual_increase_frac", 0.2)
        check(inc > thr,
              f"friction 造成 measurable mismatch: normal |r_vx|={base:.4f}, "
              f"low={low:.4f}, +{inc * 100:.1f}% (要求 >{thr * 100:.0f}%)", failures)

    print()
    if failures:
        print(f"[sanity] ✗ {len(failures)} 项失败，先排查再训练（§17）")
        sys.exit(1)
    print("[sanity] ✓ 全部通过，可以进入训练阶段")


if __name__ == "__main__":
    main()
