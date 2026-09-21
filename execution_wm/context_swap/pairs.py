"""Step 4: 构造 matched pairs（§5）。

同一 probe、同相对时刻、future command 一致、物理条件不同（cross）或相同（same，
用于 §8 control）。输出:
    pairs/pair_quality.csv    全部候选窗 + 质量指标（阈值依据）
    pairs/pair_manifest.csv   通过阈值的 pairs
    pairs/rejected_pairs.csv  被阈值拒绝的

用法: python -m execution_wm.context_swap.pairs --config execution_wm/configs/context_swap.yaml
"""
import argparse
import csv
import os

import numpy as np

from execution_wm.context_swap.common import EpisodeData, discover_all, load_cfg, out_subdir

# proprio 40 维内各组偏移（与 PROPRIO_KEYS 顺序一致）
SL = {"lin_vel": slice(0, 3), "ang_vel": slice(3, 6), "gravity": slice(6, 9),
      "joint_pos": slice(12, 24)}


def state_distance(pa, pb, weights):
    parts = {
        "lin_vel": float(np.linalg.norm(pa[SL["lin_vel"]] - pb[SL["lin_vel"]])),
        "ang_vel": float(np.linalg.norm(pa[SL["ang_vel"]] - pb[SL["ang_vel"]])),
        "gravity": float(np.linalg.norm(pa[SL["gravity"]] - pb[SL["gravity"]])),
        "joint_pos": float(np.linalg.norm(pa[SL["joint_pos"]] - pb[SL["joint_pos"]])),
    }
    d = sum(weights[k] * parts[k] for k in parts)
    return d, parts


def build_pairs(cfg):
    """{(probe_name, cond_group): [(uid, EpisodeData)]}，pool 全部 dataset_dirs。"""
    by_key = {}
    for uid, e in discover_all(cfg).items():
        m = e["meta"]
        by_key.setdefault((m["probe_name"], m["cond_group"]), []).append((uid, EpisodeData(e)))
    return by_key


def iter_windows(ep_a, ep_b, L, H, stride):
    """同相对时刻窗口：t0 从 L-1 起，两 episode 内都要装得下 future。"""
    t_max = min(ep_a.T, ep_b.T) - H - 1
    for t0 in range(L - 1, t_max + 1, stride):
        yield t0


def pair_rows(cfg, kind, cond_a, cond_b, by_key):
    """kind: 'cross'（不同条件）或 'same'（同条件不同 episode）。"""
    pc = cfg["pairs"]
    L = int(round(cfg["history_s"] * cfg["hz"]))
    H = int(round(cfg["horizon_s"] * cfg["hz"]))
    stride = int(round(pc["time_stride_s"] * cfg["hz"]))
    rows = []
    probe_names = sorted({k[0] for k in by_key})
    for pname in probe_names:
        eps_a = by_key.get((pname, cond_a), [])
        eps_b = by_key.get((pname, cond_b), []) if kind == "cross" else eps_a
        for i, (uid_a, ea) in enumerate(eps_a):
            for j, (uid_b, eb) in enumerate(eps_b):
                if kind == "same" and j <= i:
                    continue  # 同条件配对去重，且排除自己配自己
                for t0 in iter_windows(ea, eb, L, H, stride):
                    wa = ea.window(t0, L, H)
                    wb = eb.window(t0, L, H)
                    cmd_mm = float(np.abs(wa["future_action"] - wb["future_action"]).max())
                    d_state, parts = state_distance(
                        ea.proprio[t0], eb.proprio[t0], pc["state_weights"])
                    # transition / steady（§11）：窗内最大命令变化
                    du = np.abs(np.diff(wa["future_action"], axis=0)).max()
                    rows.append({
                        "kind": kind,
                        "probe": pname,
                        "cond_A": cond_a, "cond_B": cond_b,
                        "ep_A": uid_a, "ep_B": uid_b,
                        "t0": t0, "t_rel_s": round(t0 / cfg["hz"], 3),
                        "cmd_mismatch": cmd_mm,
                        "D_state": d_state,
                        "d_lin_vel": parts["lin_vel"], "d_ang_vel": parts["ang_vel"],
                        "d_gravity": parts["gravity"], "d_joint_pos": parts["joint_pos"],
                        "window_type": "transition" if du > cfg["transition"]["tau_u"] else "steady",
                        "max_du": float(du),
                    })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_cfg(args.config)
    out = out_subdir(cfg, "pairs")

    by_key = build_pairs(cfg)
    print("[pairs] probe episodes:")
    for k in sorted(by_key):
        print(f"  {k[0]:24s} {k[1]:18s} n={len(by_key[k])}")

    rows = []
    for cond_a, cond_b in cfg["pairs"]["source_target"]:
        rows += pair_rows(cfg, "cross", cond_a, cond_b, by_key)
        rows += pair_rows(cfg, "cross", cond_b, cond_a, by_key)   # 双向（§12）
        rows += pair_rows(cfg, "same", cond_a, cond_a, by_key)    # §8 control
        rows += pair_rows(cfg, "same", cond_b, cond_b, by_key)

    max_mm = cfg["pairs"]["max_command_mismatch"]
    max_ds = cfg["pairs"]["max_state_distance"]
    accepted, rejected = [], []
    for r in rows:
        ok = r["cmd_mismatch"] <= max_mm and (max_ds is None or r["D_state"] <= max_ds)
        (accepted if ok else rejected).append(r)

    fields = list(rows[0].keys())
    for name, subset in (("pair_quality.csv", rows), ("pair_manifest.csv", accepted),
                         ("rejected_pairs.csv", rejected)):
        with open(os.path.join(out, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(subset)

    ds = np.array([r["D_state"] for r in rows if r["kind"] == "cross"])
    print(f"\n[pairs] candidates={len(rows)} accepted={len(accepted)} rejected={len(rejected)}")
    print(f"[pairs] cross D_state percentiles: "
          f"p10={np.percentile(ds,10):.3f} p25={np.percentile(ds,25):.3f} "
          f"p50={np.percentile(ds,50):.3f} p75={np.percentile(ds,75):.3f} "
          f"p90={np.percentile(ds,90):.3f}")
    print(f"[pairs] max cmd_mismatch={max(r['cmd_mismatch'] for r in rows):.4f}")
    from collections import Counter
    print("[pairs] accepted by kind/window:", Counter(
        (r["kind"], r["window_type"]) for r in accepted))
    if max_ds is None:
        print("[pairs] max_state_distance=null，未按 state 过滤。"
              "根据上面分布把阈值写进 config 后重跑本脚本。")


if __name__ == "__main__":
    main()
