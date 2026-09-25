"""V0.7: manifests（episodes / windows / command_coverage）+ termination_and_masks。

输出到 results/v0_7_composition/manifests/ 与 metrics/。
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd

from execution_wm.composition_v07 import cells as C
from execution_wm.composition_v07.data_v07 import build_datasets, load_index

ROOT = os.environ.get("V07_DATA", "/media/hdd1/yuhang/datasets/execution_wm/v0_7")
OUT = os.environ.get("V07_OUT", "results/v0_7_composition")


def sha256(p, n=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(n)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    os.makedirs(f"{OUT}/manifests", exist_ok=True)
    idx = json.load(open(os.path.join(ROOT, "index.json")))
    eps = idx["episodes"]
    # ---- episodes.csv ----
    rows = []
    for e in eps:
        p20 = e["file"].replace(".npz", ".20hz.npz")
        rows.append({
            "episode_id": e["episode_id"], "group_id": e["anchor_group_id"],
            "split": e["split"], "condition": e["condition"],
            "coverage_regime": e["coverage_regime"], "script_id": e["script_id"],
            "timing_template": e["timing_template"], "reset_seed": e["reset_seed"],
            "duration_s": e["duration_s"],
            "steps_50hz": e.get("steps_50hz",
                                int(round(e["duration_s"] / 0.02))),
            "termination_reason": e["termination_reason"],
            "n_cells": len(e["command_cell_ids"]),
            "cell_types": "|".join(sorted({C.cell_type(c) for c in e["command_cell_ids"]})),
            "friction_written_static": e["friction_written_static"],
            "friction_written_dynamic": e["friction_written_dynamic"],
            "raw_sha256": sha256(e["file"]),
            "derived20hz_sha256": sha256(p20) if os.path.exists(p20) else "MISSING",
            "raw_path": e["file"].replace("/media/hdd1/yuhang", "<HDD>"),
        })
    ep_df = pd.DataFrame(rows)
    ep_df.to_csv(f"{OUT}/manifests/episodes.csv", index=False)

    # ---- windows.csv ----
    ds = build_datasets(ROOT)
    wrows = []
    for name in ("train_R0", "train_R1", "val", "test_all", "test_P0", "test_P1", "test_P2"):
        for w in ds[name].lineage_rows():
            wrows.append({"dataset": name, **w})
    pd.DataFrame(wrows).to_csv(f"{OUT}/manifests/windows.csv", index=False)

    # ---- command_coverage.csv ----
    cov = []
    for name in ("train_R0", "train_R1", "val", "test_all"):
        d = ds[name]
        cnt = {}
        for w in d.windows:
            cnt[w["dominant_cell"]] = cnt.get(w["dominant_cell"], 0) + 1
        for cid, k in sorted(cnt.items()):
            cov.append({"dataset": name, "cell_id": cid, "cell_type": C.cell_type(cid),
                        "values": C.cell_values(cid), "n_windows": k,
                        "split_of_cell": _cell_split(cid)})
    pd.DataFrame(cov).to_csv(f"{OUT}/manifests/command_coverage.csv", index=False)

    # ---- termination_and_masks.csv ----
    mrows = []
    for name in ("train_R0", "train_R1", "val", "test_P0", "test_P1", "test_P2"):
        d = ds[name]
        pur = np.array([w["future_cell_purity"] for w in d.windows]) if len(d) else np.array([])
        mrows.append({"dataset": name, "n_windows": len(d),
                      "n_groups": len({w["group_id"] for w in d.windows}),
                      "n_episodes": len({w["episode_id"] for w in d.windows}),
                      "median_future_cell_purity": float(np.median(pur)) if len(pur) else None,
                      "frac_purity_ge_0.8": float((pur >= 0.8).mean()) if len(pur) else None,
                      "valid_prefix_mask": "所有窗口 full-H 有效（无终止被截断）；"
                                           "终止样本原地保留",
                      "n_terminated_episodes": int(sum(
                          1 for e in eps if e["termination_reason"] != "schedule_end")),
                      "same_window_set_for_all_models": True})
    pd.DataFrame(mrows).to_csv(f"{OUT}/metrics/termination_and_masks.csv", index=False)

    json.dump({"plan_hash": idx["plan_hash"], "controller_hash": idx["controller_hash"],
               "data_version": idx["data_version"], "n_episodes": len(eps),
               "index_sha256": sha256(os.path.join(ROOT, "index.json")),
               "smoke": idx.get("smoke", False)},
              open(f"{OUT}/manifests/data_manifest.json", "w"), indent=1)
    print(f"episodes {len(ep_df)}, windows {len(wrows)}, coverage {len(cov)}")
    print(ep_df.groupby(["split", "coverage_regime"]).size().to_string())
    print("\nwindows per dataset:")
    print(pd.DataFrame(mrows)[["dataset", "n_windows", "n_groups", "n_episodes"]].to_string(index=False))


def _cell_split(cid):
    from execution_wm.composition_v07.cells import split_doubles, pick_triples
    d = split_doubles()
    t = pick_triples(d)
    for k, v in d.items():
        if cid in v["train"]:
            return f"{k}_train"
        if cid in v["val"]:
            return f"{k}_val"
        if cid in v["test"]:
            return f"{k}_test"
    if cid in t["val"]:
        return "triple_val"
    if cid in t["test"]:
        return "triple_test"
    return "single_or_zero" if C.cell_type(cid) in ("single", "zero") else "unused"


if __name__ == "__main__":
    main()
