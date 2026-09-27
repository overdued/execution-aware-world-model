"""V0.8 Stage C：评价与统计（预注册 §5/§6 冻结口径）。

子命令：
  --predict RUN_ID [--split val|test]   前向预测 -> predictions/<RUN_ID>_<split>_preds.npz
      （fp16 存储，逐窗血缘；指标在主进程 fp32 现算并写 metrics/predictions_per_window.csv）
  --metrics                             汇总 metrics/*.csv + stats JSON（paired group bootstrap）

主终点：test P1 2s spatial latent error（train-fit z 归一化空间 |Δ| mean over
16 t-tok×16 s-tok×1024 ch）→ 组内均值 → 16 个 test group 等权；按场景分列。
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from execution_wm.v08_visual import model_v08
from execution_wm.v08_visual.dataset_v08 import V08Windows
from execution_wm.v08_visual.train_v08 import collate

RUNS = [f"{v}_s{s}" for v in ("VDIRECT", "VAUX", "VEXEC") for s in (42, 43, 44)]
VARIANT_OF = {"VDIRECT": "V-DIRECT", "VAUX": "V-AUX", "VEXEC": "V-EXEC"}


def load_run(results, run_id, device):
    d = Path(results) / "checkpoints" / run_id
    summ = json.loads((d / "summary.json").read_text())
    variant = summ["variant"]
    model = model_v08.build(variant, seed=int(summ["seed"])).to(device).eval()
    ck = torch.load(d / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state"])
    return model, summ, ck


@torch.no_grad()
def predict(results, raw_root, run_id, split, device="cuda"):
    rr = Path(results)
    model, summ, _ = load_run(rr, run_id, device)
    ds = V08Windows(rr, raw_root, split)
    dl = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate)
    preds, rows = {}, []
    for b in dl:
        o = model(b["ctx"].to(device), b["hist"].to(device), b["fut"].to(device))
        t1 = b["tgt1s"].to(device).reshape(o["pred_1s"].shape)
        t2 = b["tgt2s"].to(device).reshape(o["pred_2s"].shape)
        e1 = (o["pred_1s"] - t1).abs().mean(dim=(1, 2)).cpu().numpy()
        e2 = (o["pred_2s"] - t2).abs().mean(dim=(1, 2)).cpu().numpy()
        pe = None
        if o["e_hat"] is not None:
            pe = (o["e_hat"] - b["e"].to(device)).abs()
            pe = pe.reshape(pe.shape[0], 40, 3).mean(dim=1).cpu().numpy()  # [B,3]
        for i, wid in enumerate(b["window_id"]):
            preds[f"{wid}/pred_1s"] = o["pred_1s"][i].half().cpu().numpy()
            preds[f"{wid}/pred_2s"] = o["pred_2s"][i].half().cpu().numpy()
            row = {"window_id": wid, "run_id": run_id, "split": split,
                   "group_id": b["group_id"][i], "level": b["level"][i],
                   "layout": b["layout"][i], "branch": int(b["branch"][i]),
                   "role": b["role"][i],
                   "eligible_for_matching": bool(b["eligible_for_matching"][i]),
                   "err_1s": float(e1[i]), "err_2s": float(e2[i])}
            if pe is not None:
                row.update({"e_err_vx": float(pe[i][0]),
                            "e_err_vy": float(pe[i][1]),
                            "e_err_wz": float(pe[i][2])})
            if o["e_hat"] is not None:
                preds[f"{wid}/e_hat"] = o["e_hat"][i].half().cpu().numpy()
            rows.append(row)
    out = rr / "predictions" / f"{run_id}_{split}_preds.npz"
    np.savez_compressed(out, **preds)
    import pandas as pd
    df = pd.DataFrame(rows)
    per_w = rr / "metrics" / f"predictions_per_window_{run_id}_{split}.csv"
    per_w.parent.mkdir(exist_ok=True)
    df.to_csv(per_w, index=False)
    print(f"[predict] {run_id}/{split}: {len(rows)} windows -> {out.name}")


def bootstrap_ci(group_vals_a, group_vals_b, n_boot=10000, seed=0):
    """paired group bootstrap：对 group 重抽，返回差值 (a-b) 的均值与 95% CI。"""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(group_vals_a), np.asarray(group_vals_b)
    n = len(a)
    idx = rng.integers(0, n, (n_boot, n))
    d = (a[idx].mean(axis=1) - b[idx].mean(axis=1))
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def metrics(results):
    import pandas as pd
    rr = Path(results)
    dfs = []
    for run in RUNS:
        for split in ("val", "test"):
            p = rr / "metrics" / f"predictions_per_window_{run}_{split}.csv"
            if p.exists():
                dfs.append(pd.read_csv(p))
    df = pd.concat(dfs, ignore_index=True)
    # 组级汇总（每 run × level × group：先窗内均值）
    g = (df.groupby(["run_id", "split", "level", "layout", "group_id"],
                    as_index=False)
           [["err_1s", "err_2s"]].mean())
    g.to_csv(rr / "metrics" / "visual_error_by_group_seed.csv", index=False)

    def _group_vals(run, level, err="err_2s"):
        sub = g[(g.run_id == run) & (g.split == "test") & (g.level == level)]
        return sub.set_index("group_id")[err]

    stats = {}
    for s in (42, 43, 44):
        a, b_ = _group_vals(f"VEXEC_s{s}", "P1"), _group_vals(f"VAUX_s{s}", "P1")
        common = a.index.intersection(b_.index)
        m, lo, hi = bootstrap_ci(b_[common].values, a[common].values)
        # 相对改善：(VAUX - VEXEC)/VAUX
        rel = float((b_[common].mean() - a[common].mean()) / b_[common].mean())
        stats[f"VEXEC_vs_VAUX_s{s}"] = {"abs_diff_mean": m, "ci95": [lo, hi],
                                        "rel_improvement": rel,
                                        "n_groups": len(common)}
        a2, b2 = _group_vals(f"VAUX_s{s}", "P1"), _group_vals(f"VDIRECT_s{s}", "P1")
        common2 = a2.index.intersection(b2.index)
        m2, lo2, hi2 = bootstrap_ci(b2[common2].values, a2[common2].values)
        rel2 = float((b2[common2].mean() - a2[common2].mean()) / b2[common2].mean())
        stats[f"VAUX_vs_VDIRECT_s{s}"] = {"abs_diff_mean": m2, "ci95": [lo2, hi2],
                                          "rel_improvement": rel2,
                                          "n_groups": len(common2)}
    (rr / "metrics" / "stats_primary.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--predict", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--metrics", action="store_true")
    args = ap.parse_args()
    if args.predict:
        predict(args.results, args.raw_root, args.predict, args.split)
    if args.metrics:
        metrics(args.results)


if __name__ == "__main__":
    main()
