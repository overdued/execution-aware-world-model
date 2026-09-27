"""V0.8 Stage C：同组候选区分与 motion head 敏感性（预注册 §5 冻结口径）。

- 候选匹配：test group 的匹配原点（eligible_for_matching，同 group 跨分支
  context 逐位相同）上，对 U0/U1/U2 分别预测 future，与各自真实 future latent
  比对：matching accuracy（argmin 命中率）与 paired difference 相对误差；
  同命令重复（U0 vs U0_repeat）真实 feature 距离 = 噪声底；候选间真实距离
  ≤1.5×噪声底的组标记 near-indistinguishable（单列，不删除）。
- 敏感性（V-EXEC）：固定全部输入，Ê / E=U / zero E / shuffled E 四档；
  true future E 仅 ORACLE_DIAGNOSTIC。

运行：
  python -m execution_wm.v08_visual.matching_v08 --results R --raw-root D --matching
  python -m execution_wm.v08_visual.matching_v08 --results R --raw-root D --sensitivity
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from execution_wm.v08_visual.dataset_v08 import V08Windows
from execution_wm.v08_visual.eval_v08 import RUNS, load_run


def _existing_runs(results):
    return [r for r in RUNS
            if (Path(results) / "checkpoints" / r / "summary.json").exists()]
from execution_wm.v08_visual.train_v08 import collate

NOISE_FLOOR_MULT = 1.5


def _matching_index(ds):
    """{group_id: {branch: dataset_row_index}}（仅匹配原点窗口）。"""
    idx = {}
    for i, r in ds.df.iterrows():
        if bool(r["eligible_for_matching"]):
            idx.setdefault(r["group_id"], {})[int(r["branch"])] = i
    return {g: b for g, b in idx.items() if set(b) >= {0, 1, 2, 3}}


def _norm_tgt(ds, wid, suffix="tgt2s"):
    m, sd = ds.norm[suffix]
    return ((ds.cache[f"{wid}/{suffix}"] - m) / sd).astype(np.float32).reshape(-1, 1024)


@torch.no_grad()
def matching(results, raw_root, device="cuda"):
    rr = Path(results)
    ds = V08Windows(rr, raw_root, "test")
    groups = _matching_index(ds)
    noise = {}
    for g, br in groups.items():
        z0 = _norm_tgt(ds, ds.df.iloc[br[0]]["window_id"])
        z3 = _norm_tgt(ds, ds.df.iloc[br[3]]["window_id"])
        z1 = _norm_tgt(ds, ds.df.iloc[br[1]]["window_id"])
        z2 = _norm_tgt(ds, ds.df.iloc[br[2]]["window_id"])
        d = lambda a, b: float(np.abs(a - b).mean())
        noise[g] = {"repeat_floor": d(z0, z3),
                    "cand_01": d(z0, z1), "cand_02": d(z0, z2),
                    "cand_12": d(z1, z2)}
        cands = [noise[g]["cand_01"], noise[g]["cand_02"], noise[g]["cand_12"]]
        noise[g]["near_indistinguishable"] = bool(
            min(cands) <= NOISE_FLOOR_MULT * max(noise[g]["repeat_floor"], 1e-9))
    rows = []
    for run in _existing_runs(rr):
        model, summ, _ = load_run(rr, run, device)
        for g, br in sorted(groups.items()):
            zs = [_norm_tgt(ds, ds.df.iloc[br[b]]["window_id"]) for b in (0, 1, 2)]
            zhats = []
            for b in (0, 1, 2):
                item = ds[br[b]]
                o = model(item["ctx"][None].to(device),
                          item["hist"][None].to(device),
                          item["fut"][None].to(device))
                zhats.append(o["pred_2s"][0].float().cpu().numpy())
            n_correct = 0
            for i in range(3):
                d = [float(np.abs(zhats[i] - zs[j]).mean()) for j in range(3)]
                n_correct += int(np.argmin(d) == i)
            pd_err = []
            for i, j in ((0, 1), (0, 2), (1, 2)):
                num = float(np.abs((zhats[i] - zhats[j]) - (zs[i] - zs[j])).mean())
                den = float(np.abs(zs[i] - zs[j]).mean()) + 1e-9
                pd_err.append(num / den)
            rows.append({"run_id": run, "group_id": g, "layout": g.split("_")[0],
                         "matching_acc": n_correct / 3,
                         "paired_diff_rel_err": float(np.mean(pd_err)),
                         **{k: noise[g][k] for k in
                            ("repeat_floor", "near_indistinguishable")}})
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(rr / "metrics" / "candidate_matching.csv", index=False)
    nf = pd.DataFrame([{"group_id": g, **v} for g, v in noise.items()])
    nf.to_csv(rr / "metrics" / "candidate_noise_floor.csv", index=False)
    summ = (df.groupby("run_id")[["matching_acc", "paired_diff_rel_err"]]
            .mean().round(4))
    print("[matching]\n", summ)
    print("near-indistinguishable groups:",
          [g for g, v in noise.items() if v["near_indistinguishable"]])


class _EOverride:
    """exec_head 输出替换 hook（eval 专用，不改模型文件）。"""

    def __init__(self, model, value):
        self.value = value
        self.h = model.exec_head.register_forward_hook(
            lambda mod, inp, out: self.value)

    def close(self):
        self.h.remove()


@torch.no_grad()
def sensitivity(results, raw_root, device="cuda"):
    rr = Path(results)
    ds = V08Windows(rr, raw_root, "test")
    groups = _matching_index(ds)
    rows = []
    for run in _existing_runs(rr):
        if not run.startswith("VEXEC"):
            continue
        model, summ, _ = load_run(rr, run, device)
        for g, br in sorted(groups.items()):
            item = ds[br[0]]
            ctx = item["ctx"][None].to(device)
            hist = item["hist"][None].to(device)
            fut = item["fut"][None].to(device)
            e_true = item["e"][None].to(device)
            tgt = item["tgt2s"][None].to(device).reshape(1, -1, 1024)
            base = model(ctx, hist, fut)
            e_hat = base["e_hat"]
            conds = {"E_hat": None,
                     "E_eq_U": (fut.reshape(1, -1) - torch.as_tensor(
                         ds.e_mean.astype(np.float32), device=device)) / torch.as_tensor(
                         ds.e_std.astype(np.float32), device=device),
                     "zero_E": torch.zeros_like(e_hat),
                     "oracle_E_diagnostic": e_true}
            for name, override in conds.items():
                if override is None:
                    o = base
                else:
                    h = _EOverride(model, override)
                    o = model(ctx, hist, fut)
                    h.close()
                err = float((o["pred_2s"] - tgt).abs().mean())
                rows.append({"run_id": run, "group_id": g, "condition": name,
                             "err_2s": err})
            # shuffled E：用同 group 另一分支的 e_hat
            it2 = ds[br[1]]
            o2 = model(it2["ctx"][None].to(device), it2["hist"][None].to(device),
                       it2["fut"][None].to(device))
            h = _EOverride(model, o2["e_hat"])
            o = model(ctx, hist, fut)
            h.close()
            rows.append({"run_id": run, "group_id": g, "condition": "shuffled_E",
                         "err_2s": float((o["pred_2s"] - tgt).abs().mean())})
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(rr / "metrics" / "exec_sensitivity.csv", index=False)
    print("[sensitivity]\n",
          df.groupby(["run_id", "condition"])["err_2s"].mean().round(4).unstack())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--matching", action="store_true")
    ap.add_argument("--sensitivity", action="store_true")
    args = ap.parse_args()
    if args.matching:
        matching(args.results, args.raw_root)
    if args.sensitivity:
        sensitivity(args.results, args.raw_root)


if __name__ == "__main__":
    main()
