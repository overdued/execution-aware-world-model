"""V0.6.2 Task 5 — Small Deployable-input Ablation（secondary）。

完全相同 split / window / 协议下训练 M0 与 M1 × 3 seeds × 2 组输入：

  privileged : 现有输入（含 simulator GT body 速度/角速度，schema [0:6]）
  deployable : **不使用** simulator GT velocity/orientation；只保留真机可获得的
               projected_gravity / IMU 线加速度 / joint pos+vel / feet contact。
               schema 中没有可部署 odom -> 不伪造替代，直接把这些通道置零并在
               manifest 中声明 `deployable_substitute: none`。

不做 condition classifier，不使用 GT friction（输入里本来也没有）。

用法:
  python -m execution_wm.validity_v062.task5_deployable --train --input-mode deployable
  python -m execution_wm.validity_v062.task5_deployable --eval
输出: metrics/deployable_gap.csv, metrics/deployable_train_logs.json
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from execution_wm.train.train_execution import MODEL_REGISTRY
from execution_wm.validity_v061.schema import PROPRIO_SCHEMA
from execution_wm.validity_v061.windows import H, WindowManifest
from execution_wm.validity_v062.common import (CKPT_V062, OUT, SPLITS_EVAL, get_manifests,
                                                load_cache, pred_from_cache, seed_mean_pred)

PRIV_OFFSETS = (0, 3)          # base_linear_velocity_body, base_angular_velocity
MCONFIG = {"hidden_dims": [256, 256], "gru_hidden": 128, "context_dim": 8,
           "predict_uncertainty": False}
SEEDS = (42, 43, 44)
MODELS = ("M0", "M1")


def apply_input_mode(hp, mode):
    """hp [N,L,40]（torch 或 numpy）-> 屏蔽 privileged 通道。"""
    if mode == "privileged":
        return hp
    if mode != "deployable":
        raise ValueError(mode)
    out = hp.copy() if isinstance(hp, np.ndarray) else hp.clone()
    # 明确按 schema 字段名屏蔽（不用魔法索引）
    for name in ("base_linear_velocity_body", "base_angular_velocity"):
        f = PROPRIO_SCHEMA.field(name)
        out[..., f.offset:f.offset + f.size] = 0.0
    return out


def forward(mname, model, hp, ha, fa):
    if mname == "M0":
        return model(hp, ha, fa)["r_hat"]
    c = model.encode_context(hp, ha)
    return model.predict_execution(model.encode_state(hp), c, fa)["r_hat"]


def train_one(mname, seed, mode, man_tr, man_va, device, max_epochs=100):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    base = "direct" if mname == "M0" else "context"
    model = MODEL_REGISTRY[base](H, MCONFIG).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lf = nn.SmoothL1Loss(beta=1.0)
    bs = 128
    hp_tr = apply_input_mode(man_tr.hp, mode)
    ha_tr = man_tr.ha
    hp_va = apply_input_mode(man_va.hp, mode)
    out_dir = os.path.join(CKPT_V062, f"{mname}_{mode}_s{seed}")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    best, best_ep, pat = float("inf"), -1, 0
    ep = -1
    for ep in range(max_epochs):
        model.train()
        idx = rng.permutation(len(man_tr))
        for i in range(0, len(idx), bs):
            b = np.sort(idx[i:i + bs])
            hp = torch.from_numpy(hp_tr[b]).to(device)
            ha = torch.from_numpy(ha_tr[b]).to(device)
            fa = torch.from_numpy(man_tr.fa[b]).to(device)
            fr = torch.from_numpy(man_tr.fr[b]).to(device)
            loss = lf(forward(mname, model, hp, ha, fa), fr)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        vtot = 0.0
        with torch.no_grad():
            for i in range(0, len(man_va), bs):
                b = np.arange(i, min(i + bs, len(man_va)))
                hp = torch.from_numpy(hp_va[b]).to(device)
                ha = torch.from_numpy(man_va.ha[b]).to(device)
                fa = torch.from_numpy(man_va.fa[b]).to(device)
                fr = torch.from_numpy(man_va.fr[b]).to(device)
                vtot += lf(forward(mname, model, hp, ha, fa), fr).item() * len(b)
        v = vtot / len(man_va)
        if v < best - 1e-5:
            best, best_ep, pat = v, ep, 0
            torch.save({"model_state": model.state_dict(), "config": {"model": MCONFIG},
                        "model_name": mname, "seed": seed, "input_mode": mode,
                        "val_loss": best, "window_manifest_hash": man_tr.manifest_hash(),
                        "deployable_substitute": "none" if mode == "deployable" else None,
                        "masked_fields": [f.name for f in PROPRIO_SCHEMA.fields[:2]]
                        if mode == "deployable" else []},
                       os.path.join(out_dir, "best.pt"))
        else:
            pat += 1
        if pat >= 15:
            break
    info = {"model": mname, "seed": seed, "input_mode": mode, "best_val": best,
            "best_epoch": best_ep, "epochs": ep + 1,
            "gpu_hours": (time.time() - t0) / 3600,
            "window_manifest_hash": man_tr.manifest_hash(),
            "n_train_windows": len(man_tr), "n_val_windows": len(man_va)}
    json.dump(info, open(os.path.join(out_dir, "train_log.json"), "w"), indent=1)
    print(f"[v062-train] {mname}/{mode}/s{seed} val={best:.6f} ep={best_ep} "
          f"gpu_h={info['gpu_hours']:.4f}", flush=True)
    return info


def load_v062(mname, mode, seed, device):
    p = os.path.join(CKPT_V062, f"{mname}_{mode}_s{seed}", "best.pt")
    if not os.path.exists(p):
        return None
    ck = torch.load(p, weights_only=False, map_location=device)
    base = "direct" if mname == "M0" else "context"
    m = MODEL_REGISTRY[base](H, ck["config"]["model"]).to(device)
    m.load_state_dict(ck["model_state"]); m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--input-mode", default="deployable",
                    choices=["privileged", "deployable"])
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    man = get_manifests()

    if args.train:
        logs = []
        for mname in MODELS:
            for seed in SEEDS:
                logs.append(train_one(mname, seed, args.input_mode, man["train"],
                                      man["val"], device))
        p = os.path.join(OUT, "metrics", "deployable_train_logs.json")
        old = json.load(open(p)) if os.path.exists(p) else []
        json.dump(old + logs, open(p, "w"), indent=1)
        return

    # ---------------- eval ----------------
    cache = load_cache()
    rows = []
    for split in SPLITS_EVAL:
        m = man[split]
        for mname in MODELS:
            for mode in ("privileged", "deployable"):
                preds = []
                for seed in SEEDS:
                    mm = load_v062(mname, mode, seed, device)
                    if mm is None:
                        continue
                    hp = torch.from_numpy(apply_input_mode(m.hp, mode)).to(device)
                    ha = torch.from_numpy(m.ha).to(device)
                    fa = torch.from_numpy(m.fa).to(device)
                    with torch.no_grad():
                        preds.append((fa + forward(mname, mm, hp, ha, fa)).cpu().numpy())
                if not preds:
                    continue
                pr = np.asarray(np.mean(preds, axis=0))
                gt = np.asarray(m.fe)
                row = {"split": split, "model": mname, "input_mode": mode,
                       "n_windows": len(m), "n_seeds": len(preds)}
                assert pr.ndim == 3 and gt.ndim == 3, (pr.shape, gt.shape)
                for i, ax in enumerate(("vx", "vy", "wz")):
                    for k, lead in ((5, "0.25s"), (10, "0.5s"), (20, "1.0s"), (40, "2.0s")):
                        row[f"MAE_{ax}@{lead}"] = float(
                            np.abs(pr[:, k - 1, i] - gt[:, k - 1, i]).mean())
                    row[f"MAE_{ax}_horizon"] = float(np.abs(pr[:, :, i] - gt[:, :, i]).mean())
                row["traj_dxy_err_m"] = float(np.abs(
                    np.cumsum(pr, axis=1)[:, -1, :2] * 0.05 -
                    np.cumsum(gt, axis=1)[:, -1, :2] * 0.05).mean())
                # 参照：现有 V0.6.1 privileged 模型（同一 split/窗口）
                rows.append(row)
        # 参照行：V0.6.1 的 M0/M1 privileged（独立训练，用于交叉核对）
        for mname in MODELS:
            pr = seed_mean_pred(cache, split, mname)
            row = {"split": split, "model": mname, "input_mode": "privileged_v061_ref",
                   "n_windows": len(m), "n_seeds": 3}
            for i, ax in enumerate(("vx", "vy", "wz")):
                for k, lead in ((5, "0.25s"), (10, "0.5s"), (20, "1.0s"), (40, "2.0s")):
                    row[f"MAE_{ax}@{lead}"] = float(
                        np.abs(pr[:, k - 1, i] - gt[:, k - 1, i]).mean())
                row[f"MAE_{ax}_horizon"] = float(np.abs(pr[:, :, i] - gt[:, :, i]).mean())
            row["traj_dxy_err_m"] = float(np.abs(
                np.cumsum(pr, axis=1)[:, -1, :2] * 0.05 -
                np.cumsum(gt, axis=1)[:, -1, :2] * 0.05).mean())
            rows.append(row)
        print(f"[v062-eval] {split} done")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "metrics", "deployable_metrics.csv"), index=False)

    # deployable_gap = deployable − privileged（同 model/split，3-seed 均值）
    gap = []
    for split in SPLITS_EVAL:
        for mname in MODELS:
            s = df[(df.split == split) & (df.model == mname)]
            pv = s[s.input_mode == "privileged"]
            dv = s[s.input_mode == "deployable"]
            if len(pv) == 0 or len(dv) == 0:
                continue
            pv, dv = pv.iloc[0], dv.iloc[0]
            r = {"split": split, "model": mname, "n_seeds": int(dv["n_seeds"])}
            for ax in ("vx", "vy", "wz"):
                for lead in ("0.25s", "0.5s", "1.0s", "2.0s"):
                    col = f"MAE_{ax}@{lead}"
                    r[f"priv_{col}"] = float(pv[col])
                    r[f"depl_{col}"] = float(dv[col])
                    r[f"gap_{col}"] = float(dv[col] - pv[col])
                    r[f"gap_pct_{col}"] = float(100 * (dv[col] - pv[col]) / max(pv[col], 1e-9))
            r["priv_traj_dxy_err_m"] = float(pv["traj_dxy_err_m"])
            r["depl_traj_dxy_err_m"] = float(dv["traj_dxy_err_m"])
            r["gap_traj_dxy_err_m"] = float(dv["traj_dxy_err_m"] - pv["traj_dxy_err_m"])
            r["gap_pct_traj"] = float(100 * (dv["traj_dxy_err_m"] - pv["traj_dxy_err_m"]) /
                                      max(pv["traj_dxy_err_m"], 1e-9))
            gap.append(r)
    g = pd.DataFrame(gap)
    g.to_csv(os.path.join(OUT, "metrics", "deployable_gap.csv"), index=False)

    # ---- 配对 episode-cluster bootstrap：gap 是否显著非零 ----
    rng = np.random.default_rng(20260923)
    brows = []
    for split in SPLITS_EVAL:
        m = man[split]
        eps = np.array([w["episode_id"] for w in m.windows])
        units = np.unique(eps); cid = np.searchsorted(units, eps)
        gt = np.asarray(m.fe)
        for mname in MODELS:
            def errs(mode):
                acc = []
                for seed in SEEDS:
                    mm = load_v062(mname, mode, seed, device)
                    if mm is None:
                        return None
                    hp = torch.from_numpy(apply_input_mode(m.hp, mode)).to(device)
                    ha = torch.from_numpy(m.ha).to(device)
                    fa = torch.from_numpy(m.fa).to(device)
                    with torch.no_grad():
                        acc.append(np.abs((fa + forward(mname, mm, hp, ha, fa)
                                           ).cpu().numpy() - gt).mean(axis=(1, 2)))
                return np.mean(acc, axis=0)
            ep_, ed_ = errs("privileged"), errs("deployable")
            if ep_ is None or ed_ is None:
                continue
            d = ed_ - ep_
            out = np.empty(2000)
            for b in range(2000):
                cnt = np.bincount(rng.integers(0, len(units), len(units)), minlength=len(units))
                w = cnt[cid]
                out[b] = (d * w).sum() / max(w.sum(), 1)
            brows.append({"split": split, "model": mname, "gap_mean": float(d.mean()),
                          "ci_low": float(np.percentile(out, 2.5)),
                          "ci_high": float(np.percentile(out, 97.5)),
                          "crosses_zero": bool(np.percentile(out, 2.5) < 0 <
                                               np.percentile(out, 97.5)),
                          "n_clusters": int(len(units)), "scope": "EXPLORATORY"})
    bdf = pd.DataFrame(brows)
    bdf.to_csv(os.path.join(OUT, "metrics", "deployable_gap_bootstrap.csv"), index=False)
    print("\n=== deployable gap 配对 bootstrap（负=deployable 更好）===")
    print(bdf.round(4).to_string(index=False))
    pd.set_option("display.width", 260)
    print("\n=== deployable gap（gap>0 = 去掉 privileged 后变差）===")
    cols = ["split", "model", "priv_MAE_vx@2.0s", "depl_MAE_vx@2.0s", "gap_MAE_vx@2.0s",
            "gap_pct_MAE_vx@2.0s", "gap_pct_MAE_vy@2.0s", "gap_pct_MAE_wz@2.0s",
            "gap_pct_traj"]
    print(g[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
