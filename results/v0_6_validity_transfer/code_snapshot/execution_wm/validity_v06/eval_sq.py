"""V0.6 阶段C 评估：模型对照（3 seeds）+ support/query 迁移 + ARX 基线。

测试集（全部只评价）:
  held-out anchor: AQ6/AQ7 的 Q1/Q2/Q3 窗口
  held-out family: 全部 anchor 的 Q4_combo 窗口
分析:
  1. M0/M1/M2/M3 × 3 seeds + command-copy + persistence + ARX(ridge, train-only)
     per-axis MAE @ leads {0.25,0.5,1,2}s，按 condition/family 分组
  2. support/query 迁移（M1/M2 结构）:
     native c（自身 history）/ cross-support c（同 condition 不同命令）/
     diff-condition support c / zero c / shuffle c / state-only(c=0 且去掉历史? 见下) /
     k=0 连续性（首步预测 vs 当前状态）
  3. two-way bootstrap（support_group × anchor_group, 2000 次）检验关键差值

输出:
  metrics/model_comparison_3seeds.csv
  metrics/support_query_transfer.csv

用法: python -m execution_wm.validity_v06.eval_sq --config execution_wm/configs/v06_validity.yaml
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import yaml

from execution_wm.train.train_execution import MODEL_REGISTRY
from execution_wm.validity_v06.train_sq import SQWindowData, SupportWindowBank, discover_sq, L, H

VEL = ["vx", "vy", "wz"]
LEADS = {"0.25s": 5, "0.5s": 10, "1s": 20, "2s": 40}
OUT_ROOT = "/media/hdd1/yuhang/checkpoints/execution_wm/v0_6"


def load_trained(name, seed, device):
    ckpt = torch.load(os.path.join(OUT_ROOT, f"{name}_s{seed}", "best.pt"),
                      weights_only=False, map_location=device)
    base = "direct" if name == "M0" else "context"
    model = MODEL_REGISTRY[base](H, ckpt["config"]["model"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    aux = None
    if ckpt.get("aux_state") is not None:
        aux = torch.nn.Linear(1, 8).to(device)
        aux.load_state_dict(ckpt["aux_state"])
        aux.eval()
    return model, aux


def predict_with(model_name, model, aux, hp, ha, fa, fric, c=None):
    with torch.no_grad():
        if model_name == "M0":
            r = model(hp, ha, fa)["r_hat"]
        else:
            if c is None:
                c = model.encode_context(hp, ha)
            if model_name == "M3":
                c = c + aux(fric.reshape(-1, 1))
            z = model.encode_state(hp)
            r = model.predict_execution(z, c, fa)["r_hat"]
    return fa + r


def fit_arx(train_data, device):
    """ridge: [last proprio(40), future cmd flat(120)] -> residual flat(120)，train-only。"""
    n = len(train_data)
    idx = np.arange(n)
    hp, ha, fa, fr, _, _, _, _ = train_data.batch(idx, "cpu")
    X = np.concatenate([hp[:, -1, :].numpy(), fa.reshape(n, -1).numpy()], axis=1)
    Y = fr.reshape(n, -1).numpy()
    Xb = np.concatenate([X, np.ones((n, 1))], axis=1)
    lam = 1.0
    W = np.linalg.solve(Xb.T @ Xb + lam * np.eye(Xb.shape[1]), Xb.T @ Y)
    return W


def arx_predict(W, hp, fa):
    n = len(hp)
    X = np.concatenate([hp[:, -1, :].cpu().numpy(), fa.reshape(n, -1).cpu().numpy()], axis=1)
    Xb = np.concatenate([X, np.ones((n, 1))], axis=1)
    return fa.cpu().numpy() + (Xb @ W).reshape(n, H, 3)


def mae_rows(pred, gt, names):
    err = np.abs(pred - gt)
    out = {}
    for lname, k in LEADS.items():
        for i, n in enumerate(VEL):
            out[f"MAE_{n}@{lname}"] = float(err[:, :k, i].mean())
    out["MAE_all@2s"] = float(err.mean())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_m = os.path.join(cfg["out_dir"], "metrics")
    os.makedirs(out_m, exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])

    sq_cfg = yaml.safe_load(open("execution_wm/configs/collect_v06_sq.yaml"))
    roles = discover_sq(sq_cfg["split"])
    held_anchor = [e for e in roles["test"] if int(e["anchor_group"][2:]) in
                   sq_cfg["split"]["test_anchors"]]
    held_family = [e for e in roles["test"] if e.get("held_out_family")]
    print(f"[eval] held-anchor eps={len(held_anchor)} held-family eps={len(held_family)}")

    test_anchor = SQWindowData(held_anchor, windows_per_ep=8, seed=1)
    test_family = SQWindowData(held_family, windows_per_ep=8, seed=1)
    train_data = SQWindowData(roles["train"] + roles["support"], seed=0)
    bank = SupportWindowBank(roles["support"])

    # ---------- 1. 模型对照 ----------
    W_arx = fit_arx(train_data, device)
    rows = []
    for split_name, data in (("held_anchor", test_anchor), ("held_family", test_family)):
        n = len(data)
        idx = np.arange(n)
        hp, ha, fa, fr_lab, fric, conds, epids, anchors = data.batch(idx, device)
        gt = fa + fr_lab
        base_preds = {
            "command-copy": fa.cpu().numpy(),
            "persistence": np.repeat(hp[:, -1, :3].cpu().numpy()[:, None, :], H, axis=1),
            "ARX_ridge": arx_predict(W_arx, hp, fa),
        }
        for mname, pr in base_preds.items():
            for cond in sorted(set(conds)):
                m = np.array([c == cond for c in conds])
                r = {"split": split_name, "model": mname, "seed": "-", "condition": cond,
                     "n_windows": int(m.sum())}
                r.update(mae_rows(pr[m], gt.cpu().numpy()[m], VEL))
                rows.append(r)
        for mname in ("M0", "M1", "M2", "M3"):
            for seed in (42, 43, 44):
                path = os.path.join(OUT_ROOT, f"{mname}_s{seed}", "best.pt")
                if not os.path.exists(path):
                    print(f"[eval] missing {mname}_s{seed}, skip")
                    continue
                model, aux = load_trained(mname, seed, device)
                if mname == "M2":
                    shp, sha = bank.sample_batch(conds, rng, device)
                    c = model.encode_context(shp, sha)
                    pr = predict_with(mname, model, aux, hp, ha, fa, fric, c=c).cpu().numpy()
                else:
                    pr = predict_with(mname, model, aux, hp, ha, fa, fric).cpu().numpy()
                for cond in sorted(set(conds)):
                    m = np.array([c2 == cond for c2 in conds])
                    r = {"split": split_name, "model": mname, "seed": seed,
                         "condition": cond, "n_windows": int(m.sum())}
                    r.update(mae_rows(pr[m], gt.cpu().numpy()[m], VEL))
                    rows.append(r)
        print(f"[eval] {split_name} done")
    comp = pd.DataFrame(rows)
    comp.to_csv(os.path.join(out_m, "model_comparison_3seeds.csv"), index=False)
    pd.set_option("display.width", 220)
    key = ["MAE_vx@1s", "MAE_vy@1s", "MAE_wz@1s", "MAE_all@2s"]
    print(comp.groupby(["split", "model"])[key].mean().round(4).to_string())

    # ---------- 2. support/query 迁移（M1 结构，seed=42） ----------
    model, _ = load_trained("M1", 42, device)
    data = test_anchor
    n = len(data)
    hp, ha, fa, fr_lab, fric, conds, epids, anchors = data.batch(np.arange(n), device)
    gt = (fa + fr_lab).cpu().numpy()
    variants = {}
    variants["native_c"] = predict_with("M1", model, None, hp, ha, fa, fric).cpu().numpy()
    # cross-support c（同 condition 不同命令）
    shp, sha = bank.sample_batch(conds, rng, device)
    variants["cross_support_c"] = predict_with(
        "M1", model, None, hp, ha, fa, fric, c=model.encode_context(shp, sha)).cpu().numpy()
    # diff-condition support c
    cond_map = {"normal": "friction_low", "friction_mid": "normal", "friction_low": "normal"}
    dconds = [cond_map.get(c, "normal") for c in conds]
    shp2, sha2 = bank.sample_batch(dconds, rng, device)
    variants["diff_condition_c"] = predict_with(
        "M1", model, None, hp, ha, fa, fric, c=model.encode_context(shp2, sha2)).cpu().numpy()
    # zero / shuffle c
    variants["zero_c"] = predict_with(
        "M1", model, None, hp, ha, fa, fric,
        c=torch.zeros(n, 8, device=device)).cpu().numpy()
    c_self = model.encode_context(hp, ha)
    variants["shuffle_c"] = predict_with(
        "M1", model, None, hp, ha, fa, fric,
        c=c_self[torch.randperm(n, device=device)]).cpu().numpy()
    tr_rows = []
    for vname, pr in variants.items():
        err = np.abs(pr - gt)
        row = {"variant": vname, "n_windows": n}
        for lname, k in LEADS.items():
            for i, vn in enumerate(VEL):
                row[f"MAE_{vn}@{lname}"] = float(err[:, :k, i].mean())
        row["MAE_all@2s"] = float(err.mean())
        tr_rows.append(row)
    tr = pd.DataFrame(tr_rows)
    tr.to_csv(os.path.join(out_m, "support_query_transfer.csv"), index=False)
    print("\n=== support/query transfer (M1, held-anchor) ===")
    print(tr[["variant", "MAE_vx@1s", "MAE_wz@1s", "MAE_all@2s"]].round(4).to_string(index=False))

    # k=0 连续性：首步预测与 persistence 的差距
    first_step = {v: float(np.abs(pr[:, 0, :] - hp[:, -1, :3].cpu().numpy()).mean())
                  for v, pr in variants.items()}
    print("k=0 |pred[0]-e[origin]| per variant:", {k: round(v, 4) for k, v in first_step.items()})
    with open(os.path.join(out_m, "k0_continuity.json"), "w") as f:
        json.dump(first_step, f, indent=1)
    print(f"[eval] -> {out_m}")


if __name__ == "__main__":
    main()
