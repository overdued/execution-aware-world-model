"""V0.7 Stage 4：单次推理 + 全量预测缓存 + 主指标表 + Δ_data / Δ_arch。

- 所有模型对所有测试窗口**只推理一次**，写 predictions/pred_cache.npz（非 object 数组）
  并记录 sha256；所有汇总/bootstrap 引用同一缓存。
- 主指标：2s FDE_xy / 2s ADE_xy / 净 yaw（圆周最短差）/ fixed-lead execution MAE。
- Δ_data = E(D,R0) − E(D,R1)；Δ_arch = E(D,R1) − E(I,R1)，分别报告。
"""
import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd
import torch

from execution_wm.composition_v07.data_v07 import H, build_datasets
from execution_wm.composition_v07.models_v07 import build
from execution_wm.composition_v07.train_v07 import ade_xy, fde_xy
from execution_wm.composition_v07.traj import integrate_xy, wrap_pi

OUT = "results/v0_7_composition"
CKPT = os.environ.get("V07_CKPT", "/media/hdd1/yuhang/checkpoints/execution_wm/v0_7")
CELLS = [("D", "R0"), ("D", "R1"), ("I", "R0"), ("I", "R1")]
SEEDS = (42, 43, 44)
LEADS = {"0.25s": 5, "0.5s": 10, "1.0s": 20, "2.0s": 40}
AXES = ("vx", "vy", "wz")
DT = 0.05


class RidgeMultiOutput:
    def __init__(self, lam=1.0):
        self.lam = lam

    def _x(self, ds):
        n = len(ds)
        return np.concatenate([ds.hp[:, -1, :], ds.fa.reshape(n, -1)], axis=1).astype(np.float64)

    def fit(self, ds):
        n = len(ds)
        Xb = np.concatenate([self._x(ds), np.ones((n, 1))], axis=1)
        Y = ds.fr.reshape(n, -1).astype(np.float64)
        self.W = np.linalg.solve(Xb.T @ Xb + self.lam * np.eye(Xb.shape[1]), Xb.T @ Y)
        return self

    def predict(self, ds):
        n = len(ds)
        Xb = np.concatenate([self._x(ds), np.ones((n, 1))], axis=1)
        return ds.fa + (Xb @ self.W).reshape(n, H, 3)


def trajectory_metrics(pred_e, true_e, yaw0, yawf):
    """平面 SE(2) 轨迹（中点积分，共享 traj.integrate_xy）。

    主口径：预测 rollout = 真值 yaw0 + **预测 wz 自回归**积分；真值 rollout = 真值 yaw 序列。
    两者同起点，故 FDE 含航向误差传播。绝不读入未来 GT yaw。
    另出 ORACLE_YAW_DIAGNOSTIC：预测速度用**真值 yaw** 积分（仅诊断，不是主能力）。
    """
    T = pred_e.shape[1]
    n = len(pred_e)
    pred_xy = np.stack([integrate_xy(pred_e[i, :, :2], yaw0[i], wz=pred_e[i, :, 2])
                        for i in range(n)])
    true_xy = np.stack([integrate_xy(true_e[i, :, :2], yaw0[i],
                                    yaw_seq=yawf[i]) for i in range(n)])
    fde = float(np.linalg.norm(pred_xy[:, -1] - true_xy[:, -1], axis=1).mean())
    ade = float(np.linalg.norm(pred_xy - true_xy, axis=2).mean())
    dyaw_pred = np.cumsum(pred_e[:, :, 2], axis=1)[:, -1] * DT
    dyaw_true = wrap_pi(yawf[:, -1] - yaw0)
    yaw_err = float(np.abs(wrap_pi(dyaw_pred - dyaw_true)).mean())
    # ORACLE_YAW_DIAGNOSTIC（预测速度 + 真值 yaw）
    orc = np.stack([integrate_xy(pred_e[i, :, :2], yaw0[i], yaw_seq=yawf[i])
                    for i in range(n)])
    fde_orc = float(np.linalg.norm(orc[:, -1] - true_xy[:, -1], axis=1).mean())
    return fde, ade, yaw_err, fde_orc


def summarize(pred_e, true_e, yaw0, yawf):
    row = {}
    for i, ax in enumerate(AXES):
        for ln, k in LEADS.items():
            row[f"lead_MAE_{ax}@{ln}"] = float(np.abs(pred_e[:, k - 1, i] -
                                                      true_e[:, k - 1, i]).mean())
    fde, ade, yaw_err, fde_orc = trajectory_metrics(pred_e, true_e, yaw0, yawf)
    row["FDE_xy_2s_m"] = fde
    row["ADE_xy_2s_m"] = ade
    row["net_yaw_err_rad"] = yaw_err
    row["FDE_xy_2s_m_ORACLE_YAW_DIAGNOSTIC"] = fde_orc
    row["MAE_all_mixed"] = float(np.abs(pred_e - true_e).mean())
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = build_datasets(args.data_root)
    evalsets = {k: ds[k] for k in ("val", "test_all", "test_P0", "test_P1", "test_P2")}
    for k, v in evalsets.items():
        print(f"[eval] {k}: {len(v)} windows")

    train_R0 = ds["train_R0"]
    ridge = {r: RidgeMultiOutput().fit(ds[f"train_{r}"]) for r in ("R0", "R1")}

    cache, rows = {}, []
    for split, dset in evalsets.items():
        n = len(dset)
        if n == 0:
            continue
        b = np.arange(n)
        hp, ha, fa, fr, conds, eps, groups, levels = dset.batch(b, device)
        gt, yaw0, yawf = dset.fe, dset.yaw0, dset.yawf
        # ---- 基线 ----
        cache[f"{split}/command-copy"] = fa.cpu().numpy()
        pers = dset.hp[:, -1, :][:, [0, 1, 5]]
        cache[f"{split}/persistence"] = np.repeat(pers[:, None, :], H, axis=1)
        for r in ("R0", "R1"):
            cache[f"{split}/ridge_{r}"] = ridge[r].predict(dset)
        # ---- 神经模型 ----
        for (mname, regime) in CELLS:
            preds = []
            for seed in SEEDS:
                p = os.path.join(CKPT, f"{mname}_{regime}_s{seed}", "best.pt")
                if not os.path.exists(p):
                    print(f"[eval] missing {mname}_{regime}_s{seed}"); continue
                ck = torch.load(p, weights_only=False, map_location=device)
                m = build(mname).to(device)
                m.load_state_dict(ck["model_state"]); m.eval()
                with torch.no_grad():
                    r = m(hp, ha, fa)
                    preds.append((fa + r).cpu().numpy())
            if not preds:
                continue
            pr = np.mean(preds, axis=0)
            cache[f"{split}/{mname}_{regime}_seedmean"] = pr
            for j, seed in enumerate(SEEDS):
                if j < len(preds):
                    cache[f"{split}/{mname}_{regime}_s{seed}"] = preds[j]
        # ---- 指标 ----
        for key in sorted(cache):
            if not key.startswith(split + "/"):
                continue
            name = key.split("/", 1)[1]
            row = {"split": split, "variant": name, "n_windows": n}
            row.update(summarize(cache[key], gt, yaw0, yawf))
            rows.append(row)
        print(f"[eval] {split} done")

    os.makedirs(os.path.join(args.out, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "metrics"), exist_ok=True)
    pth = os.path.join(args.out, "predictions", "pred_cache.npz")
    np.savez_compressed(pth, **{k.replace("/", "__"): v for k, v in cache.items()})
    sha = hashlib.sha256(open(pth, "rb").read()).hexdigest()
    json.dump({"pred_cache": pth, "sha256": sha, "keys": sorted(cache.keys()),
               "n_arrays": len(cache)}, open(os.path.join(args.out, "predictions",
                                                          "pred_cache_manifest.json"), "w"),
              indent=1)
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "metrics", "trajectory_metrics.csv"),
                              index=False)
    print(f"[eval] pred_cache sha256={sha[:16]} arrays={len(cache)}")


if __name__ == "__main__":
    main()
