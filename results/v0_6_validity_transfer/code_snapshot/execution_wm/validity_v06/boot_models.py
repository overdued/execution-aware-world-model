"""V0.6 阶段C：关键模型差值的 episode-cluster bootstrap（保留每窗误差原始数据）。

差值（paired，同一次 resample 内计算）:
  M1 - M0      （Context 相对等输入 Direct 的效益；CI 跨 0 -> INCONCLUSIVE）
  M2 - M1      （cross-support 训练相对自历史 c）
  M3 - M1      （privileged friction 相对 learned c 的差距）
  M1 - command-copy
  M1 variants  （native / cross_support / diff_condition / zero / shuffle，M1 s42）

误差定义：per-window MAE @ 2s（3 轴均值，主口径）+ per-axis @1s。
单元：episode cluster（held-anchor 36 eps / held-family 32 eps）；
      明确标注 held-anchor 只有 2 个 anchor group，确认为探索性强度。

输出:
  metrics/per_window_errors.npz          （每窗误差原始数据 + episode/anchor/family 映射）
  metrics/model_diff_bootstrap.csv

用法: python -m execution_wm.validity_v06.boot_models --config execution_wm/configs/v06_validity.yaml
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import yaml

from execution_wm.validity_v06.eval_sq import arx_predict, fit_arx, load_trained, predict_with
from execution_wm.validity_v06.train_sq import SQWindowData, SupportWindowBank, discover_sq, H

VEL = ["vx", "vy", "wz"]


def cluster_boot_diff(diff, clusters, n_boot, rng):
    units = np.unique(clusters)
    cid = np.array([np.where(units == c)[0][0] for c in clusters])
    out = np.empty(n_boot)
    for b in range(n_boot):
        cnt = np.bincount(rng.integers(0, len(units), len(units)), minlength=len(units))
        w = cnt[cid]
        out[b] = (diff * w).sum() / w.sum()
    return np.percentile(out, [2.5, 97.5]), out.mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_m = os.path.join(cfg["out_dir"], "metrics")
    rng = np.random.default_rng(cfg["seed"])
    n_boot = 2000

    sq_cfg = yaml.safe_load(open("execution_wm/configs/collect_v06_sq.yaml"))
    roles = discover_sq(sq_cfg["split"])
    held_anchor = [e for e in roles["test"] if int(e["anchor_group"][2:]) in
                   sq_cfg["split"]["test_anchors"]]
    held_family = [e for e in roles["test"] if e.get("held_out_family")]
    train_data = SQWindowData(roles["train"] + roles["support"], seed=0)
    bank = SupportWindowBank(roles["support"])

    all_err, meta_rows, cl_by_tag = {}, [], {}
    for split_name, entries in (("held_anchor", held_anchor), ("held_family", held_family)):
        data = SQWindowData(entries, windows_per_ep=8, seed=1)
        n = len(data)
        hp, ha, fa, fr_lab, fric, conds, epids, anchors = data.batch(np.arange(n), device)
        cl_by_tag[split_name] = np.array(epids)
        gt = (fa + fr_lab).cpu().numpy()
        tag = split_name
        all_err[f"{tag}/command-copy"] = np.abs(fa.cpu().numpy() - gt).mean(axis=(1, 2))
        per_axis = {}
        for mname in ("M0", "M1", "M2", "M3"):
            for seed in (42, 43, 44):
                path = f"/media/hdd1/yuhang/checkpoints/execution_wm/v0_6/{mname}_s{seed}/best.pt"
                if not os.path.exists(path):
                    continue
                model, aux = load_trained(mname, seed, device)
                if mname == "M2":
                    shp, sha = bank.sample_batch(conds, rng, device)
                    c = model.encode_context(shp, sha)
                    pr = predict_with(mname, model, aux, hp, ha, fa, fric, c=c).cpu().numpy()
                else:
                    pr = predict_with(mname, model, aux, hp, ha, fa, fric).cpu().numpy()
                err = np.abs(pr - gt)
                all_err[f"{tag}/{mname}_s{seed}"] = err.mean(axis=(1, 2))
                for i, vn in enumerate(VEL):
                    per_axis[f"{tag}/{mname}_s{seed}/{vn}"] = err[:, :20, i].mean(axis=1)
        all_err.update(per_axis)
        # M1 s42 context variants
        model, _ = load_trained("M1", 42, device)
        shp, sha = bank.sample_batch(conds, rng, device)
        cond_map = {"normal": "friction_low", "friction_mid": "normal",
                    "friction_low": "normal"}
        shp2, sha2 = bank.sample_batch([cond_map.get(c, "normal") for c in conds], rng, device)
        c_self = model.encode_context(hp, ha)
        variants = {
            "native": predict_with("M1", model, None, hp, ha, fa, fric),
            "cross_support": predict_with("M1", model, None, hp, ha, fa, fric,
                                          c=model.encode_context(shp, sha)),
            "diff_condition": predict_with("M1", model, None, hp, ha, fa, fric,
                                           c=model.encode_context(shp2, sha2)),
            "zero": predict_with("M1", model, None, hp, ha, fa, fric,
                                 c=torch.zeros(n, 8, device=device)),
            "shuffle": predict_with("M1", model, None, hp, ha, fa, fric,
                                    c=c_self[torch.randperm(n, device=device)]),
        }
        for v, pr in variants.items():
            all_err[f"{tag}/M1var_{v}"] = np.abs(pr.cpu().numpy() - gt).mean(axis=(1, 2))
        meta_rows += [{"split": tag, "ep": e_, "anchor": a_, "cond": c_}
                      for e_, a_, c_ in zip(epids, anchors, conds)]
        print(f"[boot] {tag} predictions done ({n} windows)")

    meta = pd.DataFrame(meta_rows)
    np.savez(os.path.join(out_m, "per_window_errors.npz"),
             **{k.replace("/", "__"): v for k, v in all_err.items()},
             meta_ep=meta.ep.values, meta_anchor=meta.anchor.values,
             meta_cond=meta.cond.values, meta_split=meta.split.values)

    rows = []
    for tag in ("held_anchor", "held_family"):
        cl = cl_by_tag[tag]
        seed_mean = {mn: np.mean([all_err[f"{tag}/{mn}_s{s}"] for s in (42, 43, 44)
                                  if f"{tag}/{mn}_s{s}" in all_err], axis=0)
                     for mn in ("M0", "M1", "M2", "M3")}
        pairs = [("M1-M0", seed_mean["M1"] - seed_mean["M0"]),
                 ("M2-M1", seed_mean["M2"] - seed_mean["M1"]),
                 ("M3-M1", seed_mean["M3"] - seed_mean["M1"]),
                 ("M1-ccopy", seed_mean["M1"] - all_err[f"{tag}/command-copy"]),
                 ("M1var_native-cross_support",
                  all_err[f"{tag}/M1var_native"] - all_err[f"{tag}/M1var_cross_support"]),
                 ("M1var_native-diff_condition",
                  all_err[f"{tag}/M1var_native"] - all_err[f"{tag}/M1var_diff_condition"]),
                 ("M1var_native-zero",
                  all_err[f"{tag}/M1var_native"] - all_err[f"{tag}/M1var_zero"]),
                 ("M1var_native-shuffle",
                  all_err[f"{tag}/M1var_native"] - all_err[f"{tag}/M1var_shuffle"])]
        for name, d in pairs:
            ci, est = cluster_boot_diff(d, cl, n_boot, rng)
            rows.append({"split": tag, "diff": name, "mean": float(d.mean()),
                         "ci_low": round(float(ci[0]), 4), "ci_high": round(float(ci[1]), 4),
                         "crosses_zero": bool(ci[0] < 0 < ci[1]),
                         "n_windows": int(len(d)), "n_clusters": int(len(np.unique(cl)))})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_m, "model_diff_bootstrap.csv"), index=False)
    pd.set_option("display.width", 200)
    print(df.round(4).to_string(index=False))
    print("注: 负值 = 前者误差更小（如 M1-M0<0 表示 M1 优于 M0）")
    print(f"[boot] -> {out_m}")


if __name__ == "__main__":
    main()
