"""V0.6.1 R3/R4：单次推理 + 全量预测缓存（B7）+ fixed-lead 主表（B6）。

所有模型对所有窗口**只推理一次**，写入 predictions/pred_cache.npz 并记录 sha256；
所有 summary / bootstrap 均引用同一缓存（T09）。

评估 population（B5，互不重叠）:
  A unseen_anchor_seen_family   AQ6/7 x Q1-Q3
  B seen_anchor_unseen_family   AQ0-4 x Q4
  C unseen_anchor_unseen_family AQ6/7 x Q4
  supp_AQ5xQ4（evaluation-only，单列）
基线: command-copy / 正确 persistence(schema [0,1,5]) / ridge multi-output（准确命名）
M2 四个推理对照: same-cond paired donor / wrong-cond paired donor / native / zero-context

用法: python -m execution_wm.validity_v061.eval_v061
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd
import torch

from execution_wm.train.train_execution import MODEL_REGISTRY
from execution_wm.validity_v061.metrics import (LEADS_S, all_metric_axes, fixed_lead_mae,
                                                lead_index, prefix_mae)
from execution_wm.validity_v061.schema import EXECUTION_KEYS, extract_execution_from_proprio
from execution_wm.validity_v061.splits import attach_derived_paths, classify
from execution_wm.validity_v061.support_pairs import SupportBank
from execution_wm.validity_v061.windows import H, WindowManifest

SQ = "/media/hdd1/yuhang/datasets/execution_wm/v0_6_sq"
DERIVED = "/media/hdd1/yuhang/datasets/execution_wm/v0_6_1"
CKPT = "/media/hdd1/yuhang/checkpoints/execution_wm/v0_6_1"
OUT = "results/v0_6_1_correctness"
MODELS = ("M0", "M1", "M2", "M3")
SEEDS = (42, 43, 44)
WPE = 8
COND_MAP = {"normal": "friction_low", "friction_mid": "normal", "friction_low": "normal"}


def load_entry(name, seed, device):
    path = os.path.join(CKPT, f"{name}_s{seed}", "best.pt")
    if not os.path.exists(path):
        return None, None, None
    ck = torch.load(path, weights_only=False, map_location=device)
    base = "direct" if name == "M0" else "context"
    m = MODEL_REGISTRY[base](H, ck["config"]["model"]).to(device)
    m.load_state_dict(ck["model_state"]); m.eval()
    aux = None
    if ck.get("aux_state") is not None:
        aux = torch.nn.Linear(1, 8).to(device); aux.load_state_dict(ck["aux_state"]); aux.eval()
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
    return m, aux, sha


def predict(name, m, aux, hp, ha, fa, fric, c=None):
    with torch.no_grad():
        if name == "M0":
            r = m(hp, ha, fa)["r_hat"]
        else:
            if c is None:
                c = m.encode_context(hp, ha)
            if name == "M3":
                c = c + aux(fric.reshape(-1, 1))
            r = m.predict_execution(m.encode_state(hp), c, fa)["r_hat"]
    return fa + r                                        # e_hat = u_ref + r_hat


class RidgeMultiOutput:
    """准确名称: ridge multi-output 回归, 输入 = 当前 proprio + 完整 future command。
    不是标准多滞后 ARX 全搜索。"""

    def __init__(self, lam=1.0):
        self.lam = lam

    def fit(self, man):
        n = len(man)
        X = np.concatenate([man.hp[:, -1, :], man.fa.reshape(n, -1)], axis=1).astype(np.float64)
        Y = man.fr.reshape(n, -1).astype(np.float64)
        Xb = np.concatenate([X, np.ones((n, 1))], axis=1)
        self.W = np.linalg.solve(Xb.T @ Xb + self.lam * np.eye(Xb.shape[1]), Xb.T @ Y)
        return self

    def predict(self, man):
        n = len(man)
        X = np.concatenate([man.hp[:, -1, :], man.fa.reshape(n, -1)], axis=1).astype(np.float64)
        Xb = np.concatenate([X, np.ones((n, 1))], axis=1)
        return man.fa + (Xb @ self.W).reshape(n, H, 3)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    entries = attach_derived_paths(json.load(open(os.path.join(SQ, "index.json")))["episodes"],
                                   DERIVED)
    sp = classify(entries)
    pops = {
        "A_unseen_anchor_seen_family": sp["A_unseen_anchor_seen_family"],
        "B_seen_anchor_unseen_family": sp["B_seen_anchor_unseen_family"],
        "C_unseen_anchor_unseen_family": sp["C_unseen_anchor_unseen_family"],
        "supp_AQ5xQ4": sp["supp_AQ5xQ4"],
    }
    train_man = WindowManifest(sp["train"], "train", windows_per_ep=8)
    bank = SupportBank(sp["support"], path_key="path20")
    ridge = RidgeMultiOutput().fit(train_man)

    man = {k: WindowManifest(v, k, windows_per_ep=WPE) for k, v in pops.items()}
    for k, m in man.items():
        m.assert_causal(); m.assert_label_identity()
        print(f"[eval] {k}: {len(m)} windows, eps={len({w['episode_id'] for w in m.windows})}")

    cache, rows, lineage, donor_tables = {}, [], [], {}

    for split, m in man.items():
        n = len(m)
        b = np.arange(n)
        hp, ha, fa, fr, fric, conds, eps, anchors, _splits, fams = m.batch(b, device)
        e_gt = m.fe                  # 独立物理真值（不经过 u+r 自证）
        u = m.fa
        conds_l = list(conds)
        # ---- 基线 ----
        cache[f"{split}/command-copy"] = fa.cpu().numpy()
        persis = extract_execution_from_proprio(hp[:, -1, :]).cpu().numpy()   # schema [0,1,5]
        cache[f"{split}/persistence"] = np.repeat(persis[:, None, :], H, axis=1)
        cache[f"{split}/ridge_multioutput"] = ridge.predict(m)
        # 旧（错误轴）persistence 仅作对照
        old_p = hp[:, -1, :3].cpu().numpy()
        cache[f"{split}/persistence_OLD_wrong_axis"] = np.repeat(old_p[:, None, :], H, axis=1)
        # ---- 模型 ----
        for name in MODELS:
            for seed in SEEDS:
                mm, aux, sha = load_entry(name, seed, device)
                if mm is None:
                    print(f"[eval] missing {name}_s{seed}"); continue
                pr = predict(name, mm, aux, hp, ha, fa, fric).cpu().numpy()
                cache[f"{split}/{name}_s{seed}"] = pr
                rows.append({"split": split, "model": name, "seed": seed, "sha256_16": sha,
                             "n_windows": n})
        # ---- M2 四个必要推理对照（s42 起） ----
        for seed in SEEDS:
            mm, aux, sha = load_entry("M2", seed, device)
            if mm is None:
                continue
            # 1. same-condition paired donor（固定 donor 表）
            j_same = bank.fixed_indices(conds_l)
            shp, sha_ = bank.batch(j_same, device)
            c_same = mm.encode_context(shp, sha_)
            cache[f"{split}/M2_s{seed}/donor_same_cond"] = predict(
                "M2", mm, aux, hp, ha, fa, fric, c=c_same).cpu().numpy()
            # 2. wrong-condition paired donor（同机制，换 condition）
            dconds = [COND_MAP[c] for c in conds_l]
            j_wrong = bank.fixed_indices(dconds)
            shp2, sha2 = bank.batch(j_wrong, device)
            cache[f"{split}/M2_s{seed}/donor_wrong_cond"] = predict(
                "M2", mm, aux, hp, ha, fa, fric, c=mm.encode_context(shp2, sha2)).cpu().numpy()
            # 3. native history
            cache[f"{split}/M2_s{seed}/donor_native"] = predict(
                "M2", mm, aux, hp, ha, fa, fric).cpu().numpy()
            # 4. zero context
            cache[f"{split}/M2_s{seed}/donor_zero"] = predict(
                "M2", mm, aux, hp, ha, fa, fric,
                c=torch.zeros(n, 8, device=device)).cpu().numpy()
            donor_tables[f"{split}/M2_s{seed}/donor_same_cond"] = bank.donor_ids(j_same)
            donor_tables[f"{split}/M2_s{seed}/donor_wrong_cond"] = bank.donor_ids(j_wrong)
        line = m.lineage_rows()
        for i, w in enumerate(line):
            w["row_index"] = i
        lineage += [{**w, "split_label": split} for w in line]
        print(f"[eval] {split} predictions done")

    # ---- 缓存哈希（T09） ----
    os.makedirs(os.path.join(OUT, "predictions"), exist_ok=True)
    pq_path = os.path.join(OUT, "predictions", "pred_cache.npz")
    np.savez_compressed(pq_path, **{k.replace("/", "__"): v for k, v in cache.items()})
    sha = hashlib.sha256(open(pq_path, "rb").read()).hexdigest()
    json.dump({"pred_cache": pq_path, "sha256": sha,
               "keys": sorted(cache.keys()),
               "gt_key": "每 split 的真值 = <split>__gt_execution / __gt_residual / __gt_cmd_ref"},
              open(os.path.join(OUT, "predictions", "pred_cache_manifest.json"), "w"), indent=1)
    json.dump(donor_tables, open(os.path.join(OUT, "manifests", "donor_tables.json"), "w"),
              indent=1)
    pd.DataFrame(lineage).to_csv(os.path.join(OUT, "manifests", "window_lineage.csv"), index=False)

    # ---- 指标表（同一缓存） ----
    keys = sorted(cache.keys())
    for split, m in man.items():
        gt = m.fe
        for k in keys:
            if not k.startswith(f"{split}/"):
                continue
            label = k.split("/", 1)[1]
            pr = cache[k]
            row = {"split": split, "variant": label, "n_windows": len(pr)}
            for ax in all_metric_axes():
                j = lead_index(ax["h_s"])
                a = ax["axis"]
                row[f"lead_MAE_{ax['name']}@{ax['h_s']}s"] = float(
                    np.abs(pr[:, j, a] - gt[:, j, a]).mean())
                row[f"lead_RMSE_{ax['name']}@{ax['h_s']}s"] = float(
                    np.sqrt(((pr[:, j, a] - gt[:, j, a]) ** 2).mean()))
                row[f"prefix_MAE_{ax['name']}@{ax['h_s']}s"] = float(
                    np.abs(pr[:, :j + 1, a] - gt[:, :j + 1, a]).mean())
            row["legacy_MAE_all_mixed_unit"] = float(np.abs(pr - gt).mean())
            rows.append(row)
    df = pd.DataFrame([r for r in rows if "variant" in r])
    df.to_csv(os.path.join(OUT, "metrics", "v061_main_metrics.csv"), index=False)
    # 每窗误差（供 bootstrap，仍来自同一缓存）
    per = {}
    for split, m in man.items():
        gt = m.fe
        for k in keys:
            if k.startswith(f"{split}/"):
                per[k.replace("/", "__")] = np.abs(cache[k] - gt)[:, :, :].mean(axis=(1, 2))
    for split, m in man.items():
        np.savez_compressed(os.path.join(OUT, "predictions", f"per_window_{split}.npz"),
                            **{k: v for k, v in per.items() if k.startswith(split + "__")},
                            meta_ep=np.array([w["episode_id"] for w in m.windows]),
                            meta_anchor=np.array([w["anchor_group"] for w in m.windows]),
                            meta_family=np.array([w["command_family"] for w in m.windows]),
                            meta_cond=np.array([w["condition"] for w in m.windows]))
    print(f"[eval] pred_cache sha256={sha[:16]}  metrics -> {OUT}/metrics/v061_main_metrics.csv")
    key_cols = ["split", "variant", "lead_MAE_vx@2.0s", "lead_MAE_vy@2.0s",
                "lead_MAE_wz@2.0s", "lead_MAE_wz@0.25s"]
    sub = df[df.variant.str.startswith(("M0", "M1", "M2", "M3", "command", "persistence",
                                        "ridge"))]
    sub = sub[~sub.variant.str.contains("OLD_wrong")]
    print(sub[key_cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
