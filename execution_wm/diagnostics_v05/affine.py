"""Task E: Global Affine Calibration Diagnostic（second_work §7）。

诊断实验（非正式方法）：r_cal_j = alpha_j * rhat_j + beta_j，只用 validation split 拟合，
不用任何 condition/friction 标签。然后对全部 swap pair 重算指标。

输出:
  calibration/global_affine_params.json
  calibration/affine_recomputed_swap_metrics.csv   before/after 对照
  figures/CAL_affine_before_after.png/.pdf

用法: python -m execution_wm.diagnostics_v05.affine --config execution_wm/configs/v05_diagnostics.yaml
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from execution_wm.context_swap.common import (  # noqa: E402
    EpisodeData, VEL_NAMES, load_cfg, load_context_model, out_subdir, predict,
)
from execution_wm.data.dataset import discover_episodes, split_episodes  # noqa: E402

EPS = 1e-8


def cosine_rows(a, b):
    fa, fb = a.reshape(len(a), -1), b.reshape(len(b), -1)
    return (fa * fb).sum(1) / (np.linalg.norm(fa, axis=1) * np.linalg.norm(fb, axis=1) + EPS)


def swap_metrics(u, ea, eb, rc, rs):
    """从 (u, e_a, e_b, rhat_correct, rhat_swap) 计算全部 swap 指标（overall+per-axis）。"""
    N = len(u)
    e_c, e_s = u + rc, u + rs
    d_swap, d_tgt = e_s - e_c, eb - ea
    out = {}
    E_c = np.linalg.norm((e_c - ea).reshape(N, -1), axis=1)
    E_s = np.linalg.norm((e_s - ea).reshape(N, -1), axis=1)
    D_b = np.linalg.norm((e_c - eb).reshape(N, -1), axis=1)
    D_a = np.linalg.norm((e_s - eb).reshape(N, -1), axis=1)
    out["E_correct"], out["E_swap"] = E_c, E_s
    out["dE_self"] = E_s - E_c
    out["D_before"], out["D_after"] = D_b, D_a
    out["P_target"] = D_b - D_a
    out["S_dir"] = cosine_rows(d_swap, d_tgt)
    out["S_mag"] = np.linalg.norm(d_swap.reshape(N, -1), axis=1) / (
        np.linalg.norm(d_tgt.reshape(N, -1), axis=1) + EPS)
    for i, n in enumerate(VEL_NAMES):
        out[f"P_target_{n}"] = (np.linalg.norm((e_c - eb)[:, :, i], axis=1)
                                - np.linalg.norm((e_s - eb)[:, :, i], axis=1))
        out[f"dE_self_{n}"] = (np.linalg.norm((e_s - ea)[:, :, i], axis=1)
                               - np.linalg.norm((e_c - ea)[:, :, i], axis=1))
        out[f"S_dir_{n}"] = cosine_rows(d_swap[:, :, i:i + 1], d_tgt[:, :, i:i + 1])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_cfg(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    L = int(round(cfg["history_s"] * cfg["hz"]))
    H = int(round(cfg["horizon_s"] * cfg["hz"]))
    out = out_subdir(cfg, "calibration")
    fig_dir = out_subdir(cfg, "figures")

    # ---- 7.1 在 validation split 上拟合 affine（原生预测，无任何 condition 标签输入） ----
    dc = yaml.safe_load(open(cfg["train_config"]))["dataset"]
    eps_all = [e for e in discover_episodes(cfg["dataset_dirs"][0])
               if e["meta"].get("episode_type") == "random"]
    splits = split_episodes(eps_all, dc["val_fraction"], dc["test_id_fraction"],
                            dc["ood_conditions"], True, 42)
    model = load_context_model(cfg, device)
    rng = np.random.default_rng(cfg["seed"])
    rhats, rs = [], []
    for e in splits["val"]:
        ep = EpisodeData(e)
        t0s = np.arange(L - 1, ep.T - H - 1)
        for t0 in rng.choice(t0s, size=min(40, len(t0s)), replace=False):
            w = ep.window(int(t0), L, H)
            e_hat, _ = predict(model, ep, int(t0), L, H, device)
            rhats.append(e_hat - w["future_action"])
            rs.append(w["future_residual"])
    rhat_v, r_v = np.stack(rhats), np.stack(rs)
    alpha, beta = [], []
    for i, n in enumerate(VEL_NAMES):
        x = rhat_v[:, :, i].reshape(-1)
        y = r_v[:, :, i].reshape(-1)
        A = np.stack([x, np.ones_like(x)], 1)
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]
        alpha.append(float(a))
        beta.append(float(b))
    params = {"alpha": dict(zip(VEL_NAMES, alpha)), "beta": dict(zip(VEL_NAMES, beta)),
              "fit_split": cfg["affine"]["fit_split"], "n_val_windows": len(rhat_v),
              "note": "r_cal = alpha*rhat + beta, per-axis, 只用 val split 拟合（§7.1）"}
    with open(os.path.join(out, "global_affine_params.json"), "w") as f:
        json.dump(params, f, indent=2)
    print(f"[affine] alpha={np.round(alpha,3)} beta={np.round(beta,4)} "
          f"(n_val={len(rhat_v)})")

    # ---- 7.2 重算 swap 指标（cache 上 before/after） ----
    cache = dict(np.load(os.path.join(cfg["out_dir"], "raw", "pair_pred_cache.npz")))
    manifest = pd.read_csv(os.path.join(cfg["swap_dir"], "pairs", "pair_manifest.csv"))
    cross = (manifest.kind == "cross").values
    directions = (manifest.cond_A + "->" + manifest.cond_B).values
    a_vec = np.array(alpha)[None, None, :]
    b_vec = np.array(beta)[None, None, :]
    before = swap_metrics(cache["u"], cache["e_a"], cache["e_b"],
                          cache["rhat_correct"], cache["rhat_swap"])
    after = swap_metrics(cache["u"], cache["e_a"], cache["e_b"],
                         a_vec * cache["rhat_correct"] + b_vec,
                         a_vec * cache["rhat_swap"] + b_vec)

    keys = ["E_correct", "E_swap", "dE_self", "D_before", "D_after", "P_target",
            "S_dir", "S_mag", "P_target_vx", "P_target_vy", "P_target_wz",
            "dE_self_vx", "dE_self_vy", "dE_self_wz", "S_dir_vx", "S_dir_vy", "S_dir_wz"]
    rows = []
    for direction in np.unique(directions[cross]):
        sel = cross & (directions == direction)
        row = {"direction": direction, "N": int(sel.sum())}
        for k in keys:
            row[f"{k}_before"] = float(before[k][sel].mean())
            row[f"{k}_after"] = float(after[k][sel].mean())
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "affine_recomputed_swap_metrics.csv"), index=False)
    cols = ["direction", "P_target_before", "P_target_after", "dE_self_before",
            "dE_self_after", "S_dir_before", "S_dir_after"]
    print(df[cols].round(4).to_string(index=False))

    # ---- figure: before/after P_target ----
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df))
    ax.bar(x - 0.18, df.P_target_before, 0.36, label="P_target before (raw rhat)")
    ax.bar(x + 0.18, df.P_target_after, 0.36, label="P_target after (affine r_cal)")
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("friction_", "f_") for d in df.direction], fontsize=9)
    ax.set_ylabel("P_target (mean)")
    ax.set_title("CAL: global affine calibration — swap metrics before/after")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "CAL_affine_before_after.png"), dpi=150)
    fig.savefig(os.path.join(fig_dir, "CAL_affine_before_after.pdf"))
    plt.close(fig)
    print(f"[affine] -> {out}")


if __name__ == "__main__":
    main()
