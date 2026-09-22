"""V0.6.2 Task 1 — Command-space Novelty Audit。

对每个 query future command U[t:t+H]：
  flatten 为 3H=120 向量 -> 只用 train set 拟合 normalization / PCA / 正则化协方差 ->
  NN 距离 / kNN(k=5) 均值距离 / Mahalanobis / PCA 投影 / 各 axis 联合分布。
再对每个模型（command-copy / ridge / M0 / M1 / M2）计算 error 与 novelty 的关系。

注意：命令序列是分段常数、取自小型段库 -> 120 维协方差高度奇异。
白化必须正则化（否则近零奇异值放大噪声，距离爆到 1e7）。
另报"结构性 novelty"：多轴同时非零（train 的 Q1-Q3 从未出现）。

输出: metrics/command_novelty.csv, metrics/error_vs_novelty.csv,
      metrics/command_space_structure.csv, figures/action_space_pca.png
"""
import json
import os

import numpy as np
import pandas as pd

from execution_wm.validity_v061.windows import H
from execution_wm.validity_v062.common import (OUT, SPLITS_EVAL, command_matrix,
                                                get_manifests, load_cache,
                                                pred_from_cache, seed_mean_pred)

AXES = ("vx", "vy", "wz")
LAMBDA_REG_FRAC = 1e-4          # 协方差正则化强度（相对最大特征值）


def fill_rate_stats(U, thresh=1e-6):
    nz = np.abs(U) > thresh
    n_axis = nz.sum(axis=2)
    return {
        "frac_timesteps_multi_axis": float((n_axis >= 2).mean()),
        "frac_windows_with_multi_axis": float((n_axis >= 2).any(axis=1).mean()),
        "axis_active_frac_vx": float(nz[..., 0].mean()),
        "axis_active_frac_vy": float(nz[..., 1].mean()),
        "axis_active_frac_wz": float(nz[..., 2].mean()),
        "joint_vx_wz_coactive_frac": float((nz[..., 0] & nz[..., 2]).mean()),
        "joint_vx_vy_coactive_frac": float((nz[..., 0] & nz[..., 1]).mean()),
        "joint_vy_wz_coactive_frac": float((nz[..., 1] & nz[..., 2]).mean()),
    }


def main():
    os.makedirs(os.path.join(OUT, "metrics"), exist_ok=True)
    man = get_manifests()
    cache = load_cache()

    Xtr = command_matrix(man["train"])
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    Ztr = (Xtr - mu) / sd                                   # [N,d] 标准化

    # ---- 正则化协方差（Mahalanobis）----
    C = np.cov(Ztr, rowvar=False) + 1e-12 * np.eye(Ztr.shape[1])
    lam, Vc = np.linalg.eigh(C)                             # 升序
    lam_reg = lam + LAMBDA_REG_FRAC * lam.max()
    inv_sqrt = 1.0 / np.sqrt(lam_reg)

    # ---- PCA（train only）----
    U_, S_, Vt = np.linalg.svd(Ztr - Ztr.mean(axis=0), full_matrices=False)
    evr = (S_ ** 2) / np.sum(S_ ** 2)
    rank = int((S_ > 1e-8 * S_[0]).sum())
    n90 = int(np.searchsorted(np.cumsum(evr), 0.90) + 1)
    n99 = int(np.searchsorted(np.cumsum(evr), 0.99) + 1)

    def dists(X):
        Z = (X - mu) / sd
        # standardized 空间的欧氏距离（可解释：单位 = train 命令 std）
        d2 = ((Z[:, None, :] - Ztr[None, :, :]) ** 2).sum(-1)
        d = np.sqrt(np.maximum(d2, 0))
        k = min(5, d.shape[1])
        knn = np.partition(d, k - 1, axis=1)[:, :k].mean(axis=1)
        nn = d.min(axis=1)
        # Mahalanobis（正则化）
        Zc = Z - Ztr.mean(axis=0)
        maha = np.sqrt(((Zc @ Vc) ** 2 @ (inv_sqrt ** 2)))
        # PCA 投影 + 重建误差
        Zc2 = Z - Ztr.mean(axis=0)
        proj = Zc2 @ Vt.T
        recon = np.sqrt((proj[:, 5:] ** 2).sum(axis=1))
        return nn, knn, maha, proj, recon

    # train 自距离基线（排除自身）
    Zc_tr = Ztr - Ztr.mean(axis=0)
    d2_tr = ((Ztr[:, None, :] - Ztr[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2_tr, np.inf)
    train_nn = np.sqrt(d2_tr).min(axis=1)
    nn95 = float(np.quantile(train_nn, 0.95))
    nn_max = float(train_nn.max())
    # train 命令模式的去重覆盖（命令序列高度重复：同 family/时段 -> 相同向量）
    uniq_train = np.unique(np.round(Xtr, 6), axis=0)
    train_multi_axis = ((np.abs(Xtr.reshape(len(Xtr), H, 3)) > 1e-6).sum(axis=2) >= 2)
    n_uniq_train_multi = int(len(np.unique(
        np.round(Xtr.reshape(len(Xtr), H, 3)[train_multi_axis.any(axis=1)], 6), axis=0)))

    rows, struct = [], {}
    for split, m in man.items():
        U = m.fa
        X = command_matrix(m)
        nn, knn, maha, proj, recon = dists(X)
        st = {"n_windows": len(m), **fill_rate_stats(U),
              "pca_rank_numerical": rank, "pca_n90pct": n90, "pca_n99pct": n99,
              "pca_evr_top5": [float(v) for v in evr[:5]],
              "pca_mean_pc1": float(proj[:, 0].mean()),
              "pca_std_pc1": float(proj[:, 0].std()),
              "nn_dist_std_space_mean": float(nn.mean()),
              "nn_dist_std_space_median": float(np.median(nn)),
              "frac_nn_beyond_train_nn95": float((nn > nn95).mean()),
              "frac_nn_beyond_train_nn_max": float((nn > nn_max).mean()),
              "n_distinct_command_vectors": int(len(np.unique(np.round(X, 6), axis=0))),
              "n_distinct_command_vectors_multi_axis": int(len(np.unique(
                  np.round(X.reshape(len(X), H, 3)
                           [((np.abs(X.reshape(len(X), H, 3)) > 1e-6).sum(axis=2) >= 2)
                            .any(axis=1)], 6), axis=0))),
              "mahalanobis_mean": float(maha.mean()),
              "recon_err_5pc_mean": float(recon.mean()),
              "cmd_range_vx": [float(U[..., 0].min()), float(U[..., 0].max())],
              "cmd_range_vy": [float(U[..., 1].min()), float(U[..., 1].max())],
              "cmd_range_wz": [float(U[..., 2].min()), float(U[..., 2].max())]}
        if split != "train":
            st["train_nn95_threshold"] = nn95
        struct[split] = st
        for i in range(len(m)):
            w = m.windows[i]
            rows.append({
                "split": split, "episode_id": w["episode_id"], "condition": w["condition"],
                "command_family": w["command_family"], "anchor_group": w["anchor_group"],
                "rep": w["rep"], "origin_grid": w["origin_grid"], "row_index": i,
                "nn_dist_std_space": float(nn[i]), "knn5_dist_std_space": float(knn[i]),
                "mahalanobis_dist": float(maha[i]),
                "pca_pc1": float(proj[i, 0]), "pca_pc2": float(proj[i, 1]),
                "pca_pc3": float(proj[i, 2]), "recon_err_5pc": float(recon[i]),
                "multi_axis_frac": float(((np.abs(U[i]) > 1e-6).sum(axis=1) >= 2).mean()),
                "vx_wz_coactive_frac": float(((np.abs(U[i, :, 0]) > 1e-6) &
                                              (np.abs(U[i, :, 2]) > 1e-6)).mean()),
                "cmd_mean_vx": float(U[i, :, 0].mean()),
                "cmd_mean_vy": float(U[i, :, 1].mean()),
                "cmd_mean_wz": float(U[i, :, 2].mean()),
            })
    nov = pd.DataFrame(rows)
    nov.to_csv(os.path.join(OUT, "metrics", "command_novelty.csv"), index=False)
    pd.DataFrame(struct).T.to_csv(os.path.join(OUT, "metrics", "command_space_structure.csv"))
    json.dump({"pca_evr_top10": [float(v) for v in evr[:10]],
               "pca_rank_numerical": rank, "n_pca_components_90pct": n90,
               "n_pca_components_99pct": n99, "feature_dim": int(Xtr.shape[1]),
               "n_train_windows": int(len(Xtr)), "fit_on": "train split only",
               "whitening": f"regularized eigenvalue + {LAMBDA_REG_FRAC}*max",
               "train_nn95_in_std_space": nn95, "train_nn_max_in_std_space": nn_max,
               "n_distinct_train_command_vectors": int(len(uniq_train)),
               "n_distinct_train_command_vectors_with_multi_axis": n_uniq_train_multi},
              open(os.path.join(OUT, "metrics", "command_pca_meta.json"), "w"), indent=1)

    # ---- error vs novelty ----
    erows = []
    for split in SPLITS_EVAL:
        m = man[split]
        gt = m.fe
        variants = {"command-copy": pred_from_cache(cache, split, "command-copy"),
                    "ridge_multioutput": pred_from_cache(cache, split, "ridge_multioutput")}
        for name in ("M0", "M1", "M2"):
            variants[name] = seed_mean_pred(cache, split, name)
        sub = nov[nov.split == split].reset_index(drop=True)
        for vname, pr in variants.items():
            err = np.abs(pr - gt)
            for ax, i in zip(AXES, range(3)):
                e = err[:, :, i].mean(axis=1)
                for col in ("nn_dist_std_space", "knn5_dist_std_space", "mahalanobis_dist",
                            "recon_err_5pc", "multi_axis_frac"):
                    x = sub[col].values.astype(float)
                    if x.std() < 1e-12:
                        r = sp = np.nan
                        hi = lo = np.nan
                    else:
                        r = float(np.corrcoef(x, e)[0, 1])
                        sp = float(pd.Series(x).corr(pd.Series(e), method="spearman"))
                        hi = float(e[x >= np.quantile(x, 0.75)].mean())
                        lo = float(e[x <= np.quantile(x, 0.25)].mean())
                    erows.append({"split": split, "model": vname, "axis": ax,
                                  "novelty_metric": col, "pearson_r": r, "spearman_r": sp,
                                  "n_windows": len(e), "mean_err": float(e.mean()),
                                  "mean_novelty": float(x.mean()),
                                  "err_top_quartile_novelty": hi,
                                  "err_bottom_quartile_novelty": lo})
    pd.DataFrame(erows).to_csv(os.path.join(OUT, "metrics", "error_vs_novelty.csv"), index=False)

    print("=== 命令空间结构 ===")
    cols = ["n_windows", "frac_windows_with_multi_axis", "joint_vx_wz_coactive_frac",
            "nn_dist_std_space_median", "mahalanobis_mean", "recon_err_5pc_mean",
            "frac_nn_beyond_train_nn95", "cmd_range_vx", "cmd_range_wz"]
    print(pd.DataFrame(struct).T[cols].to_string())
    print(f"\nPCA: 数值秩={rank} / 90%={n90} 维 / 99%={n99} 维 (feature dim 120)")
    print(f"train 内部 NN 距离 95 分位 = {nn95:.3f}（std 单位）")
    print("\n=== error vs novelty（Spearman，逐轴）===")
    e = pd.DataFrame(erows)
    print(e[e.novelty_metric == "nn_dist_std_space"]
          [["split", "model", "axis", "spearman_r", "mean_err"]].round(4).to_string(index=False))
    print("\n=== error vs 结构性 novelty（multi_axis_frac）===")
    print(e[e.novelty_metric == "multi_axis_frac"]
          [["split", "model", "axis", "spearman_r"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
