"""V0.6.1 B7：统计单元 —— 只读同一 pred_cache（T09），按 anchor 分组 + episode cluster。

不做"48 episode 独立"的强宣称：held-anchor(A/C) 只有 2 个 anchor group，
直接报告每组结果、training-seed 分布与探索性均值。
统计口径:
  - 主：per-anchor-group × per-seed 表（不做显著性宣称）
  - 辅：episode-cluster bootstrap（2000 次，仅探索性区间）
  - 训练 seed 与数据 seed 分离（window manifest 固定）
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd

OUT = "results/v0_6_1_correctness"
CACHE = os.path.join(OUT, "predictions", "pred_cache.npz")
SPLITS = ["A_unseen_anchor_seen_family", "B_seen_anchor_unseen_family",
          "C_unseen_anchor_unseen_family", "supp_AQ5xQ4"]
N_BOOT = 2000


def cluster_boot_mean(err, clusters, rng, n_boot=N_BOOT):
    units = np.unique(clusters)
    cid = np.searchsorted(units, clusters)
    out = np.empty(n_boot)
    for b in range(n_boot):
        cnt = np.bincount(rng.integers(0, len(units), len(units)), minlength=len(units))
        w = cnt[cid]
        out[b] = (err * w).sum() / max(w.sum(), 1)
    return np.percentile(out, [2.5, 97.5])


def cluster_boot_diff(diff, clusters, rng, n_boot=N_BOOT):
    units = np.unique(clusters)
    cid = np.searchsorted(units, clusters)
    out = np.empty(n_boot)
    for b in range(n_boot):
        cnt = np.bincount(rng.integers(0, len(units), len(units)), minlength=len(units))
        w = cnt[cid]
        out[b] = (diff * w).sum() / max(w.sum(), 1)
    return np.percentile(out, [2.5, 97.5])


def main():
    cache = np.load(CACHE)
    man = json.load(open(os.path.join(OUT, "predictions", "pred_cache_manifest.json")))
    sha = hashlib.sha256(open(CACHE, "rb").read()).hexdigest()
    assert sha == man["sha256"], "pred_cache 哈希与 manifest 不一致（T09 失败）"
    rng = np.random.default_rng(20260922)

    per_anchor_rows, diff_rows, boot_rows = [], [], []
    for split in SPLITS:
        d = np.load(os.path.join(OUT, "predictions", f"per_window_{split}.npz"))
        eps = d["meta_ep"]; anchors = d["meta_anchor"]
        keys = [k for k in d.files if k not in ("meta_ep", "meta_anchor", "meta_family",
                                                "meta_cond")]
        # 只取 aggregate（每窗 3 轴 horizon 平均）用于排序；逐轴另表
        agg = {k: d[k] for k in keys}
        # 1) per-anchor-group × model（3 seed 均值 + seed 分布）
        models = sorted({k.split("__")[1] for k in agg if k.split("__")[1].startswith("M")})
        for k, v in agg.items():
            _, name = k.split("__", 1)
            if name.endswith("OLD_wrong_axis"):
                continue
            for ag in sorted(set(anchors)):
                m = anchors == ag
                per_anchor_rows.append({
                    "split": split, "anchor_group": ag, "variant": name,
                    "n_windows": int(m.sum()),
                    "mean_err": float(v[m].mean()),
                    "seed_spread": float(np.ptp([v[m].mean()])) if v.ndim == 1 else 0.0})
        # 2) 关键差值（同一次 resample 内 -> paired）
        def get(nm):
            return agg[f"{split}__{nm}"]
        m0 = np.mean([get(f"M0_s{s}") for s in (42, 43, 44)], axis=0)
        m1 = np.mean([get(f"M1_s{s}") for s in (42, 43, 44)], axis=0)
        m2 = np.mean([get(f"M2_s{s}") for s in (42, 43, 44)], axis=0)
        m3 = np.mean([get(f"M3_s{s}") for s in (42, 43, 44)], axis=0)
        pairs = [("M1-M0", m1 - m0), ("M2-M1", m2 - m1), ("M3-M1", m3 - m1),
                 ("M1-command-copy", m1 - get("command-copy")),
                 ("M2-ridge", m2 - get("ridge_multioutput")),
                 ("persistence-correct_vs_OLD_wrong_axis",
                  get("persistence") - get("persistence_OLD_wrong_axis")),
                 ("M2donor_native-zero", get("M2_s42__donor_native") -
                  get("M2_s42__donor_zero")),
                 ("M2donor_native-same_cond", get("M2_s42__donor_native") -
                  get("M2_s42__donor_same_cond")),
                 ("M2donor_native-wrong_cond", get("M2_s42__donor_native") -
                  get("M2_s42__donor_wrong_cond")),
                 ("M2donor_same-wrong_cond", get("M2_s42__donor_same_cond") -
                  get("M2_s42__donor_wrong_cond"))]
        for name, dd in pairs:
            ci = cluster_boot_diff(dd, eps, rng)
            diff_rows.append({"split": split, "diff": name, "mean": float(dd.mean()),
                              "ci_low": float(ci[0]), "ci_high": float(ci[1]),
                              "crosses_zero": bool(ci[0] < 0 < ci[1]),
                              "n_windows": int(len(dd)),
                              "n_episode_clusters": int(len(np.unique(eps))),
                              "n_anchor_groups": int(len(np.unique(anchors))),
                              "scope": "EXPLORATORY"})
            boot_rows.append({"split": split, "variant": name, "mean_err": float(dd.mean())})
        # 3) 每 variant 的探索性区间
        for k, v in agg.items():
            _, name = k.split("__", 1)
            ci = cluster_boot_mean(v, eps, rng)
            boot_rows.append({"split": split, "variant": name, "mean_err": float(v.mean()),
                              "ci_low": float(ci[0]), "ci_high": float(ci[1]),
                              "n_episode_clusters": int(len(np.unique(eps))),
                              "n_anchor_groups": int(len(np.unique(anchors))),
                              "scope": "EXPLORATORY"})

    pd.DataFrame(per_anchor_rows).to_csv(os.path.join(OUT, "metrics", "per_anchor_group.csv"),
                                         index=False)
    dd = pd.DataFrame(diff_rows)
    dd.to_csv(os.path.join(OUT, "metrics", "v061_diff_bootstrap.csv"), index=False)
    pd.DataFrame(boot_rows).to_csv(os.path.join(OUT, "metrics", "v061_cluster_ci.csv"), index=False)
    for split in SPLITS:
        print(f"\n=== {split} ===")
        print(dd[dd.split == split][["diff", "mean", "ci_low", "ci_high", "crosses_zero",
                                     "n_windows", "n_episode_clusters",
                                     "n_anchor_groups"]].round(4).to_string(index=False))
    print("\n[注] 负值 = 前者误差更小。held-anchor(A/C) 仅 2 个 anchor group，"
          "所有区间为探索性，不构成普遍新 anchor 的显著宣称。")


if __name__ == "__main__":
    main()
