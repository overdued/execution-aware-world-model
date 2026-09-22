"""V0.6 阶段A：逐项复算 03_V0_6 §0 外部复核的 10 条声明（纯缓存，不重训不采数）。

输入: results/v0_5_diagnostics/raw/pair_pred_cache.npz + context_swap_v2 pair_manifest
输出:
  audit/metric_equivalence.csv        每条声明: claimed vs recomputed, match?
  audit/DIFFERENCE_REPORT.md          仅当存在不一致时标记（一致也写，记录容差）

声明清单（编号同 §0）:
 1  N→L P_target=-0.171499, L→N=+0.686717
 2  low source (L→N 5510行) native L2=1.349489 vs command-copy 1.308615;
    low 逐轴MAE command-copy=[0.070384,0.067221,0.090306] Context=[0.068214,0.086826,0.129350]
 3  25772行 968 种 unique actual future; N→L 424 source / 334 target
 4  (概念性, 无法从缓存复算 -> 在 A5 处理)
 5  N→L swap-vs-native direct distance 0.370617 = native target error 的 27.46%
 6  calib_slope 是 pred-on-truth; vlow/vx truth-on-pred ≈ 0.993 (pearson^2 / slope)
 7  affine 后 N→L P_target ≈ +0.00857, CI [-0.003,+0.021] 跨 0
 8  class_tau0.25 == class_tau0.5; 缓存每窗 min age ∈ {0,1,inf}
 9  N/L "真实工况差距 1.22" 实为 D_before=1.21672; 真 actual-actual = 1.485315;
    target native error 1.349489 < 1.485315
10  N→L squared gain 分解: vx +0.033757, vy +0.110085, wz -0.519081, 总 -0.375238

用法: python -m execution_wm.validity_v06.external_review_check --config execution_wm/configs/v06_validity.yaml
"""
import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd
import yaml

TOL = 5e-4  # 声明复算容差（float32 缓存 + 四舍五入）


def flat_norm(x):
    return np.linalg.norm(x.reshape(len(x), -1), axis=1)


def key_rows(arr):
    """把 [N,40,3] float32 数组按字节 hash 成行 key，用于唯一性/匹配计数。"""
    return [hashlib.sha1(r.tobytes()).hexdigest() for r in arr]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out = os.path.join(cfg["out_dir"], "audit")
    os.makedirs(out, exist_ok=True)

    cache = dict(np.load(cfg["pred_cache"]))
    man = pd.read_csv(cfg["pair_manifest"])
    N = len(man)
    u, e_a, e_b = cache["u"], cache["e_a"], cache["e_b"]
    e_c = u + cache["rhat_correct"]          # correct-context prediction
    e_s = u + cache["rhat_swap"]             # swapped prediction
    directions = (man.cond_A + "->" + man.cond_B).values
    cross = (man.kind == "cross").values

    rows = []  # (id, item, claimed, recomputed, match)

    def rec(i, item, claimed, recomputed, tol=TOL):
        ok = (claimed is None) or (abs(recomputed - claimed) <= tol * max(1.0, abs(claimed)))
        rows.append({"id": i, "item": item, "claimed": claimed,
                     "recomputed": recomputed, "match": bool(ok)})
        return recomputed

    # ---------- 1. P_target ----------
    D_before = flat_norm(e_c - e_b)
    D_after = flat_norm(e_s - e_b)
    P = D_before - D_after
    for d, claimed in (("normal->friction_low", -0.171499),
                       ("friction_low->normal", 0.686717)):
        sel = cross & (directions == d)
        rec(1, f"P_target {d}", claimed, float(P[sel].mean()))

    # ---------- 2. low source native vs command-copy ----------
    sel = cross & (directions == "friction_low->normal")
    assert sel.sum() == 5510, sel.sum()
    native_l2 = flat_norm(e_c[sel] - e_a[sel]).mean()
    ccopy_l2 = flat_norm(u[sel] - e_a[sel]).mean()
    rec(2, "low source native L2", 1.349489, float(native_l2))
    rec(2, "low source command-copy L2", 1.308615, float(ccopy_l2))
    names = ["vx", "vy", "wz"]
    for i, n in enumerate(names):
        rec(2, f"low MAE {n} command-copy", [0.070384, 0.067221, 0.090306][i],
            float(np.abs(u[sel][:, :, i] - e_a[sel][:, :, i]).mean()))
        rec(2, f"low MAE {n} Context", [0.068214, 0.086826, 0.129350][i],
            float(np.abs(e_c[sel][:, :, i] - e_a[sel][:, :, i]).mean()))

    # ---------- 3. unique futures ----------
    rec(3, "unique actual futures (all rows)", 968,
        float(len(set(key_rows(e_a)) | set(key_rows(e_b)))), tol=0.01)
    sel_nl = cross & (directions == "normal->friction_low")
    rec(3, "N->L unique source futures", 424, float(len(set(key_rows(e_a[sel_nl])))), tol=0.01)
    rec(3, "N->L unique target futures", 334, float(len(set(key_rows(e_b[sel_nl])))), tol=0.01)

    # ---------- 5. swap-vs-native direct distance (match by identical (u, e_a)) ----------
    # native target prediction = 缓存中 source future 与该行 target future 完全相同
    # 且 command future 也完全相同的行的 rhat_correct
    key_map = {}
    for i in range(N):
        k = hashlib.sha1(u[i].tobytes() + e_a[i].tobytes()).hexdigest()
        key_map.setdefault(k, i)
    dists, matched = [], 0
    for i in np.where(sel_nl)[0]:
        k = hashlib.sha1(u[i].tobytes() + e_b[i].tobytes()).hexdigest()
        if k in key_map:
            j = key_map[k]
            dists.append(flat_norm((e_s[i] - e_c[j])[None]).item())
            matched += 1
    dists = np.array(dists)
    rec(5, "N->L swap-vs-native mean distance", 0.370617, float(dists.mean()))
    rec(5, "N->L distance / native target error", 0.2746,
        float(dists.mean() / native_l2_for(cache, man, "friction_low->normal")), tol=0.02)
    n_matched_claim = 5510  # 外部复核称四个方向均完整匹配
    rec(5, "N->L matched rows", float(n_matched_claim), float(matched), tol=0.01)

    # ---------- 6. truth-on-pred slope ----------
    cal = pd.read_csv(cfg["per_axis_calibration"])
    row = cal[(cal.condition == "friction_vlow") & (cal.axis == "vx")].iloc[0]
    tot = float(row.pearson ** 2 / row.calib_slope)
    rec(6, "vlow/vx truth-on-pred slope (pearson^2/a)", 0.993, tot, tol=0.01)
    row2 = cal[(cal.condition == "friction_low") & (cal.axis == "vx")].iloc[0]
    rec(6, "low/vx truth-on-pred slope", 1.03828, float(row2.pearson ** 2 / row2.calib_slope), tol=0.01)
    row3 = cal[(cal.condition == "friction_low") & (cal.axis == "wz")].iloc[0]
    rec(6, "low/wz truth-on-pred slope", 0.84933, float(row3.pearson ** 2 / row3.calib_slope), tol=0.01)

    # ---------- 8. transition class 0.25 vs 0.5 相同; min age 取值 ----------
    tw = pd.read_csv(cfg["transition_window_manifest"])
    same_class = bool((tw["class_tau0.25"] == tw["class_tau0.5"]).all())
    rec(8, "class_tau0.25 identical to class_tau0.5", 1.0, float(same_class), tol=0.01)
    tsc_min = np.nanmin(cache["t_since_change"], axis=1)
    uniq = sorted(set(np.round(tsc_min, 6).tolist()))
    rec(8, "distinct per-window min-age values", 3.0, float(len(uniq)), tol=0.01)
    rec(8, "min-age value set == {0,1,inf}", 1.0,
        float(set(uniq) == {0.0, 1.0, float("inf")}), tol=0.01)
    rec(8, "rows with all-inf age", 10682.0, float(np.isinf(tsc_min).sum()), tol=0.01)

    # ---------- 9. D_before vs physical actual-actual ----------
    rec(9, "N->L D_before mean", 1.21672, float(D_before[sel_nl].mean()))
    phys = flat_norm(e_b - e_a)
    rec(9, "N->L ||e_B - e_A|| mean (actual-actual)", 1.485315, float(phys[sel_nl].mean()))
    sel_vl = cross & (directions == "normal->friction_vlow")
    rec(9, "N->VL D_before mean", 1.64653, float(D_before[sel_vl].mean()), tol=0.01)
    rec(9, "N->VL ||e_B - e_A|| mean", 1.88460, float(phys[sel_vl].mean()), tol=0.01)

    # ---------- 10. squared gain decomposition ----------
    d_vec = e_s - e_c                       # prediction change due to swap
    a_vec = e_b - e_c                       # target error of correct prediction
    sq_gain = 2 * (d_vec * a_vec).sum(axis=(1, 2)) - (d_vec ** 2).sum(axis=(1, 2))
    rec(10, "N->L squared_target_gain total", -0.375238, float(sq_gain[sel_nl].mean()))
    for i, n in enumerate(names):
        g = (2 * (d_vec[:, :, i] * a_vec[:, :, i]).sum(1) - (d_vec[:, :, i] ** 2).sum(1))
        rec(10, f"N->L per_axis_squared_gain {n}",
            [0.033757, 0.110085, -0.519081][i], float(g[sel_nl].mean()))

    # ---------- 汇总 ----------
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "metric_equivalence.csv"), index=False)
    n_bad = int((~df.match).sum())
    lines = ["# DIFFERENCE_REPORT — 外部复核 10 条声明逐项复算", "",
             f"复算基于 v0.5 缓存（float32）与原始 manifest，容差 {TOL}（相对）。",
             f"共 {len(df)} 项，一致 {int(df.match.sum())}，**不一致 {n_bad}**。", ""]
    if n_bad:
        lines += ["## 不一致项", "", "```", df[~df.match].to_string(index=False), "```", ""]
    lines += ["## 全量对照", "", "```", df.round(6).to_string(index=False), "```", "",
              "注：声明 4（episode-pair cluster 仍共享 episode）为概念性批评，无法由缓存复算，",
              "在 A5 依赖统计中以 node-aware / leave-one-episode-out 处理。声明 7（affine 后 CI 跨 0）",
              "与 V0.5 报告一致，已在本轮报告措辞中降级为 inconclusive。"]
    with open(os.path.join(out, "DIFFERENCE_REPORT.md"), "w") as f:
        f.write("\n".join(lines))
    print(df.round(6).to_string(index=False))
    print(f"\n[check] {len(df)} 项, 不一致 {n_bad} -> {out}")


def native_l2_for(cache, man, direction):
    u, e_a = cache["u"], cache["e_a"]
    e_c = u + cache["rhat_correct"]
    sel = (man.kind == "cross").values & ((man.cond_A + "->" + man.cond_B).values == direction)
    return float(np.linalg.norm((e_c[sel] - e_a[sel]).reshape(sel.sum(), -1), axis=1).mean())


if __name__ == "__main__":
    main()
