"""Step 15 辅助: NO-GO 诊断（§16）。不改模型，只做分析。

诊断项（对应 §16 清单）:
  1. predictor 是否绕过 c_t        -> M_shift 已证否（swap 改变预测），这里看 c 对输出的贡献率
  3. z 是否已编码全部 condition     -> 比较 direct（无 c）与 context 原生误差
  4. pair state confound           -> P_target 与 D_state 相关性
  校准性                            -> 同一批窗口上 |r_hat(c_A)| vs |r_hat(c_B)| vs 真实 |r|
  regime teleport                  -> D_after(方向 A->B) vs target 条件原生 E_correct

输出 metrics/diagnostics.json + 打印。
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from execution_wm.context_swap.common import (
    EpisodeData, discover_all, load_cfg, load_context_model, predict,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_cfg(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    L = int(round(cfg["history_s"] * cfg["hz"]))
    H = int(round(cfg["horizon_s"] * cfg["hz"]))
    rng = np.random.default_rng(cfg["seed"])

    out = os.path.join(cfg["out_dir"], "metrics")
    df = pd.read_csv(os.path.join(out, "pair_level_metrics.csv"))
    cross = df[df.kind == "cross"].copy()
    cross["direction"] = cross.cond_A + "->" + cross.cond_B
    diag = {}

    # ---- regime teleport: D_after(A->B) vs native E_correct(B) ----
    native = {}
    for cond in ("normal", "friction_low", "friction_vlow"):
        g = cross[cross.cond_A == cond]
        native[cond] = float(g.E_correct_overall.mean())
    teleport = {}
    for direction, g in cross.groupby("direction"):
        tgt = direction.split("->")[1]
        teleport[direction] = {"D_after": float(g.D_after_overall.mean()),
                               "native_E_correct_target": native[tgt],
                               "ratio": float(g.D_after_overall.mean() / native[tgt])}
    diag["regime_teleport"] = teleport

    # ---- confound: P_target vs D_state 相关 ----
    conf = {}
    for direction, g in cross.groupby("direction"):
        conf[direction] = float(np.corrcoef(g.D_state, g.P_target_overall)[0, 1])
    diag["corr_P_target_vs_D_state"] = conf

    # ---- 校准性：同一窗口 c_A vs c_B 的预测 residual 幅值 vs 真实 ----
    eps = {uid: EpisodeData(e) for uid, e in discover_all(cfg).items()}
    model = load_context_model(cfg, device)
    sub = cross[cross.direction.isin(["normal->friction_low", "friction_low->normal"])]
    sub = sub.iloc[rng.choice(len(sub), size=min(400, len(sub)), replace=False)]
    rows = {"c_normal": [], "c_low": [], "actual_normal": [], "actual_low": []}
    for r in sub.itertuples():
        ea, eb = eps[r.ep_A], eps[r.ep_B]
        t0 = int(r.t0)
        wa, wb = ea.window(t0, L, H), eb.window(t0, L, H)
        src_is_normal = r.cond_A == "normal"
        ep_n, ep_l = (ea, eb) if src_is_normal else (eb, ea)
        _, c_n = predict(model, ep_n, t0, L, H, device)
        _, c_l = predict(model, ep_l, t0, L, H, device)
        e_n, _ = predict(model, ea, t0, L, H, device,
                         context=torch.from_numpy(c_n)[None].to(device))
        e_l, _ = predict(model, ea, t0, L, H, device,
                         context=torch.from_numpy(c_l)[None].to(device))
        u = wa["future_action"]
        rows["c_normal"].append(np.abs(e_n - u).mean())
        rows["c_low"].append(np.abs(e_l - u).mean())
        rows["actual_normal"].append(np.abs(
            (wa if src_is_normal else wb)["future_execution"] - u).mean())
        rows["actual_low"].append(np.abs(
            (wb if src_is_normal else wa)["future_execution"] - u).mean())
    diag["residual_magnitude_calibration"] = {
        "pred_with_c_normal": float(np.mean(rows["c_normal"])),
        "pred_with_c_low": float(np.mean(rows["c_low"])),
        "actual_normal": float(np.mean(rows["actual_normal"])),
        "actual_low": float(np.mean(rows["actual_low"])),
        "note": "c_low 预测的 |r| 是否相对真实 low 条件过大/过小（校准性）",
    }

    # ---- z 是否已经编码 condition：direct(无c) 原生误差 vs context 原生误差 ----
    diag["z_already_encodes_condition"] = {
        "note": "V0 test_id residual MAE: direct(无 c)=0.1291 vs context=0.1278，"
                "差 1%——分布内 z/history 几乎已含全部可预测信息，c 的边际贡献小。"
                "这解释了为何 swap 误差由 target 条件原生误差主导。",
        "direct_test_id_mae": 0.1291, "context_test_id_mae": 0.1278,
    }

    with open(os.path.join(out, "diagnostics.json"), "w") as f:
        json.dump(diag, f, indent=2, ensure_ascii=False)
    print(json.dumps(diag, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
