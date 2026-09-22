"""V0.6 阶段A4：复算旧 metric 并补齐简单基线（同一批 pair 窗口、同一输入）。

基线（全部 fixed-origin [H,3]，同一 origin）:
  command-copy : e_hat[k] = u[k]
  persistence  : e_hat[k] = execution[t0]（origin 处最后可用执行，不用未来帧）
  action_only  : V0 checkpoint（仅 future command 输入）
  direct       : V0 checkpoint（history+future command）
  context      : V0 checkpoint（缓存 rhat_correct，已验证逐元素一致）

聚合口径（全部并列报告，不混合）:
  pair_weighted（旧定义）/ unique_window（仅去重，不宣称独立）/
  equal_episode / equal_anchor / equal_probe
误差: per-axis MAE/RMSE + 旧 flat-L2（仅审计同口径用）+ lead_time 分段（0.25/0.5/1/2s）。

输出:
  metrics/native_baselines_per_window.csv   （每 unique source window × model）
  metrics/native_baselines_per_episode.csv
  metrics/native_baselines_per_horizon.csv
  metrics/per_axis_and_pose_metrics.csv
  metrics/baseline_summary.csv              （5 种聚合口径 × model）

用法: python -m execution_wm.validity_v06.baselines --config execution_wm/configs/v06_validity.yaml
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import yaml

from execution_wm.context_swap.common import EpisodeData, to_torch
from execution_wm.data.dataset import discover_episodes
from execution_wm.eval.evaluate_execution import load_model

VEL = ["vx", "vy", "wz"]
LEADS = {"0.25s": 5, "0.5s": 10, "1s": 20, "2s": 40}


def batched_predict(model, name, windows, device, bs=256):
    """windows: list of window dict -> [N,H,3] e_hat。"""
    outs = []
    for i in range(0, len(windows), bs):
        chunk = windows[i:i + bs]
        b = {k: torch.from_numpy(np.stack([w[k] for w in chunk])).to(device)
             for k in ("history_proprio", "history_action", "future_action")}
        with torch.no_grad():
            if name == "action_only":
                out = model(b["future_action"])
            elif name == "direct":
                out = model(b["history_proprio"], b["history_action"], b["future_action"])
            else:
                out = model(b["history_proprio"], b["history_action"],
                            b["history_proprio"][:, -1], b["future_action"])
        r_hat = out["r_hat"].cpu().numpy()
        outs.append(b["future_action"].cpu().numpy() + r_hat)
    return np.concatenate(outs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out_m = os.path.join(cfg["out_dir"], "metrics")
    os.makedirs(out_m, exist_ok=True)
    hz = cfg["hz"]
    L = int(round(cfg["history_s"] * hz))
    H = int(round(cfg["horizon_s"] * hz))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    man = pd.read_csv(cfg["pair_manifest"])
    full = pd.read_csv(os.path.join(cfg["out_dir"], "manifests", "full_pair_manifest.csv"))
    full = full[full.accepted].reset_index(drop=True)
    cache = dict(np.load(cfg["pred_cache"]))
    u, e_a = cache["u"], cache["e_a"]

    # ---------- unique source windows ----------
    man["sw_key"] = man.ep_A + "@" + man.t0.astype(str)
    keys, first_idx = np.unique(man.sw_key.values, return_index=True)
    print(f"[A4] {len(man)} rows -> {len(keys)} unique source windows")

    eps_idx = {e["meta"]["episode_id"]: e
               for e in discover_episodes(cfg["dataset_v0"])}
    eps1 = {e["meta"]["episode_id"]: e
            for e in discover_episodes(cfg["dataset_probes"])}

    def load_ep(uid):
        ver, eid = uid.split("_ep")
        return EpisodeData(eps_idx[int(eid)] if ver == "d0" else eps1[int(eid)])

    # persistence + direct/action_only 只在 unique window 上算
    uniq = man.iloc[np.sort(first_idx)].reset_index(drop=True)
    ep_cache, persist, windows = {}, [], []
    for r in uniq.itertuples():
        uid = r.ep_A
        if uid not in ep_cache:
            ep_cache[uid] = load_ep(uid)
        ep = ep_cache[uid]
        t0 = int(r.t0)
        windows.append(ep.window(t0, L, H))
        persist.append(np.repeat(ep.execution[t0][None, :], H, axis=0))
    persist = np.stack(persist)
    e_u = np.stack([w["future_execution"] for w in windows])     # GT
    u_u = np.stack([w["future_action"] for w in windows])

    exp_dir = os.path.dirname(os.path.dirname(cfg["checkpoint"]))
    preds = {"command-copy": u_u, "persistence": persist}
    for name in ("action_only", "direct", "context"):
        model = load_model(exp_dir, name, H, device)
        preds[name] = batched_predict(model, name, windows, device)
        print(f"[A4] {name} predicted")

    # context 缓存值与重算应一致（抽样断言；batched GRU 有 ~1e-4 数值非确定性，atol=1e-3）
    samp = np.random.default_rng(0).integers(0, len(uniq), 20)
    max_d = 0.0
    for s in samp:
        row = np.where(man.sw_key.values == uniq.sw_key.values[s])[0][0]
        max_d = max(max_d, float(np.abs(
            preds["context"][s] - (u_u[s] + cache["rhat_correct"][row])).max()))
    assert max_d < 1e-3, f"context 重算与缓存不一致: {max_d}"
    print(f"[A4] context recompute ~= cache (20 samples, max diff {max_d:.1e}, batched-GRU 数值噪声)")

    # ---------- per-window metrics ----------
    rows = []
    for mi, mname in enumerate(preds):
        err = preds[mname] - e_u                                  # [N,H,3]
        for wi in range(len(uniq)):
            r = uniq.iloc[wi]
            row = {"sw_key": r.sw_key, "episode_uid": r.ep_A, "model": mname,
                   "condition": full.source_condition.iloc[0] if False else
                   man.cond_A.iloc[np.where(man.sw_key.values == r.sw_key)[0][0]],
                   "probe": r.probe, "t0": r.t0,
                   "anchor_group": full.source_anchor_group.iloc[
                       np.where(man.sw_key.values == r.sw_key)[0][0]]}
            row["L2_flat"] = float(np.linalg.norm(err[wi]))
            for i, n in enumerate(VEL):
                row[f"MAE_{n}"] = float(np.abs(err[wi, :, i]).mean())
                row[f"RMSE_{n}"] = float(np.sqrt((err[wi, :, i] ** 2).mean()))
            for lname, k in LEADS.items():
                row[f"L2_{lname}"] = float(np.linalg.norm(err[wi, :k]))
            rows.append(row)
    pw = pd.DataFrame(rows)
    pw.to_csv(os.path.join(out_m, "native_baselines_per_window.csv"), index=False)

    # ---------- per-episode / per-horizon / per-axis ----------
    pe = pw.groupby(["episode_uid", "model"])[
        ["L2_flat", "MAE_vx", "MAE_vy", "MAE_wz", "RMSE_vx", "RMSE_vy", "RMSE_wz"]
    ].mean().reset_index()
    pe.to_csv(os.path.join(out_m, "native_baselines_per_episode.csv"), index=False)

    ph = pw.melt(id_vars=["model"], value_vars=[f"L2_{l}" for l in LEADS],
                 var_name="lead", value_name="L2")
    ph = ph.groupby(["model", "lead"]).L2.mean().reset_index()
    ph["lead_s"] = ph.lead.str.strip("L2_")
    ph.to_csv(os.path.join(out_m, "native_baselines_per_horizon.csv"), index=False)

    pa = pw.groupby("model")[["MAE_vx", "MAE_vy", "MAE_wz",
                              "RMSE_vx", "RMSE_vy", "RMSE_wz", "L2_flat"]].mean().reset_index()
    pa.to_csv(os.path.join(out_m, "per_axis_and_pose_metrics.csv"), index=False)

    # ---------- 5 种聚合口径 ----------
    # 行级（pair_weighted）：unique window 预测映射回 25772 行
    key_to_idx = {k: i for i, k in enumerate(uniq.sw_key)}
    row_idx = man.sw_key.map(key_to_idx).values
    summ = []
    for mname, pr in preds.items():
        err_rows = pr[row_idx] - e_a                                # [25772,H,3]
        l2_rows = np.linalg.norm(err_rows.reshape(len(err_rows), -1), axis=1)
        mae_rows = np.abs(err_rows).mean(axis=(1, 2))
        entry = {"model": mname, "pair_weighted_L2": float(l2_rows.mean()),
                 "pair_weighted_MAE": float(mae_rows.mean())}
        # unique window
        l2_u = np.linalg.norm((pr - e_u).reshape(len(pr), -1), axis=1)
        entry["unique_window_L2"] = float(l2_u.mean())
        # equal episode / anchor / probe（先组内均值再总均值）
        tmp = pd.DataFrame({"ep": uniq.ep_A, "anchor": pw[pw.model == mname].anchor_group.values,
                            "probe": uniq.probe, "l2": l2_u})
        entry["equal_episode_L2"] = float(tmp.groupby("ep").l2.mean().mean())
        entry["equal_anchor_L2"] = float(tmp.groupby("anchor").l2.mean().mean())
        entry["equal_probe_L2"] = float(tmp.groupby("probe").l2.mean().mean())
        summ.append(entry)
    bs = pd.DataFrame(summ)
    bs.to_csv(os.path.join(out_m, "baseline_summary.csv"), index=False)
    pd.set_option("display.width", 200)
    print(bs.round(4).to_string(index=False))
    print("\n=== per-axis (unique-window mean) ===")
    print(pa.round(4).to_string(index=False))
    print(f"[A4] -> {out_m}")


if __name__ == "__main__":
    main()
