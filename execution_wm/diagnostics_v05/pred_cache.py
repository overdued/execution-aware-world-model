"""V0.5 基础设施：重算全部 pair 的预测数组并缓存（npz），供 Task A/D/E 使用。

同时充当 metric bug 检查（second_work §0/Step 3）：从缓存重算的标量指标必须与
context_swap_v2/metrics/pair_level_metrics.csv 完全一致，否则记录 metric bug。

输出: results/v0_5_diagnostics/raw/pair_pred_cache.npz
  u[H,3], e_a[H,3], e_b[H,3], rhat_correct[H,3], rhat_swap[H,3]  (每对)
  t_since_change[H]  窗内每步距上次命令变化的时间（秒）
  pair_keys: 与 pair_manifest.csv 行序一致

用法: python -m execution_wm.diagnostics_v05.pred_cache --config execution_wm/configs/v05_diagnostics.yaml
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from execution_wm.context_swap.common import (
    EpisodeData, discover_all, load_cfg, load_context_model, out_subdir, predict,
)


def time_since_change(cmd, t0, H, hz, tau_u):
    """窗内每步距 A 侧最近一次命令变化的时间（秒）。变化点: ||u_t-u_{t-1}||>tau_u。"""
    full = cmd  # [T,3]
    du = np.zeros(len(full))
    du[1:] = np.linalg.norm(np.diff(full, axis=0), axis=1)
    change_idx = np.where(du > tau_u)[0]
    ts = np.arange(t0 + 1, t0 + 1 + H)
    out = np.empty(H)
    for k, t in enumerate(ts):
        prev = change_idx[change_idx <= t]
        out[k] = (t - prev[-1]) / hz if len(prev) else np.inf
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_cfg(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    L = int(round(cfg["history_s"] * cfg["hz"]))
    H = int(round(cfg["horizon_s"] * cfg["hz"]))
    hz = cfg["hz"]
    tau_u = cfg["transition"]["tau_u"]

    model = load_context_model(cfg, device)
    eps = {uid: EpisodeData(e) for uid, e in discover_all(cfg).items()}
    manifest = pd.read_csv(os.path.join(cfg["swap_dir"], "pairs", "pair_manifest.csv"))
    N = len(manifest)
    print(f"[cache] {N} pairs")

    U = np.zeros((N, H, 3), np.float32)
    EA = np.zeros((N, H, 3), np.float32)
    EB = np.zeros((N, H, 3), np.float32)
    RC = np.zeros((N, H, 3), np.float32)
    RS = np.zeros((N, H, 3), np.float32)
    TSC = np.zeros((N, H), np.float32)

    for pi, r in enumerate(manifest.itertuples()):
        ea, eb = eps[r.ep_A], eps[r.ep_B]
        t0 = int(r.t0)
        wa, wb = ea.window(t0, L, H), eb.window(t0, L, H)
        e_correct, _ = predict(model, ea, t0, L, H, device)
        c_b = torch.from_numpy(predict(model, eb, t0, L, H, device)[1])[None].to(device)
        e_swap, _ = predict(model, ea, t0, L, H, device, context=c_b)
        U[pi] = wa["future_action"]
        EA[pi] = wa["future_execution"]
        EB[pi] = wb["future_execution"]
        RC[pi] = e_correct - wa["future_action"]     # r̂ = ê - u
        RS[pi] = e_swap - wa["future_action"]
        TSC[pi] = time_since_change(ea.cmd, t0, H, hz, tau_u)
        if (pi + 1) % 2000 == 0:
            print(f"[cache] {pi+1}/{N}", flush=True)

    out = out_subdir(cfg, "raw")
    np.savez(os.path.join(out, "pair_pred_cache.npz"),
             u=U, e_a=EA, e_b=EB, rhat_correct=RC, rhat_swap=RS, t_since_change=TSC)

    # ---- metric bug 对拍：从缓存重算标量 vs pair_level_metrics.csv ----
    old = pd.read_csv(os.path.join(cfg["swap_dir"], "metrics", "pair_level_metrics.csv"))
    assert len(old) == N, "pair_level_metrics 与 manifest 行数不一致"
    e_hat_c = U + RC
    e_hat_s = U + RS
    recompute = {
        "E_correct_overall": np.linalg.norm((e_hat_c - EA).reshape(N, -1), axis=1),
        "E_swap_overall": np.linalg.norm((e_hat_s - EA).reshape(N, -1), axis=1),
        "D_before_overall": np.linalg.norm((e_hat_c - EB).reshape(N, -1), axis=1),
        "D_after_overall": np.linalg.norm((e_hat_s - EB).reshape(N, -1), axis=1),
    }
    print("[cache] === metric bug 对拍（重算 vs v2 CSV, max abs diff）===")
    worst = 0.0
    for k, v in recompute.items():
        if k not in old:
            print(f"  {k}: v2 CSV 缺列（same pairs 只有部分列）")
            continue
        o = old[k].values
        mask = ~np.isnan(o)
        diff = np.abs(v[mask] - o[mask]).max()
        worst = max(worst, diff)
        print(f"  {k}: max|diff| = {diff:.3e} (n={mask.sum()})")
    print(f"[cache] worst diff = {worst:.3e} -> {'PASS' if worst < 1e-3 else 'FAIL: 存在 metric bug'}")
    print(f"[cache] -> {out}/pair_pred_cache.npz")


if __name__ == "__main__":
    main()
