"""Step 5/8/9/10: 执行 swap 并计算 pair-level 指标（§7/§9）。

对 pair_manifest.csv 中每个 pair:
  cross pairs: correct / swap / shuffle(×K seeds) / zero 四类预测
  same pairs:  correct / swap（§8 same-condition control）

输出:
  metrics/pair_level_metrics.csv   每对每指标（overall + vx/vy/wz 分项）
  raw/pair_predictions.npz         全部预测数组（复核用）

用法:
  python -m execution_wm.context_swap.run_swap --config execution_wm/configs/context_swap.yaml \
      [--max-pairs 50]   # Step 5 sanity: 先跑 20-50 对
"""
import argparse
import csv
import os

import numpy as np
import torch

from execution_wm.context_swap.common import (
    EpisodeData, VEL_NAMES, discover_all, load_cfg, load_context_model, out_subdir, predict,
)

EPS = 1e-8


def norms(a):
    """a: [H,3] -> overall L2 of flattened + 每维 L2。"""
    return {**{"overall": float(np.linalg.norm(a))},
            **{n: float(np.linalg.norm(a[:, i])) for i, n in enumerate(VEL_NAMES)}}


def cosine(a, b):
    fa, fb = a.reshape(-1), b.reshape(-1)
    return float(fa @ fb / (np.linalg.norm(fa) * np.linalg.norm(fb) + EPS))


def cosine_dims(a, b):
    return {n: cosine(a[:, i:i + 1], b[:, i:i + 1]) for i, n in enumerate(VEL_NAMES)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--max-pairs", type=int, default=0, help="0=全部")
    args = p.parse_args()
    cfg = load_cfg(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(cfg["seed"])

    L = int(round(cfg["history_s"] * cfg["hz"]))
    H = int(round(cfg["horizon_s"] * cfg["hz"]))
    K = cfg["shuffle_seeds"]

    model = load_context_model(cfg, device)
    eps = {uid: EpisodeData(e) for uid, e in discover_all(cfg).items()}

    import pandas as pd
    manifest = pd.read_csv(os.path.join(cfg["out_dir"], "pairs", "pair_manifest.csv"))
    if args.max_pairs > 0:
        # 分层抽样：cross/same 各半（manifest 按 kind 聚块，直接 head 会只拿到 cross）
        half = max(1, args.max_pairs // 2)
        manifest = pd.concat([
            manifest[manifest.kind == "cross"].iloc[:: max(1, (manifest.kind == "cross").sum() // half)][:half],
            manifest[manifest.kind == "same"].iloc[:: max(1, (manifest.kind == "same").sum() // half)][:half],
        ])
    print(f"[swap] {len(manifest)} pairs to run")

    # shuffle context 池：所有 probe episode 的随机 (ep, t0) context
    pool_eps = [e for e in eps.values() if e.meta.get("episode_type") == "probe"]

    rows = []
    raw = {}
    for pi, r in enumerate(manifest.itertuples()):
        ea, eb = eps[r.ep_A], eps[r.ep_B]
        t0 = int(r.t0)
        wa, wb = ea.window(t0, L, H), eb.window(t0, L, H)
        e_a, e_b = wa["future_execution"], wb["future_execution"]
        u = wa["future_action"]

        e_correct, c_a = predict(model, ea, t0, L, H, device)
        c_b = torch.from_numpy(
            predict(model, eb, t0, L, H, device)[1])[None].to(device)
        e_swap, _ = predict(model, ea, t0, L, H, device, context=c_b)
        e_zero, _ = predict(model, ea, t0, L, H, device, zero_context=True)

        row = dict(pair_idx=pi, kind=r.kind, probe=r.probe,
                   cond_A=r.cond_A, cond_B=r.cond_B, ep_A=r.ep_A, ep_B=r.ep_B,
                   t0=t0, window_type=r.window_type, D_state=r.D_state)
        if r.kind == "cross":
            shuffle_errs, shuffle_shifts = [], []
            for k in range(K):
                ep_r = pool_eps[rng.integers(len(pool_eps))]
                t_r = int(rng.integers(L - 1, ep_r.T - H - 1))
                c_r = torch.from_numpy(
                    predict(model, ep_r, t_r, L, H, device)[1])[None].to(device)
                e_sh, _ = predict(model, ea, t0, L, H, device, context=c_r)
                shuffle_errs.append(np.linalg.norm(e_sh - e_a))
                shuffle_shifts.append(np.linalg.norm(e_sh - e_correct))
            d_swap = e_swap - e_correct
            d_tgt = e_b - e_a
            m = {
                "E_correct": norms(e_correct - e_a),
                "E_swap": norms(e_swap - e_a),
                "E_zero": norms(e_zero - e_a),
                "dE_self": {k: v - norms(e_correct - e_a)[k]
                            for k, v in norms(e_swap - e_a).items()},
                "D_before": norms(e_correct - e_b),
                "D_after": norms(e_swap - e_b),
                "S_dir": {"overall": cosine(d_swap, d_tgt), **cosine_dims(d_swap, d_tgt)},
                "S_mag": {k: v / (norms(d_tgt)[k] + EPS)
                          for k, v in norms(d_swap).items()},
                "M_shift_cross": norms(d_swap),
                "E_shuffle_mean": float(np.mean(shuffle_errs)),
                "E_shuffle_std": float(np.std(shuffle_errs)),
                "M_shift_shuffle_mean": float(np.mean(shuffle_shifts)),
            }
            m["P_target"] = {k: m["D_before"][k] - m["D_after"][k]
                             for k in m["D_before"]}
            row.update({f"{mk}_{dk}": v for mk, mv in m.items() if isinstance(mv, dict)
                        for dk, v in mv.items()})
            row.update({mk: mv for mk, mv in m.items() if not isinstance(mv, dict)})
            if pi < 8 or (r.probe == "probe_1_straight" and pi < 40):
                raw[f"pair{pi}"] = dict(u=u, e_a=e_a, e_b=e_b, e_correct=e_correct,
                                        e_swap=e_swap, c_a=c_a, c_b=c_b[0].cpu().numpy(),
                                        meta=np.array([r.probe, r.cond_A, r.cond_B]))
        else:  # same-condition control
            d_swap = e_swap - e_correct
            row["M_shift_same"] = float(np.linalg.norm(d_swap))
            row["E_correct_overall"] = float(np.linalg.norm(e_correct - e_a))
            row["E_swap_overall"] = float(np.linalg.norm(e_swap - e_a))
        rows.append(row)
        if (pi + 1) % 100 == 0:
            print(f"[swap] {pi+1}/{len(manifest)}")

    out_m = out_subdir(cfg, "metrics")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_m, "pair_level_metrics.csv"), index=False)
    out_r = out_subdir(cfg, "raw")
    np.savez(os.path.join(out_r, "pair_predictions.npz"),
             **{k: {kk: vv for kk, vv in v.items()} for k, v in raw.items()})
    print(f"[swap] -> {out_m}/pair_level_metrics.csv ({len(df)} rows)")


if __name__ == "__main__":
    main()
