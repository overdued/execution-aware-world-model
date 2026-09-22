"""V0.6 阶段A6：校准（双向回归 + 分箱可靠性）、geometry 分解、per-timestep transition。

1. 校准（test split 原生预测）:
   - pred-on-truth: rhat = a*r + b（诊断用，<1 不证明均值失准——见报告教学注记）
   - truth-on-pred: r = a_cal*rhat + b_cal（部署校准方向）
   - 按预测分箱: E[truth | pred bin] ± episode-cluster CI，mean pred / mean truth / bias
   - affine 只在 val 拟合（沿用 V0.5 参数），test 只评价
   - ORACLE shrink（test 上最优标量 gamma）只标 ORACLE_DIAGNOSTIC
2. geometry（swap 分析，cache 全量）:
   d = pred_swap - pred_correct, a = e_b - pred_correct
   squared_target_gain = 2<d,a> - ||d||^2
   parallel_shift = <d, a_hat>; orthogonal_shift; overshoot = max(0, parallel - ||a||)
   per_axis_squared_gain; native-vs-swap 直接距离（绝对/相对/per-axis）
3. transition per-timestep：从 episode 完整命令事件表逐 target timestep 计算 age
   （episode 起点到首次观测变化之间标 unknown，不用 inf=steady），
   输出 per-class 的 P_target 贡献（逐 timestep |e_c-e_b| - |e_s-e_b|，per-axis + overall）。

输出:
  metrics/prediction_reliability.csv
  metrics/geometry_decomposition.csv
  metrics/transition_timestep_metrics.csv
  figures/CAL6_reliability_<cond>_<axis>.png, figures/GEO6_decomposition.png,
  figures/TRANS6_timestep.png

用法: python -m execution_wm.validity_v06.calibration_geometry --config execution_wm/configs/v06_validity.yaml
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

from execution_wm.context_swap.common import EpisodeData, load_cfg, predict  # noqa: E402
from execution_wm.data.dataset import discover_episodes, load_episode, split_episodes  # noqa: E402
from execution_wm.eval.evaluate_execution import load_model  # noqa: E402

VEL = ["vx", "vy", "wz"]
EPS = 1e-8
DIRECTIONS = ["normal->friction_low", "friction_low->normal",
              "normal->friction_vlow", "friction_vlow->normal"]


def lstsq_slope(x, y):
    A = np.stack([x, np.ones_like(x)], 1)
    return np.linalg.lstsq(A, y, rcond=None)[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_cfg(args.config)
    out_m = os.path.join(cfg["out_dir"], "metrics")
    fig_d = os.path.join(cfg["out_dir"], "figures")
    os.makedirs(out_m, exist_ok=True)
    os.makedirs(fig_d, exist_ok=True)
    hz = cfg["hz"]
    L = int(round(cfg["history_s"] * hz))
    H = int(round(cfg["horizon_s"] * hz))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(cfg["seed"])

    # ================= 1. 校准（test split，双向 + 分箱 + episode CI） =================
    tc = yaml.safe_load(open(cfg["train_config"]))["dataset"]
    eps_all = [e for e in discover_episodes(cfg["dataset_v0"])
               if e["meta"].get("episode_type") == "random"]
    splits = split_episodes(eps_all, tc["val_fraction"], tc["test_id_fraction"],
                            tc["ood_conditions"], True, 42)
    test_eps = splits["test_id"] + splits["test_ood"]
    exp_dir = os.path.dirname(os.path.dirname(cfg["checkpoint"]))
    model = load_model(exp_dir, "context", H, device)

    data = {}
    ep_of = {}
    for e in test_eps:
        cond = e["meta"]["condition"]
        uid = f"d0_ep{e['meta']['episode_id']:05d}"
        ep = EpisodeData(e)
        t0s = np.arange(L - 1, ep.T - H - 1)
        if not len(t0s):
            continue
        for t0 in rng.choice(t0s, size=min(40, len(t0s)), replace=False):
            w = ep.window(int(t0), L, H)
            e_hat, _ = predict(model, ep, int(t0), L, H, device)
            d_ = data.setdefault(cond, {"rhat": [], "r": []})
            d_["rhat"].append(e_hat - w["future_action"])
            d_["r"].append(w["future_residual"])
            ep_of.setdefault(cond, []).append(uid)

    # val affine 参数（沿用 V0.5，只评价不重拟合）
    aff = json.load(open(os.path.join(
        os.path.dirname(cfg["pred_cache"]).replace("raw", "calibration"),
        "global_affine_params.json")))

    rel_rows = []
    for cond, d_ in data.items():
        rhat = np.stack(d_["rhat"])
        r = np.stack(d_["r"])
        eps_arr = np.array(ep_of[cond])
        for i, n in enumerate(VEL):
            x, y = rhat[:, :, i].reshape(-1), r[:, :, i].reshape(-1)
            a_pt, b_pt = lstsq_slope(y, x)             # pred-on-truth: rhat = a*r + b
            a_tp, b_tp = lstsq_slope(x, y)               # truth-on-pred: r = a*rhat + b
            # 分箱可靠性 E[truth | pred bin] + episode CI
            bins = np.quantile(x, np.linspace(0, 1, 11))
            bid = np.clip(np.digitize(x, bins[1:-1]), 0, 9)
            flat_ep = np.repeat(eps_arr, r.shape[1])
            bin_stats = []
            for b_ in range(10):
                m = bid == b_
                if m.sum() < 30:
                    continue
                per_ep = pd.Series(y[m] - x[m]).groupby(flat_ep[m]).mean()
                bin_stats.append({"bin": b_, "mean_pred": float(x[m].mean()),
                                  "mean_truth": float(y[m].mean()),
                                  "bias": float((x[m] - y[m]).mean()),
                                  "ep_ci_half": float(1.96 * per_ep.std() / np.sqrt(len(per_ep)))
                                  if len(per_ep) > 2 else None,
                                  "n_ep": int(len(per_ep))})
            alpha = aff["alpha"][n]
            beta = aff["beta"][n]
            r_cal = alpha * rhat[:, :, i] + beta
            # ORACLE: test 上最优标量 gamma（只标 ORACLE_DIAGNOSTIC）
            g_or = float((rhat[:, :, i] * r[:, :, i]).sum() / ((rhat[:, :, i] ** 2).sum() + EPS))
            rel_rows.append({
                "condition": cond, "axis": n, "n_windows": len(rhat),
                "pred_on_truth_a": a_pt, "pred_on_truth_b": b_pt,
                "truth_on_pred_a": a_tp, "truth_on_pred_b": b_tp,
                "pearson": float(np.corrcoef(x, y)[0, 1]),
                "affine_alpha_val": alpha, "affine_beta_val": beta,
                "MAE_raw": float(np.abs(x - y).mean()),
                "MAE_affine_val": float(np.abs(r_cal - r[:, :, i]).mean()),
                "oracle_gamma_test_ORACLE_DIAGNOSTIC": g_or,
                "MAE_oracle": float(np.abs(g_or * x - y).mean()),
                "bin_stats_json": json.dumps(bin_stats)})
            if cond in ("friction_low", "normal") :
                fig, ax = plt.subplots(figsize=(6, 5))
                bs = pd.DataFrame(bin_stats)
                bs["ep_ci_half"] = pd.to_numeric(bs.ep_ci_half, errors="coerce").fillna(0.0)
                ax.axhline(0, color="k", lw=.8)
                ax.plot(bs.mean_pred, bs.bias, "o-", label="bias per pred-bin")
                ax.fill_between(bs.mean_pred.values,
                                (bs.bias - bs.ep_ci_half).values,
                                (bs.bias + bs.ep_ci_half).values, alpha=.25)
                ax.set_xlabel(f"mean predicted residual ({n})")
                ax.set_ylabel("bias = pred - truth")
                ax.set_title(f"CAL6 reliability {cond}/{n}: truth-on-pred a={a_tp:.2f}")
                fig.tight_layout()
                fig.savefig(os.path.join(fig_d, f"CAL6_reliability_{cond}_{n}.png"), dpi=150)
                plt.close(fig)
    rel = pd.DataFrame(rel_rows)
    rel.to_csv(os.path.join(out_m, "prediction_reliability.csv"), index=False)
    pd.set_option("display.width", 220)
    print(rel[["condition", "axis", "pred_on_truth_a", "truth_on_pred_a", "pearson",
               "MAE_raw", "MAE_affine_val", "MAE_oracle"]].round(3).to_string(index=False))

    # ================= 2. geometry 分解 =================
    cache = dict(np.load(cfg["pred_cache"]))
    man = pd.read_csv(cfg["pair_manifest"])
    u, e_a, e_b = cache["u"], cache["e_a"], cache["e_b"]
    e_c = u + cache["rhat_correct"]
    e_s = u + cache["rhat_swap"]
    d_vec = e_s - e_c                     # swap 引起的预测位移
    a_vec = e_b - e_c                     # target actual 相对 correct 预测的误差
    directions = (man.cond_A + "->" + man.cond_B).values
    cross = (man.kind == "cross").values
    N = len(man)
    a_norm = np.linalg.norm(a_vec.reshape(N, -1), axis=1) + EPS
    a_hat = a_vec / a_norm[:, None, None]
    parallel = (d_vec * a_hat).sum(axis=(1, 2))
    d_norm = np.linalg.norm(d_vec.reshape(N, -1), axis=1)
    orth = np.sqrt(np.maximum(d_norm ** 2 - parallel ** 2, 0))
    overshoot = np.maximum(parallel - a_norm, 0)
    sq_gain = 2 * (d_vec * a_vec).sum(axis=(1, 2)) - d_norm ** 2
    geo = []
    for direction in DIRECTIONS:
        sel = cross & (directions == direction)
        row = {"direction": direction, "N": int(sel.sum()),
               "squared_target_gain": float(sq_gain[sel].mean()),
               "parallel_shift": float(parallel[sel].mean()),
               "orthogonal_shift": float(orth[sel].mean()),
               "overshoot_component": float(overshoot[sel].mean()),
               "a_norm_mean": float(a_norm[sel].mean()),
               "d_norm_mean": float(d_norm[sel].mean())}
        for i, n in enumerate(VEL):
            g_ax = (2 * (d_vec[:, :, i] * a_vec[:, :, i]).sum(1)
                    - (d_vec[:, :, i] ** 2).sum(1))
            row[f"per_axis_squared_gain_{n}"] = float(g_ax[sel].mean())
        geo.append(row)
    geo = pd.DataFrame(geo)
    geo.to_csv(os.path.join(out_m, "geometry_decomposition.csv"), index=False)
    print("\n=== geometry ===")
    print(geo.round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(geo))
    ax.bar(x - .2, geo.parallel_shift, .2, label="parallel shift (toward target error)")
    ax.bar(x, geo.orthogonal_shift, .2, label="orthogonal shift")
    ax.bar(x + .2, geo.overshoot_component, .2, label="overshoot (past target)")
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("friction_", "f_") for d in geo.direction], fontsize=9)
    ax.set_title("GEO6: swap prediction shift decomposition (mean per pair)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_d, "GEO6_decomposition.png"), dpi=150)
    plt.close(fig)

    # ================= 3. transition per-timestep =================
    eps_map = {}
    for e in discover_episodes(cfg["dataset_v0"]):
        eps_map[f"d0_ep{e['meta']['episode_id']:05d}"] = e["path"]
    for e in discover_episodes(cfg["dataset_probes"]):
        eps_map[f"d1_ep{e['meta']['episode_id']:05d}"] = e["path"]
    full = pd.read_csv(os.path.join(cfg["out_dir"], "manifests", "full_pair_manifest.csv"),
                       low_memory=False)
    full = full[full.accepted].reset_index(drop=True)

    tau = 0.25  # 主口径（秒）；per-timestep 分类不再退化成整窗
    cmd_cache = {}
    cls = np.full((N, H), "unknown", dtype=object)     # per (row, k)
    age_s = np.full((N, H), np.nan)
    for pi in range(N):
        uid = full.source_episode_uid.iloc[pi]
        if uid not in cmd_cache:
            d = load_episode(eps_map[uid])
            cmd = d["cmd_vel"]
            du = np.zeros(len(cmd))
            du[1:] = np.linalg.norm(np.diff(cmd, axis=0), axis=1)
            cmd_cache[uid] = np.where(du > 0.1)[0]
        ch = cmd_cache[uid]
        t0 = int(man.t0.iloc[pi])
        ts = np.arange(t0 + 1, t0 + 1 + H)
        for k, t in enumerate(ts):
            prev = ch[ch <= t]
            if not len(prev):
                continue                          # unknown（episode 起点到首次观测变化）
            age = (t - prev[-1]) / hz
            age_s[pi, k] = age
            cls[pi, k] = "transition" if age <= tau else "steady"

    # per-timestep P_target 贡献（overall flat 与 per-axis）
    e_c_r = e_c.reshape(N, -1)
    e_s_r = e_s.reshape(N, -1)
    e_b_r = e_b.reshape(N, -1)
    d_before_t = np.abs(e_c - e_b)                # [N,H,3]
    d_after_t = np.abs(e_s - e_b)
    tr_rows = []
    for direction in DIRECTIONS:
        sel = cross & (directions == direction)
        for c in ("transition", "steady", "unknown"):
            m = sel[:, None] & (cls == c)          # [N,H]
            if m.sum() < 100:
                continue
            contrib = (np.linalg.norm(d_before_t, axis=2) - np.linalg.norm(d_after_t, axis=2))
            row = {"direction": direction, "timestep_class": c,
                   "n_timesteps": int(m.sum()),
                   "P_target_contrib_per_step": float(contrib[m].mean())}
            for i, n in enumerate(VEL):
                row[f"P_{n}_per_step"] = float(
                    (d_before_t[:, :, i] - d_after_t[:, :, i])[m].mean())
            tr_rows.append(row)
    tr = pd.DataFrame(tr_rows)
    tr.to_csv(os.path.join(out_m, "transition_timestep_metrics.csv"), index=False)
    print("\n=== transition per-timestep (τ=0.25s, 贡献口径) ===")
    print(tr.round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 5))
    for j, c in enumerate(("transition", "steady", "unknown")):
        sub = tr[tr.timestep_class == c]
        ax.bar(np.arange(len(DIRECTIONS)) + (j - 1) * .25, sub.P_target_contrib_per_step,
               .25, label=c)
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(np.arange(len(DIRECTIONS)))
    ax.set_xticklabels([d.replace("friction_", "f_") for d in DIRECTIONS], fontsize=9)
    ax.set_ylabel("P_target contribution per target-timestep")
    ax.set_title("TRANS6: per-timestep transition/steady/unknown (τ=0.25s)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_d, "TRANS6_timestep.png"), dpi=150)
    plt.close(fig)
    print(f"[A6] -> {out_m}, {fig_d}")


if __name__ == "__main__":
    main()
