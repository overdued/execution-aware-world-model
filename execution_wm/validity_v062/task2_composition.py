"""V0.6.2 Task 2 — Primitive Composition Baseline。

    r_hat[:, k, a] = g_a(state, u[t+k, a])            a in {x, y, w}

g_a 只用该轴自己的 future command（+ 一个共享 state 输入），三轴**相加**，无 interaction。
刻意保持最简：每轴 2 层 MLP（或 ridge），只用 train split 拟合，early-stop 只用 val。

目的：检验 unseen Q4（vx 与 wz 同时非零）是否只需"把两个已见 primitive 加起来"就能解决。
若 additive primitive baseline 明显改善 unseen Q4 -> ACTION_COMPOSITION_SIGNAL=SUPPORTED。
**不加入 interaction 网络**（任务书明确）。

用法: python -m execution_wm.validity_v062.task2_composition [--kind mlp|ridge]
输出: metrics/composition_baseline.csv, metrics/action_composition_signal.json
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from execution_wm.validity_v061.windows import H, L
from execution_wm.validity_v062.common import (OUT, SPLITS_EVAL, get_manifests, load_cache,
                                                per_axis_errors, pred_from_cache, seed_mean_pred)

AXES = ("vx", "vy", "wz")
# state 输入：只用 history 的最后一帧（与 M1 的 encode_state 口径一致，避免额外信息）
def state_feat(man):
    return man.hp[:, -1, :].astype(np.float32)


class PrimitiveMLP(nn.Module):
    """每轴一个 2 层 MLP，输入 = [state(40), 该轴 future command(H,1)]。"""

    def __init__(self, state_dim=40, hidden=128):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(state_dim + H, hidden), nn.ELU(),
                          nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, H))
            for _ in range(3)])

    def forward(self, s, fa):
        # fa [B,H,3] -> 每轴 [B,H,1]
        return torch.stack([self.heads[a](torch.cat([s, fa[:, :, a]], dim=-1))
                            for a in range(3)], dim=-1)          # [B,H,3]


class PrimitiveRidge:
    """每轴 ridge：[state(40), u_a(H)] -> r_a(H)。"""

    def __init__(self, lam=1.0):
        self.lam = lam

    def _X(self, s, fa, a):
        return np.concatenate([s, fa[:, :, a]], axis=1).astype(np.float64)

    def fit(self, s, fa, fr):
        self.W = {}
        for a in range(3):
            X = self._X(s, fa, a)
            Xb = np.concatenate([X, np.ones((len(X), 1))], axis=1)
            Y = fr[:, :, a].astype(np.float64)
            self.W[a] = np.linalg.solve(Xb.T @ Xb + self.lam * np.eye(Xb.shape[1]), Xb.T @ Y)
        return self

    def predict(self, s, fa):
        out = np.zeros((len(s), H, 3))
        for a in range(3):
            Xb = np.concatenate([self._X(s, fa, a), np.ones((len(s), 1))], axis=1)
            out[:, :, a] = Xb @ self.W[a]
        return out


def train_mlp(man_tr, man_va, seed=42, epochs=300, bs=128, lr=1e-3, patience=20):
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = PrimitiveMLP().to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    lf = nn.SmoothL1Loss(beta=1.0)
    s_tr = torch.from_numpy(state_feat(man_tr)).to(dev)
    fa_tr = torch.from_numpy(man_tr.fa).to(dev)
    fr_tr = torch.from_numpy(man_tr.fr).to(dev)
    s_va = torch.from_numpy(state_feat(man_va)).to(dev)
    fa_va = torch.from_numpy(man_va.fa).to(dev)
    fr_va = torch.from_numpy(man_va.fr).to(dev)
    best, best_state, pat = float("inf"), None, 0
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        m.train()
        idx = torch.randperm(len(s_tr), generator=g)
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            loss = lf(m(s_tr[b], fa_tr[b]), fr_tr[b])
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            v = lf(m(s_va, fa_va), fr_va).item()
        if v < best - 1e-6:
            best, pat = v, 0
            best_state = {k: t.detach().clone() for k, t in m.state_dict().items()}
        else:
            pat += 1
        if pat >= patience:
            break
    m.load_state_dict(best_state)
    m.eval()
    return m, best, ep + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="mlp", choices=["mlp", "ridge"])
    args = ap.parse_args()
    man = get_manifests()
    cache = load_cache()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 只用 train 拟合 ----
    if args.kind == "ridge":
        base = PrimitiveRidge().fit(state_feat(man["train"]), man["train"].fa, man["train"].fr)
        fit_info = {"kind": "ridge", "lam": 1.0, "val_loss": None, "epochs": None}
    else:
        models, vals, eps = {}, {}, {}
        for seed in (42, 43, 44):
            m, v, ep = train_mlp(man["train"], man["val"], seed=seed)
            models[seed], vals[seed], eps[seed] = m, v, ep
        fit_info = {"kind": "mlp", "val_loss": {str(k): v for k, v in vals.items()},
                    "epochs": eps, "state_dim": 40, "hidden": 128,
                    "params_per_head": None, "n_seeds": 3,
                    "architecture": "[state(40), u_a(H)] -> 128 -> 128 -> H, per axis, additive"}

    # ---- 评估（A/B/C 同一窗口）----
    rows, detail = [], {}
    for split in SPLITS_EVAL:
        m = man[split]
        gt = m.fe
        s = torch.from_numpy(state_feat(m)).to(dev)
        fa = torch.from_numpy(m.fa).to(dev)
        if args.kind == "ridge":
            with torch.no_grad():
                pr = base.predict(state_feat(m), m.fa)
        else:
            prs = []
            for seed in (42, 43, 44):
                with torch.no_grad():
                    prs.append((fa + models[seed](s, fa)).cpu().numpy())
            pr = np.mean(prs, axis=0)
            detail[split] = {"per_seed_err": [
                float(np.abs(p - gt).mean()) for p in prs]}
        variants = {
            "composition_additive": pr,
            "command-copy": pred_from_cache(cache, split, "command-copy"),
            "ridge_multioutput": pred_from_cache(cache, split, "ridge_multioutput"),
        }
        for name in ("M0", "M1", "M2"):
            variants[name] = seed_mean_pred(cache, split, name)
        for vname, p in variants.items():
            pe = per_axis_errors(p, gt)
            row = {"split": split, "model": vname, "n_windows": len(m)}
            for ax in AXES:
                row[f"MAE_{ax}"] = float(pe[ax].mean())
                for k, lead in ((5, "0.25s"), (10, "0.5s"), (20, "1.0s"), (40, "2.0s")):
                    row[f"MAE_{ax}@{lead}"] = float(np.abs(p[:, k - 1, AXES.index(ax)] -
                                                           gt[:, k - 1, AXES.index(ax)]).mean())
            row["MAE_all_mixed"] = float(np.abs(p - gt).mean())
            # trajectory endpoint（Task 4 口径的轻量版：直接积分 predicted body velocity）
            row["endpoint_xy_err"] = float(np.abs(
                np.cumsum(p, axis=1)[:, -1, :2] * 0.05 -
                np.cumsum(gt, axis=1)[:, -1, :2] * 0.05).mean())
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "metrics", "composition_baseline.csv"), index=False)

    # ---- 配对 cluster bootstrap：composition − best_other（同一次 resample）----
    rng = np.random.default_rng(20260923)
    boot = {}
    for split in SPLITS_EVAL:
        m = man[split]
        gt = m.fe
        s_ = state_feat(m)
        if args.kind == "ridge":
            comp = base.predict(s_, m.fa)
        else:
            fa_ = torch.from_numpy(m.fa).to(dev)
            st_ = torch.from_numpy(s_).to(dev)
            with torch.no_grad():
                comp = np.mean([(fa_ + models[seed](st_, fa_)).cpu().numpy()
                                for seed in (42, 43, 44)], axis=0)
        comp_e = np.abs(comp - gt).mean(axis=(1, 2))
        eps = np.array([w["episode_id"] for w in m.windows])
        units = np.unique(eps)
        cid = np.searchsorted(units, eps)
        for other in ("M0", "M1", "M2", "command-copy", "ridge_multioutput"):
            op = seed_mean_pred(cache, split, other) if other.startswith("M") else \
                pred_from_cache(cache, split, other)
            oe = np.abs(op - gt).mean(axis=(1, 2))
            d = comp_e - oe
            out = np.empty(2000)
            for b in range(2000):
                cnt = np.bincount(rng.integers(0, len(units), len(units)), minlength=len(units))
                w = cnt[cid]
                out[b] = (d * w).sum() / max(w.sum(), 1)
            boot[f"{split}|composition-{other}"] = {
                "mean": float(d.mean()), "ci_low": float(np.percentile(out, 2.5)),
                "ci_high": float(np.percentile(out, 97.5)),
                "crosses_zero": bool(np.percentile(out, 2.5) < 0 < np.percentile(out, 97.5)),
                "rel_change_pct": float(100 * d.mean() / oe.mean()),
                "n_clusters": int(len(units))}

    # ---- ACTION_COMPOSITION_SIGNAL 判定（B/C 上 vs 现有最好模型/基线）----
    sig = {}
    for split in ("B_seen_anchor_unseen_family", "C_unseen_anchor_unseen_family"):
        s = df[df.split == split].set_index("model")
        comp = s.loc["composition_additive"]
        comp_mlp_err = float(np.mean([comp["MAE_vx"], comp["MAE_vy"], comp["MAE_wz"]]))
        others = {k: float(np.mean([s.loc[k, "MAE_vx"], s.loc[k, "MAE_vy"], s.loc[k, "MAE_wz"]]))
                  for k in ("command-copy", "ridge_multioutput", "M0", "M1", "M2")}
        best_other = min(others, key=others.get)
        sig[split] = {"composition_mean_axis_MAE": comp_mlp_err,
                      "others": others, "best_other": best_other,
                      "best_other_err": others[best_other],
                      "improvement_vs_best_other": others[best_other] - comp_mlp_err,
                      "composition_wins": comp_mlp_err < others[best_other]}
    wins = [v["composition_wins"] for v in sig.values()]
    # 效果量门槛：只有当 B/C 上均优于最好对手、且配对 CI 不跨 0、且相对改善 >= 5% 才算 SUPPORTED
    strong = []
    for split in ("B_seen_anchor_unseen_family", "C_unseen_anchor_unseen_family"):
        best_other = sig[split]["best_other"]
        b = boot[f"{split}|composition-{best_other}"]
        strong.append(sig[split]["composition_wins"] and (not b["crosses_zero"])
                      and b["rel_change_pct"] <= -5.0)
        sig[split]["paired_bootstrap_vs_best_other"] = b
    verdict = ("SUPPORTED" if all(strong) else
               "PARTIAL" if any(wins) else "UNSUPPORTED")
    json.dump({"ACTION_COMPOSITION_SIGNAL": verdict, "per_split": sig,
               "paired_bootstrap": boot, "fit": fit_info, "detail": detail,
               "criterion": "SUPPORTED 需同时满足：B 与 C 上 additive baseline 优于所有对手、"
                            "配对 episode-cluster bootstrap CI 不跨 0、且相对改善 >=5%。"
                            "仅名次更高但改善甚微 -> PARTIAL（避免把小效应说成明显改善）。"},
              open(os.path.join(OUT, "metrics", "action_composition_signal.json"), "w"),
              indent=1, ensure_ascii=False)
    print(f"fit={args.kind} {fit_info.get('val_loss')}")
    cols = ["split", "model", "MAE_vx", "MAE_vy", "MAE_wz", "MAE_all_mixed", "endpoint_xy_err"]
    print(df[cols].round(4).to_string(index=False))
    print(f"\nACTION_COMPOSITION_SIGNAL = {verdict}")
    for k, v in sig.items():
        print(f"  {k}: comp={v['composition_mean_axis_MAE']:.4f} vs "
              f"best_other={v['best_other']}({v['best_other_err']:.4f}) "
              f"delta={v['improvement_vs_best_other']:+.4f}")


if __name__ == "__main__":
    main()
