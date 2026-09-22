"""V0.6.2 Task 3 — Context Disentanglement Probes。

固定 M1 checkpoint，提取 c_t；用 **train-only** 线性探针预测：
    A. friction condition（3 类）
    B. current vx/vy/wz（回归，R²）
    C. command family（train 有 Q1-Q3 三类；Q4 windows 的预测分布作为 OOD 指标）
    D. anchor ID（仅在 train 内部做 5 折 CV —— 8 个 anchor 中只有 AQ0-4 在 train）

同时对 M0 的 fast state（GRU 末隐状态）与 M1 的 state latent（encode_state，64 维）
做同样探针，另加 raw proprio 末帧串行 40 维作参照下界/上界。

实现细节：不依赖 sklearn —— ridge 回归用闭式解；分类用带 L2 的 one-vs-rest 逻辑回归
（自行实现，梯度下降，早停用 train 内部验证折）。

输出: metrics/context_probe_results.csv, metrics/context_probe_meta.json
"""
import json
import os

import numpy as np
import pandas as pd
import torch

from execution_wm.validity_v061.windows import H, L
from execution_wm.validity_v062.common import (CKPT_V061, OUT, SPLITS_EVAL, get_manifests,
                                                load_cache)
from execution_wm.train.train_execution import MODEL_REGISTRY

AXES = ("vx", "vy", "wz")
CONDITIONS = ("normal", "friction_mid", "friction_low")


# ---------------- 线性探针（不依赖 sklearn）----------------
def ridge_fit(X, Y, lam=1.0):
    Xb = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    return np.linalg.solve(Xb.T @ Xb + lam * np.eye(Xb.shape[1]), Xb.T @ Y)


def ridge_pred(W, X):
    return np.concatenate([X, np.ones((len(X), 1))], axis=1) @ W


def r2_score(y, p):
    ss_res = ((y - p) ** 2).sum()
    ss_tot = ((y - y.mean(axis=0)) ** 2).sum()
    return float(1 - ss_res / max(ss_tot, 1e-12))


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def logreg_fit(X, y, n_class, lam=1e-3, lr=0.5, iters=1500):
    """L2 正则 one-vs-rest 逻辑回归（全批量梯度下降）。返回 W[B,n_class]。"""
    Xb = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    W = np.zeros((Xb.shape[1], n_class))
    Y = np.eye(n_class)[y]
    for _ in range(iters):
        P = 1.0 / (1.0 + np.exp(-(Xb @ W)))
        G = Xb.T @ (P - Y) / len(X) + lam * W
        W -= lr * G
    return W


def logreg_pred(W, X):
    Xb = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    return (Xb @ W).argmax(axis=1)


def balanced_accuracy(y, p, n_class):
    accs = []
    for c in range(n_class):
        m = y == c
        if m.sum():
            accs.append(float((p[m] == c).mean()))
    return float(np.mean(accs)), len(accs)


# ---------------- 表示提取 ----------------
def extract_reps(models, hp, ha, device):
    """-> dict[name] = [N,d]。M1: c_t / state latent；M0: GRU 末隐状态。"""
    out = {}
    with torch.no_grad():
        for seed, m in models["M1"].items():
            c = m.encode_context(hp, ha)
            z = m.encode_state(hp)
            out.setdefault("M1_context_c_t", []).append(c.cpu().numpy())
            out.setdefault("M1_state_latent", []).append(z.cpu().numpy())
            if seed == 42:                                    # 额外：context encoder 的 GRU 隐状态
                _, h_n = m.context_encoder.gru(torch.cat([hp, ha], dim=-1))
                h = h_n[-1]
                out["M1_ctx_gru_hidden"] = [h.cpu().numpy()]
        for seed, m in models["M0"].items():
            _, h_n = m.encoder(torch.cat([hp, ha], dim=-1))
            out.setdefault("M0_fast_state_gru", []).append(h_n[-1].cpu().numpy())
    return {k: np.mean(v, axis=0) for k, v in out.items()}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    man = get_manifests()
    cache = load_cache()
    models = {"M1": {}, "M0": {}}
    for name in ("M1", "M0"):
        for seed in (42, 43, 44):
            path = os.path.join(CKPT_V061, f"{name}_s{seed}", "best.pt")
            if not os.path.exists(path):
                continue
            ck = torch.load(path, weights_only=False, map_location=device)
            base = "direct" if name == "M0" else "context"
            m = MODEL_REGISTRY[base](H, ck["config"]["model"]).to(device)
            m.load_state_dict(ck["model_state"]); m.eval()
            models[name][seed] = m

    # 一次性提取全部 split 的隐藏表示
    reps, targets = {}, {}
    for split, m in man.items():
        n = len(m)
        hp, ha, fa, fr, fric, conds, eps, anch, _, fams = m.batch(np.arange(n), device)
        reps[split] = extract_reps(models, hp, ha, device)
        reps[split]["RAW_proprio_last"] = hp[:, -1, :].cpu().numpy()
        targets[split] = {
            "condition": np.array([CONDITIONS.index(c) for c in conds]),
            "current_vxvywz": m.hp[:, -1, :][:, [0, 1, 5]],
            "family": np.array(fams),
            "anchor": np.array(anch),
            "episode_id": np.array(eps),
        }
    # 参照：correct-persistence 的误差（= 用当前状态当预测的 in-sample 上界）
    rows = []

    # ---------- 探针 1：condition（分类） ----------
    tr_X = reps["train"]["M1_context_c_t"]
    y_tr = targets["train"]["condition"]
    n_class = len(CONDITIONS)
    for rep_name in ["M1_context_c_t", "M1_state_latent", "M1_ctx_gru_hidden",
                     "M0_fast_state_gru", "RAW_proprio_last"]:
        Xtr = reps["train"][rep_name]
        if Xtr is None:
            continue
        W = logreg_fit(Xtr, y_tr, n_class)
        tr_acc = float((logreg_pred(W, Xtr) == y_tr).mean())
        tr_bal, _ = balanced_accuracy(y_tr, logreg_pred(W, Xtr), n_class)
        row = {"probe": "A_friction_condition", "representation": rep_name,
               "metric": "balanced_accuracy", "train": tr_bal, "train_raw_acc": tr_acc,
               "chance": 1.0 / n_class}
        for split in ("val",) + tuple(SPLITS_EVAL):
            X = reps[split][rep_name]
            y = targets[split]["condition"]
            p = logreg_pred(W, X)
            bal, _ = balanced_accuracy(y, p, n_class)
            row[split] = bal
        rows.append(row)

    def rmse(y, p):
        return float(np.sqrt(((y - p) ** 2).mean()))

    # ---------- 探针 2：current vx/vy/wz（回归） ----------
    # 注意：族内目标方差很小 -> R² 对微小绝对偏差极敏感。同时报物理单位 RMSE
    # 与标准化 RMSE（RMSE / 该 split 目标 std），并给出 identity 参照上界。
    Y_tr = targets["train"]["current_vxvywz"]
    for split in ("train", "val") + tuple(SPLITS_EVAL):
        Y = targets[split]["current_vxvywz"]
        Pid = reps[split]["RAW_proprio_last"][:, [0, 1, 5]]
        rows.append({"probe": "B_current_vxvywz__REFERENCE", "representation": "IDENTITY(raw last frame)",
                     "metric": "R2_mean_over_axes",
                     "train": np.nan if split != "train" else 1.0,
                     split: float(np.mean([r2_score(Y[:, i:i + 1], Pid[:, i:i + 1])
                                           for i in range(3)])),
                     f"{split}_rmse_mixed": rmse(Y, Pid),
                     f"{split}_target_std_vx": float(Y[:, 0].std()),
                     f"{split}_target_std_wz": float(Y[:, 2].std()),
                     "note": "identity 参照：区分线性映射迁移问题与表示缺失"})
    for rep_name in ["M1_context_c_t", "M1_state_latent", "M1_ctx_gru_hidden",
                     "M0_fast_state_gru", "RAW_proprio_last"]:
        Xtr = reps["train"][rep_name]
        if Xtr is None:
            continue
        W = ridge_fit(Xtr, Y_tr, lam=1.0)
        row = {"probe": "B_current_vxvywz", "representation": rep_name,
               "metric": "R2_mean_over_axes",
               "train": float(np.mean([r2_score(Y_tr[:, i:i + 1],
                                                ridge_pred(W, Xtr)[:, i:i + 1])
                                       for i in range(3)]))}
        for split in ("val",) + tuple(SPLITS_EVAL):
            X = reps[split][rep_name]
            Y = targets[split]["current_vxvywz"]
            P = ridge_pred(W, X)
            row[split] = float(np.mean([r2_score(Y[:, i:i + 1], P[:, i:i + 1])
                                        for i in range(3)]))
            row[f"{split}_rmse_mixed"] = rmse(Y, P)
            row[f"{split}_target_std_vx"] = float(Y[:, 0].std())
            row[f"{split}_target_std_wz"] = float(Y[:, 2].std())
            if rep_name == "M1_context_c_t":
                row[f"{split}_R2_vx"] = r2_score(Y[:, 0:1], P[:, 0:1])
                row[f"{split}_R2_vy"] = r2_score(Y[:, 1:2], P[:, 1:2])
                row[f"{split}_R2_wz"] = r2_score(Y[:, 2:3], P[:, 2:3])
        rows.append(row)

    # ---------- 探针 3：command family（train 只有 Q1-Q3）----------
    fams_tr = np.array(sorted(set(targets["train"]["family"])))
    y_tr_f = np.array([list(fams_tr).index(f) for f in targets["train"]["family"]])
    for rep_name in ["M1_context_c_t", "M1_state_latent", "M1_ctx_gru_hidden",
                     "M0_fast_state_gru", "RAW_proprio_last"]:
        Xtr = reps["train"][rep_name]
        W = logreg_fit(Xtr, y_tr_f, len(fams_tr))
        row = {"probe": "C_command_family(Q1-Q3 only)", "representation": rep_name,
               "metric": "balanced_accuracy", "train":
                   balanced_accuracy(y_tr_f, logreg_pred(W, Xtr), len(fams_tr))[0],
               "chance": 1.0 / len(fams_tr)}
        for split in ("val",) + tuple(SPLITS_EVAL):
            X = reps[split][rep_name]
            fam = targets[split]["family"]
            known = np.isin(fam, fams_tr)
            if known.any():
                yk = np.array([list(fams_tr).index(f) for f in fam[known]])
                bal, _ = balanced_accuracy(yk, logreg_pred(W, X[known]), len(fams_tr))
                row[split] = bal
                row[f"{split}_n_known_family"] = int(known.sum())
                row[f"{split}_n_Q4"] = int((~known).sum())
                if (~known).any():
                    pk = logreg_pred(W, X[~known])
                    prob = softmax(np.concatenate(
                        [X[~known], np.ones((int((~known).sum()), 1))], axis=1) @ W)
                    row[f"{split}_Q4_pred_entropy"] = float(
                        -(prob * np.log(prob + 1e-12)).sum(axis=1).mean())
                    row[f"{split}_Q4_n_pred_classes"] = int(len(set(pk.tolist())))
            else:
                row[split] = np.nan
        rows.append(row)

    # ---------- 探针 4：anchor ID（train 内部 5 折 CV，仅 5 个 train anchor）----------
    anchors_tr = np.array(sorted(set(targets["train"]["anchor"])))
    y_tr_a = np.array([list(anchors_tr).index(a) for a in targets["train"]["anchor"]])
    rng = np.random.default_rng(0)
    fold = rng.integers(0, 5, len(y_tr_a))
    for rep_name in ["M1_context_c_t", "M1_state_latent", "M1_ctx_gru_hidden",
                     "M0_fast_state_gru", "RAW_proprio_last"]:
        Xtr = reps["train"][rep_name]
        preds = np.zeros(len(y_tr_a), dtype=int)
        for f in range(5):
            m_tr, m_te = fold != f, fold == f
            W = logreg_fit(Xtr[m_tr], y_tr_a[m_tr], len(anchors_tr))
            preds[m_te] = logreg_pred(W, Xtr[m_te])
        bal, _ = balanced_accuracy(y_tr_a, preds, len(anchors_tr))
        rows.append({"probe": "D_anchor_id(train anchors, 5-fold CV)",
                     "representation": rep_name, "metric": "balanced_accuracy",
                     "train_cv": bal, "train_in_sample": np.nan,
                     "chance": 1.0 / len(anchors_tr),
                     "note": "train 只有 AQ0-4；val/AQ5 与 test/AQ6-7 是未见 anchor，"
                             "跨 anchor 探针不适用 -> 只报 train 内部 CV"})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "metrics", "context_probe_results.csv"), index=False)
    meta = {
        "probe_train_split": "train only（每折/每次拟合均只用 train）",
        "classifier": "自实现 L2 one-vs-rest 逻辑回归（全批量 GD，1500 iters）",
        "regressor": "闭式 ridge（lam=1.0）",
        "representations": {k: int(v.shape[1]) for k, v in reps["train"].items()},
        "n_windows": {k: len(v) for k, v in man.items()},
        "targets": {
            "A_friction_condition": "3 类：normal / friction_mid / friction_low",
            "B_current_vxvywz": "回归：proprio 末帧 [0,1,5]（schema [0,1,5]）",
            "C_command_family": "train 仅 Q1_straight/Q2_lateral/Q3_turn；Q4 为未见类",
            "D_anchor_id": "train 仅 AQ0-4，5 折 CV",
        },
        "caveat": "c_t 由 history（含当前 proprio 帧）编码 -> 探到 current state 属预期；"
                  "关键看 condition 与 family 的相对可解码性。",
    }
    json.dump(meta, open(os.path.join(OUT, "metrics", "context_probe_meta.json"), "w"),
              indent=1, ensure_ascii=False)
    pd.set_option("display.width", 240)
    show = ["probe", "representation", "metric", "train", "val",
            "A_unseen_anchor_seen_family", "B_seen_anchor_unseen_family",
            "C_unseen_anchor_unseen_family"]
    print(df[[c for c in show if c in df.columns]].round(4).to_string(index=False))
    print("\n--- 探针 D：anchor id（train 内部 5 折 CV，chance=0.2）---")
    d = df[df.probe.str.startswith("D_")]
    print(d[["representation", "train_cv", "chance"]].round(4).to_string(index=False))
    print("\n--- 探针 B：R² 与 RMSE（族内小方差 -> R² 需与 RMSE/目标 std 同看）---")
    b = df[df.probe.str.startswith("B_current")]
    print(b[["representation", "A_unseen_anchor_seen_family",
             "B_seen_anchor_unseen_family", "C_unseen_anchor_unseen_family",
             "B_seen_anchor_unseen_family_rmse_mixed",
             "B_seen_anchor_unseen_family_target_std_vx"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
