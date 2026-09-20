"""可视化：Figures A–E + context PCA（first_work.md §15/§16）。

用法:
    python -m execution_wm.eval.visualize_execution --config execution_wm/configs/train_v0.yaml
输出到 exp_dir/figures/
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from execution_wm.data.dataset import (  # noqa: E402
    PROPRIO_KEYS, discover_episodes, episode_proprio, load_episode, split_episodes,
)
from execution_wm.data.window_dataset import WindowDataset  # noqa: E402
from execution_wm.eval.evaluate_execution import load_model  # noqa: E402
from execution_wm.train.train_execution import MODEL_REGISTRY  # noqa: E402

VEL_NAMES = ["vx", "vy", "wz"]


def rollout_episode(model, model_name, d, hz, L_s, H_s, device):
    """在单个 episode 上做滑动窗预测，返回 (t, cmd, exec, pred_exec, resid, pred_resid)（H=单步前预测拼接首步）。"""
    ds = WindowDataset([{"path": None, "meta": {}}], hz=hz, history_s=L_s, horizon_s=H_s)
    # 手动构造窗口（直接用 episode 数据）
    L, H = ds.L, ds.H
    proprio = episode_proprio(d)
    cmd, exe, res = d["cmd_vel"], d["execution"], d["residual"]
    preds, targets, ts = [], [], []
    for t0 in range(L - 1, len(cmd) - H, H):  # 不重叠窗口，取每窗首步预测
        batch = {
            "history_proprio": torch.from_numpy(proprio[t0 - L + 1:t0 + 1])[None].to(device),
            "history_action": torch.from_numpy(cmd[t0 - L + 1:t0 + 1].astype(np.float32))[None].to(device),
            "current_state": torch.from_numpy(proprio[t0])[None].to(device),
            "future_action": torch.from_numpy(cmd[t0 + 1:t0 + 1 + H].astype(np.float32))[None].to(device),
        }
        with torch.no_grad():
            out = model(**batch)
        preds.append(out["r_hat"][0].cpu().numpy())        # [H,3]
        ts.append(t0 + 1 + np.arange(H))
    if not preds:
        return None
    return np.concatenate(ts), cmd, exe, res, np.concatenate(preds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--model", default="context")
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = cfg["train"]["device"] if torch.cuda.is_available() else "cpu"
    dc = cfg["dataset"]
    hz, L_s, H_s = dc["hz"], dc["history_s"], dc["train_horizon_s"]
    H = int(round(H_s * hz))
    fig_dir = os.path.join(cfg["output"]["exp_dir"], "figures")
    os.makedirs(fig_dir, exist_ok=True)

    eps = discover_episodes(dc["dataset_dir"])
    splits = split_episodes(eps, dc["val_fraction"], dc["test_id_fraction"],
                            dc["ood_conditions"], True, cfg["seed"])
    model = load_model(cfg["output"]["exp_dir"], args.model, H, device)
    model.eval()

    # ---------- Figure A & B: 随机挑 test_id episode 画 cmd/actual/pred 与 residual ----------
    rng = np.random.default_rng(0)
    for cond in ("normal", "friction_low"):
        cands = [e for e in splits["test_id"] if e["meta"]["condition"] == cond] or \
                [e for e in eps if e["meta"]["condition"] == cond and e["meta"]["episode_type"] == "random"]
        if not cands:
            continue
        e = cands[0]
        d = load_episode(e["path"])
        out = rollout_episode(model, args.model, d, hz, L_s, H_s, device)
        if out is None:
            continue
        ts, cmd, exe, res, pres = out
        t_axis = np.arange(len(cmd)) / hz
        # Figure A
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for i, n in enumerate(VEL_NAMES):
            axes[i].plot(t_axis, cmd[:, i], "k--", label="commanded", alpha=0.7)
            axes[i].plot(t_axis, exe[:, i], "b", label="actual")
            axes[i].plot(ts / hz, pres[:, i] + cmd[ts, i], "r", label="predicted", alpha=0.8)
            axes[i].set_ylabel(n)
            axes[i].legend(loc="upper right")
        axes[0].set_title(f"Figure A: cmd vs actual vs predicted ({cond}, ep{e['meta']['episode_id']})")
        axes[-1].set_xlabel("t [s]")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"figA_cmd_actual_pred_{cond}.png"), dpi=150)
        plt.close(fig)
        # Figure B
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for i, n in enumerate(VEL_NAMES):
            axes[i].plot(t_axis, res[:, i], "b", label="true residual")
            axes[i].plot(ts / hz, pres[:, i], "r", label="predicted residual", alpha=0.8)
            axes[i].set_ylabel(f"r_{n}")
            axes[i].legend(loc="upper right")
        axes[0].set_title(f"Figure B: true vs predicted residual ({cond})")
        axes[-1].set_xlabel("t [s]")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"figB_residual_{cond}.png"), dpi=150)
        plt.close(fig)

    # ---------- Figure C: prediction error vs friction / severity ----------
    conds, frics, errs = [], [], []
    for e in splits["test_id"] + splits["test_ood"]:
        d = load_episode(e["path"])
        out = rollout_episode(model, args.model, d, hz, L_s, H_s, device)
        if out is None:
            continue
        ts, cmd, exe, res, pres = out
        errs.append(np.abs(pres - res[ts]).mean())
        frics.append(float(e["meta"]["friction"]) if np.isscalar(e["meta"]["friction"]) else np.mean(e["meta"]["friction"]))
        conds.append(e["meta"]["condition"])
    if frics:
        fig, ax = plt.subplots(figsize=(7, 5))
        for cond in sorted(set(conds)):
            xs = [f for f, c in zip(frics, conds) if c == cond]
            ys = [y for y, c in zip(errs, conds) if c == cond]
            ax.scatter(xs, ys, label=cond, alpha=0.7)
        ax.set_xlabel("friction coefficient")
        ax.set_ylabel("residual prediction MAE")
        ax.set_title("Figure C: prediction error vs friction")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "figC_error_vs_friction.png"), dpi=150)
        plt.close(fig)

    # ---------- Figure D: context embedding PCA（§15） ----------
    if args.model == "context":
        ctxs, fric_c, act_c, dist_c = [], [], [], []
        for e in splits["test_id"] + splits["test_ood"]:
            d = load_episode(e["path"])
            ds = WindowDataset([e], hz=hz, history_s=L_s, horizon_s=H_s)
            if len(ds) == 0:
                continue
            idx = np.linspace(0, len(ds) - 1, min(50, len(ds))).astype(int)
            for i in idx:
                s = ds[i]
                batch = {k: v[None].to(device) for k, v in s.items()
                         if isinstance(v, torch.Tensor)}
                with torch.no_grad():
                    out = model(**batch)
                ctxs.append(out["context"][0].cpu().numpy())
                m = e["meta"]
                fric_c.append(float(m["friction"]) if np.isscalar(m["friction"]) else np.mean(m["friction"]))
                act_c.append(float(m["actuator_scale"]) if np.isscalar(m["actuator_scale"]) else np.mean(m["actuator_scale"]))
                dist_c.append("disturbance" if m.get("disturbance") else "none")
        if ctxs:
            X = np.stack(ctxs)
            Xc = X - X.mean(0)
            _, _, vt = np.linalg.svd(Xc, full_matrices=False)
            Z = Xc @ vt[:2].T
            for values, name, title in (
                (fric_c, "friction", "colored by friction"),
                (act_c, "actuator", "colored by actuator scale"),
            ):
                fig, ax = plt.subplots(figsize=(6, 5))
                sc = ax.scatter(Z[:, 0], Z[:, 1], c=values, cmap="viridis", alpha=0.6, s=8)
                fig.colorbar(sc, ax=ax, label=name)
                ax.set_title(f"Figure D: context PCA ({title})")
                fig.tight_layout()
                fig.savefig(os.path.join(fig_dir, f"figD_context_pca_{name}.png"), dpi=150)
                plt.close(fig)
            # disturbance 类别着色
            fig, ax = plt.subplots(figsize=(6, 5))
            for cls, marker in (("none", "o"), ("disturbance", "^")):
                pts = Z[np.array(dist_c) == cls]
                ax.scatter(pts[:, 0], pts[:, 1], label=cls, alpha=0.6, s=8, marker=marker)
            ax.legend()
            ax.set_title("Figure D: context PCA (colored by disturbance)")
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, "figD_context_pca_disturbance.png"), dpi=150)
            plt.close(fig)

    # ---------- Figure E: matched command 跨条件 execution 对比（§6/§16） ----------
    probe_eps = {}
    for e in splits["probe"]:
        m = e["meta"]
        probe_eps.setdefault((m["probe_name"], m["condition"]), []).append(e)
    probe_names = sorted({k[0] for k in probe_eps})
    cond_names = sorted({k[1] for k in probe_eps})
    for pname in probe_names:
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        plotted = False
        for cond in cond_names:
            cands = probe_eps.get((pname, cond))
            if not cands:
                continue
            d = load_episode(cands[0]["path"])
            t = d["timestamp"] - d["timestamp"][0]
            for i, n in enumerate(VEL_NAMES):
                axes[i].plot(t, d["execution"][:, i], label=f"{cond}" if i == 0 else None)
            if not plotted:  # 命令只画一次（matched）
                for i, n in enumerate(VEL_NAMES):
                    axes[i].plot(t, d["cmd_vel"][:, i], "k--", alpha=0.5,
                                 label="command" if i == 0 else None)
                plotted = True
        for i, n in enumerate(VEL_NAMES):
            axes[i].set_ylabel(n)
        axes[0].legend(loc="upper right")
        axes[0].set_title(f"Figure E: same command, different conditions ({pname})")
        axes[-1].set_xlabel("t [s]")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"figE_matched_{pname}.png"), dpi=150)
        plt.close(fig)

    print(f"[viz] figures -> {fig_dir}")


if __name__ == "__main__":
    main()
