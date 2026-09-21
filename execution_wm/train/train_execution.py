"""训练 execution predictor（first_work.md §8-§14）。

用法（不需要仿真，直接 python）:
    python -m execution_wm.train.train_execution \
        --config execution_wm/configs/train_v0.yaml --model context
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from execution_wm.data.dataset import discover_episodes, split_episodes, PROPRIO_DIM  # noqa: E402
from execution_wm.data.window_dataset import WindowDataset, window_collate  # noqa: E402
from execution_wm.models.baseline_action import ActionOnlyBaseline  # noqa: E402
from execution_wm.models.baseline_direct import DirectPredictor  # noqa: E402
from execution_wm.models.execution_context_model import ExecutionContextModel  # noqa: E402

MODEL_REGISTRY = {
    "action_only": lambda H, mc: ActionOnlyBaseline(H, tuple(mc["hidden_dims"]), mc["predict_uncertainty"]),
    "direct": lambda H, mc: DirectPredictor(PROPRIO_DIM, H, mc["gru_hidden"], tuple(mc["hidden_dims"]), mc["predict_uncertainty"]),
    "context": lambda H, mc: ExecutionContextModel(PROPRIO_DIM, H, mc["context_dim"], mc["gru_hidden"], tuple(mc["hidden_dims"]), mc["predict_uncertainty"]),
}

VEL_NAMES = ["vx", "vy", "wz"]


def model_forward(model, name, batch):
    return model(
        history_proprio=batch.get("history_proprio"),
        history_action=batch.get("history_action"),
        current_state=batch.get("current_state"),
        future_action=batch["future_action"],
    )


def compute_metrics(r_hat, r_true, future_action):
    """§12: MAE/RMSE，分 vx/vy/wz + overall。"""
    m = {}
    err = r_hat - r_true
    e_hat = future_action + r_hat
    e_true = future_action + r_true
    verr = e_hat - e_true
    for i, n in enumerate(VEL_NAMES):
        m[f"residual_MAE_{n}"] = err[..., i].abs().mean().item()
        m[f"residual_RMSE_{n}"] = err[..., i].pow(2).mean().sqrt().item()
        m[f"velocity_MAE_{n}"] = verr[..., i].abs().mean().item()
        m[f"velocity_RMSE_{n}"] = verr[..., i].pow(2).mean().sqrt().item()
    m["residual_MAE"] = err.abs().mean().item()
    m["residual_RMSE"] = err.pow(2).mean().sqrt().item()
    m["velocity_MAE"] = verr.abs().mean().item()
    m["velocity_RMSE"] = verr.pow(2).mean().sqrt().item()
    return m


def gaussian_nll(r_hat, logvar, r_true):
    """§13: 0.5 * [(r-mu)^2/sigma^2 + log sigma^2]（默认不启用）"""
    return 0.5 * ((r_true - r_hat).pow(2) / logvar.exp() + logvar).mean()


@torch.no_grad()
def evaluate(model, name, loader, device):
    model.eval()
    agg, n = {}, 0
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        out = model_forward(model, name, batch)
        m = compute_metrics(out["r_hat"], batch["future_residual"], batch["future_action"])
        bs = batch["future_residual"].shape[0]
        for k, v in m.items():
            agg[k] = agg.get(k, 0.0) + v * bs
        n += bs
    return {k: v / n for k, v in agg.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--model", choices=list(MODEL_REGISTRY), default=None)
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_name = args.model or cfg["model"]["name"]
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = cfg["train"]["device"] if torch.cuda.is_available() else "cpu"

    exp_dir = os.path.join(cfg["output"]["exp_dir"], model_name)
    os.makedirs(exp_dir, exist_ok=True)

    # ---- 数据（episode-level split, §14） ----
    dc = cfg["dataset"]
    eps = discover_episodes(dc["dataset_dir"])
    splits = split_episodes(eps, dc["val_fraction"], dc["test_id_fraction"],
                            dc["ood_conditions"], dc["exclude_probe_from_train"], cfg["seed"])
    hz, L_s, H_s = dc["hz"], dc["history_s"], dc["train_horizon_s"]
    H = int(round(H_s * hz))
    loaders = {}
    sizes = {}
    for key in ("train", "val", "test_id", "test_ood"):
        ds = WindowDataset(splits[key], hz=hz, history_s=L_s, horizon_s=H_s)
        sizes[key] = {"episodes": len(splits[key]), "windows": len(ds)}
        loaders[key] = DataLoader(ds, batch_size=cfg["train"]["batch_size"],
                                  shuffle=(key == "train"),
                                  num_workers=cfg["train"]["num_workers"],
                                  collate_fn=window_collate,
                                  drop_last=(key == "train"))
    print(f"[train:{model_name}] sizes: {sizes}")

    model = MODEL_REGISTRY[model_name](H, cfg["model"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    huber = nn.HuberLoss(delta=cfg["train"]["huber_delta"])

    history = {"train_loss": [], "val_loss": []}
    best_val, bad_epochs = float("inf"), 0
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        tot, nb = 0.0, 0
        for batch in loaders["train"]:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            out = model_forward(model, model_name, batch)
            loss = huber(out["r_hat"], batch["future_residual"])          # §12
            if cfg["model"]["predict_uncertainty"] and "logvar_r" in out:  # §13
                loss = gaussian_nll(out["r_hat"], out["logvar_r"], batch["future_residual"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        train_loss = tot / max(nb, 1)

        model.eval()
        vtot, vnb = 0.0, 0
        with torch.no_grad():
            for batch in loaders["val"]:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                out = model_forward(model, model_name, batch)
                vtot += huber(out["r_hat"], batch["future_residual"]).item()
                vnb += 1
        val_loss = vtot / max(vnb, 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"[{model_name}] epoch {epoch:03d} train={train_loss:.5f} val={val_loss:.5f}")

        if val_loss < best_val - 1e-5:
            best_val, bad_epochs = val_loss, 0
            torch.save({"model_state": model.state_dict(), "config": cfg,
                        "model_name": model_name, "horizon": H},
                       os.path.join(exp_dir, "best.pt"))
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["train"]["early_stop_patience"]:
                print(f"[{model_name}] early stop at epoch {epoch}")
                break

    # ---- 最终评估（best checkpoint，ID + OOD, §14/§18.F） ----
    ckpt = torch.load(os.path.join(exp_dir, "best.pt"), weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    metrics = {k: evaluate(model, model_name, loaders[k], device)
               for k in ("val", "test_id", "test_ood") if sizes[k]["windows"] > 0}

    try:
        commit = subprocess.check_output(["git", "-C", PROJECT_ROOT, "rev-parse", "HEAD"],
                                         text=True).strip()
    except Exception:
        commit = "unknown"
    summary = {
        "git_commit": commit,
        "model": model_name,
        "config": cfg,
        "dataset_sizes": sizes,
        "best_val_loss": best_val,
        "history": history,
        "metrics": metrics,
    }
    with open(os.path.join(exp_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[train:{model_name}] done -> {exp_dir}")
    print(json.dumps({k: {m: round(v, 4) for m, v in mm.items() if m.endswith(('MAE', 'RMSE'))}
                      for k, mm in metrics.items()}, indent=2))


if __name__ == "__main__":
    main()
