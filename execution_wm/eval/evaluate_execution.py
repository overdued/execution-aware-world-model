"""评估三个模型并输出对比指标（first_work.md §18.E/F）。

用法:
    python -m execution_wm.eval.evaluate_execution --config execution_wm/configs/train_v0.yaml
"""
import argparse
import json
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from execution_wm.data.dataset import discover_episodes, split_episodes  # noqa: E402
from execution_wm.data.window_dataset import WindowDataset  # noqa: E402
from execution_wm.train.train_execution import MODEL_REGISTRY, compute_metrics, evaluate  # noqa: E402


def load_model(exp_root, model_name, H, device):
    path = os.path.join(exp_root, model_name, "best.pt")
    ckpt = torch.load(path, weights_only=False)
    model = MODEL_REGISTRY[model_name](H, ckpt["config"]["model"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--models", nargs="*", default=["action_only", "direct", "context"])
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = cfg["train"]["device"] if torch.cuda.is_available() else "cpu"

    dc = cfg["dataset"]
    eps = discover_episodes(dc["dataset_dir"])
    splits = split_episodes(eps, dc["val_fraction"], dc["test_id_fraction"],
                            dc["ood_conditions"], dc["exclude_probe_from_train"], cfg["seed"])
    H = int(round(dc["train_horizon_s"] * dc["hz"]))

    results = {}
    for split_name in ("test_id", "test_ood"):
        ds = WindowDataset(splits[split_name], hz=dc["hz"], history_s=dc["history_s"],
                           horizon_s=dc["train_horizon_s"])
        loader = DataLoader(ds, batch_size=512, shuffle=False)
        for mn in args.models:
            model = load_model(cfg["output"]["exp_dir"], mn, H, device)
            results.setdefault(split_name, {})[mn] = evaluate(model, mn, loader, device)

    out_path = os.path.join(cfg["output"]["exp_dir"], "evaluation.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    for split_name, per_model in results.items():
        print(f"\n===== {split_name} =====")
        header = f"{'model':<14}" + "".join(f"{m:>22}" for m in ("residual_MAE", "residual_RMSE", "velocity_MAE", "velocity_RMSE"))
        print(header)
        for mn, m in per_model.items():
            print(f"{mn:<14}" + "".join(f"{m[k]:>22.4f}" for k in ("residual_MAE", "residual_RMSE", "velocity_MAE", "velocity_RMSE")))
    print(f"\n[eval] -> {out_path}")


if __name__ == "__main__":
    main()
