"""V0.7 Stage 3：2×2 主实验训练（D/I × R0/R1，各 3 seed，共 12 run）。

公平性（PRE_REGISTRATION_V07 §5.3）:
  同一数据（同 cell 内 R0/R1 各自的训练池）、同一窗口 manifest、同一输入、同一 horizon、
  同一优化器/早停/步数上限/seed；checkpoint 只用共同验证指标 val FDE_xy 选择。
损失: L_E = Huber((ê − e)/s_E)，ê = u_ref + r̂；s_E 取自 **R0 训练池**（R0 ⊆ R1，故任何
  一格都不使用自身训练数据之外的信息），四个 cell 共用同一 s_E。
本轮 λ_xy = λ_ψ = 0（不加相对位姿 loss；预注册允许，记录为共同配置）。

用法: python -m execution_wm.composition_v07.train_v07 --data-root ... --model D --regime R0 --seed 42
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from execution_wm.composition_v07.data_v07 import build_datasets, H
from execution_wm.composition_v07.models_v07 import build, n_params

OUT = "results/v0_7_composition"
CKPT = "/media/hdd1/yuhang/checkpoints/execution_wm/v0_7"
MAX_EPOCHS = 100
PATIENCE = 15
BS = 128
LR = 1e-3


def fde_xy(pred_e, true_e, yaw0, n=H):
    """2s FDE_xy（m）：把体速度积分到局部坐标系原点朝向下（不读未来 GT yaw）。"""
    dp = np.cumsum(pred_e[:, :n, :2], axis=1)[:, -1] * 0.05
    dt_ = np.cumsum(true_e[:, :n, :2], axis=1)[:, -1] * 0.05
    return float(np.linalg.norm(dp - dt_, axis=1).mean())


def ade_xy(pred_e, true_e, n=H):
    dp = np.cumsum(pred_e[:, :n, :2], axis=1) * 0.05
    dt_ = np.cumsum(true_e[:, :n, :2], axis=1) * 0.05
    return float(np.linalg.norm(dp - dt_, axis=2).mean())


@torch.no_grad()
def evaluate(model, ds, device, s_e):
    if len(ds) == 0:
        return float("nan"), float("nan"), float("nan")
    model.eval()
    preds, trues = [], []
    for i in range(0, len(ds), 256):
        b = np.arange(i, min(i + 256, len(ds)))
        hp, ha, fa, fr, _, _, _, _ = ds.batch(b, device)
        r = model(hp, ha, fa)
        preds.append((fa + r).cpu().numpy()); trues.append(ds.fe[b])
    p, t = np.concatenate(preds), np.concatenate(trues)
    return fde_xy(p, t, None), ade_xy(p, t), float(np.abs(p - t).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--model", required=True, choices=["D", "I"])
    ap.add_argument("--regime", required=True, choices=["R0", "R1"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = build_datasets(args.data_root)
    tr = ds[f"train_{args.regime}"]
    va = ds["val"]
    print(f"[{args.model}/{args.regime}/s{args.seed}] train {len(tr)} val {len(va)} windows",
          flush=True)
    # s_E 来自 R0 训练池（四格共用）
    s_e = float(np.mean([ds["train_R0"].fe[:, :, i].std() for i in range(3)]))
    s_e = max(s_e, 1e-3)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = build(args.model, seed=args.seed).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.HuberLoss(delta=1.0)          # 输入已除以 s_E
    out_dir = os.path.join(CKPT, f"{args.model}_{args.regime}_s{args.seed}{args.tag}")
    os.makedirs(out_dir, exist_ok=True)

    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
    t0 = time.time()
    best_fde, best_ep, pat = float("inf"), -1, 0
    hist = []
    ep = -1
    for ep in range(args.max_epochs):
        model.train()
        idx = rng.permutation(len(tr))
        tot = 0.0
        for i in range(0, len(idx), BS):
            b = np.sort(idx[i:i + BS])
            hp, ha, fa, fr, _, _, _, _ = tr.batch(b, device)
            r = model(hp, ha, fa)
            loss = lossf(r / s_e, fr / s_e)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item()) * len(b)
        fde, ade, mae = evaluate(model, va, device, s_e)
        hist.append({"epoch": ep, "train_loss": tot / len(tr), "val_fde_xy": fde,
                     "val_ade_xy": ade, "val_mae": mae})
        if fde < best_fde - 1e-6:
            best_fde, best_ep, pat = fde, ep, 0
            torch.save({"model_state": model.state_dict(), "model_name": args.model,
                        "regime": args.regime, "seed": args.seed,
                        "val_fde_xy": best_fde, "s_E": s_e,
                        "window_manifest_hash": tr.manifest_hash(),
                        "n_params": n_params(model)}, os.path.join(out_dir, "best.pt"))
        else:
            pat += 1
        if ep % 10 == 0 or pat == 0:
            print(f"  ep{ep} loss {tot/len(tr):.5f} val_fde {fde:.4f} ade {ade:.4f} "
                  f"mae {mae:.4f}", flush=True)
        if pat >= PATIENCE:
            break
    gpu_h = (time.time() - t0) / 3600
    peak_mb = (torch.cuda.max_memory_allocated() / 2 ** 20) if device == "cuda" else 0.0
    meta = {"model": args.model, "regime": args.regime, "seed": args.seed,
            "n_train_windows": len(tr), "n_val_windows": len(va),
            "n_params": n_params(model), "s_E": s_e,
            "best_val_fde_xy": best_fde, "best_epoch": best_ep, "epochs": ep + 1,
            "wall_seconds": time.time() - t0, "gpu_hours": gpu_h,
            "peak_gpu_mem_mb": peak_mb, "window_manifest_hash": tr.manifest_hash(),
            "lambda_xy": 0.0, "lambda_yaw": 0.0,
            "checkpoint_selection": "val FDE_xy (common metric for all cells)",
            "input_mode": "simulated deployable-candidate (GT body vel/ang vel masked)"}
    json.dump(meta, open(os.path.join(out_dir, "train_log.json"), "w"), indent=1)
    json.dump(hist, open(os.path.join(out_dir, "history.json"), "w"), indent=1)
    print(f"[{args.model}/{args.regime}/s{args.seed}] done val_fde={best_fde:.4f} "
          f"ep={best_ep} gpu_h={gpu_h:.4f} peak={peak_mb:.0f}MB -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
