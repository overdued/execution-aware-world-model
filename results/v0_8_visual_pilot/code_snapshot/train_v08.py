"""V0.8 Stage C：训练（预注册 §3/§4 冻结口径）。

- AdamW lr 3e-4 wd 1e-4，cosine→0，warmup 500，batch 64，30,000 步，fp32；
- L_vis = 归一化目标 patch L1（1s/2s 双头等权）；V-AUX/V-EXEC: + λ_E=1.0 · L_exec；
- checkpoint = validation 2s visual loss 最优（不用 test）；
- R-GRU 按 seed 匹配从 V0.7 D/R1 checkpoint 初始化（形状严格一致才允许；
  G 形状不一致 → 随机初始化，DEVIATIONS D2）；
- 运行：python -m execution_wm.v08_visual.train_v08 --variant V-AUX --seed 42 \
    --results results/v0_8_visual_pilot --raw-root /media/.../v0_8_pilot
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from execution_wm.v08_visual import model_v08
from execution_wm.v08_visual.dataset_v08 import V08Windows

V07_CKPT = Path("/home/yuhang/cvpr_embed-v07/results/v0_7_composition/checkpoints")
STEPS, BATCH, LR, WD, WARMUP = 30000, 64, 3e-4, 1e-4, 500
LAMBDA_E = 1.0
EVAL_EVERY = 1000


def init_from_v07(model, variant, seed, log):
    """冻结规则：R-GRU 形状严格一致 → seed 匹配初始化；G 不一致 → 随机。"""
    ck = V07_CKPT / f"D_R1_s{seed}" / "best.pt"
    if not ck.exists():
        log["v07_init"] = "ckpt_missing_random"
        return
    sd = torch.load(ck, map_location="cpu", weights_only=False)["model_state"]
    mine = model.hist_gru.state_dict()
    src = {k.replace("enc.gru.", ""): v for k, v in sd.items()
           if k.startswith("enc.gru.")}
    if set(src) == set(mine) and all(src[k].shape == mine[k].shape for k in src):
        model.hist_gru.load_state_dict(src)
        log["v07_init"] = {"gru": f"D_R1_s{seed}",
                           "sha": hashlib.sha256(ck.read_bytes()).hexdigest()[:16],
                           "G": "shape_mismatch_random"}
    else:
        log["v07_init"] = "shape_mismatch_random"


def collate(batch):
    out = {}
    for k in batch[0]:
        if isinstance(batch[0][k], torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch])
        else:
            out[k] = [b[k] for b in batch]
    return out


@torch.no_grad()
def eval_vis2s(model, dl, device, max_batches=0):
    model.eval()
    tot, n = 0.0, 0
    for bi, b in enumerate(dl):
        if max_batches and bi >= max_batches:
            break
        o = model(b["ctx"].to(device), b["hist"].to(device), b["fut"].to(device))
        t = b["tgt2s"].to(device).reshape(o["pred_2s"].shape)
        tot += (o["pred_2s"] - t).abs().mean().item()
        n += 1
    model.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    choices=["V-DIRECT", "V-AUX", "V-EXEC"])
    ap.add_argument("--seed", type=int, required=True, choices=[42, 43, 44])
    ap.add_argument("--results", required=True)
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    assert args.variant in ("V-DIRECT", "V-AUX", "V-EXEC")
    rr = Path(args.results)
    run_id = f"{args.variant.replace('-', '')}_s{args.seed}"
    out_dir = rr / "checkpoints" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    log = {"run_id": run_id, "variant": args.variant, "seed": args.seed,
           "steps": args.steps, "batch": BATCH, "lr": LR, "wd": WD,
           "warmup": WARMUP, "lambda_E": LAMBDA_E, "dtype": "fp32"}

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_ds = V08Windows(rr, args.raw_root, "train", fit_norm=True)
    val_ds = V08Windows(rr, args.raw_root, "val",
                        norm_stats=(train_ds.e_mean, train_ds.e_std))
    g = torch.Generator().manual_seed(args.seed)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                          generator=g, drop_last=True, collate_fn=collate,
                          num_workers=2, persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=128, shuffle=False,
                        collate_fn=collate)
    log["n_train_windows"], log["n_val_windows"] = len(train_ds), len(val_ds)

    model = model_v08.build(args.variant, seed=args.seed).to(args.device)
    init_from_v07(model, args.variant, args.seed, log)
    log["n_params"] = model_v08.n_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    def lr_at(step):
        if step < WARMUP:
            return LR * step / WARMUP
        p = (step - WARMUP) / max(1, args.steps - WARMUP)
        return LR * 0.5 * (1 + np.cos(np.pi * p))

    best_val, rows = float("inf"), []
    t0, it = time.time(), iter(train_dl)
    model.train()
    for step in range(1, args.steps + 1):
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step)
        try:
            b = next(it)
        except StopIteration:
            it = iter(train_dl)
            b = next(it)
        o = model(b["ctx"].to(args.device), b["hist"].to(args.device),
                  b["fut"].to(args.device))
        t1 = b["tgt1s"].to(args.device).reshape(o["pred_1s"].shape)
        t2 = b["tgt2s"].to(args.device).reshape(o["pred_2s"].shape)
        l_vis = ((o["pred_1s"] - t1).abs().mean() +
                 (o["pred_2s"] - t2).abs().mean()) / 2
        loss = l_vis
        l_exec = torch.zeros((), device=args.device)
        if args.variant != "V-DIRECT":
            l_exec = (o["e_hat"] - b["e"].to(args.device)).abs().mean()
            loss = l_vis + LAMBDA_E * l_exec
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 100 == 0:
            rows.append({"step": step, "l_vis": l_vis.item(),
                         "l_exec": l_exec.item(), "lr": lr_at(step)})
        if step % EVAL_EVERY == 0 or step == args.steps:
            v = eval_vis2s(model, val_dl, args.device)
            if v < best_val:
                best_val = v
                torch.save({"model_state": model.state_dict(),
                            "variant": args.variant, "seed": args.seed,
                            "step": step, "val_vis2s": v},
                           out_dir / "best.pt")
            rows.append({"step": step, "val_vis2s": v})
            print(f"  [{run_id}] step {step} val_vis2s={v:.4f} "
                  f"best={best_val:.4f} ({(time.time()-t0)/60:.1f}min)", flush=True)

    # 延迟/显存实测（同步 CUDA 计时）
    model.eval()
    b = next(iter(val_dl))
    ctx = b["ctx"][:1].to(args.device)
    hist = b["hist"][:1].to(args.device)
    fut = b["fut"][:1].to(args.device)
    torch.cuda.synchronize()
    for _ in range(10):
        model(ctx, hist, fut)
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(50):
        model(ctx, hist, fut)
    torch.cuda.synchronize()
    lat_b1 = (time.time() - t1) / 50 * 1000
    ctx64, hist64, fut64 = (b["ctx"][:64].to(args.device),
                            b["hist"][:64].to(args.device),
                            b["fut"][:64].to(args.device))
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(20):
        model(ctx64, hist64, fut64)
    torch.cuda.synchronize()
    lat_b64 = (time.time() - t1) / 20 * 1000
    log.update({
        "best_val_vis2s": best_val,
        "wall_min": (time.time() - t0) / 60,
        "peak_mem_mib": torch.cuda.max_memory_allocated() / 2**20,
        "latency_batch1_ms": lat_b1, "latency_batch64_ms": lat_b64,
        "ckpt_sha": hashlib.sha256((out_dir / "best.pt").read_bytes()).hexdigest(),
    })
    import pandas as pd
    pd.DataFrame(rows).to_csv(out_dir / "train_log.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(log, indent=2, default=str))
    print(json.dumps(log, indent=2, default=str))


if __name__ == "__main__":
    main()
