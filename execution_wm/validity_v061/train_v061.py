"""V0.6.1 R2/R3：公平重跑训练（固定 window manifest，训练 seed 只影响初始化与打乱）。

模型结构不变:
  M0 Direct            : history -> r_hat（等输入对照）
  M1 native Context    : c 来自 query 自身 history（因果）
  M2 paired-support    : c 来自**同 condition 的另一条 support episode**（成对窗口，B1 修复）
  M3 privileged        : M1 + 零初始化 W·真实摩擦（诊断基线，非上界）

数据: /media/hdd1/yuhang/datasets/execution_wm/v0_6_1（B2/B3 修复后的 20Hz 派生）
目标: residual_label = execution_label - command_reference（恒等式成立）
split: A/B/C 三分 + val(AQ5 x Q1-Q3)，train = AQ0-4 x Q1-Q3

用法: python -m execution_wm.validity_v061.train_v061 --model M1 --seed 42
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from execution_wm.train.train_execution import MODEL_REGISTRY
from execution_wm.validity_v061.splits import attach_derived_paths, classify, TRAIN_ANCHORS
from execution_wm.validity_v061.support_pairs import SupportBank
from execution_wm.validity_v061.windows import H, L, WindowManifest

SQ = "/media/hdd1/yuhang/datasets/execution_wm/v0_6_sq"
DERIVED = "/media/hdd1/yuhang/datasets/execution_wm/v0_6_1"
OUT_ROOT = "/media/hdd1/yuhang/checkpoints/execution_wm/v0_6_1"
WINDOWS_PER_EP_TRAIN, WINDOWS_PER_EP_VAL = 8, 8
MCONFIG = {"hidden_dims": [256, 256], "gru_hidden": 128, "context_dim": 8,
           "predict_uncertainty": False}


def build_data():
    entries = attach_derived_paths(json.load(open(os.path.join(SQ, "index.json")))["episodes"],
                                   DERIVED)
    sp = classify(entries)
    train = WindowManifest(sp["train"], "train", windows_per_ep=WINDOWS_PER_EP_TRAIN)
    val = WindowManifest(sp["val"], "val", windows_per_ep=WINDOWS_PER_EP_VAL)
    bank = SupportBank(sp["support"], path_key="path20")
    return sp, train, val, bank


def forward_model(model_name, model, hp, ha, fa, friction, model_aux=None, sup=None):
    if model_name == "M0":
        return model(hp, ha, fa)["r_hat"]
    if model_name == "M2":
        shp, sha = sup
        c = model.encode_context(shp, sha)          # 成对 support 窗口（B1）
    elif model_name == "M3":
        c = model.encode_context(hp, ha) + model_aux(friction.reshape(-1, 1))
    else:  # M1
        c = model.encode_context(hp, ha)
    z = model.encode_state(hp)
    return model.predict_execution(z, c, fa)["r_hat"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["M0", "M1", "M2", "M3"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--tag", default="")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    sp, train_data, val_data, bank = build_data()
    train_data.assert_causal(); train_data.assert_label_identity()
    val_data.assert_causal(); val_data.assert_label_identity()
    train_eps = {w["episode_id"] for w in train_data.windows}

    base = "direct" if args.model == "M0" else "context"
    model = MODEL_REGISTRY[base](H, MCONFIG).to(device)
    model_aux = None
    if args.model == "M3":
        model_aux = nn.Linear(1, 8).to(device)
        nn.init.zeros_(model_aux.weight); nn.init.zeros_(model_aux.bias)
    params = list(model.parameters()) + (list(model_aux.parameters()) if model_aux else [])
    opt = torch.optim.Adam(params, lr=1e-3)
    loss_fn = nn.SmoothL1Loss(beta=1.0)
    bs = 128
    print(f"[{args.model}/s{args.seed}] train {len(train_data)} val {len(val_data)} windows "
          f"(固定 manifest; support bank {len(bank)} 成对窗口)", flush=True)

    out_dir = os.path.join(OUT_ROOT, f"{args.model}_s{args.seed}{args.tag}")
    os.makedirs(out_dir, exist_ok=True)
    best_val, best_ep, patience = float("inf"), -1, 0
    ep = -1
    for ep in range(args.max_epochs):
        model.train()
        idx = rng.permutation(len(train_data))
        tot = 0.0
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            hp, ha, fa, fr, fric, conds, eps, _, _, _ = train_data.batch(b, device)
            sup = None
            if args.model == "M2":
                j = bank.sample_indices(conds, rng,
                                        exclude_episodes=self_exclusion(bank, eps))
                sup = bank.batch(j, device)
            r_hat = forward_model(args.model, model, hp, ha, fa, fric, model_aux, sup)
            loss = loss_fn(r_hat, fr)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(hp)
        model.eval()
        vtot = 0.0
        with torch.no_grad():
            for i in range(0, len(val_data), bs):
                b = np.arange(i, min(i + bs, len(val_data)))
                hp, ha, fa, fr, fric, conds, eps, _, _, _ = val_data.batch(b, device)
                sup = None
                if args.model == "M2":
                    j = bank.sample_indices(conds, rng, exclude_episodes=self_exclusion(bank, eps))
                    sup = bank.batch(j, device)
                vtot += loss_fn(forward_model(args.model, model, hp, ha, fa, fric,
                                              model_aux, sup), fr).item() * len(hp)
        vloss = vtot / max(len(val_data), 1)
        if vloss < best_val - 1e-5:
            best_val, best_ep, patience = vloss, ep, 0
            torch.save({"model_state": model.state_dict(),
                        "aux_state": model_aux.state_dict() if model_aux else None,
                        "config": {"model": MCONFIG}, "model_name": args.model,
                        "seed": args.seed, "val_loss": best_val,
                        "window_manifest_hash": train_data.manifest_hash(),
                        "target": "residual_label = execution_label - command_reference",
                        "derive_version": "v061.2"}, os.path.join(out_dir, "best.pt"))
        else:
            patience += 1
        if ep % 10 == 0 or patience == 0:
            print(f"  ep{ep} train {tot/len(train_data):.6f} val {vloss:.6f}", flush=True)
        if patience >= 15:
            break
    gpu_h = (time.time() - t0) / 3600
    meta = {"model": args.model, "seed": args.seed, "best_val": best_val,
            "best_epoch": best_ep, "epochs": ep + 1, "gpu_hours": gpu_h,
            "n_train_windows": len(train_data), "n_val_windows": len(val_data),
            "n_params": sum(p_.numel() for p_ in params),
            "window_manifest_hash": train_data.manifest_hash(),
            "support_bank_size": len(bank), "target": "residual_label (B2 修复)",
            "data_dir": DERIVED, "derive_version": "v061.2"}
    with open(os.path.join(out_dir, "train_log.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[{args.model}/s{args.seed}] done best_val={best_val:.6f} ep={best_ep} "
          f"gpu_h={gpu_h:.4f} -> {out_dir}", flush=True)


_BANK_EPS_CACHE = {}


def self_exclusion(bank, query_eps):
    """B1 self-exclusion：若某 query window 本身来自 support episode，
    跨 episode 试验必须排除该 episode 作为 donor；并禁止抽 query 自身未来。
    本数据集 query/support episode 不重叠 -> 该排除条件为真空但保持生效。"""
    key = id(bank)
    if key not in _BANK_EPS_CACHE:
        _BANK_EPS_CACHE[key] = {r.support_episode_id for r in bank.records}
    bank_eps = _BANK_EPS_CACHE[key]
    return [e if e in bank_eps else None for e in query_eps]


if __name__ == "__main__":
    main()
