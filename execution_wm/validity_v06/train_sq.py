"""V0.6 阶段C：公平模型对照训练（M0-M3 × 3 seeds，PRE_REGISTRATION 冻结协议）。

四组（同数据、同 history/H、同 target、同 loss、同 early-stop、同预算）:
  M0 direct   : history -> r_hat（等输入 Direct recurrent）
  M1 context  : 现有 Context 结构（c 来自自身 history）
  M2 sq_context: 同 M1 结构，训练时 c 来自同 condition 的**另一条 support episode**
                 （cross-support/query 训练；decoder 与 M1 相同）
  M3 privileged: M1 + 真实 friction 标量输入（诊断用，非部署、非保证上界）

数据：d2 support/query（20Hz 派生版：inputs 因果 / labels 抗混叠）。
split：anchor group 级（train AQ0-4 / val AQ5 / test AQ6-7），Q4_combo held-out（test-only）。
early stop 与 scaler 只由 train/val 决定。

用法:
  python -m execution_wm.validity_v06.train_sq --model M1 --seed 42 \
      --config execution_wm/configs/v06_validity.yaml
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import yaml

from execution_wm.train.train_execution import MODEL_REGISTRY

SQ_DIR = "/media/hdd1/yuhang/datasets/execution_wm/v0_6_sq"
OUT_ROOT = "/media/hdd1/yuhang/checkpoints/execution_wm/v0_6"
L, H = 20, 40


def discover_sq(split_cfg):
    """返回 dict role -> [episode entry]（20hz 派生文件）。"""
    idx = json.load(open(os.path.join(SQ_DIR, "index.json")))["episodes"]
    roles = {"train": [], "val": [], "test": [], "support": []}
    for e in idx:
        e = dict(e)
        e["path20"] = e["file"].replace(".npz", ".20hz.npz")
        if not os.path.exists(e["path20"]):
            continue
        if e["episode_type"] == "support":
            roles["support"].append(e)
            continue
        ag = int(e["anchor_group"][2:])
        if e.get("held_out_family"):
            role = "test"
        elif ag in split_cfg["train_anchors"]:
            role = "train"
        elif ag in split_cfg["val_anchors"]:
            role = "val"
        else:
            role = "test"
        roles[role].append(e)
    return roles


class SQWindowData:
    """从 20hz 派生 npz 构建窗口张量。"""

    PROPRIO = ["base_linear_velocity_body", "base_angular_velocity", "projected_gravity",
               "imu_linear_acceleration", "joint_position", "joint_velocity", "feet_contact"]

    def __init__(self, entries, windows_per_ep=4, seed=0, require_query_phase=True):
        self.samples = []     # (hist_p, hist_a, fut_a, fut_r, friction, cond, ep_id)
        rng = np.random.default_rng(seed)
        for e in entries:
            d = np.load(e["path20"])
            proprio = np.concatenate([d[f"in_{k}"].reshape(len(d["timestamp"]), -1)
                                      for k in self.PROPRIO], axis=-1).astype(np.float32)
            cmd = d["cmd_vel"].astype(np.float32)
            res = d["lb_residual"].astype(np.float32)
            phase = d["phase"]
            T = len(cmd)
            t0s = [t for t in range(L - 1, T - H)
                   if not require_query_phase or phase[t] == 1]
            if not t0s:
                continue
            pick = rng.choice(t0s, size=min(windows_per_ep, len(t0s)), replace=False)
            for t0 in sorted(pick):
                self.samples.append((
                    proprio[t0 - L + 1:t0 + 1], cmd[t0 - L + 1:t0 + 1],
                    cmd[t0 + 1:t0 + 1 + H], res[t0 + 1:t0 + 1 + H],
                    float(e["friction"]), e["condition"], int(e["episode_id"]),
                    e.get("anchor_group") or f"SP{e.get('support_seed')}"))

    def __len__(self):
        return len(self.samples)

    def batch(self, idx, device):
        s = [self.samples[i] for i in idx]
        t = lambda a: torch.from_numpy(np.stack(a)).to(device)
        return (t([x[0] for x in s]), t([x[1] for x in s]), t([x[2] for x in s]),
                t([x[3] for x in s]), torch.tensor([x[4] for x in s],
                dtype=torch.float32, device=device),
                [x[5] for x in s], [x[6] for x in s], [x[7] for x in s])


class SupportWindowBank:
    """每个 condition 的 support 历史窗口张量池（M2 在线编码 c，梯度可回传）。"""

    def __init__(self, support_entries):
        self.by_cond = {}
        for e in support_entries:
            d = np.load(e["path20"])
            proprio = np.concatenate([d[f"in_{k}"].reshape(len(d["timestamp"]), -1)
                                      for k in SQWindowData.PROPRIO], axis=-1).astype(np.float32)
            cmd = d["cmd_vel"].astype(np.float32)
            T = len(cmd)
            for t0 in range(L - 1, T, 10):          # 每 0.5s 一个 support context 点
                self.by_cond.setdefault(e["condition"], []).append(
                    (proprio[t0 - L + 1:t0 + 1], cmd[t0 - L + 1:t0 + 1]))
        for c in self.by_cond:
            hp = np.stack([x[0] for x in self.by_cond[c]])
            ha = np.stack([x[1] for x in self.by_cond[c]])
            self.by_cond[c] = (hp, ha)

    def sample_batch(self, conds, rng, device):
        hp = np.stack([self.by_cond[c][0][rng.integers(0, len(self.by_cond[c][0]))]
                       for c in conds])
        ha = np.stack([self.by_cond[c][1][rng.integers(0, len(self.by_cond[c][1]))]
                       for c in conds])
        return (torch.from_numpy(hp).to(device), torch.from_numpy(ha).to(device))


def forward_model(model_name, model, hp, ha, fa, friction, model_aux=None,
                  support_hp=None, support_ha=None):
    if model_name == "M0":
        return model(hp, ha, fa)["r_hat"]
    if model_name == "M2":
        # c 来自同 condition 的 support 窗口（在线编码，梯度回传到 encoder）
        c = model.encode_context(support_hp, support_ha)
    elif model_name == "M3":
        # privileged：learned c + W·friction（W 为零初始化 8x1，参数差 +8，已声明）
        c = model.encode_context(hp, ha) + model_aux(friction.reshape(-1, 1))
    else:  # M1
        c = model.encode_context(hp, ha)
    z = model.encode_state(hp)
    return model.predict_execution(z, c, fa)["r_hat"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["M0", "M1", "M2", "M3"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--max-epochs", type=int, default=100)
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    t_start = time.time()

    sq_cfg = yaml.safe_load(open("execution_wm/configs/collect_v06_sq.yaml"))
    roles = discover_sq(sq_cfg["split"])
    train_data = SQWindowData(roles["train"] + roles["support"], seed=args.seed)
    val_data = SQWindowData(roles["val"], windows_per_ep=8, seed=0)
    print(f"[{args.model}/s{args.seed}] train {len(train_data)} val {len(val_data)} windows")

    base = "direct" if args.model == "M0" else "context"
    mcfg = {"hidden_dims": [256, 256], "gru_hidden": 128, "context_dim": 8,
            "predict_uncertainty": False}
    model = MODEL_REGISTRY[base](H, mcfg).to(device)
    model_aux = nn.Linear(1, 8).to(device) if args.model == "M3" else None
    if model_aux is not None:
        nn.init.zeros_(model_aux.weight)
        nn.init.zeros_(model_aux.bias)
    params = list(model.parameters()) + (list(model_aux.parameters()) if model_aux else [])
    opt = torch.optim.Adam(params, lr=1e-3)
    loss_fn = nn.SmoothL1Loss(beta=1.0)

    bank = SupportWindowBank(roles["support"]) if args.model == "M2" else None

    out_dir = os.path.join(OUT_ROOT, f"{args.model}_s{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    best_val, best_ep, patience = float("inf"), -1, 0
    bs = 128
    for ep in range(args.max_epochs):
        model.train()
        idx = rng.permutation(len(train_data))
        tot = 0.0
        for i in range(0, len(idx), bs):
            hp, ha, fa, fr_lab, fric, conds, _, _a = train_data.batch(idx[i:i + bs], device)
            shp, sha = (bank.sample_batch(conds, rng, device) if bank else (None, None))
            r_hat = forward_model(args.model, model, hp, ha, fa, fric, model_aux,
                                  shp, sha)
            loss = loss_fn(r_hat, fr_lab)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(hp)
        # val
        model.eval()
        vtot = 0.0
        with torch.no_grad():
            for i in range(0, len(val_data), bs):
                hp, ha, fa, fr_lab, fric, conds, _, _a = val_data.batch(
                    np.arange(i, min(i + bs, len(val_data))), device)
                shp, sha = (bank.sample_batch(conds, rng, device) if bank else (None, None))
                vtot += loss_fn(forward_model(args.model, model, hp, ha, fa, fric,
                                              model_aux, shp, sha),
                                fr_lab).item() * len(hp)
        vloss = vtot / max(len(val_data), 1)
        if vloss < best_val - 1e-5:
            best_val, best_ep, patience = vloss, ep, 0
            torch.save({"model_state": model.state_dict(),
                        "aux_state": model_aux.state_dict() if model_aux else None,
                        "config": {"model": mcfg}, "model_name": args.model,
                        "seed": args.seed, "val_loss": best_val},
                       os.path.join(out_dir, "best.pt"))
        else:
            patience += 1
        if ep % 10 == 0 or patience == 0:
            print(f"  ep{ep} train {tot/len(train_data):.4f} val {vloss:.4f}", flush=True)
        if patience >= 15:
            break
    gpu_h = (time.time() - t_start) / 3600
    with open(os.path.join(out_dir, "train_log.json"), "w") as f:
        json.dump({"model": args.model, "seed": args.seed, "best_val": best_val,
                   "best_epoch": best_ep, "epochs": ep + 1, "gpu_hours": gpu_h,
                   "n_train_windows": len(train_data), "n_val_windows": len(val_data),
                   "n_params": sum(p_.numel() for p_ in params)}, f, indent=1)
    print(f"[{args.model}/s{args.seed}] done best_val={best_val:.4f} "
          f"ep={best_ep} gpu_h={gpu_h:.3f}")


if __name__ == "__main__":
    main()
