"""context_swap 共享工具：配置、模型加载、episode 窗口构造。"""
import hashlib
import json
import os
import subprocess

import numpy as np
import torch
import yaml

from execution_wm.data.dataset import discover_episodes, episode_proprio, load_episode
from execution_wm.eval.evaluate_execution import load_model

VEL_NAMES = ["vx", "vy", "wz"]


def discover_all(cfg):
    """从 cfg 的全部 dataset_dirs 发现 probe episodes，返回 {uid: ep_entry}。

    uid = f"d{数据集序号}_ep{episode_id:05d}" —— 不同数据集 episode_id 会撞，必须加前缀。
    同时在 ep_entry["meta"]["cond_group"] 写入按 friction 映射的统一条件名。
    """
    dirs = cfg.get("dataset_dirs") or [cfg["dataset_dir"]]
    groups = {float(k): v for k, v in cfg["condition_groups"].items()}
    out = {}
    for di, d in enumerate(dirs):
        for e in discover_episodes(d):
            m = e["meta"]
            if m.get("episode_type") != "probe":
                continue
            fr = round(float(m["friction"]), 2)
            if fr not in groups:
                continue
            m["cond_group"] = groups[fr]
            m["dataset_idx"] = di
            uid = f"d{di}_ep{int(m['episode_id']):05d}"
            out[uid] = e
    return out


def load_cfg(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def out_subdir(cfg, name):
    d = os.path.join(cfg["out_dir"], name)
    os.makedirs(d, exist_ok=True)
    return d


def load_context_model(cfg, device):
    """加载 V0 context checkpoint（唯一，不做任何挑选）。"""
    model = load_model(cfg["exp_dir"], cfg["model"],
                       int(round(cfg["horizon_s"] * cfg["hz"])), device)
    model.eval()
    return model


def checkpoint_info(cfg):
    path = os.path.join(cfg["exp_dir"], cfg["model"], "best.pt")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    summary_path = os.path.join(cfg["exp_dir"], cfg["model"], "summary.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
    return {
        "path": path,
        "sha256": h.hexdigest(),
        "size_bytes": os.path.getsize(path),
        "best_val_loss": summary.get("best_val_loss"),
        "train_config": summary.get("config"),
    }


def git_commit(repo_root):
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    except Exception:
        return "unknown"


class EpisodeData:
    """一个 episode 的 numpy 视图 + 窗口切片。"""

    def __init__(self, ep_entry):
        self.meta = ep_entry["meta"]
        d = load_episode(ep_entry["path"])
        self.proprio = episode_proprio(d)          # [T, 40]
        self.cmd = d["cmd_vel"].astype(np.float32)        # [T, 3]
        self.execution = d["execution"].astype(np.float32)
        self.residual = d["residual"].astype(np.float32)
        self.T = len(self.cmd)

    def window(self, t0, L, H):
        """t0 = 当前时刻（history 最后一帧）。返回 dict of np arrays。"""
        h0 = t0 - L + 1
        f0 = t0 + 1
        return {
            "history_proprio": self.proprio[h0:t0 + 1],       # [L, 40]
            "history_action": self.cmd[h0:t0 + 1],            # [L, 3]
            "future_action": self.cmd[f0:f0 + H],             # [H, 3]
            "future_execution": self.execution[f0:f0 + H],    # [H, 3]
            "future_residual": self.residual[f0:f0 + H],
        }


def to_torch(window, device):
    return {k: torch.from_numpy(np.ascontiguousarray(v))[None].to(device)
            for k, v in window.items() if k in
            ("history_proprio", "history_action", "future_action")}


@torch.no_grad()
def predict(model, ep, t0, L, H, device, context=None, zero_context=False):
    """对 episode ep 在 t0 做预测。context=None 用自身 c；否则用给定 [8] tensor。"""
    w = ep.window(t0, L, H)
    b = to_torch(w, device)
    z = model.encode_state(b["history_proprio"])
    if zero_context:
        c = torch.zeros(1, model.context_encoder.context_dim, device=device)
    elif context is not None:
        c = context
    else:
        c = model.encode_context(b["history_proprio"], b["history_action"])
    out = model.predict_execution(z, c, b["future_action"])
    e_hat = (b["future_action"] + out["r_hat"])[0].cpu().numpy()   # [H, 3]
    return e_hat, c[0].cpu().numpy()
