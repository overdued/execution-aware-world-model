"""Episode 加载与 episode-level 划分（first_work.md §7, §14）。

不依赖 isaaclab，可在普通 python 下运行。
"""
import glob
import json
import os

import numpy as np

# 每个 episode 用于模型输入的 proprioception 字段（§10/§11: 不含任何仿真 metadata）
PROPRIO_KEYS = [
    "base_linear_velocity_body",   # 3
    "base_angular_velocity",       # 3
    "projected_gravity",           # 3
    "imu_linear_acceleration",     # 3
    "joint_position",              # 12
    "joint_velocity",              # 12
    "feet_contact",                # 4
]
PROPRIO_DIM = 40


def load_episode(npz_path):
    """读取一个 episode npz -> dict of [T, ...] arrays。"""
    with np.load(npz_path) as z:
        return {k: z[k] for k in z.files}


def load_meta(json_path):
    with open(json_path) as f:
        return json.load(f)


def discover_episodes(dataset_dir):
    """遍历 dataset_dir/<condition>/ep_*.npz，返回 [{path, meta_path, meta}, ...]。"""
    eps = []
    for npz in sorted(glob.glob(os.path.join(dataset_dir, "*", "ep_*.npz"))):
        meta_path = npz.replace(".npz", ".json")
        if not os.path.exists(meta_path):
            continue
        eps.append({"path": npz, "meta_path": meta_path, "meta": load_meta(meta_path)})
    return eps


def episode_proprio(ep):
    """拼接 proprioception 特征 [T, PROPRIO_DIM]。"""
    feats = []
    for k in PROPRIO_KEYS:
        v = ep[k]
        feats.append(v.reshape(len(v), -1))
    return np.concatenate(feats, axis=-1).astype(np.float32)


def split_episodes(eps, val_fraction, test_id_fraction, ood_conditions,
                   exclude_probe_from_train=True, seed=42):
    """episode-level split（§14）。

    返回 dict: train / val / test_id / test_ood / probe（均为 episode 列表）。
    - ood_conditions 整组进 test_ood
    - probe episodes 单独返回（matched-command 评估用），不进 train/val/test_id
    """
    rng = np.random.default_rng(seed)
    train, val, test_id, test_ood, probe = [], [], [], [], []
    # 按 condition 分组，每个 condition 内部做 episode-level 划分
    by_cond = {}
    for e in eps:
        by_cond.setdefault(e["meta"]["condition"], []).append(e)
    for cond, items in by_cond.items():
        random_eps = [e for e in items if e["meta"].get("episode_type") == "random"]
        probe_eps = [e for e in items if e["meta"].get("episode_type") == "probe"]
        probe.extend(probe_eps)
        if cond in ood_conditions:
            test_ood.extend(random_eps)
            continue
        rng.shuffle(random_eps)
        n = len(random_eps)
        nv = max(1, int(round(n * val_fraction))) if n >= 3 else 0
        nt = max(1, int(round(n * test_id_fraction))) if n >= 3 else 0
        val.extend(random_eps[:nv])
        test_id.extend(random_eps[nv:nv + nt])
        train.extend(random_eps[nv + nt:])
    return {"train": train, "val": val, "test_id": test_id,
            "test_ood": test_ood, "probe": probe}
