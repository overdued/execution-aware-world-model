"""V0.6.2 公共设施：复用 V0.6.1 的 split / 固定 window manifest / 单次推理缓存。

原则（继承 V0.6.1）：
- 不重新推理：所有模型预测读 predictions/pred_cache.npz（sha256 固定）；
- window 与 V0.6.1 完全一致（固定 manifest，无 RNG）；
- 只用 train split 拟合任何 normalization / PCA / probe / baseline。
"""
import hashlib
import json
import os

import numpy as np
import torch

from execution_wm.validity_v061.splits import attach_derived_paths, classify
from execution_wm.validity_v061.windows import H, L, WindowManifest

SQ = "/media/hdd1/yuhang/datasets/execution_wm/v0_6_sq"
DERIVED = "/media/hdd1/yuhang/datasets/execution_wm/v0_6_1"
CKPT_V061 = "/media/hdd1/yuhang/checkpoints/execution_wm/v0_6_1"
CKPT_V062 = "/media/hdd1/yuhang/checkpoints/execution_wm/v0_6_2"
V061 = "results/v0_6_1_correctness"
OUT = "results/v0_6_2_generalization"
CACHE = os.path.join(V061, "predictions", "pred_cache.npz")
WPE = 8
SPLITS_EVAL = ["A_unseen_anchor_seen_family", "B_seen_anchor_unseen_family",
               "C_unseen_anchor_unseen_family", "supp_AQ5xQ4"]
SEEDS = (42, 43, 44)
MODELS = ("M0", "M1", "M2", "M3")


def get_entries():
    return attach_derived_paths(
        json.load(open(os.path.join(SQ, "index.json")))["episodes"], DERIVED)


def get_manifests():
    """-> dict[split] = WindowManifest（与 V0.6.1 完全相同的窗口）。"""
    sp = classify(get_entries())
    m = {"train": WindowManifest(sp["train"], "train", windows_per_ep=WPE),
         "val": WindowManifest(sp["val"], "val", windows_per_ep=WPE)}
    for k in SPLITS_EVAL:
        m[k] = WindowManifest(sp[k], k, windows_per_ep=WPE)
    return m


def load_cache(verify=True):
    man = json.load(open(os.path.join(V061, "predictions", "pred_cache_manifest.json")))
    if verify:
        sha = hashlib.sha256(open(CACHE, "rb").read()).hexdigest()
        assert sha == man["sha256"], "pred_cache 哈希不一致 —— 不得在此重算"
    return np.load(CACHE)


def pred_from_cache(cache, split, variant):
    """读同一缓存里的预测（不重新推理）。"""
    key = f"{split}__{variant}"
    assert key in cache.files, f"缓存缺少 {key}"
    return cache[key]


def seed_mean_pred(cache, split, model, seeds=SEEDS):
    arrs = [pred_from_cache(cache, split, f"{model}_s{s}") for s in seeds
            if f"{split}__{model}_s{s}" in cache.files]
    return np.mean(arrs, axis=0)


def command_matrix(man):
    """future command [N,H,3] -> flatten [N,3H]。"""
    return man.fa.reshape(len(man), -1).astype(np.float64)


def per_axis_errors(pred, gt):
    """[N,H,3] -> dict axis -> (N,) horizon-mean abs err。"""
    e = np.abs(pred - gt)
    return {ax: e[:, :, i].mean(axis=1) for i, ax in enumerate(("vx", "vy", "wz"))}


def load_model_v061(name, seed, device):
    from execution_wm.train.train_execution import MODEL_REGISTRY
    path = os.path.join(CKPT_V061, f"{name}_s{seed}", "best.pt")
    ck = torch.load(path, weights_only=False, map_location=device)
    base = "direct" if name == "M0" else "context"
    m = MODEL_REGISTRY[base](H, ck["config"]["model"]).to(device)
    m.load_state_dict(ck["model_state"])
    m.eval()
    return m, ck


def simpson_diversity(counts):
    return 1.0 - float(np.sum((np.asarray(counts, dtype=np.float64) /
                               max(np.sum(counts), 1)) ** 2))
