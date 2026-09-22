"""V0.6 阶段A：代码/数据盘点（§0 要求记录 resolved paths / commit / SHA256 / 训练配置）。

输出: audit/source_inventory.json
用法: python -m execution_wm.validity_v06.inventory --config execution_wm/configs/v06_validity.yaml
"""
import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter

import yaml


def sha256(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        n = 0
        while True:
            b = f.read(1 << 20)
            if not b or (limit and n >= limit):
                break
            h.update(b)
            n += len(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out = os.path.join(cfg["out_dir"], "audit")
    os.makedirs(out, exist_ok=True)

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                            capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo,
                           capture_output=True, text=True).stdout.strip()

    ckpts = {}
    for name in ("action_only", "direct", "context"):
        pth = os.path.join(os.path.dirname(cfg["checkpoint"]), name, "best.pt")
        ckpts[name] = {"path": pth, "sha256": sha256(pth),
                       "size_mb": round(os.path.getsize(pth) / 1e6, 2)}

    datasets = {}
    for key in ("dataset_v0", "dataset_probes"):
        idx_path = os.path.join(cfg[key], "index.json")
        info = {"path": cfg[key], "resolved": os.path.realpath(cfg[key])}
        if os.path.exists(idx_path):
            idx = json.load(open(idx_path))
            eps = idx["episodes"]
            info["n_episodes"] = len(eps)
            info["by_condition"] = dict(Counter(e["condition"] for e in eps))
            info["by_type"] = dict(Counter(e["episode_type"] for e in eps))
            info["by_termination"] = dict(Counter(e["termination_reason"] for e in eps))
            info["index_sha256"] = sha256(idx_path)
        datasets[key] = info

    inv = {
        "date": "2026-09-22",
        "repo": repo,
        "git_commit": commit,
        "git_dirty_files": dirty.splitlines() if dirty else [],
        "checkpoints": ckpts,
        "datasets": datasets,
        "train_config": {"path": cfg["train_config"],
                         "sha256": sha256(cfg["train_config"]),
                         "content": yaml.safe_load(open(cfg["train_config"]))},
        "derived_artifacts": {
            "pred_cache": {"path": cfg["pred_cache"], "sha256": sha256(cfg["pred_cache"])},
            "pair_manifest": {"path": cfg["pair_manifest"],
                              "sha256": sha256(cfg["pair_manifest"])},
        },
        "note": "所有路径为解析后绝对路径；旧结果目录保留未动；本轮新增仅 results/v0_6_validity_transfer/",
    }
    with open(os.path.join(out, "source_inventory.json"), "w") as f:
        json.dump(inv, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in inv.items() if k != "train_config"}, indent=1)[:1200])
    print(f"[inventory] -> {out}/source_inventory.json")


if __name__ == "__main__":
    main()
