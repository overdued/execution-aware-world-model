"""Step 5/6 quick look: 画 command / actual / residual，确认 mismatch 真实存在。

用法: python -m execution_wm.eval.quick_look --config execution_wm/configs/collect_v0.yaml
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from ..data.dataset import discover_episodes, load_episode  # noqa: E402

VEL = [("vx", 0), ("vy", 1), ("wz", 2)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    out_dir = args.out or os.path.join(cfg["output"]["dataset_dir"], "quick_look")
    os.makedirs(out_dir, exist_ok=True)

    eps = discover_episodes(cfg["output"]["dataset_dir"])
    conds = sorted({e["meta"]["condition"] for e in eps})
    # 每个 condition 挑一个 random episode 画三联图
    for cond in conds:
        cands = [e for e in eps if e["meta"]["condition"] == cond
                 and e["meta"]["episode_type"] == "random"]
        if not cands:
            continue
        d = load_episode(cands[0]["path"])
        t = d["timestamp"] - d["timestamp"][0]
        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        for ax, (name, i) in zip(axes, VEL):
            ax.plot(t, d["cmd_vel"][:, i], "k--", label="cmd", alpha=0.7)
            ax.plot(t, d["execution"][:, i], "b", label="actual")
            ax.plot(t, d["residual"][:, i], "r", label="residual", alpha=0.6)
            ax.set_ylabel(name)
            ax.legend(loc="upper right", fontsize=8)
        axes[0].set_title(f"{cond} (friction={cands[0]['meta']['friction']}, "
                          f"act={cands[0]['meta']['actuator_scale']})")
        axes[-1].set_xlabel("t [s]")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"quick_look_{cond}.png"), dpi=120)
        plt.close(fig)
    # 汇总表：每条件 |residual| 均值
    print(f"{'condition':<18}{'|r_vx|':>10}{'|r_vy|':>10}{'|r_wz|':>10}")
    for cond in conds:
        rs = [np.abs(load_episode(e["path"])["residual"]).mean(axis=0)
              for e in eps if e["meta"]["condition"] == cond
              and e["meta"]["episode_type"] == "random"]
        if rs:
            m = np.mean(rs, axis=0)
            print(f"{cond:<18}{m[0]:>10.4f}{m[1]:>10.4f}{m[2]:>10.4f}")
    print(f"\n[quick_look] -> {out_dir}")


if __name__ == "__main__":
    main()
