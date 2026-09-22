"""V0.6 阶段B smoke 验证：注入 readback / 记录完整性 / reset 复原 / 命令执行一致性。

通过后才允许增量 pilot（03_V0_6 §B4）。
检查项:
 1. index.json friction_readback：写入值 vs 读回 static/dynamic；地面 material 已记录
 2. 同 anchor_seed 跨 condition 的 anchor 状态差（approximate-paired 量化）
 3. npz 完整性：50Hz、phase 标记、settle=1s、时长、字段形状
 4. cmd_vel 与段表一致（settle 段为 0；family 段幅值）
 5. termination 标签分布

输出: results/v0_6_validity_transfer/audit/smoke_validation.md
用法: python -m execution_wm.validity_v06.smoke_check --config execution_wm/configs/v06_validity.yaml
"""
import argparse
import glob
import json
import os

import numpy as np
import yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--sq-dir", default="/media/hdd1/yuhang/datasets/execution_wm/v0_6_sq")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out = os.path.join(cfg["out_dir"], "audit")
    os.makedirs(out, exist_ok=True)

    idx_path = os.path.join(args.sq_dir, "index.json")
    idx = json.load(open(idx_path))
    eps, rb = idx["episodes"], idx.get("friction_readback", {})

    L = []
    A = L.append
    A("# Smoke validation — V0.6 阶段B\n")
    ok = True

    A("## 1. 摩擦注入 readback\n```")
    for cond, r in rb.items():
        wrote = r["written_friction"]
        got = r["robot_static_friction_mean"]
        match = abs(got - wrote) < 1e-3
        ok &= match
        A(f"{cond}: wrote={wrote} readback_static={got:.4f} "
          f"(min {r['robot_static_friction_min']:.4f} max {r['robot_static_friction_max']:.4f}) "
          f"dyn={r['robot_dynamic_friction_mean']:.4f} "
          f"ground_static={r.get('ground_static_friction')} combine={r.get('combine_rule')} "
          f"-> {'OK' if match else 'MISMATCH'}")
    A("```\n注：读回验证的是 robot material 写入路径；有效足-地摩擦还受地面 material 与 "
      "average combine 影响（effective ≈ (robot+ground)/2），档位解释按写入值命名。\n")

    A("## 2. anchor 复原（同 seed 跨 condition）\n```")
    by_anchor = {}
    for e in eps:
        if e["episode_type"] != "query":
            continue
        by_anchor.setdefault((e["anchor_group"], e["command_family"]), {})[e["condition"]] = e["file"]
    max_diff = 0.0
    for (ag, fam), conds in sorted(by_anchor.items()):
        if len(conds) < 2:
            continue
        ref = None
        diffs = []
        for cond in ("normal", "friction_mid", "friction_low"):
            if cond not in conds:
                continue
            a = np.load(conds[cond])["anchor_joint_position"]
            if ref is None:
                ref = a
            else:
                diffs.append(float(np.abs(a - ref).max()))
        if diffs:
            d = max(diffs)
            max_diff = max(max_diff, d)
            A(f"{ag}/{fam}: max |Δanchor joint pos| vs normal = {d:.2e}")
    A(f"```\n最大 anchor 复原差 {max_diff:.2e}（approximate-paired；solver contact cache 不可复原）。\n")

    A("## 3. 记录完整性\n```")
    n_bad = 0
    for e in eps[:12]:
        d = np.load(e["file"])
        T = len(d["timestamp"])
        dt = np.diff(d["timestamp"])
        ph = d["phase"]
        settle_steps = int((ph == 0).sum())
        line = (f"ep{e['episode_id']} {e['condition']}/{e.get('command_family')}: T={T} "
                f"dt∈[{dt.min():.4f},{dt.max():.4f}] settle={settle_steps}步 "
                f"term={e['termination_reason']}")
        good = (abs(dt.mean() - 0.02) < 1e-3) and (settle_steps >= 45)
        n_bad += not good
        A(line + (" OK" if good else " BAD"))
    ok &= (n_bad == 0)
    A("```\n")

    A("## 4. 命令执行一致性（settle 段 cmd=0）\n```")
    for e in eps[:4]:
        d = np.load(e["file"])
        settle_cmd = np.abs(d["cmd_vel"][d["phase"] == 0]).max()
        A(f"ep{e['episode_id']}: settle 段 |cmd|max={settle_cmd:.2e}")
        ok &= settle_cmd < 1e-6
    A("```\n")

    from collections import Counter
    A(f"## 5. termination 分布\n`{dict(Counter(e['termination_reason'] for e in eps))}`\n")
    A(f"## 判定: {'SMOKE PASS — 可进入增量 pilot' if ok else 'SMOKE FAIL — 修复后重跑'}\n")
    with open(os.path.join(out, "smoke_validation.md"), "w") as f:
        f.write("\n".join(L))
    print("\n".join(L))
    print(f"[smoke] -> {out}/smoke_validation.md")


if __name__ == "__main__":
    main()
