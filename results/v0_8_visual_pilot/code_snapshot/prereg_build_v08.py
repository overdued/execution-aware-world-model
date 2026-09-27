"""V0.8 预注册构建器：生成 split_plan_v08.json（采集前冻结）。

设计真源（细节见 report/V0_8_PRE_REGISTRATION.md，本文件只生成机器可读计划）：
- 6 场景布局 L0..L5（train L0-L2 / val L3 / test L4-L5），周界视觉地标几何不同；
- 每布局 8 个独立 group：reset_seed、摩擦、外观独立分配（外观与摩擦不相关）；
- 每 group 4 分支：U0/U1/U2 三个候选命令 + U0 重复（同初态/动作/工况）；
- 命令脚本复用 V0.7 split_plan.json 的 cell/模板冻结值，首个 segment 统一钉为
  公共前缀 cell c422（vx=+0.65），dwell 多重集不变；
- 192 episodes = 6×8×4；窗口规则与 V0.7 相同 + 匹配原点 tick 75。

运行：python -m execution_wm.v08_visual.prereg_build_v08 --out <plan.json>
"""
import argparse
import hashlib
import json
from pathlib import Path

V07_PLAN = Path(__file__).resolve().parents[2] / \
    "results/v0_7_composition/prereg/split_plan.json"

COMMON_PREFIX_CELL = "c422"          # vx=+0.65, vy=0, wz=0
EPISODES = {"settle_s": 1.0, "schedule_s": 12.6, "total_s": 13.6,
            "control_hz": 50, "dataset_hz": 20}
MATCH_ORIGIN_TICK = 75               # 1.5s：settle(1.0s)+0.5s，所有模板首段内
RESET_SEED_BASE = 9000               # v08 命名空间（v07 用 7000）
FRICTION_CYCLE = ["nominal", "mid", "low", "nominal", "mid", "low", "nominal", "mid"]

# 8 个外观 palette：panel 主色 / 地板色调 / 光强缩放。与摩擦分配不相关（见 group 表）。
PALETTES = [
    {"panels": (0.82, 0.36, 0.30), "floor": (0.42, 0.42, 0.44), "light": 1.00},
    {"panels": (0.30, 0.56, 0.82), "floor": (0.48, 0.45, 0.40), "light": 0.92},
    {"panels": (0.36, 0.72, 0.42), "floor": (0.40, 0.44, 0.46), "light": 1.06},
    {"panels": (0.85, 0.70, 0.28), "floor": (0.46, 0.42, 0.42), "light": 0.96},
    {"panels": (0.62, 0.40, 0.78), "floor": (0.44, 0.46, 0.40), "light": 1.03},
    {"panels": (0.30, 0.72, 0.70), "floor": (0.41, 0.41, 0.41), "light": 0.90},
    {"panels": (0.88, 0.50, 0.20), "floor": (0.47, 0.43, 0.45), "light": 1.08},
    {"panels": (0.55, 0.55, 0.60), "floor": (0.43, 0.45, 0.43), "light": 0.94},
]

# 场景几何：半径 12m 八方位 slot（45° 间隔）上的面板；L4 另有半径 10m 圆柱。
# panel: (slot, width_m, height_m)；slot 角度 = 45°×slot。
LAYOUTS = {
    "L0": {"split": "train", "panels": [(s, 4.0, 3.0) for s in (0, 2, 4, 6)], "cylinders": []},
    "L1": {"split": "train", "panels": [(s, 4.0, 3.0) for s in (1, 3, 5, 7)], "cylinders": []},
    "L2": {"split": "train", "panels": [(s, 4.0, 2.0 if s % 2 == 0 else 4.0)
                                        for s in range(8)], "cylinders": []},
    "L3": {"split": "val",   "panels": [(s, 4.0, 3.0) for s in (0, 3, 6)], "cylinders": []},
    "L4": {"split": "test",  "panels": [(s, 4.0, 3.0) for s in (1, 5)],
           "cylinders": [(s, 0.5, 4.0) for s in (0, 2, 4, 6)]},
    "L5": {"split": "test",  "panels": [(s, 4.0, 2.5) for s in (0, 1, 2, 4, 5, 6)],
           "cylinders": []},
}

# 分支 -> （角色, V0.7 脚本 id）
BRANCHES = [("U0", "sc06"), ("U1", "sc07"), ("U2", "sc08"), ("U0_repeat", "sc06")]
BRANCHES_EVAL = [("U0", "sc00"), ("U1", "sc04"), ("U2", "sc08"), ("U0_repeat", "sc00")]


def _pin_prefix(script):
    segs = [dict(s) for s in script["segments"]]
    segs[0]["cell"] = COMMON_PREFIX_CELL
    return {"timing_template": script["timing_template"], "segments": segs}


def build_plan():
    v07 = json.load(open(V07_PLAN))
    v07_scripts = v07["scripts"]
    groups, episodes = [], []
    for lay_idx, (lay_id, lay) in enumerate(sorted(LAYOUTS.items())):
        split = lay["split"]
        for g in range(8):
            gid = f"{lay_id}_G{g:02d}"
            friction = FRICTION_CYCLE[g]
            appearance_id = (3 * g + lay_idx) % 8     # 与摩擦周期 3 不相关
            reset_seed = RESET_SEED_BASE + 37 * (8 * lay_idx + g)
            if split == "train":
                src = f"G{(lay_idx * 8 + g) % 12:02d}"          # v07 train groups
                branch_src = BRANCHES
            elif split == "val":
                src = f"G{12 + g % 6:02d}"                       # v07 val groups
                branch_src = BRANCHES_EVAL
            else:
                off = g % 6 if lay_id == "L4" else (g + 3) % 6
                src = f"G{18 + off:02d}"                         # v07 test groups
                branch_src = BRANCHES_EVAL
            src_scripts = {s["script_id"]: s for s in v07_scripts[src]}
            levels = ({"U0": "R1", "U1": "R1", "U2": "R1", "U0_repeat": "R1"}
                      if split == "train" else
                      {"U0": "P0", "U1": "P1", "U2": "P2", "U0_repeat": "P0"})
            groups.append({"group_id": gid, "layout": lay_id, "split": split,
                           "reset_seed": reset_seed, "friction": friction,
                           "appearance_id": appearance_id,
                           "v07_script_source_group": src})
            for b, (role, sc) in enumerate(branch_src):
                script = _pin_prefix(src_scripts[sc])
                episodes.append({
                    "episode_id": f"{gid}_b{b}",
                    "group_id": gid, "branch": b, "role": role,
                    "level": levels[role], "layout": lay_id, "split": split,
                    "friction": friction, "appearance_id": appearance_id,
                    "reset_seed": reset_seed,
                    "v07_source": {"group": src, "script_id": sc},
                    "script": script,
                })
    plan = {
        "version": "v0.8-pilot",
        "episodes_cfg": EPISODES,
        "common_prefix": {"cell": COMMON_PREFIX_CELL,
                          "note": "首个 segment 统一钉为该 cell；dwell 多重集不变"},
        "match_origin_tick": MATCH_ORIGIN_TICK,
        "layouts": LAYOUTS, "palettes": PALETTES,
        "frictions": v07["frictions"],
        "windows": dict(v07["windows"]),
        "groups": groups, "episodes": episodes,
    }
    payload = json.dumps(plan, sort_keys=True).encode()
    plan["plan_sha256"] = hashlib.sha256(payload).hexdigest()
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    plan = build_plan()
    n = len(plan["episodes"])
    splits = {}
    for e in plan["episodes"]:
        splits[e["split"]] = splits.get(e["split"], 0) + 1
    assert n == 192, n
    assert splits == {"train": 96, "val": 32, "test": 64}, splits
    # 摩擦-外观不相关核验：每个摩擦值至少对应 3 种外观
    from collections import defaultdict
    fa = defaultdict(set)
    for g in plan["groups"]:
        fa[g["friction"]].add(g["appearance_id"])
    assert all(len(v) >= 3 for v in fa.values()), dict(fa)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(plan, indent=1))
    print(f"episodes={n} splits={splits} groups={len(plan['groups'])} "
          f"sha256={plan['plan_sha256'][:16]}...")


if __name__ == "__main__":
    main()
