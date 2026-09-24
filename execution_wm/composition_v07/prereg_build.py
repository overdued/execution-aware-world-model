"""V0.7 Stage 1：采集前冻结 —— 生成 PRE_REGISTRATION_V07.md / command_cells.json / split_plan.json。

预算：24 groups × 3 friction × 12 scripts = 864 episodes（≤900）。
R0/R1 为等脚本、等回合、等窗口预算；两者只有"cell 组合覆盖"不同。
"""
import json
import os

import numpy as np

from execution_wm.composition_v07 import cells as C

OUT = "results/v0_7_composition/prereg"
N_GROUPS = 24
N_TRAIN_G, N_VAL_G, N_TEST_G = 12, 6, 6
N_SCRIPTS = 12
FRICTIONS = {"nominal": 1.0, "mid": 0.6, "low": 0.3}     # 写入值（combine=multiply, ground=1.0）
SETTLE_S = 1.0
N_TRIPLE_VAL, N_TRIPLE_TEST = 5, 9
ANCHOR_SEED_BASE, ANCHOR_SEED_STRIDE = 7000, 37


def build_train_scripts(gidx, all_train_doubles):
    """训练 group 的 12 条脚本：sc00-05 = R0，sc06-11 = R1。

    R0 数据集 = R0 脚本；R1 数据集 = R0 脚本 + R1 脚本（R1 是 R0 的超集）。
    要匹配每轴占空比 d，需要 R1 脚本贡献的**轴激活总量** == R0 脚本的：
        d_R0  = s1 / (3 * 72)
        d_R1  = (s1 + a2) / (3 * 144)        令相等 -> a2 = s1
    30 个 train double 本身贡献 60 次激活 -> a2 >= 60 -> s1 = 60。
    取 s1 = 60（12 个 single 各 5 次，R0 脚本 60 非零 + 12 零）
       a2 = 60（R1 脚本 = 30 double + 0 single + 42 零）
    -> 每轴占空比精确相等（0.2778）；共激活按设计不同（R0=0, R1=0.2083）；
       零占比不可避免不同（0.167 vs 0.375），如实报告。
    """
    singles = [c for c in C.all_cells() if C.cell_type(c) == "single"]
    zeros = [C.cell_id(2, 2, 2)]
    tpl = C.timing_templates()

    r0_pool = [s for s in singles for _ in range(5)] + zeros * 12          # 60 + 12 = 72
    r0_pool = list(np.random.default_rng(1000 + gidx).permutation(r0_pool))
    r1_pool = list(all_train_doubles) + zeros * 42                          # 30 + 42 = 72
    r1_pool = list(np.random.default_rng(2000 + gidx).permutation(r1_pool))

    scripts = []
    for s in range(N_SCRIPTS):
        regime = "R0" if s < 6 else "R1"
        pool = r0_pool if regime == "R0" else r1_pool
        tname = list(tpl.keys())[s % 6]
        take = pool[:12]; del pool[:12]
        scripts.append({"script_id": f"sc{s:02d}", "timing_template": tname,
                        "regime": regime,
                        "segments": [{"cell": c, "duration_s": float(d)}
                                     for c, d in zip(take, tpl[tname])]})
    assert not r0_pool and not r1_pool, (len(r0_pool), len(r1_pool))
    return scripts


def build_eval_scripts(gidx, kind, held, triples):
    """val/test group 的 12 条脚本：4 P0(seen cell) / 4 P1(held-out pair) / 4 P2(triple)。

    每条脚本 8 个非零 slot + 4 个 zero slot（固定 dwell 多重集）。
    """
    singles = [c for c in C.all_cells() if C.cell_type(c) == "single"]
    zeros = [C.cell_id(2, 2, 2)]
    tpl = C.timing_templates()

    def make_pool(nonzero_cells, n_slots, seed):
        """把 nonzero_cells 循环铺满 n_slots，再插入 zeros 到 12 槽/脚本。"""
        reps = int(np.ceil(n_slots / max(len(nonzero_cells), 1)))
        pool = (list(nonzero_cells) * reps)[:n_slots]
        pool = list(np.random.default_rng(seed + gidx).permutation(pool))
        out = []
        for i in range(0, n_slots, 8):
            out += pool[i:i + 8] + zeros * 4
        return out

    n_scripts_per_level = 4
    n_slots_per_level = n_scripts_per_level * 8
    p0 = make_pool(list(singles) + list(held["all_train_doubles"]), n_slots_per_level, 3000)
    p1 = make_pool(list(held[kind]), n_slots_per_level, 4000)
    p2 = make_pool(list(triples[kind]), n_slots_per_level, 5000)

    scripts = []
    for s in range(N_SCRIPTS):
        tname = list(tpl.keys())[s % 6]
        if s < 4:
            pool, lvl = p0, "P0_seen_cell"
        elif s < 8:
            pool, lvl = p1, "P1_heldout_pair"
        else:
            pool, lvl = p2, "P2_triple"
        take = pool[:12]; del pool[:12]
        scripts.append({"script_id": f"sc{s:02d}", "timing_template": tname, "level": lvl,
                        "segments": [{"cell": c, "duration_s": float(d)}
                                     for c, d in zip(take, tpl[tname])]})
    assert not p0 and not p1 and not p2, (len(p0), len(p1), len(p2))
    return scripts


def dwell_stats(scripts):
    d = [s["duration_s"] for sc in scripts for s in sc["segments"]]
    return {"n_segments": len(d), "mean_dwell_s": float(np.mean(d)),
            "min_dwell_s": float(np.min(d)), "max_dwell_s": float(np.max(d)),
            "n_switches": len(d)}


def cell_activity(scripts):
    """按轴统计占空比与共激活率。"""
    n_slot = 0
    act = np.zeros(3)
    multi = 0
    types = {}
    for sc in scripts:
        for seg in sc["segments"]:
            n_slot += 1
            t = C.cell_type(seg["cell"])
            types[t] = types.get(t, 0) + 1
            for i, v in enumerate(C.parse(seg["cell"])):
                if v != C.ZERO_IDX:
                    act[i] += 1
            if t in ("double", "triple"):
                multi += 1
    return {"n_slots": n_slot, "axis_duty_vx": float(act[0] / n_slot),
            "axis_duty_vy": float(act[1] / n_slot), "axis_duty_wz": float(act[2] / n_slot),
            "coactivation_slot_frac": float(multi / n_slot), "type_counts": types}


def main():
    os.makedirs(OUT, exist_ok=True)
    doubles = C.split_doubles()
    triples = C.pick_triples(doubles, n_val=N_TRIPLE_VAL, n_test=N_TRIPLE_TEST)
    all_train_doubles = sorted({c for v in doubles.values() for c in v["train"]})

    groups, scripts_by_group = [], {}
    for g in range(N_GROUPS):
        gid = f"G{g:02d}"
        if g < N_TRAIN_G:
            split = "train"
            scripts = build_train_scripts(g, all_train_doubles)
        else:
            split = "val" if g < N_TRAIN_G + N_VAL_G else "test"
            kind = "val" if split == "val" else "test"
            hd = {"all_train_doubles": all_train_doubles, "val": [], "test": []}
            for k, v in doubles.items():
                hd[kind] = hd[kind] + v[kind]
            scripts = build_eval_scripts(g, kind, hd,
                                         {"val": triples["val"], "test": triples["test"]})
        scripts_by_group[gid] = scripts
        groups.append({"group_id": gid, "split": split,
                       "reset_seed": ANCHOR_SEED_BASE + ANCHOR_SEED_STRIDE * g,
                       "n_scripts": N_SCRIPTS,
                       "regimes": sorted({s.get("regime") for s in scripts if s.get("regime")})
                       if split == "train" else ["eval"],
                       "activity": cell_activity(scripts), "dwell": dwell_stats(scripts)})

    eps_table = {"n_groups": N_GROUPS, "n_train_groups": N_TRAIN_G,
                 "n_val_groups": N_VAL_G, "n_test_groups": N_TEST_G,
                 "n_frictions": len(FRICTIONS), "n_scripts_per_group": N_SCRIPTS,
                 "episodes_formal": N_GROUPS * len(FRICTIONS) * N_SCRIPTS,
                 "smoke_budget_max": 12,
                 "total_cap": 900,
                 "anchor_seed_base": ANCHOR_SEED_BASE,
                 "anchor_seed_stride": ANCHOR_SEED_STRIDE,
                 "settle_s": SETTLE_S,
                 "episode_duration_s": SETTLE_S + sum(C.DWELL_MULTISET),
                 "control_hz": 50, "dataset_hz": 20}
    assert eps_table["episodes_formal"] + eps_table["smoke_budget_max"] <= 900

    cells_doc = {
        "value_set": {"values": C.VALUES, "A": C.A_MAG, "B": C.B_MAG,
                      "units": ["m/s", "m/s", "rad/s"],
                      "source": "Stage 0 实测 controller ranges ±1.0；a=0.35,b=0.65"},
        "zero_cell": C.cell_id(2, 2, 2),
        "singles": sorted([c for c in C.all_cells() if C.cell_type(c) == "single"]),
        "doubles": {k: {"pair": v["pair"], "train": sorted(v["train"]),
                        "val": sorted(v["val"]), "test": sorted(v["test"]),
                        "balance_score": v["balance_score"],
                        "balance_note": "3 cell 对 4 个取值，最优不平衡度 6.0（无法为 0），"
                                        "已穷举取最小；逐值频数见 count_check"}
                    for k, v in doubles.items()},
        "triples": {"available": triples["available"], "n_available": triples["n_available"],
                    "val": triples["val"], "test": triples["test"],
                    "requirement": "三个二元投影都在 R1 train（已逐条核验）",
                    "val_projections": triples["val_projections"],
                    "test_projections": triples["test_projections"]},
        "levels": {"P0_seen_cell": "cell 在训练出现，新 anchor / 新时序模板",
                   "P1_heldout_pair": "二元轴类型在训练出现，具体符号/幅值元组从未出现",
                   "P2_triple": "三轴同时未训练；三个二元投影都在 R1 train"},
    }
    # 逐值频数核查
    check = {}
    for k, v in doubles.items():
        for grp in ("train", "val", "test"):
            cnt = {ax: {} for ax in (0, 1, 2)}
            for c in v[grp]:
                for ax, idx in enumerate(C.parse(c)):
                    if idx != C.ZERO_IDX:
                        cnt[ax][C.VALUES[idx]] = cnt[ax].get(C.VALUES[idx], 0) + 1
            check[f"{k}_{grp}"] = {str(a): {str(kk): vv for kk, vv in cnt[a].items()}
                                   for a in (0, 1, 2)}
    cells_doc["count_check"] = check

    json.dump(cells_doc, open(os.path.join(OUT, "command_cells.json"), "w"),
              indent=1, ensure_ascii=False)
    json.dump({"episode_table": eps_table, "frictions": FRICTIONS,
               "groups": groups, "scripts": scripts_by_group,
               "templates": C.timing_templates(), "template_split": C.TEMPLATE_SPLIT,
               "windows": {"L": 20, "H": 40, "windows_per_ep_cap": 6,
                           "selection": "query 相位内确定性等距取 6 个 origin（无 RNG）"}},
              open(os.path.join(OUT, "split_plan.json"), "w"), indent=1, ensure_ascii=False)

    # ---- R0 vs R1 边际对照（§4.3 必须报告差异）----
    r0s = [s for gid in scripts_by_group for s in scripts_by_group[gid]
           if s.get("regime") == "R0"]
    r1s = [s for gid in scripts_by_group for s in scripts_by_group[gid]
           if s.get("regime") == "R1"]
    a0, a1 = cell_activity(r0s), cell_activity(r1s)
    print("=== R0 vs R1 边际对照（每 12 组训练 group）===")
    for k in ("n_slots", "axis_duty_vx", "axis_duty_vy", "axis_duty_wz",
              "coactivation_slot_frac"):
        print(f"  {k:26s} R0={a0[k]:.4f}  R1={a1[k]:.4f}")
    print("  type_counts R0:", a0["type_counts"], " R1:", a1["type_counts"])
    print(f"\nepisodes: {eps_table['episodes_formal']} (+smoke ≤12) / cap 900")
    print(f"groups: {N_TRAIN_G} train / {N_VAL_G} val / {N_TEST_G} test")
    print(f"triples available {triples['n_available']}, val {triples['val']}, test {triples['test']}")
    print(f"-> {OUT}/command_cells.json, split_plan.json")

    json.dump({"R0": a0, "R1": a1,
               "note": "两 regime 的脚本数/段数/dwell 多重集完全相同；"
                       "轴占空比尽量对齐，共激活率按设计不同（R0=0）"},
              open(os.path.join(OUT, "r0_vs_r1_marginals.json"), "w"),
              indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
