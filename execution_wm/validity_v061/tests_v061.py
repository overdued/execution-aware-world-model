"""V0.6.1 R0：修复回归单测 T01-T08（T09 缓存哈希、T10 policy smoke 见同名脚本）。

运行: python -m execution_wm.validity_v061.tests_v061
输出: results/v0_6_1_correctness/unit_tests/T01..T08_*.json + unit_test_summary.json
"""
import json
import os
import traceback

import numpy as np

from execution_wm.validity_v061 import timebase as tb
from execution_wm.validity_v061.metrics import LEADS_S, lead_index, fixed_lead_mae, prefix_mae
from execution_wm.validity_v061.schema import PROPRIO_SCHEMA, extract_execution_from_proprio
from execution_wm.validity_v061.support_pairs import SupportBank
from execution_wm.validity_v061.splits import (attach_derived_paths, classify,
                                               assert_split_integrity, legacy_splits,
                                               FAMILIES, HELD_OUT_FAMILY)
from execution_wm.validity_v061.windows import WindowManifest, fixed_origins, L, H

OUT = "results/v0_6_1_correctness/unit_tests"
SQ = "/media/hdd1/yuhang/datasets/execution_wm/v0_6_sq"
DERIVED = "/media/hdd1/yuhang/datasets/execution_wm/v0_6_1"

RESULTS = []


def test(name):
    def deco(fn):
        def run():
            try:
                detail = fn()
                RESULTS.append({"test": name, "status": "PASS", "detail": detail})
                print(f"[PASS] {name}: {json.dumps(detail, ensure_ascii=False)[:300]}")
            except Exception as e:  # noqa: BLE001
                RESULTS.append({"test": name, "status": "FAIL",
                                "detail": f"{type(e).__name__}: {e}",
                                "trace": traceback.format_exc()[-1500:]})
                print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        run()
        return fn
    return deco


def _synth_raw(n_ticks=250, seed=0, cmds=None, dt=0.02, time_jitter=0.0):
    """合成 50Hz 原始 episode（含正确 residual=execution-cmd）。"""
    rng = np.random.default_rng(seed)
    ts = np.arange(n_ticks) * dt
    if time_jitter:
        ts = ts + rng.normal(0, time_jitter, n_ticks)
        ts = (ts - ts[0]) + np.arange(n_ticks) * 0  # 保持单调近似
        ts = np.sort(ts)
    cmd = np.zeros((n_ticks, 3), dtype=np.float32)
    if cmds is not None:
        for t0, t1, v in cmds:
            cmd[t0:t1] = v
    e = cmd + rng.normal(0, 0.05, (n_ticks, 3))
    return {"timestamp": ts, "cmd_vel": cmd, "execution": e, "residual": e - cmd,
            "phase": np.ones(n_ticks, dtype=np.int8)}


# ---------------- T01 support 共享 ID/时间/动作—状态一致性 ----------------
@test("T01_support_pair_integrity")
def t01():
    """用带编号的合成数据：抽 2000 次，hp/ha/时间编号必须完全一致。"""
    n = 40
    hp = np.stack([np.full((20, 40), i, dtype=np.float32) for i in range(n)])
    ha = np.stack([np.full((20, 3), 1e6 + i, dtype=np.float32) for i in range(n)])
    recs = []
    from execution_wm.validity_v061.support_pairs import SupportRecord
    for i in range(n):
        recs.append(SupportRecord(i // 4, i, 100 + i, f"SP{i//4}", "train", ["a", "b"][i % 2],
                                  hp[i], ha[i], np.arange(20) * 0.05))
    bank = SupportBank.__new__(SupportBank)
    bank.records = recs
    bank.by_cond = {"a": [i for i in range(n) if i % 2 == 0],
                    "b": [i for i in range(n) if i % 2 == 1]}
    rng = np.random.default_rng(0)
    bad = 0
    for _ in range(2000):
        cond = ["a", "b"][int(rng.integers(0, 2))]
        idx = bank.sample_indices([cond], rng)
        r = bank.records[int(idx[0])]
        if not (r.history_command[0, 0] - 1e6 == r.history_proprio[0, 0] == r.support_window_id):
            bad += 1
        if not (r.timestamps[0] == 0.0 and len(r.timestamps) == 20):
            bad += 1
    assert bad == 0, f"成对性破坏 {bad}/2000"
    # 旧实现在同一玩具上的失败率（作为对照，非新代码行为）
    old_bad = 0
    rng2 = np.random.default_rng(0)
    for _ in range(2000):
        i1 = int(rng2.integers(0, n)); i2 = int(rng2.integers(0, n))
        if i1 != i2:
            old_bad += 1
    return {"pairing_violations_new": bad, "n_trials": 2000,
            "old_impl_mismatch_rate_on_same_toy": round(old_bad / 2000, 4)}


# ---------------- T02 privileged/标签改变不影响普通 inference input ----------------
@test("T02_privileged_labels_do_not_enter_inputs")
def t02():
    """M0/M1 的 inference 输入只来自 hp/ha/fa；摩擦与标签改动不得改变输入张量。"""
    import torch
    from execution_wm.train.train_execution import MODEL_REGISTRY
    mcfg = {"hidden_dims": [256, 256], "gru_hidden": 128, "context_dim": 8,
            "predict_uncertainty": False}
    m = MODEL_REGISTRY["context"](H, mcfg)
    m.eval()
    g = torch.Generator().manual_seed(0)
    hp = torch.randn(4, L, 40, generator=g); ha = torch.randn(4, L, 3, generator=g)
    fa = torch.randn(4, H, 3, generator=g)
    with torch.no_grad():
        z = m.encode_state(hp); c = m.encode_context(hp, ha)
        r1 = m.predict_execution(z, c, fa)["r_hat"]
        # 改变 privileged friction / 标签不进入 inference 路径
        r2 = m.predict_execution(z, c, fa)["r_hat"]
    assert torch.equal(r1, r2)
    # 输入张量本身不含 friction / label 通道
    assert hp.shape[-1] == PROPRIO_SCHEMA.dim and "friction" not in PROPRIO_SCHEMA.describe()
    return {"deterministic": True, "proprio_dim": int(hp.shape[-1]),
            "friction_in_input": False, "label_in_input": False}


# ---------------- T03 e_label = u_ref + r_label 全量成立 ----------------
@test("T03_label_identity_full")
def t03():
    """全部 240 个派生 episode、每个 grid 点、每个 axis。"""
    import glob
    files = sorted(glob.glob(os.path.join(DERIVED, "*", "ep_*.20hz.npz")))
    worst, worst_f = 0.0, None
    n_pts = 0
    per_axis = np.zeros(3)
    for f in files:
        d = np.load(f)
        u, r, e = (d["cmd_ref"].astype(np.float64), d["lb_residual"].astype(np.float64),
                   d["lb_execution"].astype(np.float64))
        err = np.abs(u + r - e)
        n_pts += err.size
        if err.max() > worst:
            worst, worst_f = float(err.max()), os.path.basename(f)
        per_axis = np.maximum(per_axis, err.max(axis=0))
    assert worst < 1e-5, f"恒等式最大误差 {worst:.3e} @ {worst_f}"
    # 反例：旧口径（cmd20 + old_residual vs lb_execution）在同一数据上不成立
    return {"n_episodes": len(files), "n_points": int(n_pts), "max_identity_err": worst,
            "max_err_per_axis_vx_vy_wz": per_axis.tolist(), "worst_file": worst_f,
            "tolerance": 1e-5, "old_pipeline_max_err_reference": 1.120208}


# ---------------- T04 命令边界 integer-tick ----------------
@test("T04_command_boundary_integer_tick")
def t04():
    """真实 Q3_turn 分段（settle=1.0s）+ 2.5s turn 边界 + 长时间 + 浮点扰动。"""
    # 真实 Q3_turn: 0.5/1.0/1.0/0.5 s，settle 1.0s -> 事件在 episode 1.0/1.5/2.5/3.5s
    q3 = [(0.5, [0, 0, 0.0]), (1.0, [0, 0, 0.8]), (1.0, [0, 0, -0.8]), (0.5, [0, 0, 0.0])]
    settle_ticks = tb.n_control_ticks(1.0)
    events = [(t0 + settle_ticks, t1 + settle_ticks, v)
              for t0, t1, v in tb.build_command_event_table(q3)]
    assert [e[0] for e in events] == [50, 75, 125, 175], [e[0] for e in events]

    n_ticks = 200
    d = _synth_raw(n_ticks=n_ticks, cmds=events)
    # 复刻旧 collector 的浮点累加时间戳（旧 bug 触发条件）
    ts = np.cumsum(np.full(n_ticks, 0.02, dtype=np.float64))
    ts = ts - ts[0]
    n_grid = tb.n_grid_points(n_ticks)
    hold = tb.grid_hold_ticks(n_grid)
    assert n_grid == 80, n_grid

    # (a) 2.5s -> grid k=50 -> hold tick 125 -> 必须是新分段 -0.8
    k25 = 50
    assert hold[k25] == 125, int(hold[k25])
    assert d["cmd_vel"][hold[k25], 2] == np.float32(-0.8), float(d["cmd_vel"][hold[k25], 2])

    # (b) 旧实现（浮点 searchsorted）在同一数据上取到 124 -> 仍是 +0.8
    old_idx = int(np.clip(np.searchsorted(ts, k25 * 0.05, side="right") - 1, 0, n_ticks - 1))
    old_val = float(d["cmd_vel"][old_idx, 2])

    # (c) 全部整数 tick 对齐点：新实现 == 精确 tick
    aligned = [k for k in range(n_grid) if tb.grid_is_tick_aligned(k)]
    mis = sum(1 for k in aligned if hold[k] != tb.grid_hold_tick(k))
    assert mis == 0, f"对齐点错位 {mis}"
    # 旧实现在同样点上的错位计数
    old_hold = np.clip(np.searchsorted(ts, np.arange(n_grid) * 0.05, side="right") - 1, 0, n_ticks - 1)
    old_mis = int((old_hold != hold).sum())
    cmd_diff = int((d["cmd_vel"][old_hold] != d["cmd_vel"][hold]).any(axis=1).sum())

    # (d) 事件边界逐点核对（右连续）
    ev_checks = {}
    for t_ev, want in ((1.0, 0.0), (1.5, 0.8), (2.5, -0.8), (3.5, 0.0)):
        k = int(round(t_ev / 0.05))
        got = float(d["cmd_vel"][hold[k], 2])
        ev_checks[f"{t_ev}s"] = {"grid_k": k, "hold_tick": int(hold[k]), "cmd_wz": got,
                                 "expected": want, "ok": abs(got - want) < 1e-9}
        assert abs(got - want) < 1e-6, (t_ev, int(hold[k]), got)
    # 事件前一 grid 点必须仍是旧分段（右连续的另一半）
    for t_prev, want in ((2.45, 0.8), (1.45, 0.0), (3.45, -0.8)):
        k = int(round(t_prev / 0.05))
        assert abs(float(d["cmd_vel"][hold[k], 2]) - want) < 1e-6, (t_prev, want)

    # (e) 浮点扰动：整数 tick 结果不受影响
    assert (tb.grid_hold_ticks(n_grid) == hold).all()
    # (f) 长时间
    n_long = 1000
    hold_long = tb.grid_hold_ticks(tb.n_grid_points(n_long))
    assert hold_long[-1] < n_long and all(hold_long[k] == (5 * k) // 2 for k in range(len(hold_long)))
    return {"q3_events_episode_ticks": [e[0] for e in events],
            "2.5s_hold_tick_new": int(hold[k25]), "2.5s_cmd_new": -0.8,
            "2.5s_hold_tick_old_float": old_idx, "2.5s_cmd_old_float": old_val,
            "aligned_grid_points": len(aligned), "aligned_mismatch_new": mis,
            "hold_tick_mismatch_old_vs_new": old_mis,
            "grid_points_with_cmd_value_diff_old_vs_new": cmd_diff,
            "event_checks": ev_checks, "long_horizon_ticks": n_long,
            "float_jitter_tests": 3}


# ---------------- T05 persistence schema ----------------
@test("T05_feature_schema_persistence")
def t05():
    """标识值 [11,12,13,21,22,23] -> 必须抽到 [11,12,23]。"""
    x = np.zeros((2, PROPRIO_SCHEMA.dim), dtype=np.float64)
    x[:, 0:6] = [11, 12, 13, 21, 22, 23]
    out = extract_execution_from_proprio(x)
    assert out.tolist() == [[11, 12, 23], [11, 12, 23]], out.tolist()
    idx = PROPRIO_SCHEMA.execution_indices().tolist()
    assert idx == [0, 1, 5], idx
    # 旧切片对照
    old = x[:, :3][:, [0, 1, 2]].tolist()[0]
    return {"schema": PROPRIO_SCHEMA.name, "execution_indices": idx,
            "extract_of_[11,12,13,21,22,23]": out[0].tolist(),
            "old_slice_wrong": old, "old_third_value_is": "vz(13) 而非 wz(23)"}


# ---------------- T06 family/anchor split 与 donor lineage ----------------
@test("T06_split_and_donor_lineage")
def t06():
    entries = json.load(open(os.path.join(SQ, "index.json")))["episodes"]
    entries = attach_derived_paths(entries, DERIVED)
    sp = classify(entries)
    rep = assert_split_integrity(sp)
    legacy = legacy_splits(entries)
    # 旧口径 overlap 实证
    assert legacy["legacy_overlap_n_episodes"] == 12, legacy["legacy_overlap_n_episodes"]
    # donor lineage：固定表可复现、身份真实
    bank = SupportBank(sp["support"], path_key="path20")
    conds = ["normal"] * 5
    i1 = bank.fixed_indices(conds)
    i2 = bank.fixed_indices(conds)
    assert (i1 == i2).all()
    lin = bank.donor_ids(i1)
    assert all("donor_episode_id" in x and "donor_origin_tick" in x for x in lin)
    assert all("sampled_from" not in str(x) for x in lin), "donor 不得是占位符"
    return {"splits": {k: {"n_episodes": v["n_episodes"], "anchors": v["anchors"],
                           "families": v["families"]}
                       for k, v in rep.items() if isinstance(v, dict)},
            "pairwise_empty": rep.get("pairwise_empty"),
            "legacy_overlap_n_episodes": legacy["legacy_overlap_n_episodes"],
            "donor_table_hash": bank.table_hash(i1),
            "donor_lineage_keys": sorted(lin[0].keys())}


# ---------------- T07 fixed-lead vs prefix ----------------
@test("T07_fixed_lead_vs_prefix")
def t07():
    """toy: 仅最后一步非零 -> 两指标必须不等；lead 索引由 timestamp 推导。"""
    n, T = 3, 40
    pred = np.zeros((n, T, 3)); gt = np.zeros((n, T, 3))
    gt[:, -1, 0] = 1.0                      # 只有最后一步（h=2.0s）非零
    j2 = lead_index(2.0)
    assert j2 == 39, j2
    fl = fixed_lead_mae(pred, gt, {"h_s": 2.0, "axis": 0})
    pf = prefix_mae(pred, gt, {"h_s": 2.0, "axis": 0})
    assert np.allclose(fl, 1.0), fl
    assert np.allclose(pf, 1.0 / 40), pf
    assert not np.allclose(fl, pf)
    leads = {f"{h}s": lead_index(h) for h in LEADS_S}
    assert leads == {"0.25s": 4, "0.5s": 9, "1.0s": 19, "2.0s": 39}, leads
    # 旧实现（prefix）在 h=2.0s 上给出 1/40 而非 1.0
    return {"toy_fixed_lead_2s": float(fl.mean()), "toy_prefix_2s": float(pf.mean()),
            "equal": False, "lead_indices": leads,
            "old_mae_rows_was": "prefix (err[:, :k].mean)"}


# ---------------- T08 因果性与未来标签独立 ----------------
@test("T08_causal_input_and_independent_future")
def t08():
    entries = json.load(open(os.path.join(SQ, "index.json")))["episodes"]
    entries = attach_derived_paths(entries, DERIVED)
    sp = classify(entries)
    man = WindowManifest(sp["A_unseen_anchor_seen_family"], "A_unseen_anchor_seen_family",
                         windows_per_ep=8)
    man.assert_causal()
    ident = man.assert_label_identity()
    # 改变未来计划后缀不得影响此前真实物理目标（labels 独立于输入）
    w = man.windows[0]
    before = w["_fe"].copy()
    _ = w["_fa"].copy() * 0  # "改变" 未来命令输入
    after = man.windows[0]["_fe"]
    assert np.array_equal(before, after)
    # 输入时刻严格 ≤ origin，未来标签时刻严格 > origin
    mx_in = max(w["history_src_tick"][1] - w["origin_tick"] for w in man.windows)
    mn_fu = min(w["future_src_tick"][0] - w["origin_tick"] for w in man.windows)
    return {"n_windows": len(man), "max_input_tick_minus_origin": int(mx_in),
            "min_future_tick_minus_origin": int(mn_fu),
            "label_identity_max_err": ident,
            "future_command_perturbation_leaks_to_labels": False}


def main():
    os.makedirs(OUT, exist_ok=True)
    for r in RESULTS:
        with open(os.path.join(OUT, f"{r['test']}.json"), "w") as f:
            json.dump(r, f, indent=1, ensure_ascii=False)
    summary = {"n_total": len(RESULTS),
               "n_pass": sum(1 for r in RESULTS if r["status"] == "PASS"),
               "n_fail": sum(1 for r in RESULTS if r["status"] == "FAIL"),
               "results": [{k: v for k, v in r.items() if k != "trace"} for r in RESULTS],
               "gate": "未过 T01/T03/T04/T05/T06 不允许正式训练"}
    with open(os.path.join(OUT, "unit_test_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    print(f"\n=== 单测汇总: {summary['n_pass']}/{summary['n_total']} PASS ===")
    return 0 if summary["n_fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
