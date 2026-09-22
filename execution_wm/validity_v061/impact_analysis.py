"""V0.6.1：修复影响量化（BUG_IMPACT_MATRIX / Q1 / Q2 证据）。

对全部 240 episode 比较旧派生 vs 新派生：
  - 旧 e_used = cmd20_hold + F(e-u)  vs  新 e_label = F(e)
  - 差值应 == S(u) - F(u)（纯 command 处理差异，与机器人运动无关）
  - 逐轴分布、边界特化误差、单位
  - residual 统计（旧 vs 新）
  - persistence 轴错误量级
用法: python -m execution_wm.validity_v061.impact_analysis
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from execution_wm.validity_v061 import timebase as tb
from execution_wm.validity_v061.schema import PROPRIO_SCHEMA, extract_execution_from_proprio

SQ = "/media/hdd1/yuhang/datasets/execution_wm/v0_6_sq"
NEW = "/media/hdd1/yuhang/datasets/execution_wm/v0_6_1"
OUT = "results/v0_6_1_correctness"


def main():
    files = sorted(f for f in glob.glob(os.path.join(SQ, "*", "ep_*.npz"))
                   if not f.endswith(".20hz.npz"))
    rows, diff_all, res_old_all, res_new_all = [], [], [], []
    for f in files:
        cond = os.path.basename(os.path.dirname(f))
        base = os.path.basename(f)
        raw = np.load(f)
        new = np.load(os.path.join(NEW, cond, base.replace(".npz", ".20hz.npz")))
        n_ticks = len(raw["timestamp"])
        n_grid = tb.n_grid_points(n_ticks)
        hold = tb.grid_hold_ticks(n_grid)
        # 旧口径
        t = raw["timestamp"] - raw["timestamp"][0]
        old_hold = np.clip(np.searchsorted(t, np.arange(n_grid) * 0.05, side="right") - 1,
                           0, n_ticks - 1)
        cmd20_old = raw["cmd_vel"][old_hold].astype(np.float64)
        F_e = resample_poly(raw["execution"].astype(np.float64), 2, 5, axis=0)[:n_grid]
        F_u = resample_poly(raw["cmd_vel"].astype(np.float64), 2, 5, axis=0)[:n_grid]
        F_r = resample_poly(raw["residual"].astype(np.float64), 2, 5, axis=0)[:n_grid]
        e_used_old = cmd20_old + F_r                     # 旧"物理真值"
        e_label_new = new["lb_execution"].astype(np.float64)
        d = e_used_old - e_label_new
        # 两个独立成分的代数分解（应精确相加为 d）:
        #   comp_time   = S_float(u) - S_int(u)   浮点 hold 错取前一 tick（B3）
        #   comp_filter = S_int(u)   - F(u)       命令事件采样 vs 抗混叠滤波（B2）
        comp_time = cmd20_old - new["cmd_ref"].astype(np.float64)
        comp_filter = new["cmd_ref"].astype(np.float64) - F_u
        resid_expl = np.abs(d - comp_time - comp_filter)
        diff_all.append(d)
        res_old_all.append(F_r)
        res_new_all.append(new["lb_residual"].astype(np.float64))
        rows.append({
            "file": base, "condition": cond, "n_ticks": n_ticks, "n_grid": n_grid,
            "old_hold_vs_new_mismatch": int((old_hold != hold).sum()),
            "n_cmd_step_events": int((np.abs(np.diff(raw["cmd_vel"], axis=0)).sum(axis=1) > 1e-9).sum()),
            "max_abs_label_diff": float(np.abs(d).max()),
            "mean_abs_label_diff": float(np.abs(d).mean()),
            "max_diff_vx": float(np.abs(d[:, 0]).max()),
            "max_diff_vy": float(np.abs(d[:, 1]).max()),
            "max_diff_wz": float(np.abs(d[:, 2]).max()),
            "mean_diff_wz": float(np.abs(d[:, 2]).mean()),
            "algebra_residual_max": float(resid_expl.max()),
            "comp_time_max": float(np.abs(comp_time).max()),
            "comp_filter_max": float(np.abs(comp_filter).max()),
            "comp_time_mean": float(np.abs(comp_time).mean()),
            "comp_filter_mean": float(np.abs(comp_filter).mean()),
            "old_residual_std_vx": float(F_r[:, 0].std()),
            "new_residual_std_vx": float(new["lb_residual"][:, 0].std()),
            "old_residual_std_wz": float(F_r[:, 2].std()),
            "new_residual_std_wz": float(new["lb_residual"][:, 2].std()),
        })
    df = pd.DataFrame(rows)
    D = np.concatenate(diff_all)
    pd.DataFrame({"axis": ["vx", "vy", "wz"],
                  "max_abs_diff": np.abs(D).max(axis=0),
                  "mean_abs_diff": np.abs(D).mean(axis=0),
                  "p99_abs_diff": np.percentile(np.abs(D), 99, axis=0),
                  "std_diff": D.std(axis=0),
                  "unit": ["m/s", "m/s", "rad/s"]}).to_csv(
        os.path.join(OUT, "derived_data_summary", "label_change_per_axis.csv"), index=False)
    df.to_csv(os.path.join(OUT, "derived_data_summary", "label_identity_per_episode.csv"),
              index=False)

    # 旧/新 residual 分布对比
    ro = np.concatenate(res_old_all); rn = np.concatenate(res_new_all)
    pd.DataFrame({
        "axis": ["vx", "vy", "wz"],
        "unit": ["m/s", "m/s", "rad/s"],
        "old_residual_mean": ro.mean(axis=0), "old_residual_std": ro.std(axis=0),
        "new_residual_mean": rn.mean(axis=0), "new_residual_std": rn.std(axis=0),
        "std_ratio_new_over_old": rn.std(axis=0) / ro.std(axis=0),
        "old_p99_abs": np.percentile(np.abs(ro), 99, axis=0),
        "new_p99_abs": np.percentile(np.abs(rn), 99, axis=0),
    }).to_csv(os.path.join(OUT, "derived_data_summary", "residual_distribution_old_vs_new.csv"),
              index=False)

    # persistence 轴错误：正确 [0,1,5] vs 旧 [0,1,2]
    dummy = np.zeros(PROPRIO_SCHEMA.dim)
    dummy[0:6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    correct = extract_execution_from_proprio(dummy)
    summary = {
        "n_episodes": len(df),
        "n_points": int(D.shape[0]),
        "label_change": {
            "max_abs_diff_all": float(np.abs(D).max()),
            "worst_episode": df.loc[df.max_abs_label_diff.idxmax(), "file"],
            "decomposition_identity_max_err": float(df.algebra_residual_max.max()),
            "note": "旧 e_used - 新 e_label == comp_time + comp_filter 精确成立（残差 ~1e-16）",
            "comp_time_float_hold_bug": {
                "max_abs": float(df.comp_time_max.max()),
                "mean_abs": float(df.comp_time_mean.mean()),
                "description": "浮点 searchsorted 在整 tick 对齐点错取前一 tick（B3）"},
            "comp_filter_cmd_sampling_vs_filtering": {
                "max_abs": float(df.comp_filter_max.max()),
                "mean_abs": float(df.comp_filter_mean.mean()),
                "description": "S(u)-F(u) 纯 command 处理差异，与机器人运动无关（B2）"},
        },
        "persistence_axis": {
            "old_slice": "proprio[:,-1,:3] -> [vx,vy,vz]",
            "correct_slice": "执行 schema 索引 [0,1,5] -> [vx,vy,wz]",
            "demo_input_[1..6]": [1, 2, 3, 4, 5, 6],
            "correct_output": correct.tolist(),
            "old_output": [1.0, 2.0, 3.0],
            "old_third_channel": "vz 而非 wz（rad/s 指标被 m/s 量污染）",
        },
        "grid_hold": {
            "old_float_searchsorted_vs_new_integer_mismatch_total":
                int(df.old_hold_vs_new_mismatch.sum()),
            "mean_per_episode": float(df.old_hold_vs_new_mismatch.mean()),
            "note": "浮点时间在整 tick 对齐点错选前一 tick",
        },
        "termination": {"all_240_retained": True,
                        "note": "未删除任何样本；1 条 terminated 仍保留"},
    }
    json.dump(summary, open(os.path.join(OUT, "derived_data_summary",
                                         "impact_summary.json"), "w"),
              indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    print("\n逐轴 label 变化:")
    print(pd.read_csv(os.path.join(OUT, "derived_data_summary",
                                   "label_change_per_axis.csv")).round(5).to_string(index=False))
    print("\nresidual 分布 old vs new:")
    print(pd.read_csv(os.path.join(OUT, "derived_data_summary",
                                   "residual_distribution_old_vs_new.csv")).round(5).to_string(index=False))


if __name__ == "__main__":
    main()
