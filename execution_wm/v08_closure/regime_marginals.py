"""V0.8 Stage A4：R0/R1 训练池的时序边际描述统计（只描述，不重配、不重训）。

占空比相同 ≠ 时序分布一致。逐 episode 从 20Hz cmd_ref / lb_execution 计算：
  停/走比例（命令零 slot + 执行低速比例）、command switch rate、dwell time、
  |Δu|（切换跳变）、初始速度（settle 后）、稳态/瞬态比例（切换后 0.3s 内为瞬态）。
输出：metrics/v07_regime_marginals.csv（逐 episode）+ 汇总打印。

运行：python -m execution_wm.v08_closure.regime_marginals
"""
import json
import os
import sys

import numpy as np
import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ)

DATA = os.environ.get("V07_DATA", "/media/hdd1/yuhang/datasets/execution_wm/v0_7")
OUT = os.environ.get("V08_OUT", "results/v0_8_visual_pilot")
DT = 0.05
TRANSIENT_S = 0.3
MOVE_TH = 0.05          # |cmd| 或 |exec| 低于此视为停


def episode_stats(path):
    d = np.load(path)
    cmd = d["cmd_ref"].astype(np.float64)          # [T,3] 20Hz
    exe = d["lb_execution"].astype(np.float64)
    T = len(cmd)
    active = np.abs(cmd).max(axis=1) > MOVE_TH
    moving = np.linalg.norm(exe[:, :2], axis=1) > MOVE_TH
    # 切换：相邻 grid 点命令变化
    dcmd = np.linalg.norm(np.diff(cmd, axis=0), axis=1)
    switches = dcmd > 1e-6
    n_sw = int(switches.sum())
    dur_s = T * DT
    # dwell：连续不变段长度
    runs, run = [], 1
    for k in range(1, T):
        if switches[k - 1]:
            runs.append(run); run = 1
        else:
            run += 1
    runs.append(run)
    dwell = np.array(runs) * DT
    # 初始速度：settle 后 1s（grid 20 附近）的执行速度
    i0 = min(20, T - 1)
    v0 = float(np.linalg.norm(exe[i0, :2]))
    # 瞬态：切换后 TRANSIENT_S 内的 grid 点比例
    tr = np.zeros(T, bool)
    sw_idx = np.where(switches)[0] + 1
    k_tr = int(round(TRANSIENT_S / DT))
    for k in sw_idx:
        tr[k:k + k_tr] = True
    return {"zero_cmd_frac": float((~active).mean()),
            "low_speed_frac": float((~moving).mean()),
            "switch_per_s": n_sw / dur_s,
            "dwell_mean_s": float(dwell.mean()), "dwell_min_s": float(dwell.min()),
            "dwell_max_s": float(dwell.max()),
            "abs_du_mean": float(dcmd[switches].mean()) if n_sw else 0.0,
            "abs_du_max": float(dcmd[switches].max()) if n_sw else 0.0,
            "init_speed": v0,
            "transient_frac": float(tr.mean()),
            "coact_frac": float((np.sum(np.abs(cmd) > MOVE_TH, axis=1) >= 2).mean())}


def main():
    idx = json.load(open(f"{DATA}/index.json"))["episodes"]
    rows = []
    for e in idx:
        if e["split"] != "train":
            continue
        p = os.path.join(DATA, e["file"].replace(".npz", ".20hz.npz")
                         if not e["file"].startswith("/") else e["file"].replace(".npz", ".20hz.npz"))
        if not os.path.exists(p):
            p = e["file"].replace(".npz", ".20hz.npz")
        st = episode_stats(p)
        st.update(episode_id=e["episode_id"], regime=e["coverage_regime"],
                  group_id=e["anchor_group_id"], condition=e["condition"],
                  script_id=e["script_id"])
        rows.append(st)
    df = pd.DataFrame(rows)
    # 两个口径：脚本 regime（R0 脚本 vs R1 脚本）与数据集池（R1 数据集 = R0 + R1 脚本）
    df["pool"] = "R1_dataset"
    df.loc[df.regime == "R0", "pool"] = "R0_dataset"
    df = pd.concat([df, df.assign(pool="R1_dataset")[df.regime == "R0"]], ignore_index=True)
    df.loc[(df.regime == "R1"), "pool"] = "R1_dataset"
    os.makedirs(f"{OUT}/metrics", exist_ok=True)
    df.to_csv(f"{OUT}/metrics/v07_regime_marginals.csv", index=False)

    cols = ["zero_cmd_frac", "low_speed_frac", "switch_per_s", "dwell_mean_s",
            "dwell_min_s", "dwell_max_s", "abs_du_mean", "abs_du_max", "init_speed",
            "transient_frac", "coact_frac"]
    pd.set_option("display.width", 240)
    print(f"train episodes: R0 脚本={int((df.regime=='R0').sum()/2)}, "
          f"R1 脚本={int((df.regime=='R1').sum())}（R0 脚本同时属于两个数据集池）")
    print("\n=== 按脚本 regime ===")
    print(df.drop_duplicates(["episode_id"]).groupby("regime")[cols].agg(["mean", "std"]).round(4).to_string())
    print("\n=== 按数据集池（R1_dataset = R0 + R1 脚本，对应冻结 marginals 口径）===")
    print(df.groupby("pool")[cols].agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    main()
