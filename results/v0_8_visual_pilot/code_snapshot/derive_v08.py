"""V0.8：20Hz 派生包装（复用 derive_v07.derive，适配 v08 文件命名）。

v08 采集文件名为 <episode_id>.npz（非 ep_*.npz），派生规则/代码/参数与 V0.7
完全相同（预注册 §2），仅文件枚举不同。
运行：python -m execution_wm.v08_visual.derive_v08 --root <pilot_root>
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

from execution_wm.composition_v07.derive_v07 import derive


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    files = sorted(f for f in glob.glob(os.path.join(args.root, "*", "*", "*", "*.npz"))
                   if not f.endswith(".20hz.npz"))
    rows = []
    for f in files:
        out, meta = derive(f)
        np.savez_compressed(f.replace(".npz", ".20hz.npz"), **out)
        rows.append(meta)
    if not rows:
        print("[derive] no new episodes")
        return
    df = pd.DataFrame(rows)
    audit = os.path.join(args.root, "derive_identity_audit.csv")
    df.to_csv(audit, mode="a", header=not os.path.exists(audit), index=False)
    print(f"[derive] {len(files)} episodes; identity max err = {df.identity_max_err.max():.2e}; "
          f"ts grid max err = {df.timestamp_grid_max_err.max():.2e}; "
          f"quat raw norm dev max = {df.quat_norm_max_dev_raw.max():.4f}")


if __name__ == "__main__":
    main()
