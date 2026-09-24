"""V0.7 后处理：param_check 表、主逐轴/逐时域表、代码快照与 git diff。"""
import json
import os
import subprocess
import time

import numpy as np
import pandas as pd
import torch

from execution_wm.composition_v07.data_v07 import H, build_datasets
from execution_wm.composition_v07.models_v07 import build, n_params

OUT = os.environ.get("V07_OUT", "results/v0_7_composition")


def main():
    # ---- 参数量 / 延迟 / FLOPs 估算 ----
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d, i = build("D").to(dev), build("I").to(dev)
    hp = torch.randn(64, 20, 40, device=dev); ha = torch.randn(64, 20, 3, device=dev)
    fa = torch.randn(64, H, 3, device=dev)
    res = {}
    for name, m in (("D", d), ("I", i)):
        m.eval()
        with torch.no_grad():
            for _ in range(3):
                m(hp, ha, fa)
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(20):
                m(hp, ha, fa)
            if dev == "cuda":
                torch.cuda.synchronize()
            lat = (time.time() - t0) / 20 / 64 * 1e3        # ms/sample
        res[name] = {"n_params": n_params(m), "latency_ms_per_sample_batch64": lat}
    res["param_ratio_I_over_D"] = res["I"]["n_params"] / res["D"]["n_params"]
    res["within_pm10pct"] = abs(res["param_ratio_I_over_D"] - 1) <= 0.10
    json.dump(res, open(f"{OUT}/metrics/param_check.json", "w"), indent=1)
    print(json.dumps(res, indent=1))

    # ---- 主逐轴/逐时域表（从 pred_cache 复算，保证同一缓存） ----
    cache_p = f"{OUT}/predictions/pred_cache.npz"
    if os.path.exists(cache_p):
        cache = np.load(cache_p)
        ds = build_datasets(os.environ.get("V07_DATA", "/media/hdd1/yuhang/datasets/execution_wm/v0_7"))
        rows = []
        for split in ("test_all", "test_P0", "test_P1", "test_P2"):
            dset = ds[split]
            if len(dset) == 0:
                continue
            for key in cache.files:
                if not key.startswith(split + "__"):
                    continue
                pr = cache[key]
                gt = dset.fe
                for ax, ai in enumerate(("vx", "vy", "wz")):
                    for ln, k in (("0.25s", 5), ("0.5s", 10), ("1.0s", 20), ("2.0s", 40)):
                        rows.append({"split": split, "variant": key.split("__", 1)[1],
                                     "axis": ax, "lead_s": ln.replace("s", ""),
                                     "MAE": float(np.abs(pr[:, k - 1, ai] -
                                                         gt[:, k - 1, ai]).mean()),
                                     "RMSE": float(np.sqrt(((pr[:, k - 1, ai] -
                                                             gt[:, k - 1, ai]) ** 2).mean()))})
        pd.DataFrame(rows).to_csv(f"{OUT}/metrics/main_axis_lead.csv", index=False)
        print(f"main_axis_lead.csv: {len(rows)} rows")

    # ---- 代码快照 + git diff ----
    os.makedirs(f"{OUT}/code_snapshot", exist_ok=True)
    subprocess.run(["bash", "-c",
                    f"cp -r execution_wm/composition_v07 {OUT}/code_snapshot/ && "
                    f"git diff HEAD~1 --stat > {OUT}/code_snapshot/git_diff_stat.txt; "
                    f"git rev-parse HEAD > {OUT}/code_snapshot/git_commit.txt"], check=False)
    print("code snapshot written")


if __name__ == "__main__":
    main()
