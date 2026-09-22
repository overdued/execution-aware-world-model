"""V0.6.1 R1（可选但有价值）：旧 checkpoint + 正确 execution 标签离线重评。

标注 OLD_MODEL_NEW_EVALUATION_ONLY：只说明"评价变化"，不能代替正确标签重训。
做法：加载 V0.6 的 12 个 checkpoint，在 V0.6.1 的**同一窗口**上推理，
      e_hat = cmd_ref(new) + r_hat_old，与独立真值 lb_execution(new) 比较。
旧模型训练目标是旧 residual -> 这里衡量的是"旧残差预测器放在正确真值下的表现"。
"""
import glob
import hashlib
import json
import os

import numpy as np
import pandas as pd
import torch

from execution_wm.train.train_execution import MODEL_REGISTRY
from execution_wm.validity_v061.metrics import all_metric_axes, lead_index
from execution_wm.validity_v061.splits import attach_derived_paths, classify
from execution_wm.validity_v061.windows import H, WindowManifest

OLD_CKPT = "/media/hdd1/yuhang/checkpoints/execution_wm/v0_6"
SQ = "/media/hdd1/yuhang/datasets/execution_wm/v0_6_sq"
DER = "/media/hdd1/yuhang/datasets/execution_wm/v0_6_1"
OUT = "results/v0_6_1_correctness"
MODELS = ("M0", "M1", "M2", "M3")
SEEDS = (42, 43, 44)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    entries = attach_derived_paths(json.load(open(f"{SQ}/index.json"))["episodes"], DER)
    sp = classify(entries)
    pops = {k: sp[k] for k in ("A_unseen_anchor_seen_family", "B_seen_anchor_unseen_family",
                               "C_unseen_anchor_unseen_family")}
    rows, shas = [], {}
    for split, ents in pops.items():
        man = WindowManifest(ents, split, windows_per_ep=8)
        n = len(man)
        hp, ha, fa, fr, fric, conds, eps, anch, _, fams = man.batch(np.arange(n), device)
        gt = man.fe                       # 新正确真值
        for name in MODELS:
            for seed in SEEDS:
                path = f"{OLD_CKPT}/{name}_s{seed}/best.pt"
                if not os.path.exists(path):
                    continue
                ck = torch.load(path, weights_only=False, map_location=device)
                base = "direct" if name == "M0" else "context"
                m = MODEL_REGISTRY[base](H, ck["config"]["model"]).to(device)
                m.load_state_dict(ck["model_state"]); m.eval()
                aux = None
                if ck.get("aux_state") is not None:
                    aux = torch.nn.Linear(1, 8).to(device)
                    aux.load_state_dict(ck["aux_state"]); aux.eval()
                shas[f"{name}_s{seed}"] = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
                with torch.no_grad():
                    if name == "M0":
                        r = m(hp, ha, fa)["r_hat"]
                    else:
                        c = m.encode_context(hp, ha)
                        if name == "M3":
                            c = c + aux(fric.reshape(-1, 1))
                        r = m.predict_execution(m.encode_state(hp), c, fa)["r_hat"]
                pred = (fa + r).cpu().numpy()
                row = {"split": split, "model": name, "seed": seed, "n_windows": n,
                       "tag": "OLD_MODEL_NEW_EVALUATION_ONLY",
                       "old_ckpt_sha256_16": shas[f"{name}_s{seed}"]}
                for ax in all_metric_axes():
                    j = lead_index(ax["h_s"]); a = ax["axis"]
                    row[f"lead_MAE_{ax['name']}@{ax['h_s']}s"] = float(
                        np.abs(pred[:, j, a] - gt[:, j, a]).mean())
                row["legacy_MAE_all_mixed_unit"] = float(np.abs(pred - gt).mean())
                rows.append(row)
        print(f"[R1] {split} done")
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/metrics/r1_old_models_new_labels.csv", index=False)
    print(df.groupby(["split", "model"])[["lead_MAE_vx@1.0s", "lead_MAE_wz@1.0s",
                                          "legacy_MAE_all_mixed_unit"]]
          .mean().round(4).to_string())
    json.dump({"tag": "OLD_MODEL_NEW_EVALUATION_ONLY",
               "old_ckpt_root": OLD_CKPT, "old_ckpt_sha256_16": shas,
               "new_labels": DER, "note": "旧模型 + 新标签；只说明评价变化，"
               "不能代替正确标签重训。逐元素预测已写入 R1 流程（本表为汇总）。"},
              open(f"{OUT}/metrics/r1_manifest.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
