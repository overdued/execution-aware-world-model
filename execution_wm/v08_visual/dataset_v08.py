"""V0.8 Stage C：训练数据集（窗口张量装配，全部确定性、无 RNG）。

输入/目标口径见 V0_8_PRE_REGISTRATION.md §2：
- hist_phys [L=20, 43]：in_* 本体（privileged 屏蔽，同 V0.7 PRIV_FIELDS）+ 过去 cmd_ref；
- fut_cmd  [H=40, 3]：未来公开候选命令 cmd_ref(k, k+40]；
- ctx/tgt 特征：feature_cache.npz（冻结 encoder，4×4 空间 pool），train-fit z 归一化；
- e_star   [120]：lb_execution(k, k+40] 展平，train-fit 归一化（统计落盘 manifests/）。
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from execution_wm.composition_v07.data_v07 import mask_privileged
from execution_wm.validity_v061.support_pairs import PROPRIO_KEYS

L_HIST, H_FUT = 20, 40


def _load_derived_map(raw_root, df):
    """急切加载本 split 需要的全部 20hz 文件到内存（主进程内完成，
    fork 后只读 COW 共享，杜绝 worker 间共享 fd 的 seek 竞争）。"""
    out = {}
    for _, r in df.iterrows():
        p = str(Path(raw_root) / r["split"] / r["group_id"] /
                f"b{r['branch']}" / f"{r['episode_id']}.20hz.npz")
        if p not in out:
            npz = np.load(p)
            out[p] = {k: npz[k] for k in npz.files}
    return lambda split, gid, branch, eid: out[
        str(Path(raw_root) / split / gid / f"b{branch}" / f"{eid}.20hz.npz")]


class V08Windows(Dataset):
    def __init__(self, results_root, raw_root, split, norm_stats=None,
                 fit_norm=False):
        rr = Path(results_root)
        import pandas as pd
        df = pd.read_csv(rr / "manifests/windows.csv")
        self.df = df[df.split == split].reset_index(drop=True)
        # 急切加载到内存：NpzFile 惰性 zip 读取在 DataLoader 多 worker fork 下共享
        # fd，并发 seek/decompress 会数据竞争（曾导致 zlib 解压错误）。只读数组
        # fork 后是 COW 共享页，不额外占内存。
        npz = np.load(rr / "predictions/feature_cache.npz")
        self.cache = {k: npz[k] for k in npz.files}
        self.get20 = _load_derived_map(raw_root, self.df)
        self.norm = {s: (self.cache[f"_norm/{s}_mean"], self.cache[f"_norm/{s}_std"])
                     for s in ("ctx", "tgt1s", "tgt2s")}
        if norm_stats is not None:
            self.e_mean, self.e_std = norm_stats
        elif fit_norm:
            self.e_mean, self.e_std = self._fit_e_norm()
            (rr / "manifests").mkdir(exist_ok=True)
            (rr / "manifests/e_norm.json").write_text(json.dumps(
                {"mean": self.e_mean.tolist(), "std": self.e_std.tolist(),
                 "fit_split": split}, indent=2))
        else:
            nj = rr / "manifests/e_norm.json"
            d = json.loads(nj.read_text())
            self.e_mean, self.e_std = np.array(d["mean"]), np.array(d["std"])

    def _fit_e_norm(self):
        es = []
        for _, r in self.df.iterrows():
            d = self.get20(r["split"], r["group_id"], r["branch"], r["episode_id"])
            k = int(r["origin_k20"])
            es.append(d["lb_execution"][k + 1:k + 1 + H_FUT].reshape(-1))
        a = np.stack(es)
        return a.mean(axis=0).astype(np.float32), (a.std(axis=0) + 1e-6).astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        d = self.get20(r["split"], r["group_id"], r["branch"], r["episode_id"])
        k = int(r["origin_k20"])
        proprio = np.concatenate(
            [d[f"in_{kk}"].reshape(len(d["timestamp"]), -1) for kk in PROPRIO_KEYS],
            axis=-1).astype(np.float32)
        proprio = mask_privileged(proprio)
        cmd = d["cmd_ref"].astype(np.float32)
        hist = np.concatenate([proprio[k - L_HIST + 1:k + 1],
                               cmd[k - L_HIST + 1:k + 1]], axis=-1)      # [20,43]
        fut = cmd[k + 1:k + 1 + H_FUT]                                    # [40,3]
        e = d["lb_execution"][k + 1:k + 1 + H_FUT].reshape(-1).astype(np.float32)
        e = (e - self.e_mean) / self.e_std
        wid = r["window_id"]

        def _feat(s):
            m, sd = self.norm[s]
            return ((self.cache[f"{wid}/{s}"] - m) / sd).astype(np.float32)
        return {
            "ctx": torch.from_numpy(_feat("ctx")),                        # [8,16,1024]
            "hist": torch.from_numpy(hist), "fut": torch.from_numpy(fut),
            "tgt1s": torch.from_numpy(_feat("tgt1s")),
            "tgt2s": torch.from_numpy(_feat("tgt2s")),
            "e": torch.from_numpy(e),
            "window_id": wid, "group_id": r["group_id"], "level": r["level"],
            "layout": r["layout"], "branch": int(r["branch"]),
            "role": r["role"], "eligible_for_matching":
                bool(r["eligible_for_matching"]),
        }
