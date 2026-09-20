"""Sliding window dataset（first_work.md §7）。

输出每个样本:
    history_proprio   [L, D]   过去 1 秒 proprioception
    history_action    [L, 3]   过去 commanded action
    future_action     [H, 3]   未来 commanded action
    future_execution  [H, 3]   未来实际 body velocity
    future_residual   [H, 3]   未来 residual = execution - command
    current_state     [D]      当前 proprio（context model 的 state latent 输入）
    metadata          dict     （默认不进模型，仅评估/可视化用）
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import episode_proprio, load_episode


class WindowDataset(Dataset):
    def __init__(self, episodes, hz=20, history_s=1.0, horizon_s=2.0, cache=True):
        self.L = int(round(history_s * hz))
        self.H = int(round(horizon_s * hz))
        self.cache = cache
        self.samples = []   # (ep_idx, start_t)
        self._data = []
        for i, e in enumerate(episodes):
            d = load_episode(e["path"])
            T = len(d["timestamp"])
            if T < self.L + self.H:
                continue
            if cache:
                self._data.append({
                    "proprio": episode_proprio(d),
                    "cmd": d["cmd_vel"].astype(np.float32),
                    "execution": d["execution"].astype(np.float32),
                    "residual": d["residual"].astype(np.float32),
                    "meta": e["meta"],
                })
            else:
                self._data.append(e)
            for t0 in range(self.L - 1, T - self.H):
                self.samples.append((i, t0))

    def __len__(self):
        return len(self.samples)

    def _get(self, i):
        if self.cache:
            return self._data[i]
        e = self._data[i]
        d = load_episode(e["path"])
        return {
            "proprio": episode_proprio(d),
            "cmd": d["cmd_vel"].astype(np.float32),
            "execution": d["execution"].astype(np.float32),
            "residual": d["residual"].astype(np.float32),
            "meta": e["meta"],
        }

    def __getitem__(self, idx):
        i, t0 = self.samples[idx]
        d = self._get(i)
        h0 = t0 - self.L + 1
        f0 = t0 + 1
        sample = {
            "history_proprio": torch.from_numpy(d["proprio"][h0:t0 + 1]),
            "history_action": torch.from_numpy(d["cmd"][h0:t0 + 1]),
            "current_state": torch.from_numpy(d["proprio"][t0]),
            "future_action": torch.from_numpy(d["cmd"][f0:f0 + self.H]),
            "future_execution": torch.from_numpy(d["execution"][f0:f0 + self.H]),
            "future_residual": torch.from_numpy(d["residual"][f0:f0 + self.H]),
            "metadata": d["meta"],
        }
        return sample
