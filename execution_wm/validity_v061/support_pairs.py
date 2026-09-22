"""V0.6.1 B1：support 成对抽样 —— 旧实现分别抽 hp/ha，可跨 episode/origin。

SupportRecord 是唯一抽样单位：抽一次 window_id，读取全部配套字段。
三种可辨别模式（禁止混用）:
    A native      : c 来自 query 自身 history（因果）
    B paired      : c 来自同 condition 的另一条 support episode（成对动作-响应）
    C mismatched  : 故意错配/打乱（仅负对照）
"""
import hashlib
from dataclasses import dataclass

import numpy as np
import torch

from execution_wm.validity_v061 import timebase as tb

L_DEFAULT = 20


@dataclass
class SupportRecord:
    support_episode_id: int
    support_window_id: int
    support_origin_tick: int          # 整数控制 tick 身份
    support_session_id: str
    split: str
    condition: str
    history_proprio: np.ndarray       # [L, 40]
    history_command: np.ndarray       # [L, 3]
    timestamps: np.ndarray            # [L] = origin_tick 对应的真实采样时刻（秒）

    def key(self):
        return (self.support_session_id, self.support_window_id)


class SupportBank:
    """按 condition 分组的成对窗口池；索引即 SupportRecord，字段永不错配。"""

    def __init__(self, support_entries, path_key="path20", L=L_DEFAULT, stride=10,
                 split="reference_bank_train"):
        self.L = L
        self.records = []
        for e in support_entries:
            d = np.load(e[path_key])
            proprio = np.concatenate(
                [d[f"in_{k}"].reshape(len(d["timestamp"]), -1) for k in PROPRIO_KEYS],
                axis=-1).astype(np.float32)
            cmd = d["cmd_ref"].astype(np.float32)
            ticks = d["hold_ctrl_tick"] if "hold_ctrl_tick" in d.files else tb.grid_hold_ticks(len(cmd))
            T = len(cmd)
            for wid, t0 in enumerate(range(L - 1, T, stride)):
                self.records.append(SupportRecord(
                    support_episode_id=int(e["episode_id"]),
                    support_window_id=len(self.records),
                    support_origin_tick=int(ticks[t0]),
                    support_session_id=e.get("support_session_id",
                                              f"SP{e.get('support_seed', e['episode_id'])}"),
                    split=split, condition=e["condition"],
                    history_proprio=proprio[t0 - L + 1:t0 + 1].copy(),
                    history_command=cmd[t0 - L + 1:t0 + 1].copy(),
                    timestamps=(ticks[t0 - L + 1:t0 + 1] * tb.CTRL_DT_S).copy()))
        self.by_cond = {}
        for i, r in enumerate(self.records):
            self.by_cond.setdefault(r.condition, []).append(i)
        # 自洽检查：每条 record 内部三模态必须同源同 tick
        for r in self.records:
            assert len(r.history_proprio) == L and len(r.history_command) == L
            assert len(r.timestamps) == L

    def __len__(self):
        return len(self.records)

    def sample_indices(self, conds, rng, exclude_episodes=None):
        """每个 query 抽一个 window_id（一次抽取决定全部模态）。返回 record 索引。"""
        out = []
        for c, ex in zip(conds, exclude_episodes or [None] * len(conds)):
            pool = self.by_cond[c]
            if ex is not None:
                pool = [i for i in pool if self.records[i].support_episode_id != ex]
                if not pool:
                    raise ValueError(f"condition {c} 排除 episode {ex} 后无可用 donor")
            out.append(pool[int(rng.integers(0, len(pool)))])
        return np.array(out, dtype=np.int64)

    def fixed_indices(self, conds, seed=20260922, exclude_episodes=None):
        """固定 donor 表：对每个 (condition, 排除集) 用固定 rng 抽一次并缓存。"""
        key = (tuple(conds), tuple(exclude_episodes or [None] * len(conds)))
        if not hasattr(self, "_fixed_cache"):
            self._fixed_cache = {}
        if key not in self._fixed_cache:
            rng = np.random.default_rng(seed)
            self._fixed_cache[key] = self.sample_indices(conds, rng, exclude_episodes)
        return self._fixed_cache[key]

    def batch(self, idx, device):
        hp = np.stack([self.records[i].history_proprio for i in idx])
        ha = np.stack([self.records[i].history_command for i in idx])
        return torch.from_numpy(hp).to(device), torch.from_numpy(ha).to(device)

    def donor_ids(self, idx):
        """donor lineage（可独立复现每个 query 的 donor）。"""
        return [{"query_donor_window_id": int(i),
                 "donor_episode_id": self.records[i].support_episode_id,
                 "donor_session_id": self.records[i].support_session_id,
                 "donor_origin_tick": self.records[i].support_origin_tick,
                 "donor_condition": self.records[i].condition,
                 "donor_split": self.records[i].split} for i in idx]

    def table_hash(self, idx):
        h = hashlib.sha256()
        for i in idx:
            r = self.records[i]
            h.update(f"{r.support_episode_id}|{r.support_window_id}|{r.support_origin_tick}|"
                     f"{r.condition}|{r.split}".encode())
        return h.hexdigest()[:16]


PROPRIO_KEYS = ["base_linear_velocity_body", "base_angular_velocity", "projected_gravity",
                "imu_linear_acceleration", "joint_position", "joint_velocity", "feet_contact"]
