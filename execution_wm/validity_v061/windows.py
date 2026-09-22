"""V0.6.1 B7：固定 window manifest —— 训练 seed 不再改变数据样本。

旧实现 `SQWindowData(..., seed=args.seed)` 让 3 个训练 seed 同时换掉了训练窗口，
使"3 seeds 排除优化问题"的说法不成立。本版窗口选择完全确定性（无 RNG），
所有模型/所有 seed 共用同一 manifest。
"""
import numpy as np
import torch

from execution_wm.validity_v061 import timebase as tb
from execution_wm.validity_v061.schema import PROPRIO_SCHEMA, extract_execution_from_proprio
from execution_wm.validity_v061.support_pairs import PROPRIO_KEYS
from execution_wm.validity_v061.schema import EXECUTION_KEYS

L, H = 20, 40


def load_episode(path):
    d = np.load(path)
    proprio = np.concatenate(
        [d[f"in_{k}"].reshape(len(d["timestamp"]), -1) for k in PROPRIO_KEYS],
        axis=-1).astype(np.float32)
    assert proprio.shape[1] == PROPRIO_SCHEMA.dim, proprio.shape
    return {
        "proprio": proprio,
        "cmd": d["cmd_ref"].astype(np.float32),
        "e_label": d["lb_execution"].astype(np.float32),      # 唯一评估真值
        "r_label": d["lb_residual"].astype(np.float32),       # = e_label - cmd_ref
        "e_raw": d["lbra_execution"].astype(np.float32),      # 未平滑真值（对照）
        "tick": d["hold_ctrl_tick"].astype(np.int64),
        "phase": d["phase"],
        "T": len(d["timestamp"]),
    }


def fixed_origins(T, phase, windows_per_ep, require_query_phase=True, stride=None):
    """确定性 origin 选择（无 RNG）。返回升序 grid 索引列表。"""
    cand = [k for k in range(L - 1, T - H) if (not require_query_phase) or phase[k] == 1]
    if not cand:
        return []
    if stride:
        return cand[::stride]
    if len(cand) <= windows_per_ep:
        return cand
    idx = np.linspace(0, len(cand) - 1, windows_per_ep).round().astype(int)
    return [cand[i] for i in sorted(set(idx.tolist()))]


class WindowManifest:
    """固定窗口集合；暴露 window 级 lineage（origin_tick/origin_time/索引）。"""

    def __init__(self, entries, split_name, windows_per_ep=8, require_query_phase=True,
                 path_key="path20"):
        self.split = split_name
        self.windows = []
        for e in entries:
            ep = load_episode(e[path_key])
            origins = fixed_origins(ep["T"], ep["phase"], windows_per_ep, require_query_phase)
            for k0 in origins:
                self.windows.append(self._make(e, ep, k0, split_name))
        self._stack()

    def _make(self, e, ep, k0, split_name):
        return {
            "episode_id": int(e["episode_id"]), "condition": e["condition"],
            "command_family": e["command_family"], "anchor_group": e.get("anchor_group"),
            "rep": e.get("rep"), "friction": float(e["friction"]),
            "split": split_name, "origin_grid": int(k0),
            "origin_tick": int(ep["tick"][k0]),
            "origin_time": float(ep["tick"][k0]) * tb.CTRL_DT_S,
            "history_grid": (int(k0 - L + 1), int(k0)),
            "future_grid": (int(k0 + 1), int(k0 + H)),
            "history_src_tick": (int(ep["tick"][k0 - L + 1]), int(ep["tick"][k0])),
            "future_src_tick": (int(ep["tick"][k0 + 1]), int(ep["tick"][k0 + H])),
            "_hp": ep["proprio"][k0 - L + 1:k0 + 1],
            "_ha": ep["cmd"][k0 - L + 1:k0 + 1],
            "_fa": ep["cmd"][k0 + 1:k0 + H + 1],
            "_fr": ep["r_label"][k0 + 1:k0 + H + 1],
            "_fe": ep["e_label"][k0 + 1:k0 + H + 1],
            "_feraw": ep["e_raw"][k0 + 1:k0 + H + 1],
        }

    def _stack(self):
        n = len(self.windows)
        cat = lambda key, dt: (np.stack([w[key] for w in self.windows]).astype(dt) if n else
                               np.zeros((0,), dtype=dt))
        self.hp = cat("_hp", np.float32)
        self.ha = cat("_ha", np.float32)
        self.fa = cat("_fa", np.float32)
        self.fr = cat("_fr", np.float32)
        self.fe = cat("_fe", np.float32)
        self.fe_raw = cat("_feraw", np.float32)

    def __len__(self):
        return len(self.windows)

    def batch(self, idx, device):
        t = lambda a: torch.from_numpy(np.asarray(a)).to(device)
        return (t(self.hp[idx]), t(self.ha[idx]), t(self.fa[idx]), t(self.fr[idx]),
                torch.tensor([self.windows[i]["friction"] for i in idx],
                             dtype=torch.float32, device=device),
                [self.windows[i]["condition"] for i in idx],
                [self.windows[i]["episode_id"] for i in idx],
                [self.windows[i]["anchor_group"] for i in idx],
                [self.windows[i]["split"] for i in idx],
                [self.windows[i]["command_family"] for i in idx])

    def manifest_hash(self):
        """固定窗口清单的哈希（用于声明"所有模型/seed 同窗口"）。"""
        import hashlib
        h = hashlib.sha256()
        for w in self.windows:
            h.update(f"{w['episode_id']}|{w['origin_grid']}|{w['origin_tick']}|{w['split']}"
                     .encode())
        return h.hexdigest()[:16]

    def lineage_rows(self):
        return [{k: v for k, v in w.items() if not k.startswith("_")} for w in self.windows]

    def assert_causal(self):
        """T08: 每个 origin 的输入源 tick ≤ origin tick；未来标签源 tick > origin tick。"""
        bad = []
        for w in self.windows:
            if w["history_src_tick"][1] > w["origin_tick"]:
                bad.append(("history", w))
            if w["future_src_tick"][0] <= w["origin_tick"]:
                bad.append(("future", w))
        assert not bad, f"因果性违规 {len(bad)} 例，首例: {bad[0] if bad else None}"
        return True

    def assert_label_identity(self, atol=1e-5):
        """T03: e_label == cmd_ref + r_label 在全部窗口成立。"""
        err = np.abs(self.ha * 0 + np.nan)
        # 逐窗口（history/future 拼接后仍成立）
        e = np.concatenate([self.windows[i]["_fe"] for i in range(len(self.windows))])
        r = np.concatenate([self.windows[i]["_fr"] for i in range(len(self.windows))])
        u = np.concatenate([self.windows[i]["_fa"] for i in range(len(self.windows))])
        err = float(np.abs(u + r - e).max())
        assert err < atol, f"标签恒等式误差 {err:.3e} >= {atol}"
        return err
