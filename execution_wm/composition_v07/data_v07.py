"""V0.7 数据管线：固定窗口 manifest + R0/R1 数据集构造。

- 窗口选择**无 RNG**（确定性等距），所有模型/seed 共用同一 manifest。
- R0 数据集 = 来自 R0 regime 脚本的窗口；R1 = R0 脚本 + R1 脚本（R1 ⊇ R0）。
- 输入为 simulated deployable-candidate：屏蔽 schema 中的 privileged GT 通道。
- 评价层级 P0/P1/P2 取自脚本层级；同时记录未来 horizon 内的 cell 类型以报告纯度。
"""
import json
import os

import numpy as np
import torch

from execution_wm.validity_v061.schema import PROPRIO_SCHEMA
from execution_wm.validity_v061.support_pairs import PROPRIO_KEYS

L, H = 20, 40
WPE = 6
PRIV_FIELDS = ("base_linear_velocity_body", "base_angular_velocity")


def load_episode(path20):
    d = np.load(path20)
    proprio = np.concatenate(
        [d[f"in_{k}"].reshape(len(d["timestamp"]), -1) for k in PROPRIO_KEYS],
        axis=-1).astype(np.float32)
    assert proprio.shape[1] == PROPRIO_SCHEMA.dim
    return {"proprio": proprio, "cmd": d["cmd_ref"].astype(np.float32),
            "e": d["lb_execution"].astype(np.float32),
            "r": d["lb_residual"].astype(np.float32),
            "yaw": d["lb_yaw"].astype(np.float32),
            "pos": d["lb_base_position"].astype(np.float32),
            "tick": d["hold_ctrl_tick"].astype(np.int64),
            "phase": d["phase"], "T": len(d["timestamp"])}


def mask_privileged(proprio):
    out = np.array(proprio, copy=True)
    for name in PRIV_FIELDS:
        f = PROPRIO_SCHEMA.field(name)
        out[..., f.offset:f.offset + f.size] = 0.0
    return out


def fixed_origins(T, phase, wpe=WPE):
    cand = [k for k in range(L - 1, T - H) if phase[k] == 1]
    if not cand:
        return []
    if len(cand) <= wpe:
        return cand
    idx = np.linspace(0, len(cand) - 1, wpe).round().astype(int)
    return [cand[i] for i in sorted(set(idx.tolist()))]


class V07Windows:
    """从 20Hz 派生文件构建窗口；lineage 逐窗保存。"""

    def __init__(self, entries, mask_priv=True):
        self.windows = []
        self.mask_priv = mask_priv
        for e in entries:
            ep = load_episode(e["path20"])
            for k0 in fixed_origins(ep["T"], ep["phase"]):
                cells = e["command_cell_ids"]
                seg_durs = np.array([s["duration_s"] for s in e["segments"]])
                tq = k0 * 0.05 - e["settle_s"]
                acc, dom = 0.0, cells[0]
                for c, dur in zip(cells, seg_durs):
                    if tq < acc + dur:
                        dom = c; break
                    acc += dur
                # future horizon 内的 cell 集合（评估纯度）
                fut_cells = []
                for j in range(k0 + 1, k0 + H + 1):
                    tqj = j * 0.05 - e["settle_s"]
                    acc2 = 0.0
                    for c, dur in zip(cells, seg_durs):
                        if tqj < acc2 + dur:
                            fut_cells.append(c); break
                        acc2 += dur
                self.windows.append({
                    "episode_id": e["episode_id"], "group_id": e["anchor_group_id"],
                    "split": e["split"], "condition": e["condition"],
                    # 采集器对 train 写 regime(R0/R1)，对 val/test 写 level(P0/P1/P2)，
                    # 二者都存在 coverage_regime 字段里；此处解析出语义化名称。
                    "regime": e["coverage_regime"],
                    "level": e.get("level") or e.get("regime") or e["coverage_regime"],
                    "script_id": e["script_id"], "timing_template": e["timing_template"],
                    "origin_grid": int(k0), "origin_tick": int(ep["tick"][k0]),
                    "dominant_cell": dom,
                    "future_cells_distinct": sorted(set(fut_cells)),
                    "future_cell_purity": float(
                        max(np.bincount([cells.index(c) for c in fut_cells]).max()
                            / len(fut_cells), 0.0)) if fut_cells else 0.0,
                    "_hp": ep["proprio"][k0 - L + 1:k0 + 1],
                    "_ha": ep["cmd"][k0 - L + 1:k0 + 1],
                    "_fa": ep["cmd"][k0 + 1:k0 + H + 1],
                    "_fr": ep["r"][k0 + 1:k0 + H + 1],
                    "_fe": ep["e"][k0 + 1:k0 + H + 1],
                    "_yaw0": float(ep["yaw"][k0]),
                    "_yawf": ep["yaw"][k0 + 1:k0 + H + 1],
                })
        self._stack()

    def _stack(self):
        n = len(self.windows)
        hp = (np.stack([w["_hp"] for w in self.windows]).astype(np.float32) if n
              else np.zeros((0, L, PROPRIO_SCHEMA.dim), np.float32))
        self.hp = mask_privileged(hp) if self.mask_priv else hp
        cat = lambda k: np.stack([w[k] for w in self.windows]).astype(np.float32) if n \
            else np.zeros((0, H, 3), np.float32)
        self.ha = cat("_ha"); self.fa = cat("_fa"); self.fr = cat("_fr"); self.fe = cat("_fe")
        self.yaw0 = np.array([w["_yaw0"] for w in self.windows]) if n else np.zeros(0)
        self.yawf = cat("_yawf") if n else np.zeros((0, H, 1), np.float32)

    def __len__(self):
        return len(self.windows)

    def batch(self, idx, device):
        t = lambda a: torch.from_numpy(np.asarray(a)).to(device)
        i = np.asarray(idx)
        return (t(self.hp[i]), t(self.ha[i]), t(self.fa[i]), t(self.fr[i]),
                [self.windows[k]["condition"] for k in i],
                [self.windows[k]["episode_id"] for k in i],
                [self.windows[k]["group_id"] for k in i],
                [self.windows[k]["level"] for k in i])

    def subset(self, pred):
        w = [x for x in self.windows if pred(x)]
        new = V07Windows.__new__(V07Windows)
        new.windows = w; new.mask_priv = self.mask_priv; new._stack()
        return new

    def manifest_hash(self):
        import hashlib
        h = hashlib.sha256()
        for w in self.windows:
            h.update(f"{w['episode_id']}|{w['origin_grid']}".encode())
        return h.hexdigest()[:16]

    def lineage_rows(self):
        return [{k: v for k, v in w.items() if not k.startswith("_")} for w in self.windows]


def load_index(root):
    idx = json.load(open(os.path.join(root, "index.json")))["episodes"]
    for e in idx:
        e["path20"] = e["file"].replace(".npz", ".20hz.npz")
    return idx


def build_datasets(root):
    """-> dict: train_R0, train_R1, val, test_P0, test_P1, test_P2（+ test_all）"""
    idx = [e for e in load_index(root) if e["split"] in ("train", "val", "test")]
    tr = [e for e in idx if e["split"] == "train"]
    va = [e for e in idx if e["split"] == "val"]
    te = [e for e in idx if e["split"] == "test"]
    r0_eps = [e for e in tr if e["coverage_regime"] == "R0"]
    r1_eps = [e for e in tr if e["coverage_regime"] == "R1"]
    ds = {
        "train_R0": V07Windows(r0_eps),
        "train_R1": V07Windows(r0_eps + r1_eps),
        "val": V07Windows(va),
        "test_all": V07Windows(te),
    }
    ds["test_P0"] = ds["test_all"].subset(lambda w: w["level"] == "P0_seen_cell")
    ds["test_P1"] = ds["test_all"].subset(lambda w: w["level"] == "P1_heldout_pair")
    ds["test_P2"] = ds["test_all"].subset(lambda w: w["level"] == "P2_triple")
    return ds
