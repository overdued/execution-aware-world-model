"""V0.8 视觉窗口构造：context/target clip 的 clip 级因果分离。

关键设计约束（B1 审计发现）：V-JEPA2 encoder 在 clip 内部是双向 attention，
**不是时间因果的**（改 clip 后半段会改变前半段 token）。因此因果性必须在
clip 级保证：

- context clip：仅由 capture_time ≤ origin 的帧组成，单独一次 encoder 调用；
- target clip：仅由 (origin, origin+lead] 内的帧组成，另一次 encoder 调用；
- 两个 clip 在任何前向路径中不得混合编码。

帧时间戳取自采集器记录的 rgb_sim_time_first_seen（真实捕获时刻，
不是名义 20Hz 网格——实际帧间隔受渲染网格量化为 0.06s，如实使用）。
"""
from dataclasses import dataclass, field

import numpy as np

CTX_FRAMES = 16            # context clip 帧数（encoder 16 帧 -> 8 时域 token）
LEADS_S = (1.0, 2.0)       # target lead，与任务书 §C 主终点一致
WPE = 6                    # 每 episode 确定性等距 origin 数（同 v07 约定）


@dataclass
class VisualWindow:
    episode_path: str
    group_id: str
    script_id: str
    origin_tick: int                 # 50Hz control tick
    origin_time: float               # sim_time[origin_tick]
    ctx_frame_idx: np.ndarray        # 指向 rgb 帧的下标（全部 ≤ origin_time）
    tgt_frame_idx: dict              # lead_s -> np.ndarray（全部在 (origin, origin+lead]）
    meta: dict = field(default_factory=dict)


def load_episode_visual(path):
    d = np.load(path)
    return {
        "rgb": d["rgb"],
        "frame_time": d["rgb_sim_time_first_seen"].astype(np.float64),
        "frame_counter": d["rgb_frame_counter"].astype(np.int64),
        "sim_time": d["sim_time"].astype(np.float64),
        "phase": d["phase"],
        "cmd_vel": d["cmd_vel"].astype(np.float32),
        "execution": d["execution"].astype(np.float32),
        "base_position": d["base_position"].astype(np.float32),
        "group_id": str(d["group_id"]) if "group_id" in d.files else None,
        "script_id": str(d["script_id"]) if "script_id" in d.files else None,
    }


def fixed_origins(T, phase, wpe=WPE):
    """与 v07 相同的确定性等距 origin 选择（无 RNG）。"""
    cand = [k for k in range(T) if phase[k] == 1]
    if len(cand) <= wpe:
        return cand
    idx = np.linspace(0, len(cand) - 1, wpe).round().astype(int)
    return [cand[i] for i in sorted(set(idx.tolist()))]


def build_windows(ep, leads_s=LEADS_S, ctx_frames=CTX_FRAMES, wpe=WPE):
    """从单条 episode 构建视觉窗口。不满足因果约束的 origin 被跳过。"""
    ft, st = ep["frame_time"], ep["sim_time"]
    wins = []
    for k in fixed_origins(len(st), ep["phase"], wpe):
        t0 = st[k]
        ctx = np.nonzero(ft <= t0)[0]
        if len(ctx) < ctx_frames:
            continue
        ctx = ctx[-ctx_frames:]
        tgt = {}
        for lead in leads_s:
            idx = np.nonzero((ft > t0) & (ft <= t0 + lead + 1e-9))[0]
            if len(idx) >= 2:
                tgt[float(lead)] = idx
        if len(tgt) != len(leads_s):
            continue
        wins.append(VisualWindow(
            episode_path=str(ep.get("path", "")),
            group_id=ep.get("group_id"), script_id=ep.get("script_id"),
            origin_tick=int(k), origin_time=float(t0),
            ctx_frame_idx=ctx, tgt_frame_idx=tgt,
            meta={"n_frames": int(len(ft))},
        ))
    return wins


def check_window_causality(win: VisualWindow, frame_time, leads_s=LEADS_S):
    """逐窗口硬校验：context 全帧 ≤ origin；target 全帧 ∈ (origin, origin+lead]。"""
    t0 = win.origin_time
    assert frame_time[win.ctx_frame_idx].max() <= t0 + 1e-9, \
        f"context leak: {frame_time[win.ctx_frame_idx].max()} > {t0}"
    for lead, idx in win.tgt_frame_idx.items():
        assert frame_time[idx].min() > t0, f"target starts before origin (lead={lead})"
        assert frame_time[idx].max() <= t0 + lead + 1e-9, \
            f"target beyond lead: {frame_time[idx].max()} > {t0 + lead}"
    return True
