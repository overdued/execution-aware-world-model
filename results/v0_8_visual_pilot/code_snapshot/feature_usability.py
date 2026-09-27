"""Stage B5：冻结 V-JEPA2 特征可用性检查（smoke 数据）。

设计（利用 smoke 结构：同 group 4 条 episode 初态逐位相同、命令前 50 tick=1.0s
逐位相同、之后分叉）：
- 同命令重复基线：origin=tick50，context clip 跨 4 条 episode 应逐位一致（确定性
  仿真 + 确定性 encoder），给出"同命令"差异基线（期望 0）；
- 动作依赖信号：同 origin 的 target clip（lead 1s，全部落在命令分叉之后）跨
  episode 两两差异 —— 唯一变化来源是未来命令；
- 语义扰动参照：同一 context clip 平移 1 帧重编码的差异，作为"最小语义变化"参照；
- 通道方差：全部 clip token 的逐通道方差，统计死通道；
- 特征缓存：所有编码结果写 npz 供 Stage C 复用。

运行：python -m execution_wm.v08_visual.feature_usability --out <json> --cache <npz>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from execution_wm.v08_visual.encoder_v08 import encode_frames

SMOKE_ROOT = Path("/media/hdd1/yuhang/datasets/execution_wm/v0_8_smoke/train")
ORIGIN_TICK = 50          # 1.0s：settle 结束、命令分叉点
LEAD_S = 1.0
N_CTX, N_TGT = 16, 16


def _select_frames(frame_time, lo, hi, n):
    idx = np.nonzero((frame_time > lo) & (frame_time <= hi + 1e-9))[0]
    return idx[:n] if len(idx) >= n else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    groups = {}
    for p in sorted(SMOKE_ROOT.rglob("ep_*.npz")):
        groups.setdefault(p.parents[1].name, []).append(p)

    cache, report = {}, {"origin_tick": ORIGIN_TICK, "lead_s": LEAD_S,
                         "n_ctx": N_CTX, "n_tgt": N_TGT, "groups": {}}
    all_tokens = []
    for g, paths in sorted(groups.items()):
        eps = [np.load(p) for p in paths]
        # 前置断言：初态与命令前缀逐位一致（smoke 结构假设）
        for d in eps[1:]:
            assert np.array_equal(eps[0]["base_position"][:ORIGIN_TICK + 1],
                                  d["base_position"][:ORIGIN_TICK + 1])
            assert np.array_equal(eps[0]["cmd_vel"][:ORIGIN_TICK],
                                  d["cmd_vel"][:ORIGIN_TICK])
        ctx_feats, tgt_feats = [], []
        for i, d in enumerate(eps):
            ft = d["rgb_sim_time_first_seen"].astype(np.float64)
            t0 = float(d["sim_time"][ORIGIN_TICK])
            ci = _select_frames(ft, -1, t0, N_CTX + 1)      # 含余量再取末 16
            ci = ci[-N_CTX:]
            ti = _select_frames(ft, t0, t0 + LEAD_S, N_TGT)
            fc = encode_frames(d["rgb"][ci], args.device)   # [8,1024]
            ftgt = encode_frames(d["rgb"][ti], args.device) # [8,1024]
            ctx_feats.append(fc)
            tgt_feats.append(ftgt)
            cache[f"{g}/ep{i:02d}_ctx"] = fc.numpy()
            cache[f"{g}/ep{i:02d}_tgt"] = ftgt.numpy()
            cache[f"{g}/ep{i:02d}_ctx_frames"] = ci
            cache[f"{g}/ep{i:02d}_tgt_frames"] = ti
            all_tokens += [fc, ftgt]
            if i == 0:  # 1 帧平移语义参照
                fs = encode_frames(d["rgb"][ci + 1], args.device)
                report["groups"].setdefault(g, {})["shift1_maxdiff"] = \
                    float((fc - fs).abs().max())
        ctx_stack = torch.stack(ctx_feats)   # [4,8,1024]
        tgt_stack = torch.stack(tgt_feats)
        rep = report["groups"][g]
        rep["same_cmd_ctx_maxdiff"] = float((ctx_stack - ctx_stack[0:1]).abs().max())
        pair = (tgt_stack.unsqueeze(0) - tgt_stack.unsqueeze(1)).abs().amax(dim=(2, 3))
        rep["diff_cmd_tgt_pairwise_max"] = float(pair.max())
        rep["diff_cmd_tgt_pairwise_mean"] = float(pair[np.triu_indices(4, 1)].mean())

    tok = torch.cat([t.reshape(-1, 1024) for t in all_tokens])  # [N,1024]
    var = tok.var(dim=0, unbiased=False)
    report["channel_var"] = {
        "n_tokens": int(tok.shape[0]),
        "min": float(var.min()), "median": float(var.median()),
        "max": float(var.max()),
        "dead_channels_var_lt_1e-8": int((var < 1e-8).sum()),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    np.savez_compressed(args.cache, **cache)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
