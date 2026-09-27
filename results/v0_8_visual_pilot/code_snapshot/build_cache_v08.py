"""V0.8 Stage C：窗口 manifest + 冻结特征缓存构建（采集后、训练前）。

流程（对应预注册 §2）：
1. 读 20Hz 派生（.20hz.npz，derive_v07 同代码同参数产物）+ 原始 npz（RGB 与帧时间戳）；
2. 回归原点：V0.7 同规则（phase==1 内确定性等距 ≤6，无 RNG）；
3. 匹配原点：每 episode 钉定 50Hz tick 75（1.5s；断言 phase==1，否则取 ≤110 的
   最大 phase==1 tick 并记录）——同 group 跨分支 context 逐位相同的唯一点；
4. clip 选取（tick 级边界）：
   context = first_seen_tick ≤ origin_tick50 的最后 16 帧；
   target  = origin_tick50 < first_seen_tick ≤ origin_tick50 + h·50 的前 16/32 帧；
5. 冻结 V-JEPA2 编码（clip 级分离，context/target 独立调用），4×4 空间 mean-pool；
6. 目标归一化统计仅由 train split 窗口拟合，写入 manifest；
7. 输出：manifests/windows.csv + splits.json + hashes.json，
   predictions/feature_cache.npz（含逐窗血缘）+ feature_cache_manifest.json（sha256）。

运行：python -m execution_wm.v08_visual.build_cache_v08 \
  --raw-root /media/hdd1/yuhang/datasets/execution_wm/v0_8_pilot \
  --plan results/v0_8_visual_pilot/prereg/split_plan_v08.json \
  --out-results results/v0_8_visual_pilot
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from execution_wm.v08_visual.encoder_v08 import encode_frames

L_HIST, H_FUT = 20, 40                 # 20Hz：过去 1s / 未来 2s
WPE = 6                                # 每 episode 回归原点上限（V0.7 同规则）
CTX_FRAMES = 16
TGT_FRAMES = {1.0: 16, 2.0: 32}
MATCH_TICK50 = 75
MATCH_TICK50_FALLBACK_MAX = 110


def fixed_origins(T, phase, wpe=WPE, lo=L_HIST - 1, hi_margin=H_FUT):
    """V0.7 同规则：phase==1 候选内确定性等距（无 RNG），需满足历史/未来长度。"""
    cand = [k for k in range(lo, T - hi_margin) if phase[k] == 1]
    if not cand:
        return []
    if len(cand) <= wpe:
        return cand
    idx = np.linspace(0, len(cand) - 1, wpe).round().astype(int)
    return [cand[i] for i in sorted(set(idx.tolist()))]


def matching_origin(phase50):
    """50Hz phase 上的钉定匹配原点。"""
    if phase50[MATCH_TICK50] == 1:
        return MATCH_TICK50, False
    cand = [t for t in range(MATCH_TICK50_FALLBACK_MAX + 1) if phase50[t] == 1]
    assert cand, "no phase==1 tick for matching origin"
    return cand[-1], True


def select_clip(frame_tick, origin_tick50, lead_s, n):
    if lead_s is None:  # context
        idx = np.nonzero(frame_tick <= origin_tick50)[0]
        return idx[-n:] if len(idx) >= n else None
    idx = np.nonzero((frame_tick > origin_tick50) &
                     (frame_tick <= origin_tick50 + int(lead_s * 50)))[0]
    return idx[:n] if len(idx) >= n else None


def encode_clip_tokens(rgb, idx, s_pool=4):
    """[T,256,256,3] -> [T//2, (256/s_pool)^2, 1024] float32 numpy。"""
    frames = rgb[idx]
    if len(frames) % 2:
        frames = frames[:-1]
    import torch as _t
    model_input = frames
    # 复用 encoder_v08 的 processor/模型，但保留空间 token 再 4×4 pool
    from execution_wm.v08_visual.encoder_v08 import load_encoder
    model, proc = load_encoder("cuda")
    with _t.no_grad():
        inputs = proc(list(model_input), return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        out = model(**inputs).last_hidden_state        # [1,(T/2)*256,1024]
        n_tok = len(frames) // 2
        tok = out.reshape(1, n_tok, 16, 16, 1024)      # 16×16 空间网格
        tok = tok.reshape(1, n_tok, s_pool, 16 // s_pool, s_pool, 16 // s_pool, 1024)
        tok = tok.mean(dim=(3, 5))                     # [1, n_tok, s_pool, s_pool, 1024]
        tok = tok.reshape(1, n_tok, s_pool * s_pool, 1024)
    return tok[0].float().cpu().numpy()


def build_windows_for_episode(npz_path, plan_ep):
    raw = np.load(npz_path)
    der = np.load(str(npz_path).replace(".npz", ".20hz.npz"))
    T20 = len(der["timestamp"])
    phase20 = der["phase"] if "phase" in der.files else raw["phase"][::2][:T20]
    frame_tick = raw["rgb_ctrl_tick_first_seen"].astype(np.int64)
    frame_time = raw["rgb_sim_time_first_seen"].astype(np.float64)
    mo_tick50, mo_fallback = matching_origin(raw["phase"])
    # 20Hz 索引（origin_time = k*0.05 -> origin_tick50 = floor(t/0.02)）
    wins = []
    for k in fixed_origins(T20, phase20):
        t = float(der["timestamp"][k])
        wins.append((k, t, int(t / 0.02 + 1e-9), False))
    t_m = mo_tick50 * 0.02
    k_m = int(round(t_m / 0.05))
    wins.append((k_m, t_m, mo_tick50, True))
    out = []
    for k20, t0, tick50, is_match in wins:
        ci = select_clip(frame_tick, tick50, None, CTX_FRAMES)
        t1 = select_clip(frame_tick, tick50, 1.0, TGT_FRAMES[1.0])
        t2 = select_clip(frame_tick, tick50, 2.0, TGT_FRAMES[2.0])
        if ci is None or t1 is None or t2 is None:
            continue
        # 因果硬校验（保守方向：捕获 ≤ first_seen ≤ origin）
        assert frame_time[ci].max() <= t0 + 0.02 + 1e-9
        assert frame_time[t2].max() <= t0 + 2.0 + 0.02 + 1e-9
        out.append({
            "episode_id": plan_ep["episode_id"], "group_id": plan_ep["group_id"],
            "branch": plan_ep["branch"], "role": plan_ep["role"],
            "level": plan_ep["level"], "layout": plan_ep["layout"],
            "split": plan_ep["split"], "friction": plan_ep["friction"],
            "appearance_id": plan_ep["appearance_id"],
            "origin_k20": int(k20), "origin_time": t0, "origin_tick50": int(tick50),
            "eligible_for_matching": bool(is_match),
            "matching_origin_fallback": bool(is_match and mo_fallback),
            "ctx_frames": ci, "tgt1s_frames": t1, "tgt2s_frames": t2,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out-results", required=True)
    ap.add_argument("--limit-eps", type=int, default=0)
    args = ap.parse_args()
    plan = json.load(open(args.plan))
    raw_root = Path(args.raw_root)
    out_root = Path(args.out_results)

    rows, cache = [], {}
    eps = plan["episodes"][:args.limit_eps or None]
    for i, ep in enumerate(eps):
        npz = raw_root / ep["split"] / ep["group_id"] / f"b{ep['branch']}" / \
            f"{ep['episode_id']}.npz"
        assert npz.exists(), npz
        for w in build_windows_for_episode(npz, ep):
            wid = f"{ep['episode_id']}_k{w['origin_k20']:03d}"
            key = ep["episode_id"], w["origin_k20"]
            raw = np.load(npz)
            cache[f"{wid}/ctx"] = encode_clip_tokens(raw["rgb"], w["ctx_frames"])
            cache[f"{wid}/tgt1s"] = encode_clip_tokens(raw["rgb"], w["tgt1s_frames"])
            cache[f"{wid}/tgt2s"] = encode_clip_tokens(raw["rgb"], w["tgt2s_frames"])
            rows.append({k: v for k, v in w.items()
                         if not k.endswith("_frames")} | {"window_id": wid})
        if (i + 1) % 16 == 0:
            print(f"  [{i+1}/{len(eps)}] windows={len(rows)}", flush=True)

    # train-only 归一化统计（逐通道，跨时域/空间共享）
    tr = [r["window_id"] for r in rows if r["split"] == "train"]
    def _stats(suffix):
        arrs = [cache[f"{w}/{suffix}"] for w in tr]
        a = np.concatenate([x.reshape(-1, 1024) for x in arrs])
        return a.mean(axis=0), a.std(axis=0) + 1e-6
    stats = {s: _stats(s) for s in ("ctx", "tgt1s", "tgt2s")}
    for s, (m, sd) in stats.items():
        cache[f"_norm/{s}_mean"] = m.astype(np.float32)
        cache[f"_norm/{s}_std"] = sd.astype(np.float32)

    (out_root / "manifests").mkdir(parents=True, exist_ok=True)
    (out_root / "predictions").mkdir(parents=True, exist_ok=True)
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out_root / "manifests/windows.csv", index=False)
    np.savez_compressed(out_root / "predictions/feature_cache.npz", **cache)
    h = hashlib.sha256((out_root / "predictions/feature_cache.npz").read_bytes())
    mani = {"n_windows": len(rows), "n_episodes": len(eps),
            "feature_cache_sha256": h.hexdigest(),
            "norm_fit_split": "train",
            "clip_rules": {"ctx": "first_seen<=origin last16",
                           "tgt1s": "(origin,origin+50] first16",
                           "tgt2s": "(origin,origin+100] first32"},
            "spatial_pool": "4x4 mean", "encoder": "vjepa2-vitl-fpc64-256@b3c1679"}
    (out_root / "predictions/feature_cache_manifest.json").write_text(
        json.dumps(mani, indent=2))
    print(json.dumps({k: v for k, v in mani.items() if k != "clip_rules"}, indent=2))


if __name__ == "__main__":
    main()
