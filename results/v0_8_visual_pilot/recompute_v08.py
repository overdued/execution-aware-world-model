#!/usr/bin/env python
"""V0.8 独立复算脚本：只读 predictions/pred_cache.npz + predictions/feature_cache.npz
重算主终点（P1 err_2s，test 16 group）的组级表与 paired group bootstrap，
并与 metrics/stats_primary.json 对拍。不加载模型、不依赖训练/采集代码。

口径（与 execution_wm/v08_visual/eval_v08.py 一致）：
- 目标 = feature_cache.npz 中 <window_id>/tgt2s，经 train-fit z 归一化
  （_norm/tgt2s_mean, _norm/tgt2s_std，float32）；
- err_2s = |pred_2s - tgt2s| 全元素均值（pred_2s 以 fp16 存储，量化误差 << 容差）；
- 组级 = 窗均值 → 按 group 聚合；bootstrap = paired group 重抽 10,000 次，seed 0。

用法（worktree 根 ~/cvpr_embed-v08 下）：
    python results/v0_8_visual_pilot/recompute_v08.py
退出码 0 = 全部 MATCH。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RR = Path(__file__).resolve().parent


def window_errors(run, split):
    cache = np.load(RR / 'predictions' / 'pred_cache.npz')
    feats = np.load(RR / 'predictions' / 'feature_cache.npz')
    nm, ns = feats['_norm/tgt2s_mean'].astype(np.float32), feats['_norm/tgt2s_std'].astype(np.float32)
    w = pd.read_csv(RR / 'manifests' / 'windows.csv')
    w = w[w.split == split]
    rows = []
    for _, r in w.iterrows():
        key = f"{run}/{split}/{r['window_id']}/pred_2s"
        if key not in cache:
            continue
        pred = cache[key].astype(np.float32).reshape(-1)
        tgt = (feats[f"{r['window_id']}/tgt2s"].astype(np.float32) - nm) / ns
        n = min(pred.size, tgt.size)
        rows.append(dict(run_id=run, level=r['level'], layout=r['layout'],
                         group_id=r['group_id'], window_id=r['window_id'],
                         err_2s=float(np.abs(pred[:n] - tgt.reshape(-1)[:n]).mean())))
    return pd.DataFrame(rows)


def bootstrap(a, b, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(a) - np.asarray(b)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    m = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    frames = []
    for s in (42, 43, 44):
        for v in ('VDIRECT', 'VAUX', 'VEXEC'):
            frames.append(window_errors(f'{v}_s{s}', 'test'))
    df = pd.concat(frames, ignore_index=True)
    g = (df.groupby(['run_id', 'level', 'layout', 'group_id'], as_index=False)
           ['err_2s'].mean())
    stats = {}
    for s in (42, 43, 44):
        ve = g[(g.run_id == f'VEXEC_s{s}') & (g.level == 'P1')].set_index('group_id')['err_2s']
        va = g[(g.run_id == f'VAUX_s{s}') & (g.level == 'P1')].set_index('group_id')['err_2s']
        vd = g[(g.run_id == f'VDIRECT_s{s}') & (g.level == 'P1')].set_index('group_id')['err_2s']
        c = ve.index.intersection(va.index)
        m, lo, hi = bootstrap(va[c].values, ve[c].values)
        stats[f'VEXEC_vs_VAUX_s{s}'] = dict(abs_diff_mean=m, ci95=[lo, hi],
            rel_improvement=float((va[c].mean() - ve[c].mean()) / va[c].mean()))
        c2 = va.index.intersection(vd.index)
        m2, lo2, hi2 = bootstrap(vd[c2].values, va[c2].values)
        stats[f'VAUX_vs_VDIRECT_s{s}'] = dict(abs_diff_mean=m2, ci95=[lo2, hi2],
            rel_improvement=float((vd[c2].mean() - va[c2].mean()) / vd[c2].mean()))
    ref = json.loads((RR / 'metrics' / 'stats_primary.json').read_text())
    ok = True
    for k, v in stats.items():
        rel_r = ref[k]['rel_improvement']
        match = abs(v['rel_improvement'] - rel_r) < 5e-3
        ok &= match
        print(f"{k}: rel={v['rel_improvement']:+.4f} (ref {rel_r:+.4f}) "
              f"CI=[{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}] {'MATCH' if match else 'DIFF'}")
    print('RECOMPUTE', 'PASS' if ok else 'MISMATCH(>5e-3)')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
