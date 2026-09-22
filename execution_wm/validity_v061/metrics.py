"""V0.6.1 B6：fixed-lead 与 prefix-average 分离（旧 mae_rows 实为 prefix）。

    MAE_lead(h)   = E | e_hat[:, j(h)] - e[:, j(h)] |
    MAE_prefix(h) = (1/j(h)) Σ_{i<=j(h)} E | e_hat[:, i] - e[:, i] |

lead 索引由 timestamp 推导（h/0.05 - 1），不盲假固定长度。
旧 MAE_all（m/s 与 rad/s 混平均）仅作历史兼容，不做主门。
"""
import numpy as np

from execution_wm.validity_v061.schema import EXECUTION_KEYS

LEADS_S = (0.25, 0.5, 1.0, 2.0)
GRID_DT = 0.05


def lead_index(h_s, dt=GRID_DT, n_steps=None):
    """lead 秒 -> future 数组索引（future 第 j 步时刻 = (j+1)*dt）。"""
    x = h_s / dt
    j = int(round(x)) - 1
    if abs(x - round(x)) > 1e-9:
        raise ValueError(f"lead {h_s}s 不是网格步长 {dt}s 的整数倍")
    if j < 0 or (n_steps is not None and j >= n_steps):
        raise ValueError(f"lead {h_s}s 越界（索引 {j}）")
    return j


def fixed_lead_mae(pred, gt, axis):
    """pred/gt [N,T,3]，单点。返回 (N,) 误差。"""
    j = lead_index(axis["h_s"], axis.get("dt", GRID_DT), pred.shape[1])
    a = axis["axis"]
    return np.abs(pred[:, j, a] - gt[:, j, a])


def prefix_mae(pred, gt, axis):
    """从第一步到 j(h) 的平均。返回 (N,) 误差。"""
    j = lead_index(axis["h_s"], axis.get("dt", GRID_DT), pred.shape[1])
    a = axis["axis"]
    return np.abs(pred[:, :j + 1, a] - gt[:, :j + 1, a]).mean(axis=1)


def rmse_fixed_lead(pred, gt, axis):
    j = lead_index(axis["h_s"], axis.get("dt", GRID_DT), pred.shape[1])
    a = axis["axis"]
    d = pred[:, j, a] - gt[:, j, a]
    return d ** 2


def all_metric_axes():
    return [{"h_s": h, "axis": i, "name": EXECUTION_KEYS[i]} for h in LEADS_S for i in range(3)]


def metric_row(pred, gt, axis, prefix=False):
    fn = prefix_mae if prefix else fixed_lead_mae
    return fn(pred, gt, axis)


def summarize(pred, gt, prefix=False):
    """返回 {f"{'prefix' if prefix else 'lead'}_MAE_{axis}@{h}s": value}。"""
    out = {}
    tag = "prefix" if prefix else "lead"
    for ax in all_metric_axes():
        v = metric_row(pred, gt, ax, prefix)
        out[f"{tag}_MAE_{ax['name']}@{ax['h_s']}s"] = float(np.mean(v))
    out["legacy_MAE_all (m/s+rad/s 混平均, 仅历史兼容)"] = float(np.abs(pred - gt).mean())
    return out


def per_window_errors(pred, gt):
    """[N,T,3] -> 每窗每轴每 lead 的误差（供 bootstrap）。"""
    out = {}
    for ax in all_metric_axes():
        out[f"lead_{ax['name']}@{ax['h_s']}s"] = fixed_lead_mae(pred, gt, ax)
        out[f"prefix_{ax['name']}@{ax['h_s']}s"] = prefix_mae(pred, gt, ax)
    return out
