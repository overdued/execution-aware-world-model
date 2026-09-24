"""V0.7 轨迹积分（单一真源，tests 与 eval 共用）。

平面 SE(2)：体速度 v[k] 在区间 [t_k, t_k+dt) 内作用，用**中点 yaw**
    yaw_mid = yaw[k] + wz[k]*dt/2
做旋转，即梯形/中点法则。前向欧拉（用 yaw[k]）在恒定转弯下有 O(dt) 偏差
（dt=0.05, R=1.2m, θ=1rad 时约 1.3cm），中点法则降到 O(dt²)。
不使用任何未来 GT yaw —— yaw 由起点真值 + 预测 wz 自回归积分得到。
"""
import numpy as np

DT = 0.05


def wrap_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def integrate_xy(v_xy, yaw0, wz=None, dt=DT, yaw_seq=None):
    """v_xy [T,2]（体系），返回世界系位移轨迹 [T,2]（累积）。

    yaw_seq 给出时直接使用（用于真值侧 "给定 yaw" 诊断）；
    否则由 yaw0 + 自回归积分 wz 得到（预测侧主口径）。
    """
    T = len(v_xy)
    if yaw_seq is None:
        yaw_seq = yaw0 + np.concatenate([[0.0], np.cumsum(wz[:-1]) * dt])
    yaw_mid = yaw_seq + (0.0 if wz is None else wz * dt / 2.0)
    c, s = np.cos(yaw_mid), np.sin(yaw_mid)
    dx = (c * v_xy[:, 0] - s * v_xy[:, 1]) * dt
    dy = (s * v_xy[:, 0] + c * v_xy[:, 1]) * dt
    return np.cumsum(np.stack([dx, dy], axis=1), axis=0)


def integrate_xy_batch(v_xy, yaw0, wz=None, yaw_seq=None, dt=DT):
    """向量化版本：v_xy [N,T,2]，yaw0 [N]，wz [N,T] 或 yaw_seq [N,T]。返回 [N,T,2]。"""
    N, T = v_xy.shape[0], v_xy.shape[1]
    if yaw_seq is None:
        yaw_seq = yaw0[:, None] + np.concatenate(
            [np.zeros((N, 1)), np.cumsum(wz[:, :-1], axis=1) * dt], axis=1)
    yaw_mid = yaw_seq + (0.0 if wz is None else wz * dt / 2.0)
    c, s = np.cos(yaw_mid), np.sin(yaw_mid)
    dx = (c * v_xy[:, :, 0] - s * v_xy[:, :, 1]) * dt
    dy = (s * v_xy[:, :, 0] + c * v_xy[:, :, 1]) * dt
    return np.cumsum(np.stack([dx, dy], axis=2), axis=1)
