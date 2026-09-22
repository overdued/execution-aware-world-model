"""V0.6.1 B3：整数 tick 时间身份 —— 不用浮点秒做等式比较。

约定（PRE_REGISTRATION_V061 §时间语义）：
- 原始 50Hz：控制 tick i 的时刻 = i * CTRL_DT_S 秒（i 为整数身份，浮点秒仅展示）。
- 20Hz 查询 grid：grid 点 k 的时刻 = k * 0.05 s = 5k/100 s。
  50→20 是有理数关系 5k/100 ÷ 2/100 = 5k/2 个控制 tick。
- hold 索引（"不超过 t_k 的最后一个控制 tick"）= (5*k)//2，纯整数运算：
    k 偶 -> 正好落在控制 tick 5k/2 上（该 tick 就是 t_k 时刻的样本）；
    k 奇 -> t_k 严格位于 tick floor(2.5k) 与下一 tick 之间，取 floor(2.5k)。
- command 语义：控制 tick i 的 command 在区间 [i*0.02, (i+1)*0.02) 内生效（右连续）。
  u_ref[k] = cmd[tick_k]，tick_k = (5*k)//2。事件边界（如 2.5s -> tick 125）取新分段。
"""
import numpy as np

CTRL_DT_S = 0.02          # 50 Hz 控制周期
GRID_DT_S = 0.05          # 20 Hz 派生网格
_RATIO_NUM, _RATIO_DEN = 5, 1   # t_k = 5k/100 s；tick = (5k)/2 -> 用 (5*k)//2


def n_control_ticks(duration_s):
    """时长 -> 控制 tick 数（严格整数，不用浮点比较）。"""
    return int(round(duration_s / CTRL_DT_S))


def n_grid_points(n_ticks):
    """N 个控制 tick（时长 N*0.02 s）-> 20Hz grid 点数 = ceil(N*0.02/0.05)。"""
    return -((-2 * n_ticks) // 5)          # ceil(2N/5)


def grid_hold_tick(k):
    """grid 点 k 的 hold 控制 tick 索引（纯整数）。"""
    return (5 * k) // 2


def grid_is_tick_aligned(k):
    """grid 点 k 是否正好落在控制 tick 上。"""
    return (5 * k) % 2 == 0


def grid_hold_ticks(n_grid):
    return np.array([grid_hold_tick(k) for k in range(n_grid)], dtype=np.int64)


def tick_event_index(event_s):
    """命令事件时刻（秒，相对 episode 起点的控制时钟）-> tick 索引。

    事件表由分段常数构成，事件时刻定义在控制 tick 网格上（秒 -> tick 用有理数判断）。
    """
    num = int(round(event_s * 100))        # 以 1/100 s 为单位
    if abs(event_s * 100 - num) > 1e-9:
        raise ValueError(f"事件时刻 {event_s}s 不在 1/100 s 有理网格上")
    return num // 2, num % 2               # tick = num/2（0.02s/tick）


def build_command_event_table(segments, settle_s=0.0):
    """分段常数命令 -> 事件表 [(tick_start, tick_end_exclusive, cmd[3])]。"""
    events, tick = [], 0
    for dur, cmd in segments:
        n = n_control_ticks(dur)
        events.append((tick, tick + n, np.asarray(cmd, dtype=np.float64)))
        tick += n
    return events


def command_from_events(events, tick):
    """右连续：返回 tick 处生效的命令。"""
    for t0, t1, cmd in events:
        if t0 <= tick < t1:
            return cmd
    return np.zeros(3, dtype=np.float64)


def assert_tick_grid_consistent(timestamps, ctrl_dt=CTRL_DT_S, atol=1e-6, label=""):
    """校验记录时间戳与整数 tick 网格一致（浮点秒仅展示，不做等式判断）。"""
    n = len(timestamps)
    rel = timestamps - timestamps[0]
    expect = np.arange(n) * ctrl_dt
    err = float(np.abs(rel - expect).max()) if n else 0.0
    if err > atol:
        raise AssertionError(
            f"{label} 时间戳与整数 tick 网格不一致: max|err|={err:.3e} > {atol:.1e} "
            f"(n={n})。禁止用浮点秒做 tick 身份。")
    return n, err
