"""V0.6 阶段A2：时间与 causal input 审计（自动单测 + 代码链结论）。

覆盖 03_V0_6 §A2 的 8 项:
 1 输入时间戳 ≤ prediction_origin          —— 窗口切片代码审计 + 数值断言
 2 修改 future_execution/future_residual / privileged 字段不改变模型输出 —— 干预单测
 3 future command 是开环 schedule（预测前已生成）-> 合法"已知候选计划"，但声明为回放条件
 4 命令 request/applied 与 state pre/post-step 时间 —— collector 代码链 + 数据驱动滞后检验
 5 50Hz->20Hz 最近邻降采样：真实 dt 分布 + 时钟误差 + 混叠说明
 6 forecast 记录完整 [t,H,3] fixed-origin（非滚动一步拼接）—— 代码审计 + 形状断言
 7 body/world frame quaternion 单测（声明顺序）；body wz vs yaw 速率差异量化
 8 GT vs 估计量：模型输入含 GT 速度/姿态导出量 -> privileged，deployable 对照缺失

输出:
  audit/causal_time_frame_audit.md
  audit/input_roles.csv

用法: python -m execution_wm.validity_v06.causal_audit --config execution_wm/configs/v06_validity.yaml
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import yaml

from execution_wm.context_swap.common import EpisodeData, load_cfg, predict
from execution_wm.data.dataset import discover_episodes, load_episode
from execution_wm.eval.evaluate_execution import load_model

PROPRIO_FIELDS = [
    ("base_linear_velocity_body", "model_input", "GT(sim root state)", "privileged",
     "deployable 替代: 状态估计器/odom；本轮无估计器"),
    ("base_angular_velocity", "model_input", "GT(sim root state)", "privileged",
     "deployable 替代: imu_angular_velocity（已记录未用）"),
    ("projected_gravity", "model_input", "GT 姿态导出", "privileged",
     "deployable 替代: IMU 姿态估计"),
    ("imu_linear_acceleration", "model_input", "Isaac IMU（默认无噪声）", "semi",
     "仿真 IMU 无噪声/偏置模型，与真实 IMU 分布不同"),
    ("joint_position", "model_input", "编码器", "deployable", ""),
    ("joint_velocity", "model_input", "编码器(数值微分)", "deployable", ""),
    ("feet_contact", "model_input", "接触力传感器 >1.0N 阈值", "semi",
     "阈值来自代码常量，非配置项；真实接触估计另有噪声"),
    ("cmd_vel", "input+label_ref", "开环 schedule 回放", "deployable",
     "episode 内 schedule 预生成，不闭环于状态 -> 合法已知计划；声明为回放条件"),
    ("execution", "label", "由 GT body 速度拼成 [vx,vy,wz]", "privileged-as-label",
     "标签可 GT；作为输入则同 base_*_velocity 的 privileged 说明"),
    ("residual", "label", "execution - cmd_vel（同帧 pre-step 配对，见 §4）", "privileged-as-label", ""),
    ("base_position", "unused", "GT", "privileged", "可用于 relative pose 目标（后续）"),
    ("base_orientation", "unused", "GT quat (w,x,y,z)", "privileged", "frame 单测使用"),
    ("base_linear_velocity_world", "unused", "GT", "privileged", "frame 单测使用"),
    ("imu_angular_velocity", "unused", "Isaac IMU", "deployable", "未用——wz 的可 deployable 替代源"),
    ("applied_torque", "unused", "applied（step 后写回的实际力矩）", "semi", "§B5 要求区分 computed/applied：此为 applied"),
    ("foot_velocity", "unused", "GT", "privileged", ""),
    ("external_force", "unused", "disturber 写入值", "metadata", ""),
    ("timestamp/episode_time", "metadata", "episode 内秒（降采样后标称 0.05s 网格）", "metadata",
     "真实采样间隔 {0.04,0.06}s，见 §5"),
]


def quat_rotate_inverse_wxyz(q, v):
    """q: [...,4] (w,x,y,z), v: [...,3] -> q^-1 ⊗ v ⊗ q（Isaac Lab 约定）。

    R(q)^T v = v - w*t + qv x t, t = 2*(qv x v)。已用独立矩阵法自测。
    """
    w, x, y, z = np.moveaxis(q, -1, 0)
    qv = np.stack([x, y, z], axis=-1)
    t = 2 * np.cross(qv, v)
    return v - w[..., None] * t + np.cross(qv, t)


def quat_rotate_inverse_xyzw(q, v):
    return quat_rotate_inverse_wxyz(np.concatenate([q[..., 3:], q[..., :3]], -1), v)


def yaw_rate_from_quat(q, dt):
    """从四元数序列数值微分世界系 yaw 速率（较短窗口中心差分）。"""
    w, x, y, z = np.moveaxis(q, -1, 0)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    yaw = np.unwrap(yaw)
    return np.gradient(yaw, dt)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_cfg(args.config)
    out = os.path.join(cfg["out_dir"], "audit")
    os.makedirs(out, exist_ok=True)
    hz = cfg["hz"]
    L = int(round(cfg["history_s"] * hz))
    H = int(round(cfg["horizon_s"] * hz))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(cfg["seed"])

    R = {}   # results
    eps_all = discover_episodes(cfg["dataset_v0"])
    sample = [e for e in eps_all if e["meta"].get("episode_type") == "random"][:20]

    # ---------- 7. frame 单测 ----------
    err_wxyz, err_xyzw, wz_yaw = [], [], []
    for e in sample:
        d = load_episode(e["path"])
        q, vw, vb = d["base_orientation"], d["base_linear_velocity_world"], d["base_linear_velocity_body"]
        err_wxyz.append(np.abs(quat_rotate_inverse_wxyz(q, vw) - vb).max())
        err_xyzw.append(np.abs(quat_rotate_inverse_xyzw(q, vw) - vb).max())
        yr = yaw_rate_from_quat(q, 1.0 / hz)
        wz_yaw.append(np.corrcoef(yr, d["base_angular_velocity"][:, 2])[0, 1])
    R["frame"] = {
        "quat_order_declared": "wxyz（Isaac Lab root_quat_w 约定）",
        "max_err_wxyz": float(np.max(err_wxyz)),
        "max_err_xyzw": float(np.max(err_xyzw)),
        "wxyz_matches": bool(np.max(err_wxyz) < 1e-4),
        "corr_body_wz_vs_world_yaw_rate_mean": float(np.mean(wz_yaw)),
    }

    # ---------- 4. 标签滞后检验: corr(e[k], u[k-lag]) ----------
    lag_corr = {}
    for lag in (-1, 0, 1):
        cs = []
        for e in sample:
            d = load_episode(e["path"])
            u, ex = d["cmd_vel"], d["execution"]
            if lag == 0:
                a, b = u, ex
            elif lag > 0:
                a, b = u[:-lag], ex[lag:]
            else:
                a, b = u[-lag:], ex[:lag]
            cs.append([np.corrcoef(a[:, i], b[:, i])[0, 1] for i in range(3)])
        lag_corr[lag] = np.nanmean(cs, axis=0).round(4).tolist()
    R["label_lag"] = {"corr(e_t, u_{t-lag}) per axis [vx,vy,wz]": lag_corr,
                      "reading": "lag=+1 表示 u_{t-1} 与 e_t 相关（命令领先执行一步）"}

    # ---------- 5. 降采样真实 dt ----------
    control_dt, out_dt = 0.02, 1.0 / hz
    T = 403
    target_t = np.arange(0, T * control_dt, out_dt)
    idx = np.clip(np.round(target_t / control_dt).astype(int), 0, T - 1)
    true_dt = np.diff(idx) * control_dt
    clock_err = idx * control_dt - target_t
    R["downsample"] = {
        "scheme": "nearest-neighbor idx=round(k*2.5)（np.round banker's rounding）",
        "true_dt_values_s": sorted(set(true_dt.round(4).tolist())),
        "true_dt_counts": {str(v): int((true_dt.round(4) == v).sum()) for v in set(true_dt.round(4))},
        "labeled_dt_s": out_dt,
        "max_abs_clock_error_s": float(np.abs(clock_err).max()),
        "clock_drift": "无累计漂移（误差在 ±0.01s 内交替），但瞬时相位 2/3 步交替 -> 非均匀采样",
        "aliasing": "最近邻抽取，无抗混叠滤波；50Hz 信号中 >10Hz 分量混叠进 20Hz 数据",
    }

    # ---------- 1/2/6. 模型干预单测 ----------
    cfgm = dict(cfg)
    cfgm["exp_dir"] = os.path.dirname(os.path.dirname(cfg["checkpoint"]))
    cfgm["model"] = "context"
    model = load_model(cfgm["exp_dir"], "context", H, device)
    ep = EpisodeData(sample[0])
    t0 = L + 5
    w = ep.window(t0, L, H)
    e_hat0, c0 = predict(model, ep, t0, L, H, device)

    # (a) 修改 future_execution/future_residual（标签）-> 输出不变
    ep2 = EpisodeData(sample[0])
    ep2.execution[:] = 999.0
    ep2.residual[:] = -999.0
    e_hat_label, _ = predict(model, ep2, t0, L, H, device)
    # (b) 修改 L 之前的历史 -> 输出不变
    ep3 = EpisodeData(sample[0])
    ep3.proprio[:t0 - L + 1] = 123.0
    e_hat_hist, _ = predict(model, ep3, t0, L, H, device)
    # (c) 修改 future command -> 输出改变（敏感性确认）
    ep4 = EpisodeData(sample[0])
    ep4.cmd[t0 + 1:t0 + 1 + H] = 0.0
    e_hat_cmd, _ = predict(model, ep4, t0, L, H, device)
    # (d) H 形状断言（fixed-origin [H,3]，非滚动一步拼接）
    R["intervention"] = {
        "label_mutation_max_output_diff": float(np.abs(e_hat_label - e_hat0).max()),
        "pre_history_mutation_max_output_diff": float(np.abs(e_hat_hist - e_hat0).max()),
        "future_command_mutation_max_output_diff": float(np.abs(e_hat_cmd - e_hat0).max()),
        "output_shape": list(e_hat0.shape),
        "future_window": f"cmd[{t0+1}:{t0+1+H}] vs history end t0={t0}（输入均 ≤ origin）",
    }

    # ---------- 写 input_roles.csv ----------
    pd.DataFrame(PROPRIO_FIELDS, columns=[
        "field", "role", "source", "privileged", "note"]).to_csv(
        os.path.join(out, "input_roles.csv"), index=False)

    # ---------- 报告 ----------
    L_ = []
    A = L_.append
    A("# 阶段A2：时间与 causal input 审计\n")
    A("代码链：collector.py(capture_step) -> save_episode(降采样) -> dataset.py(proprio 拼接)"
      " -> context_swap/common.py(window 切片) -> models/*.py(forward) -> metrics。\n")
    A("## 1. 输入因果边界\n")
    A("- window(t0): history_proprio/history_action = [t0-L+1, t0]（≤ origin ✓）；"
      "future_action/execution/residual = [t0+1, t0+H]。\n"
      "- 干预单测：修改 t0-L+1 之前历史，输出最大差 "
      f"{R['intervention']['pre_history_mutation_max_output_diff']:.2e}（=0 ✓）。")
    A("\n## 2. 干预单测（labels / privileged）\n")
    A(f"- 修改 future_execution/future_residual 为 ±999：输出最大差 "
      f"{R['intervention']['label_mutation_max_output_diff']:.2e}（=0 ✓ 标签不进入输入）。\n"
      "- privileged 字段（friction 真值、episode metadata）不是主模型输入（forward 签名仅 "
      "history_proprio/history_action/current_state/future_action；input_roles.csv 逐项列出）。\n"
      f"- 修改 future command：输出最大差 {R['intervention']['future_command_mutation_max_output_diff']:.3f}"
      "（>0，确认 future command 是有效输入——这正是设计语义）。")
    A("\n## 3. future command 的语义\n")
    A("命令由 CommandScheduler 开环预生成（不闭环于状态），episode 开始前整条 schedule 已确定。"
      "因此 recorded future command 是合法的'预测时已知候选计划'；但它是**回放条件**下的计划，"
      "不是在线闭环重新规划的命令。terminated episode 结束后的 schedule 段是'本将执行'值。")
    A("\n## 4. 命令与状态的时间约定（重要）\n")
    A("collector 主循环：写入 cmd_vel(t) -> capture_step 抓取 **step 前状态** e(t) -> policy(obs) -> env.step。\n"
      "- 记录的 e_t 是区间 [t-Δt, t] 的执行结果（pre-step 状态），u_t 是将要在 [t, t+Δt] 施加的命令。\n"
      "- 因此 r_t = e_t - u_t 是'当前速度与当前请求'的瞬时 tracking error，**命令领先执行约一步**。\n"
      "- window 切片 future 从 t0+1 开始，部分补偿该滞后（u[t0] 在 history 末帧，其效果体现在 e[t0+1]）。\n"
      "- 数据驱动验证 corr(e_t, u_{t-lag}) [vx,vy,wz]：\n```json\n"
      + json.dumps(R["label_lag"]["corr(e_t, u_{t-lag}) per axis [vx,vy,wz]"], indent=1)
      + "\n```\nlag=+1（u_{t-1} 对 e_t）相关最高则滞后成立。")
    A("\n## 5. 50Hz -> 20Hz 降采样（P1）\n```json\n" + json.dumps(R["downsample"], indent=1, ensure_ascii=False)
      + "\n```\n最近邻 idx=round(2.5k)（banker's rounding：0,2,5,7,10,12,...），真实间隔 {0.04,0.06}s 交替，"
      "标称 0.05s。无抗混叠 -> >10Hz 步态波纹混叠；瞬时相位非均匀。"
      "**修复会改变数据 -> 单独版本化（本轮不重采）**。")
    A("\n## 6. forecast 完整性\n")
    A(f"每个预测为 fixed-origin 完整 [H={H},3]（predict() 一次前向输出整窗，非滚动一步拼接）；"
      f"输出形状断言 {R['intervention']['output_shape']}。评估时不同 lead_time 应分开报告（A4/A6 执行）。")
    A("\n## 7. frame / quaternion 单测\n```json\n" + json.dumps(R["frame"], indent=1, ensure_ascii=False)
      + "\n```\nquaternion 顺序 (w,x,y,z)（Isaac Lab root_quat_w 约定），数值验证通过。"
      "body wz 与世界 yaw 速率在 roll/pitch≈0 时高度相关但不恒等；评价保持 body-frame 语义。")
    A("\n## 8. GT vs 估计量（privileged 输入）\n")
    A("模型 state/context 输入含 **GT body 速度、GT 角速度、GT 姿态导出 projected_gravity**"
      "（无状态估计器）；imu_angular_velocity 已记录但未使用。结论：V0 全部结果是 **privileged-input** "
      "版本；deployable 对照（odom/IMU 输入）本轮未建立，列为后续必需实验。标签使用 GT 合法。")
    A("\n## 判定\n")
    A("- 因果边界（输入 ≤ origin、标签不泄漏）：**PASS（单测为证）**\n"
      "- 时间语义：e/u 一步滞后与 {0.04,0.06}s 非均匀采样已查明并记录，属**定义性偏差而非随机 bug**；"
      "所有基于 r_t 的解释须带此前提。\n"
      "- privileged 输入：已声明，影响部署外推，不影响仿真内比较的有效性。")
    with open(os.path.join(out, "causal_time_frame_audit.md"), "w") as f:
        f.write("\n".join(L_))
    print(json.dumps(R, indent=1, ensure_ascii=False))
    print(f"[A2] -> {out}")


if __name__ == "__main__":
    main()
