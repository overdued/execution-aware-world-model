# 阶段A2：时间与 causal input 审计

代码链：collector.py(capture_step) -> save_episode(降采样) -> dataset.py(proprio 拼接) -> context_swap/common.py(window 切片) -> models/*.py(forward) -> metrics。

## 1. 输入因果边界

- window(t0): history_proprio/history_action = [t0-L+1, t0]（≤ origin ✓）；future_action/execution/residual = [t0+1, t0+H]。
- 干预单测：修改 t0-L+1 之前历史，输出最大差 0.00e+00（=0 ✓）。

## 2. 干预单测（labels / privileged）

- 修改 future_execution/future_residual 为 ±999：输出最大差 0.00e+00（=0 ✓ 标签不进入输入）。
- privileged 字段（friction 真值、episode metadata）不是主模型输入（forward 签名仅 history_proprio/history_action/current_state/future_action；input_roles.csv 逐项列出）。
- 修改 future command：输出最大差 0.773（>0，确认 future command 是有效输入——这正是设计语义）。

## 3. future command 的语义

命令由 CommandScheduler 开环预生成（不闭环于状态），episode 开始前整条 schedule 已确定。因此 recorded future command 是合法的'预测时已知候选计划'；但它是**回放条件**下的计划，不是在线闭环重新规划的命令。terminated episode 结束后的 schedule 段是'本将执行'值。

## 4. 命令与状态的时间约定（重要）

collector 主循环：写入 cmd_vel(t) -> capture_step 抓取 **step 前状态** e(t) -> policy(obs) -> env.step。
- 记录的 e_t 是区间 [t-Δt, t] 的执行结果（pre-step 状态），u_t 是将要在 [t, t+Δt] 施加的命令。
- 因此 r_t = e_t - u_t 是'当前速度与当前请求'的瞬时 tracking error，**命令领先执行约一步**。
- window 切片 future 从 t0+1 开始，部分补偿该滞后（u[t0] 在 history 末帧，其效果体现在 e[t0+1]）。
- 数据驱动验证 corr(e_t, u_{t-lag}) [vx,vy,wz]：
```json
{
 "-1": [
  0.7825,
  0.7125,
  0.744
 ],
 "0": [
  0.8345,
  0.7448,
  0.7777
 ],
 "1": [
  0.8873,
  0.7754,
  0.8072
 ]
}
```
lag=+1（u_{t-1} 对 e_t）相关最高则滞后成立。

## 5. 50Hz -> 20Hz 降采样（P1）
```json
{
 "scheme": "nearest-neighbor idx=round(k*2.5)（np.round banker's rounding）",
 "true_dt_values_s": [
  0.04,
  0.06
 ],
 "true_dt_counts": {
  "0.04": 81,
  "0.06": 80
 },
 "labeled_dt_s": 0.05,
 "max_abs_clock_error_s": 0.010000000000000675,
 "clock_drift": "无累计漂移（误差在 ±0.01s 内交替），但瞬时相位 2/3 步交替 -> 非均匀采样",
 "aliasing": "最近邻抽取，无抗混叠滤波；50Hz 信号中 >10Hz 分量混叠进 20Hz 数据"
}
```
最近邻 idx=round(2.5k)（banker's rounding：0,2,5,7,10,12,...），真实间隔 {0.04,0.06}s 交替，标称 0.05s。无抗混叠 -> >10Hz 步态波纹混叠；瞬时相位非均匀。**修复会改变数据 -> 单独版本化（本轮不重采）**。

## 6. forecast 完整性

每个预测为 fixed-origin 完整 [H=40,3]（predict() 一次前向输出整窗，非滚动一步拼接）；输出形状断言 [40, 3]。评估时不同 lead_time 应分开报告（A4/A6 执行）。

## 7. frame / quaternion 单测
```json
{
 "quat_order_declared": "wxyz（Isaac Lab root_quat_w 约定）",
 "max_err_wxyz": 2.384185791015625e-07,
 "max_err_xyzw": 2.161564826965332,
 "wxyz_matches": true,
 "corr_body_wz_vs_world_yaw_rate_mean": 0.8727843569767323
}
```
quaternion 顺序 (w,x,y,z)（Isaac Lab root_quat_w 约定），数值验证通过。body wz 与世界 yaw 速率在 roll/pitch≈0 时高度相关但不恒等；评价保持 body-frame 语义。

## 8. GT vs 估计量（privileged 输入）

模型 state/context 输入含 **GT body 速度、GT 角速度、GT 姿态导出 projected_gravity**（无状态估计器）；imu_angular_velocity 已记录但未使用。结论：V0 全部结果是 **privileged-input** 版本；deployable 对照（odom/IMU 输入）本轮未建立，列为后续必需实验。标签使用 GT 合法。

## 判定

- 因果边界（输入 ≤ origin、标签不泄漏）：**PASS（单测为证）**
- 时间语义：e/u 一步滞后与 {0.04,0.06}s 非均匀采样已查明并记录，属**定义性偏差而非随机 bug**；所有基于 r_t 的解释须带此前提。
- privileged 输入：已声明，影响部署外推，不影响仿真内比较的有效性。