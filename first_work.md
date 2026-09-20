你现在需要在当前已经安装完成的 Isaac Lab / Isaac Sim 环境中实现一个
“Execution-Aware World Model”第一阶段实验。

不要大规模重构现有 Isaac Lab。
不要删除已有代码。
优先复用当前仓库已有的 Unitree Go2、velocity command、locomotion controller、
sensor/state API。
如果具体 API 名称和这里描述不同，请先检查当前 Isaac Lab 版本，并按照实际 API 适配。

============================================================
0. 研究目标
============================================================

当前阶段只验证：

给定机器人过去一段时间的物理状态 history，以及未来 commanded action，
是否可以预测机器人未来真正的 physical execution，
尤其是在 friction、actuator degradation、external disturbance 等情况下。

核心定义：

u_t = [vx_cmd, vy_cmd, wz_cmd]

e_t = [vx_actual, vy_actual, wz_actual]

r_t = e_t - u_t

其中：

u_t = commanded action
e_t = physical execution
r_t = execution residual / execution mismatch

第一阶段不做：
- V-JEPA 2
- RGB world model
- online correction
- MDA
- LLM agent
- memory
- navigation planner

第一阶段首先证明 execution mismatch 是可学习和可预测的。

============================================================
1. 项目结构
============================================================

请根据当前仓库结构适配，但逻辑上至少建立以下模块：

execution_wm/
    configs/
        collect_v0.yaml
        train_v0.yaml

    data/
        collector.py
        dataset.py
        window_dataset.py

    models/
        baseline_action.py
        baseline_direct.py
        context_encoder.py
        execution_predictor.py
        execution_context_model.py

    train/
        train_execution.py

    eval/
        evaluate_execution.py
        visualize_execution.py

    scripts/
        collect_v0.sh
        train_v0.sh
        eval_v0.sh

不要强制改变现有代码结构；如果当前项目已有对应目录，融入已有结构。

============================================================
2. Robot 和 Action
============================================================

Robot:
Unitree Go2

优先复用已有的 Go2 velocity tracking / locomotion controller。

High-level action 定义统一为：

u_t = [vx_cmd, vy_cmd, wz_cmd]

不要直接把 12-dimensional joint torque 当作当前研究 action。

command 必须记录原始控制命令，而不是控制执行后的 velocity。

command 的取值范围不要硬编码超出已有 controller 的正常工作范围。
请读取当前环境已有的 command limits，并默认采样其安全范围的大约 60%-80%。

需要包含：
- forward
- backward（如果 controller 支持）
- lateral
- rotation
- combinations

command 使用 piecewise-constant 或 smooth random command，
每段持续约 0.5~2 秒。

============================================================
3. Physical Execution
============================================================

定义：

e_t = [vx_body, vy_body, wz_body]

其中 linear velocity 转换到 robot body frame。

不要直接使用 world-frame vx/vy 与 body-frame command 做 subtraction。

必须保证：

u_t 与 e_t 位于同一个 coordinate frame。

然后计算：

r_t = e_t - u_t

保存：
- commanded velocity
- actual body velocity
- execution residual

============================================================
4. 数据记录
============================================================

simulation 内部可以高频运行，但 V0 dataset 统一按照 20 Hz 输出训练数据。

必须保留 timestamp。

每一个 timestep 至少保存：

timestamp

cmd_vel:
    vx_cmd
    vy_cmd
    wz_cmd

base state:
    base_position
    base_orientation
    base_linear_velocity_world
    base_linear_velocity_body
    base_angular_velocity

IMU:
    linear_acceleration
    angular_velocity
    projected_gravity（如果现有环境支持）

joint:
    joint_position
    joint_velocity
    applied_torque 或 motor effort（如果可以获得）

feet:
    contact state for 4 feet
    foot velocity（如果可以获得）

execution:
    vx_actual
    vy_actual
    wz_actual

residual:
    vx_actual - vx_cmd
    vy_actual - vy_cmd
    wz_actual - wz_cmd

ground truth / simulation metadata:
    terrain type
    friction parameter
    actuator strength parameter
    applied external force
    episode id
    timestep
    termination reason

注意：
simulation metadata 是实验标注。
它不能作为主模型 inference input。

============================================================
5. 环境扰动
============================================================

V0 先实现四类：

A. Normal
正常 friction
正常 actuator
无 external force

B. Friction perturbation
不同 friction coefficient

C. Actuator perturbation
降低 motor strength / torque capability / actuator effectiveness

D. External disturbance
机器人运动过程中施加随机 lateral / longitudinal external force

所有 perturbation 范围必须写在 YAML 中，不要散落在 Python 文件里。

不要一开始加入大量复杂 terrain。

第一版先让 execution mismatch 来源清楚。

后续再增加：
- rough terrain
- slope
- compliant terrain
- sensor corruption

============================================================
6. 非常重要：Matched Command Experiment
============================================================

必须实现一组 fixed probe command sequences。

例如逻辑上包括：

probe_1:
    straight velocity

probe_2:
    acceleration / deceleration

probe_3:
    left/right turn

probe_4:
    lateral movement（如果 controller 支持）

同一组 probe command 必须分别在：

normal
low friction
actuator degradation

下重复执行。

目的：

保持 command 尽可能相同，
只改变 execution condition。

这样之后我们才能证明：

same command
+
different execution context
→
different actual execution

这组数据非常重要。

============================================================
7. Episode 和 Dataset
============================================================

V0 sanity dataset：

先生成约 100 episodes，
每个 10~20 秒。

确认所有数据正确后，
再扩展到约 1000~3000 episodes。

训练数据采用 sliding window。

history:

L = 1 second

20 Hz 时：
history_steps = 20

future prediction horizon 至少支持：

0.5 sec = 10 steps
1.0 sec = 20 steps
2.0 sec = 40 steps

Dataset output：

history_proprio
history_action
future_action
future_execution
future_residual
metadata

注意：
metadata 默认不能进入模型。

============================================================
8. Baseline 0：Action-only
============================================================

输入：

future commanded action

输出：

future execution residual

模型可以先使用简单 MLP / temporal MLP。

目标：

证明 command alone 无法充分解释不同 physical condition 下的 execution。

============================================================
9. Baseline 1：Direct History Predictor
============================================================

输入：

history proprioception
history action
future commanded action

直接预测：

future residual

形式：

r_hat =
F(history, future_action)

这是普通 direct predictor。

不要加入显式 execution context。

============================================================
10. Ours V0：Execution Context Model
============================================================

构建一个 ContextEncoder：

c_t = C_phi(history)

history 当前优先使用：

past commanded action
past actual body velocity
IMU
joint state
contact

context latent dimension 默认：

context_dim = 8

必须可以通过 YAML 修改。

然后：

r_hat =
G_theta(current_state_latent,
        c_t,
        future_command)

最终：

e_hat = future_command + r_hat

当前 V0 使用 deterministic c_t 即可。

============================================================
11. Context Encoder
============================================================

第一版使用：

GRU / temporal encoder

输入：
过去 1 秒 history

输出：
c_t ∈ R^8

不要把 simulation friction coefficient、
actuator strength、
terrain ID 直接送入 ContextEncoder。

这些只能用来：

- evaluation
- visualization
- later auxiliary ablation

============================================================
12. Execution Prediction Loss
============================================================

默认使用 Huber loss：

L_exec =
Σ_k Huber(r_hat_{t+k} - r_{t+k})

同时记录：

velocity prediction MAE
velocity prediction RMSE
residual MAE
residual RMSE

分别输出：

vx
vy
wz

和 overall metric。

============================================================
13. Optional uncertainty
============================================================

代码结构需要预留 uncertainty 输出，
但第一轮训练默认关闭。

后续支持：

predict:
mu_r
logvar_r

Gaussian NLL：

0.5 * [
    (r - mu)^2 / sigma^2
    + log sigma^2
]

配置项：

predict_uncertainty: false

当前不要影响 deterministic MVP。

============================================================
14. Train / Val / Test split
============================================================

不要只做 random timestep split。

必须 episode-level split。

此外实现 OOD test。

例如：

train:
normal + selected friction/actuator ranges

test_id:
seen parameter ranges, unseen episodes

test_ood:
unseen friction / actuator range

具体数值通过 config 定义。

============================================================
15. Context 可视化
============================================================

训练后保存：

c_t embeddings

至少提供：

PCA 2D visualization

颜色按照：
- friction
- actuator strength
- disturbance type

这些 metadata 只用于 visualization。

目标是观察：

Execution Context 是否自然形成与执行条件相关的结构。

============================================================
16. 必须输出的图
============================================================

Figure A:
commanded velocity vs actual velocity vs predicted velocity

Figure B:
true residual vs predicted residual

Figure C:
prediction error vs friction / disturbance severity

Figure D:
context embedding PCA

Figure E:
同一个 matched command 在不同 physical conditions 下的 execution comparison

============================================================
17. Sanity checks
============================================================

数据采完以后首先运行自动检查：

1. cmd 和 actual 是否同 frame
2. timestamp 是否单调
3. residual == actual - command
4. contact 是否合理
5. velocity unit 是否正确
6. episode boundary 是否正确
7. NaN / Inf
8. perturbation metadata 是否正确
9. perturbation 是否真的造成 measurable execution mismatch

如果 friction 改变以后 execution 基本不变，
不要继续训练模型。

需要首先确认 locomotion controller 是否过强、
friction range 是否无效、
或者 simulation material 没有正确应用。

============================================================
18. 最终 acceptance criteria
============================================================

第一阶段不是追求 SOTA。

完成条件：

A.
能够稳定生成 execution dataset。

B.
在 matched command 条件下可以看到：
different context → different execution residual。

C.
Direct baseline 可以训练并收敛。

D.
Execution Context model 可以训练并收敛。

E.
能够比较：

Action only
vs
Direct history predictor
vs
Execution-context predictor

F.
能够输出 ID + OOD evaluation。

G.
所有实验参数都放在 config，而不是散落 hard-code。

H.
输出 experiment summary，包括：
git commit
config
dataset size
training curves
evaluation metrics
plots

============================================================
19. 执行顺序
============================================================

不要一次全部完成。

Step 1:
确认 Go2 velocity tracking 正常。

Step 2:
完成 Normal + Friction 数据采集。

Step 3:
仅采 10 个 episodes，人工检查数据。

Step 4:
采 100 episodes sanity dataset。

Step 5:
画 command / actual / residual。

Step 6:
确认 execution mismatch 真实存在。

Step 7:
训练 Action-only baseline。

Step 8:
训练 Direct predictor。

Step 9:
实现 ContextEncoder。

Step 10:
比较 Direct vs Context model。

Step 11:
再加入 actuator / external force。

Step 12:
生成 ID / OOD 实验。

每完成一个阶段都保存运行命令和结果。