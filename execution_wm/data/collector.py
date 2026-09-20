"""Execution dataset collector (first_work.md §2-§7).

在 Isaac Lab Go2 flat velocity 环境上，用已训练的 locomotion policy 作为 controller，
按 YAML 中的条件（normal / friction / actuator / disturbance）采集 commanded velocity
与 physical execution 数据。包含 matched probe command 实验（§6）。

运行方式（必须用 isaaclab.sh 启动仿真）:
    cd ~/IsaacLab
    OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh -p \
        ~/cvpr_embed/execution_wm/data/collector.py \
        --config ~/cvpr_embed/execution_wm/configs/collect_v0.yaml \
        [--conditions normal friction_low] [--num-episodes 10]
"""
import argparse
import sys

# --- 必须在导入 isaaclab 之前启动 App --------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--conditions", nargs="*", default=None, help="只跑指定条件")
parser.add_argument("--num-episodes", type=int, default=None, help="覆盖每条件 episode 数（调试用）")
parser.add_argument("--num-envs", type=int, default=None)
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# --- 其余导入 ---------------------------------------------------------------
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(PROJECT_ROOT))

import gymnasium as gym  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401  注册 gym 环境
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.envs import mdp  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.sensors import ImuCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import (  # noqa: E402
    UnitreeGo2FlatPPORunnerCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (  # noqa: E402
    UnitreeGo2FlatEnvCfg,
)
from rsl_rl.runners import OnPolicyRunner  # noqa: E402


@configclass
class Go2ExecCollectEnvCfg(UnitreeGo2FlatEnvCfg):
    """在 Go2 flat velocity env 上做最小改动用于数据采集。"""

    def __post_init__(self):
        super().__post_init__()
        # 数据采集期间关闭无关随机化，让 execution mismatch 来源清楚（§5）
        self.events.physics_material = None      # friction 由 collector 显式设定
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        # 命令改为直接速度模式（关闭 heading 模式），由 collector 每步写入
        cmd = self.commands.base_velocity
        cmd.heading_command = False
        cmd.rel_heading_envs = 0.0
        cmd.rel_standing_envs = 0.0
        cmd.resampling_time_range = (1.0e6, 1.0e6)  # 禁用自动重采样
        cmd.debug_vis = False
        # episode 长度覆盖
        self.episode_length_s = 25.0
        # 加 IMU（§4 数据记录）
        self.scene.imu = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/base")


# ---------------------------------------------------------------------------
class CommandScheduler:
    """为每个 env 生成/执行 piecewise-constant 命令序列。"""

    def __init__(self, cfg, limits, device, num_envs):
        self.cfg = cfg
        self.limits = limits  # [vx, vy, wz] 正向幅值
        self.device = device
        self.num_envs = num_envs
        # 每个 env 当前 episode 的 schedule: (durations[s], cmds[3]) 与起始 step
        self.schedules = [None] * num_envs
        self.start_step = np.zeros(num_envs, dtype=np.int64)
        self.episode_type = ["random"] * num_envs   # random | probe 名
        self.rng = np.random.default_rng(cfg["seed"])

    def _sample_random_schedule(self, total_s, dt):
        lo, hi = self.cfg["command"]["limit_fraction"]
        d_lo, d_hi = self.cfg["command"]["segment_duration_s"]
        segs, t = [], 0.0
        while t < total_s:
            dur = float(self.rng.uniform(d_lo, d_hi))
            frac = self.rng.uniform(lo, hi, size=3)
            signs = self.rng.choice([-1.0, 1.0], size=3)
            # 有一定概率某些轴为 0，制造单轴/组合混合
            zero_mask = self.rng.random(3) < 0.25
            cmd = frac * signs * self.limits
            cmd[zero_mask] = 0.0
            segs.append((dur, cmd.astype(np.float32)))
            t += dur
        return segs

    def assign(self, env_id, episode_type, probe_segments=None, total_s=20.0, dt=0.02, step=0):
        if episode_type == "random":
            segs = self._sample_random_schedule(total_s, dt)
        else:
            segs = [(float(d), np.asarray(c, dtype=np.float32)) for d, c in probe_segments]
        self.schedules[env_id] = segs
        self.start_step[env_id] = step
        self.episode_type[env_id] = episode_type

    def duration(self, env_id):
        return sum(d for d, _ in self.schedules[env_id])

    def command_at(self, env_id, t):
        """t: 该 episode 内经过秒数。"""
        acc = 0.0
        for dur, cmd in self.schedules[env_id]:
            if t < acc + dur:
                return cmd
            acc += dur
        return np.zeros(3, dtype=np.float32)


# ---------------------------------------------------------------------------
def set_friction(env, friction):
    dyn = max(0.05, 0.7 * friction)
    mdp.randomize_rigid_body_material(
        env, torch.arange(env.num_envs, device=env.device),
        static_friction_range=(friction, friction),
        dynamic_friction_range=(dyn, dyn),
        restitution_range=(0.0, 0.0),
        asset_cfg=SceneEntityCfg("robot", body_names=".*"),
        num_buckets=1,
    )


class ActuatorScaler:
    """缩放 actuator 力矩上限，模拟执行器退化（§5.C）。"""

    def __init__(self, env):
        self.act = env.scene["robot"].actuators["base_legs"]
        self.backup = {}
        for name in ("effort_limit", "saturation_effort"):
            v = getattr(self.act, name, None)
            if isinstance(v, torch.Tensor):
                self.backup[name] = v.clone()

    def set_scale(self, s):
        for name, orig in self.backup.items():
            setattr(self.act, name, orig * s)


class Disturber:
    """随机侧向/纵向外力（§5.D）。"""

    def __init__(self, env, force_range, resample_range, rng):
        self.env = env
        self.robot = env.scene["robot"]
        self.f_lo, self.f_hi = force_range
        self.r_lo, self.r_hi = resample_range
        self.rng = rng
        self.base_id = self.robot.find_bodies("base")[0]
        self.force = torch.zeros(env.num_envs, 1, 3, device=env.device)
        self.torque = torch.zeros(env.num_envs, 1, 3, device=env.device)
        self.next_resample = np.zeros(env.num_envs)
        self.enabled = False

    def maybe_apply(self, t_abs, dt):
        if not self.enabled:
            if self.force.abs().sum() > 0:
                self.force.zero_()
                self.torque.zero_()
                self._write()
            return
        for e in range(self.env.num_envs):
            if t_abs >= self.next_resample[e]:
                mag = self.rng.uniform(self.f_lo, self.f_hi)
                ang = self.rng.uniform(0, 2 * math.pi)   # 水平面随机方向
                self.force[e, 0, 0] = mag * math.cos(ang)
                self.force[e, 0, 1] = mag * math.sin(ang)
                self.force[e, 0, 2] = 0.0
                self.next_resample[e] = t_abs + self.rng.uniform(self.r_lo, self.r_hi)
        self._write()

    def _write(self):
        self.robot.set_external_force_and_torque(
            self.force, self.torque, body_ids=self.base_id, is_global=True
        )

    def current(self):
        return self.force[:, 0, :].cpu().numpy()


# ---------------------------------------------------------------------------
FOOT_EXPR = ".*FOOT"


def capture_step(env, cmd_np, disturber, t_abs, ep_time, foot_ids, sensor_foot_ids):
    """从 env 抓取一个 timestep 的全部记录字段（§4）。"""
    robot = env.scene["robot"]
    data = robot.data
    contact_sensor = env.scene["contact_forces"]
    imu = env.scene["imu"]
    foot_force = contact_sensor.data.net_forces_w[:, sensor_foot_ids]      # [N,4,3]
    contact = (foot_force.norm(dim=-1) > 1.0).float()                       # [N,4]
    rec = {
        "timestamp": np.full(env.num_envs, t_abs, dtype=np.float64),
        "episode_time": np.full(env.num_envs, ep_time, dtype=np.float64),
        "cmd_vel": cmd_np.copy(),                                           # [N,3]
        "base_position": data.root_pos_w.cpu().numpy(),
        "base_orientation": data.root_quat_w.cpu().numpy(),
        "base_linear_velocity_world": data.root_lin_vel_w.cpu().numpy(),
        "base_linear_velocity_body": data.root_lin_vel_b.cpu().numpy(),
        "base_angular_velocity": data.root_ang_vel_b.cpu().numpy(),
        "imu_linear_acceleration": imu.data.lin_acc_b.cpu().numpy(),
        "imu_angular_velocity": imu.data.ang_vel_b.cpu().numpy(),
        "projected_gravity": data.projected_gravity_b.cpu().numpy(),
        "joint_position": data.joint_pos.cpu().numpy(),
        "joint_velocity": data.joint_vel.cpu().numpy(),
        "applied_torque": data.applied_torque.cpu().numpy(),
        "feet_contact": contact.cpu().numpy(),
        "foot_velocity": data.body_lin_vel_w[:, foot_ids].cpu().numpy(),    # [N,4,3]
        "external_force": disturber.current(),
    }
    # execution e_t 与 residual r_t（§3）：body frame，与 command 同 frame
    e = np.stack([rec["base_linear_velocity_body"][:, 0],
                  rec["base_linear_velocity_body"][:, 1],
                  rec["base_angular_velocity"][:, 2]], axis=-1)
    rec["execution"] = e
    rec["residual"] = e - rec["cmd_vel"]
    return rec


def save_episode(out_dir, ep_id, meta, buffer, control_dt, output_hz):
    T = len(buffer["timestamp"])
    if T == 0:
        return None
    arrs = {k: np.stack(v, axis=0) for k, v in buffer.items()}              # [T,N=1,...]
    arrs = {k: v[:, 0] for k, v in arrs.items()}
    # 50Hz -> 20Hz 最近邻降采样（§4）
    dur = T * control_dt
    target_t = np.arange(0, dur, 1.0 / output_hz)
    idx = np.clip(np.round(target_t / control_dt).astype(int), 0, T - 1)
    arrs = {k: v[idx] for k, v in arrs.items()}
    arrs["timestamp"] = arrs["episode_time"]                                # 统一用 episode 内时间
    path = os.path.join(out_dir, f"ep_{ep_id:05d}.npz")
    np.savez_compressed(path, **arrs)
    meta_path = os.path.join(out_dir, f"ep_{ep_id:05d}.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return {"file": path, "meta_file": meta_path, "steps_20hz": len(idx), **meta}


# ---------------------------------------------------------------------------
def main():
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    seed = cfg["seed"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    out_root = cfg["output"]["dataset_dir"]
    os.makedirs(out_root, exist_ok=True)

    env_cfg = Go2ExecCollectEnvCfg()
    env_cfg.scene.num_envs = args.num_envs or cfg["sim"]["num_envs"]
    env_cfg.sim.dt = cfg["sim"]["control_dt"] / cfg["sim"]["decimation"]
    env_cfg.decimation = cfg["sim"]["decimation"]
    env_cfg.seed = seed

    env = gym.make("Isaac-Velocity-Flat-Unitree-Go2-v0", cfg=env_cfg)
    unwrapped: ManagerBasedRLEnv = env.unwrapped
    env = RslRlVecEnvWrapper(env)
    device = unwrapped.device
    control_dt = unwrapped.step_dt

    # locomotion controller（复用已训练 policy，§2）
    agent_cfg = UnitreeGo2FlatPPORunnerCfg()
    agent_cfg.device = device
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    ckpt = cfg["output"]["policy_checkpoint"]
    print(f"[INFO] 加载 locomotion controller: {ckpt}")
    runner.load(ckpt)
    policy = runner.get_inference_policy(device=device)

    # controller command limits（§2：运行时读取，不硬编码）
    cmd_term = unwrapped.command_manager.get_term("base_velocity")
    r = cmd_term.cfg.ranges
    limits = np.array([r.lin_vel_x[1], r.lin_vel_y[1], r.ang_vel_z[1]], dtype=np.float32)
    print(f"[INFO] controller command limits: vx=±{limits[0]}, vy=±{limits[1]}, wz=±{limits[2]}")

    robot = unwrapped.scene["robot"]
    foot_ids = robot.find_bodies(FOOT_EXPR)[0]
    sensor_foot_ids = unwrapped.scene["contact_forces"].find_bodies(FOOT_EXPR)[0]
    actuator = ActuatorScaler(unwrapped)
    scheduler = CommandScheduler(cfg, limits, device, unwrapped.num_envs)

    # ---- 组装 episode 队列 -------------------------------------------------
    cond_cfgs = cfg["conditions"]
    if args.conditions:
        cond_cfgs = {k: v for k, v in cond_cfgs.items() if k in args.conditions}
    index = []       # 全局 episode 索引
    ep_global = 0

    def episode_meta(ep_id, cond_name, cond, ep_type, probe_name=None):
        return {
            "episode_id": ep_id,
            "condition": cond_name,
            "episode_type": ep_type,                # random | probe
            "probe_name": probe_name,
            "friction": float(cond["friction"]) if np.isscalar(cond["friction"]) else list(cond["friction"]),
            "actuator_scale": float(cond["actuator_scale"]) if np.isscalar(cond["actuator_scale"]) else list(cond["actuator_scale"]),
            "disturbance": bool(cond.get("disturbance", False)),
            "termination_reason": None,
        }

    # 每个条件一轮仿真：物理参数在该轮内固定/逐 episode 采样
    for cond_name, cond in cond_cfgs.items():
        count = args.num_episodes or cond["count"]
        # ---------- probe 部分（§6）----------
        probe_jobs = []
        if cond_name in cfg.get("probe_conditions", {}):
            pcond = cfg["probe_conditions"][cond_name]
            for pname, segs in cfg["probes"].items():
                for rep in range(cfg.get("probe_repeats", 1)):
                    probe_jobs.append((pname, segs, pcond, rep))
        # 本条件队列: ("random"|pname, cond_dict)
        queue = [("random", cond)] * count + [(p[0], p[2]) for p in probe_jobs]
        probe_map = {p[0]: p[1] for p in probe_jobs}

        friction_c, act_c = cond["friction"], cond["actuator_scale"]
        dist_cfg = cond
        print(f"\n[INFO] === 条件 {cond_name}: {count} random + {len(probe_jobs)} probe episodes ===")

        # 物理参数：标量 -> 整轮固定；[lo,hi] -> 每个 episode 重采样
        def sample_param(p):
            return float(rng.uniform(*p)) if isinstance(p, list) else float(p)

        # 初始化本条件物理参数
        cur_friction = sample_param(friction_c)
        cur_act = sample_param(act_c)
        set_friction(unwrapped, cur_friction)
        actuator.set_scale(cur_act)
        disturber = Disturber(
            unwrapped,
            dist_cfg.get("disturbance_force_range_n", [30.0, 60.0]),
            dist_cfg.get("disturbance_resample_s", [0.5, 1.0]),
            rng,
        )
        disturber.enabled = bool(dist_cfg.get("disturbance", False))

        obs, _ = env.get_observations()
        unwrapped.reset()

        # 逐 env episode 状态
        buffers = [dict() for _ in range(unwrapped.num_envs)]
        metas = [None] * unwrapped.num_envs
        queue_idx = 0
        finished = 0
        total_target = len(queue)
        t_abs = 0.0

        def start_episode(env_id, step_t):
            nonlocal queue_idx, ep_global
            spec_type, c = queue[queue_idx % len(queue)]
            queue_idx += 1
            is_probe = spec_type != "random"
            total_s = 10.0 if is_probe else cfg["episode"]["max_length_s"]
            segs = probe_map.get(spec_type) if is_probe else None
            scheduler.assign(env_id, spec_type if is_probe else "random",
                             probe_segments=segs, total_s=total_s, dt=control_dt, step=step_t)
            metas[env_id] = episode_meta(ep_global, cond_name, c,
                                         "probe" if is_probe else "random",
                                         spec_type if is_probe else None)
            buffers[env_id] = {k: [] for k in (
                "timestamp", "episode_time", "cmd_vel", "base_position", "base_orientation",
                "base_linear_velocity_world", "base_linear_velocity_body", "base_angular_velocity",
                "imu_linear_acceleration", "imu_angular_velocity", "projected_gravity",
                "joint_position", "joint_velocity", "applied_torque", "feet_contact",
                "foot_velocity", "external_force", "execution", "residual")}
            ep_global += 1

        for e in range(unwrapped.num_envs):
            start_episode(e, 0)

        step = 0
        while finished < total_target:
            # 1) 写入本步命令与外力
            cmd_np = np.zeros((unwrapped.num_envs, 3), dtype=np.float32)
            for e in range(unwrapped.num_envs):
                ep_t = step * control_dt - scheduler.start_step[e] * control_dt
                cmd_np[e] = scheduler.command_at(e, ep_t)
            cmd_term.vel_command_b = torch.as_tensor(cmd_np, device=device)
            disturber.maybe_apply(t_abs, control_dt)

            # 2) 抓取当前状态（作为本 step 样本）
            ep_times = np.array([(step - scheduler.start_step[e]) * control_dt
                                 for e in range(unwrapped.num_envs)])
            rec = capture_step(unwrapped, cmd_np, disturber, t_abs, 0.0, foot_ids, sensor_foot_ids)
            rec["episode_time"] = ep_times

            # 3) policy 前向 + 环境步进
            with torch.inference_mode():
                actions = policy(obs)
            obs, _, dones, extras = env.step(actions)
            t_abs += control_dt
            step += 1

            # 4) 写入样本 & episode 收尾
            time_outs = extras.get("time_outs", torch.zeros(unwrapped.num_envs, dtype=torch.bool, device=device))
            for e in range(unwrapped.num_envs):
                ep_t = ep_times[e]
                sched_dur = scheduler.duration(e)
                if metas[e] is None:
                    continue
                if ep_t <= sched_dur:
                    for k in buffers[e]:
                        v = rec[k][e]
                        buffers[e][k].append(v.copy() if isinstance(v, np.ndarray) else v)
                done = bool(dones[e]) or ep_t >= sched_dur
                if done:
                    reason = "timeout" if bool(time_outs[e]) else (
                        "terminated" if bool(dones[e]) else "schedule_end")
                    n = len(buffers[e]["timestamp"])
                    dur_s = n * control_dt
                    keep_min = cfg["episode"]["terminated_keep_s"] if reason == "terminated" \
                        else cfg["episode"]["min_keep_s"]
                    if dur_s >= keep_min:
                        metas[e]["termination_reason"] = reason
                        metas[e]["duration_s"] = dur_s
                        metas[e]["env_id"] = e
                        cond_dir = os.path.join(out_root, cond_name)
                        os.makedirs(cond_dir, exist_ok=True)
                        info = save_episode(cond_dir, metas[e]["episode_id"], metas[e],
                                            buffers[e], control_dt, cfg["sim"]["output_hz"])
                        index.append(info)
                        finished += 1
                        if finished % 10 == 0 or finished == total_target:
                            print(f"[{cond_name}] {finished}/{total_target} episodes "
                                  f"(last: {reason}, {dur_s:.1f}s)")
                    else:
                        # 太短丢弃，但不计 finished，补一个新 episode
                        queue.append(("random", cond))
                        total_target += 1
                    metas[e] = None
                    if finished < total_target:
                        start_episode(e, step)

        # 条件结束：恢复外力清零
        disturber.enabled = False
        disturber.maybe_apply(t_abs, control_dt)

    with open(os.path.join(out_root, "index.json"), "w") as f:
        json.dump({"episodes": index, "config": cfg}, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] 采集完成: {len(index)} episodes -> {out_root}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
