"""Controlled probe 补采（03.1 v2 §6）。

V0 的 probe 存在两个缺口：(1) friction_vlow(0.15) 没有 probe episode；
(2) probe_2_accel_decel 在非 normal 条件下缺失（采集过采样时被长 probe 挤出）。
本脚本按 §6 原则补采：same reset pose / same initial velocity / same command sequence /
same seed set，只有物理条件变化。

每个 (condition, probe) 一波：torch.manual_seed 固定 -> env.reset() 得到可复现初始状态，
num_envs 个并行副本即多个 seed。每波结束按 env 保存 episode。

运行:
    cd ~/IsaacLab && OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 ./isaaclab.sh -p \
        ~/cvpr_embed/execution_wm/data/collect_controlled_probes.py \
        --config ~/cvpr_embed/execution_wm/configs/collect_v0.yaml \
        --out-dataset /media/hdd1/yuhang/datasets/execution_wm/v0_controlled_probes \
        --manifest /media/hdd1/yuhang/checkpoints/execution_wm/v0/context_swap_v2/extra_data/controlled_probe_manifest.csv
"""
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="collect_v0.yaml（复用 probes/sim/policy）")
parser.add_argument("--out-dataset", required=True)
parser.add_argument("--manifest", required=True)
parser.add_argument("--num-envs", type=int, default=8)
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import csv  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(PROJECT_ROOT))

import gymnasium as gym  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
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

# 与 collector.py 相同的扰动条件（friction_low=0.3 与 V0 probe 条件 low_friction 一致）
CONDITIONS = {
    "normal": 1.0,
    "friction_low": 0.3,
    "friction_vlow": 0.15,
}
FOOT_EXPR = ".*_foot"
# probe -> 固定种子偏移（不用 hash()，它随进程随机）
PROBE_SEED_OFFSET = {"probe_1_straight": 101, "probe_2_accel_decel": 202,
                     "probe_3_turn": 303, "probe_4_lateral": 404}


@configclass
class Go2ControlledProbeEnvCfg(UnitreeGo2FlatEnvCfg):
    """与 v0 collector 相同的环境裁剪（DR 关闭、命令直写、IMU）。"""

    def __post_init__(self):
        super().__post_init__()
        self.events.randomize_rigid_body_material = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        cmd = self.commands.base_velocity
        cmd.heading_command = False
        cmd.rel_heading_envs = 0.0
        cmd.rel_standing_envs = 0.0
        cmd.resampling_time_range = (1.0e6, 1.0e6)
        cmd.debug_vis = False
        self.episode_length_s = 1.0e6
        self.scene.imu = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/base")


def set_friction(env, friction):
    dyn = max(0.05, 0.7 * friction)
    robot = env.scene["robot"]
    view = robot.root_physx_view
    materials = view.get_material_properties()
    env_ids = torch.arange(env.num_envs, dtype=torch.long)
    materials[env_ids, :, 0] = friction
    materials[env_ids, :, 1] = dyn
    materials[env_ids, :, 2] = 0.0
    view.set_material_properties(materials, env_ids)


def capture_step(env, cmd_np, foot_ids, sensor_foot_ids, ep_t):
    robot = env.scene["robot"]
    data = robot.data
    contact_sensor = env.scene["contact_forces"]
    imu = env.scene["imu"]
    foot_force = contact_sensor.data.net_forces_w[:, sensor_foot_ids]
    contact = (foot_force.norm(dim=-1) > 1.0).float()
    rec = {
        "timestamp": np.full(env.num_envs, ep_t, dtype=np.float64),
        "episode_time": np.full(env.num_envs, ep_t, dtype=np.float64),
        "cmd_vel": cmd_np.copy(),
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
        "foot_velocity": data.body_lin_vel_w[:, foot_ids].cpu().numpy(),
        "external_force": np.zeros((env.num_envs, 3), dtype=np.float32),
    }
    e = np.stack([rec["base_linear_velocity_body"][:, 0],
                  rec["base_linear_velocity_body"][:, 1],
                  rec["base_angular_velocity"][:, 2]], axis=-1)
    rec["execution"] = e
    rec["residual"] = e - rec["cmd_vel"]
    return rec


BUFFER_KEYS = ("timestamp", "episode_time", "cmd_vel", "base_position", "base_orientation",
               "base_linear_velocity_world", "base_linear_velocity_body", "base_angular_velocity",
               "imu_linear_acceleration", "imu_angular_velocity", "projected_gravity",
               "joint_position", "joint_velocity", "applied_torque", "feet_contact",
               "foot_velocity", "external_force", "execution", "residual")


def save_episode(out_dir, ep_id, meta, buffer, control_dt, output_hz):
    T = len(buffer["timestamp"])
    if T == 0:
        return None
    arrs = {k: np.stack(v, axis=0) for k, v in buffer.items()}
    dur = T * control_dt
    target_t = np.arange(0, dur, 1.0 / output_hz)
    idx = np.clip(np.round(target_t / control_dt).astype(int), 0, T - 1)
    arrs = {k: v[idx] for k, v in arrs.items()}
    arrs["timestamp"] = arrs["episode_time"]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"ep_{ep_id:05d}.npz")
    np.savez_compressed(path, **arrs)
    meta_path = os.path.join(out_dir, f"ep_{ep_id:05d}.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return path


def probe_cmd_at(segs, t):
    acc = 0.0
    for dur, cmd in segs:
        if t < acc + dur:
            return cmd
        acc += dur
    return np.zeros(3, dtype=np.float32)


def main():
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    seed = cfg["seed"]
    torch.manual_seed(seed)
    os.makedirs(args.out_dataset, exist_ok=True)
    os.makedirs(os.path.dirname(args.manifest), exist_ok=True)

    env_cfg = Go2ControlledProbeEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.dt = cfg["sim"]["control_dt"] / cfg["sim"]["decimation"]
    env_cfg.decimation = cfg["sim"]["decimation"]
    env_cfg.seed = seed

    env = gym.make("Isaac-Velocity-Flat-Unitree-Go2-v0", cfg=env_cfg)
    unwrapped: ManagerBasedRLEnv = env.unwrapped
    env = RslRlVecEnvWrapper(env)
    device = unwrapped.device
    control_dt = unwrapped.step_dt

    agent_cfg = UnitreeGo2FlatPPORunnerCfg()
    agent_cfg.device = device
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(cfg["output"]["policy_checkpoint"])
    policy = runner.get_inference_policy(device=device)

    cmd_term = unwrapped.command_manager.get_term("base_velocity")
    robot = unwrapped.scene["robot"]
    foot_ids = robot.find_bodies(FOOT_EXPR)[0]
    sensor_foot_ids = unwrapped.scene["contact_forces"].find_bodies(FOOT_EXPR)[0]

    probes = {name: [(float(d), np.asarray(c, dtype=np.float32)) for d, c in segs]
              for name, segs in cfg["probes"].items()}
    manifest = []
    ep_global = 0

    for cond_name, friction in CONDITIONS.items():
        set_friction(unwrapped, friction)
        for pname, segs in probes.items():
            probe_dur = sum(d for d, _ in segs)
            # 固定种子 -> 每次 reset 初始状态可复现，跨条件一致（§6 same seed set）
            wave_seed = seed + PROBE_SEED_OFFSET.get(pname, 0)
            torch.manual_seed(wave_seed)
            env.reset()
            obs = env.get_observations()
            n_steps = int(math.ceil(probe_dur / control_dt))
            buffers = [{k: [] for k in BUFFER_KEYS} for _ in range(unwrapped.num_envs)]
            alive = np.ones(unwrapped.num_envs, dtype=bool)
            term_reason = ["schedule_end"] * unwrapped.num_envs
            for step in range(n_steps):
                t = step * control_dt
                cmd_np = np.stack([probe_cmd_at(segs, t)] * unwrapped.num_envs).astype(np.float32)
                cmd_term.vel_command_b = torch.as_tensor(cmd_np, device=device)
                rec = capture_step(unwrapped, cmd_np, foot_ids, sensor_foot_ids, t)
                with torch.inference_mode():
                    actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                for e in range(unwrapped.num_envs):
                    if not alive[e]:
                        continue
                    for k in BUFFER_KEYS:
                        v = rec[k][e]
                        buffers[e][k].append(v.copy() if isinstance(v, np.ndarray) else v)
                    if bool(dones[e]):
                        alive[e] = False
                        term_reason[e] = "terminated"
            cond_dir = os.path.join(args.out_dataset, cond_name)
            for e in range(unwrapped.num_envs):
                dur_s = len(buffers[e]["timestamp"]) * control_dt
                meta = {
                    "episode_id": ep_global,
                    "condition": cond_name,
                    "episode_type": "probe",
                    "probe_name": pname,
                    "friction": friction,
                    "actuator_scale": 1.0,
                    "disturbance": False,
                    "termination_reason": term_reason[e],
                    "duration_s": dur_s,
                    "seed": wave_seed,
                    "reset_state": "default_pose_zero_vel",
                    "controlled": True,
                }
                save_episode(cond_dir, ep_global, meta, buffers[e], control_dt,
                             cfg["sim"]["output_hz"])
                manifest.append({"episode_id": ep_global, "probe_name": pname,
                                 "condition": cond_name, "friction": friction,
                                 "seed": wave_seed, "reset_state": meta["reset_state"],
                                 "duration_s": round(dur_s, 3),
                                 "termination_reason": term_reason[e]})
                ep_global += 1
            n_term = sum(1 for r in term_reason if r == "terminated")
            print(f"[ctrl] {cond_name}/{pname}: {unwrapped.num_envs} eps "
                  f"(terminated mid-probe: {n_term})", flush=True)

    with open(args.manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)
    print(f"[ctrl] 完成: {ep_global} episodes -> {args.out_dataset}", flush=True)
    print(f"[ctrl] manifest -> {args.manifest}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    os._exit(0)  # simulation_app.close() 已知挂起，直接退
