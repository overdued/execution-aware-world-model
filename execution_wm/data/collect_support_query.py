"""V0.6 阶段B：support/query pilot 采集器（PRE_REGISTRATION 冻结协议）。

每个 (condition, anchor_seed, rep)：env.reset() -> 记录 anchor 状态 -> settle(1s) ->
query family(3s)，50Hz 原始记录。support：独立 session（reset -> 8s random dwell）。
摩擦写入后 readback 并记录；地面 material / combine rule 记录进 smoke 报告。

运行（先 smoke）:
    cd ~/IsaacLab && OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 ./isaaclab.sh -p \
        ~/cvpr_embed/execution_wm/data/collect_support_query.py \
        --config ~/cvpr_embed/execution_wm/configs/collect_v06_sq.yaml --smoke
    全量: 去掉 --smoke
"""
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--smoke", action="store_true")
parser.add_argument("--num-envs", type=int, default=None)
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

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

# 本地副本（不能 import collector.py——其模块级 argparse 会在 import 时解析 sys.argv）
FOOT_EXPR = ".*_foot"


def set_friction(env, friction):
    """直接写 PhysX material buffer（与 collector.py 相同写入路径，便于 readback 对照）。"""
    dyn = max(0.05, 0.7 * friction)
    robot = env.scene["robot"]
    view = robot.root_physx_view
    materials = view.get_material_properties()
    env_ids = torch.arange(env.num_envs, dtype=torch.long)
    materials[env_ids, :, 0] = friction
    materials[env_ids, :, 1] = dyn
    materials[env_ids, :, 2] = 0.0
    view.set_material_properties(materials, env_ids)


@configclass
class Go2SQEnvCfg(UnitreeGo2FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.events.physics_material = None
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


BUFFER_KEYS = (
    "timestamp", "episode_time", "phase", "cmd_vel", "base_position", "base_orientation",
    "base_linear_velocity_world", "base_linear_velocity_body", "base_angular_velocity",
    "imu_linear_acceleration", "imu_angular_velocity", "projected_gravity",
    "joint_position", "joint_velocity", "applied_torque", "feet_contact",
    "foot_velocity", "execution", "residual")


def capture(env, cmd_np, phase, t_abs, ep_t, foot_ids, sensor_foot_ids):
    robot = env.scene["robot"]
    data = robot.data
    contact_sensor = env.scene["contact_forces"]
    imu = env.scene["imu"]
    foot_force = contact_sensor.data.net_forces_w[:, sensor_foot_ids]
    contact = (foot_force.norm(dim=-1) > 1.0).float()
    rec = {
        "timestamp": np.full(env.num_envs, t_abs, dtype=np.float64),
        "episode_time": np.full(env.num_envs, ep_t, dtype=np.float64),
        "phase": np.full(env.num_envs, phase, dtype=np.int8),
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
    }
    e = np.stack([rec["base_linear_velocity_body"][:, 0],
                  rec["base_linear_velocity_body"][:, 1],
                  rec["base_angular_velocity"][:, 2]], axis=-1)
    rec["execution"] = e
    rec["residual"] = e - rec["cmd_vel"]
    return rec


def anchor_state(env):
    robot = env.scene["robot"]
    data = robot.data
    return {
        "anchor_base_position": data.root_pos_w.cpu().numpy(),
        "anchor_base_orientation": data.root_quat_w.cpu().numpy(),
        "anchor_base_lin_vel_w": data.root_lin_vel_w.cpu().numpy(),
        "anchor_base_ang_vel_w": data.root_ang_vel_w.cpu().numpy(),
        "anchor_joint_position": data.joint_pos.cpu().numpy(),
        "anchor_joint_velocity": data.joint_vel.cpu().numpy(),
    }


def friction_readback(env):
    view = env.scene["robot"].root_physx_view
    m = view.get_material_properties().cpu().numpy()   # (N, max_shapes, 3)
    return {"robot_static_friction_mean": float(m[..., 0].mean()),
            "robot_static_friction_min": float(m[..., 0].min()),
            "robot_static_friction_max": float(m[..., 0].max()),
            "robot_dynamic_friction_mean": float(m[..., 1].mean()),
            "robot_restitution_mean": float(m[..., 2].mean())}


def ground_material_info(env):
    """地面 material（cfg 级）+ combine rule（PhysX 默认 average，代码注明）。"""
    info = {"combine_rule": "physx_default_average（代码常量注明，未从引擎枚举读回）"}
    try:
        pm = env.cfg.scene.terrain.physics_material
        info.update({"ground_static_friction": float(pm.static_friction),
                     "ground_dynamic_friction": float(pm.dynamic_friction),
                     "ground_restitution": float(pm.restitution),
                     "source": "scene.terrain.physics_material cfg"})
    except Exception as e:  # noqa: BLE001
        info["source"] = f"unavailable: {e}"
    return info


def cmd_at(segs, t):
    acc = 0.0
    for dur, cmd in segs:
        if t < acc + dur:
            return cmd
        acc += dur
    return np.zeros(3, dtype=np.float32)


def jittered(segs, rng):
    out = []
    for dur, cmd in segs:
        d = max(0.2, float(dur) + float(rng.uniform(-0.1, 0.1)))
        out.append((d, cmd))
    return out


def make_support_segments(cfg, seed, limits):
    rng = np.random.default_rng(seed)
    segs = []
    for _ in range(cfg["support_segments"]):
        frac = rng.uniform(0.4, 0.8, size=3)
        signs = rng.choice([-1.0, 1.0], size=3)
        cmd = frac * signs * limits
        cmd[rng.random(3) < 0.25] = 0.0
        segs.append((cfg["support_seg_dur_s"], cmd.astype(np.float32)))
    return segs


def run_wave(env, unwrapped, policy, cmd_term, segs, settle_s, control_dt,
             foot_ids, sensor_foot_ids):
    """执行一条 settle+schedule，返回每 env 的 buffer/term_reason。"""
    n_steps = int(math.ceil((settle_s + sum(d for d, _ in segs)) / control_dt))
    buffers = [{k: [] for k in BUFFER_KEYS} for _ in range(unwrapped.num_envs)]
    alive = np.ones(unwrapped.num_envs, dtype=bool)
    term = ["schedule_end"] * unwrapped.num_envs
    obs = env.get_observations()
    t_abs = 0.0
    for step in range(n_steps):
        t = step * control_dt
        if t < settle_s:
            cmd_np = np.zeros((unwrapped.num_envs, 3), dtype=np.float32)
            phase = 0
        else:
            cmd_np = np.stack([cmd_at(segs, t - settle_s)] * unwrapped.num_envs)
            phase = 1
        cmd_term.vel_command_b = torch.as_tensor(cmd_np, device=unwrapped.device)
        rec = capture(unwrapped, cmd_np, phase, t_abs, t, foot_ids, sensor_foot_ids)
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
        t_abs += control_dt
        for e in range(unwrapped.num_envs):
            if not alive[e]:
                continue
            for k in BUFFER_KEYS:
                v = rec[k][e]
                buffers[e][k].append(v.copy() if isinstance(v, np.ndarray) else v)
            if bool(dones[e]):
                alive[e] = False
                term[e] = "terminated"
    return buffers, term


def save_episode_50hz(out_dir, ep_id, meta, buffer, extra):
    if len(buffer["timestamp"]) == 0:
        return None
    arrs = {k: np.stack(v, axis=0) for k, v in buffer.items()}
    arrs.update(extra)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"ep_{ep_id:05d}.npz")
    np.savez_compressed(path, **arrs)
    with open(os.path.join(out_dir, f"ep_{ep_id:05d}.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return {"file": path, "steps_50hz": len(arrs["timestamp"]), **meta}


def main():
    cfg = yaml.safe_load(open(args.config))
    seed = cfg["seed"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    env_cfg = Go2SQEnvCfg()
    env_cfg.scene.num_envs = args.num_envs or cfg["sim"]["num_envs"]
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
    r = cmd_term.cfg.ranges
    limits = np.array([r.lin_vel_x[1], r.lin_vel_y[1], r.ang_vel_z[1]], dtype=np.float32)
    foot_ids = unwrapped.scene["robot"].find_bodies(FOOT_EXPR)[0]
    sensor_foot_ids = unwrapped.scene["contact_forces"].find_bodies(FOOT_EXPR)[0]

    out_root = cfg["output"]["dataset_dir"]
    os.makedirs(out_root, exist_ok=True)
    ground_info = ground_material_info(unwrapped)

    families = {k: [(float(d), np.asarray(c, dtype=np.float32)) for d, c in segs]
                for k, segs in cfg["query_families"].items()}
    anchors = list(range(cfg["n_anchor"]))
    reps = list(range(cfg["reps"]))
    if args.smoke:
        families = {k: v for k, v in families.items() if k in cfg["smoke"]["families"]}
        anchors = cfg["smoke"]["anchors"]
        reps = [0]

    index, ep_global = [], 0
    readback_log = {}

    for cond_name, cond in cfg["conditions"].items():
        set_friction(unwrapped, float(cond["friction"]))
        rb = friction_readback(unwrapped)
        readback_log[cond_name] = {**rb, **ground_info,
                                   "written_friction": float(cond["friction"])}
        print(f"[readback] {cond_name}: wrote {cond['friction']} -> "
              f"static mean {rb['robot_static_friction_mean']:.4f} "
              f"dyn {rb['robot_dynamic_friction_mean']:.4f} | ground {ground_info}")

        # ---------- query waves ----------
        for ai in anchors:
            aseed = cfg["anchor_seed_base"] + cfg["anchor_seed_stride"] * ai
            for rep in reps:
                for fname, segs in families.items():
                    segs_run = segs
                    if rep == 1:
                        jrng = np.random.default_rng(aseed + 5000)
                        segs_run = jittered(segs, jrng)
                    torch.manual_seed(aseed)
                    env.reset()
                    anchors_np = anchor_state(unwrapped)
                    buffers, term = run_wave(env, unwrapped, policy, cmd_term, segs_run,
                                             cfg["episode"]["settle_s"], control_dt,
                                             foot_ids, sensor_foot_ids)
                    for e in range(unwrapped.num_envs):
                        dur = len(buffers[e]["timestamp"]) * control_dt
                        if dur < cfg["episode"]["min_keep_s"]:
                            continue
                        meta = {"episode_id": ep_global, "dataset_version": "d2",
                                "condition": cond_name, "episode_type": "query",
                                "command_family": fname,
                                "friction": float(cond["friction"]),
                                "actuator_scale": float(cond["actuator_scale"]),
                                "disturbance": False,
                                "termination_reason": term[e],
                                "duration_s": dur, "env_id": e,
                                "anchor_group": f"AQ{ai}", "anchor_seed": aseed,
                                "rep": rep, "seed": aseed,
                                "reset_state": "default_pose_zero_vel",
                                "approximate_paired": True,
                                "held_out_family": fname == cfg["held_out_family"],
                                "friction_readback": rb}
                        info = save_episode_50hz(
                            os.path.join(out_root, cond_name), ep_global, meta,
                            buffers[e], {k: v[e] for k, v in anchors_np.items()})
                        if info:
                            index.append(info)
                            ep_global += 1
                print(f"[{cond_name}] anchor AQ{ai} rep{rep} done ({len(index)} eps)")

        # ---------- support waves ----------
        if not args.smoke:
            for j in range(cfg["n_support_seed"]):
                sseed = cfg["support_seed_base"] + cfg["support_seed_stride"] * j
                segs = make_support_segments(cfg, sseed, limits)
                torch.manual_seed(sseed)
                env.reset()
                anchors_np = anchor_state(unwrapped)
                buffers, term = run_wave(env, unwrapped, policy, cmd_term, segs,
                                         0.0, control_dt, foot_ids, sensor_foot_ids)
                for e in range(unwrapped.num_envs):
                    dur = len(buffers[e]["timestamp"]) * control_dt
                    if dur < cfg["episode"]["min_keep_s"]:
                        continue
                    meta = {"episode_id": ep_global, "dataset_version": "d2",
                            "condition": cond_name, "episode_type": "support",
                            "command_family": "random_dwell",
                            "friction": float(cond["friction"]),
                            "actuator_scale": float(cond["actuator_scale"]),
                            "disturbance": False,
                            "termination_reason": term[e],
                            "duration_s": dur, "env_id": e,
                            "support_seed": sseed, "seed": sseed,
                            "reset_state": "default_pose_zero_vel",
                            "friction_readback": rb}
                    info = save_episode_50hz(
                        os.path.join(out_root, cond_name), ep_global, meta,
                        buffers[e], {k: v[e] for k, v in anchors_np.items()})
                    if info:
                        index.append(info)
                        ep_global += 1
            print(f"[{cond_name}] support waves done ({len(index)} eps total)")

    with open(os.path.join(out_root, "index.json"), "w") as f:
        json.dump({"episodes": index, "config": cfg,
                   "friction_readback": readback_log}, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] 完成: {len(index)} episodes -> {out_root}")
    env.close()


if __name__ == "__main__":
    main()
    os._exit(0)  # simulation_app.close() 在 headless 下可能挂起（CLAUDE.md 已知坑）
