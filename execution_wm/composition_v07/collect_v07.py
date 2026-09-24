"""V0.7 Stage 2：结构化命令覆盖采集器（PRE_REGISTRATION_V07 冻结协议）。

每个 (group, friction, script) 一条 episode：
  env.reset(seed=group.reset_seed) -> 记录 anchor 状态 -> settle(1.0s) -> schedule(12.6s)
50 Hz 原始记录，逐 tick 保存 u_requested / u_consumed / joint_command / 物理量。

运行（先 smoke）:
  cd ~/IsaacLab && OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 ./isaaclab.sh -p \
    ~/cvpr_embed-v07/execution_wm/composition_v07/collect_v07.py \
    --plan ~/cvpr_embed-v07/results/v0_7_composition/prereg/split_plan.json \
    --out-root /media/hdd1/yuhang/datasets/execution_wm/v0_7 [--smoke]
"""
import argparse
import json
import math
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--plan", required=True)
ap.add_argument("--out-root", required=True)
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--num-envs", type=int, default=1)
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.headless = True
simulation_app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ)                      # 本文件比 collector.py 深一层 -> PROJ 即仓库根
from execution_wm.composition_v07 import cells as C  # noqa: E402

import gymnasium as gym  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
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

FOOT_EXPR = ".*_foot"
POLICY_CKPT = os.path.expanduser(
    "~/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-09-20_22-52-26/model_299.pt")
DATA_VERSION = "v07.1"


def set_friction(env, friction):
    dyn = max(0.05, 0.7 * friction)
    view = env.scene["robot"].root_physx_view
    mats = view.get_material_properties()
    eids = torch.arange(env.num_envs, dtype=torch.long)
    mats[eids, :, 0] = friction
    mats[eids, :, 1] = dyn
    mats[eids, :, 2] = 0.0
    view.set_material_properties(mats, eids)


@configclass
class Go2V07EnvCfg(UnitreeGo2FlatEnvCfg):
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
    "physics_tick", "control_tick", "sim_time", "wall_time",
    "cmd_vel", "u_consumed", "joint_command",
    "base_position", "base_orientation", "base_linear_velocity_world",
    "base_linear_velocity_body", "base_angular_velocity",
    "imu_linear_acceleration", "imu_angular_velocity", "projected_gravity",
    "joint_position", "joint_velocity", "applied_torque", "feet_contact",
    "foot_velocity", "execution", "residual", "phase")


def capture(env, cmd_np, u_consumed_np, joint_cmd_np, phase, sim_t, wall_t,
            physics_tick, ctrl_tick, foot_ids, sensor_foot_ids):
    robot = env.scene["robot"]
    d = robot.data
    contact = (env.scene["contact_forces"].data.net_forces_w[:, sensor_foot_ids].norm(dim=-1)
               > 1.0).float()
    imu = env.scene["imu"]
    rec = {
        "physics_tick": np.full(env.num_envs, physics_tick, dtype=np.int64),
        "control_tick": np.full(env.num_envs, ctrl_tick, dtype=np.int64),
        "sim_time": np.full(env.num_envs, sim_t, dtype=np.float64),
        "wall_time": np.full(env.num_envs, wall_t, dtype=np.float64),
        "cmd_vel": cmd_np.copy(),
        "u_consumed": u_consumed_np.copy(),
        "joint_command": joint_cmd_np.copy(),
        "base_position": d.root_pos_w.cpu().numpy(),
        "base_orientation": d.root_quat_w.cpu().numpy(),
        "base_linear_velocity_world": d.root_lin_vel_w.cpu().numpy(),
        "base_linear_velocity_body": d.root_lin_vel_b.cpu().numpy(),
        "base_angular_velocity": d.root_ang_vel_b.cpu().numpy(),
        "imu_linear_acceleration": imu.data.lin_acc_b.cpu().numpy(),
        "imu_angular_velocity": imu.data.ang_vel_b.cpu().numpy(),
        "projected_gravity": d.projected_gravity_b.cpu().numpy(),
        "joint_position": d.joint_pos.cpu().numpy(),
        "joint_velocity": d.joint_vel.cpu().numpy(),
        "applied_torque": d.applied_torque.cpu().numpy(),
        "feet_contact": contact.cpu().numpy(),
        "foot_velocity": d.body_lin_vel_w[:, foot_ids].cpu().numpy(),
        "phase": np.full(env.num_envs, phase, dtype=np.int8),
    }
    e = np.stack([rec["base_linear_velocity_body"][:, 0],
                  rec["base_linear_velocity_body"][:, 1],
                  rec["base_angular_velocity"][:, 2]], axis=-1)
    rec["execution"] = e
    rec["residual"] = e - rec["cmd_vel"]
    return rec


def anchor_state(env):
    d = env.scene["robot"].data
    return {"anchor_base_position": d.root_pos_w.cpu().numpy(),
            "anchor_base_orientation": d.root_quat_w.cpu().numpy(),
            "anchor_base_lin_vel_w": d.root_lin_vel_w.cpu().numpy(),
            "anchor_base_ang_vel_w": d.root_ang_vel_w.cpu().numpy(),
            "anchor_joint_position": d.joint_pos.cpu().numpy(),
            "anchor_joint_velocity": d.joint_vel.cpu().numpy()}


def material_readback(env):
    m = env.scene["robot"].root_physx_view.get_material_properties().cpu().numpy()
    return {"robot_static_mean": float(m[..., 0].mean()),
            "robot_dynamic_mean": float(m[..., 1].mean()),
            "robot_static_min": float(m[..., 0].min()),
            "robot_static_max": float(m[..., 0].max()),
            "n_shapes_per_env": int(m.shape[1])}


def run_episode(env, unwrapped, policy, cmd_term, cmd_slice, segments, settle_s,
                control_dt, foot_ids, sensor_foot_ids):
    total = settle_s + sum(s["duration_s"] for s in segments)
    n_steps = int(math.ceil(total / control_dt))
    buffers = [{k: [] for k in BUFFER_KEYS}]
    alive = True
    term = "schedule_end"
    obs_td = env.get_observations()
    wall0 = time.time()
    for step in range(n_steps):
        t = step * control_dt
        if t < settle_s:
            cmd_np = np.zeros((1, 3), dtype=np.float32); phase = 0
        else:
            tq = t - settle_s
            acc, cmd_v = 0.0, [0.0, 0.0, 0.0]
            for s in segments:
                if tq < acc + s["duration_s"]:
                    cmd_v = s["cmd_values"]; break
                acc += s["duration_s"]
            cmd_np = np.array([cmd_v], dtype=np.float32); phase = 1
        cmd_term.vel_command_b = torch.as_tensor(cmd_np, device=unwrapped.device)
        # policy 实际消费的 observation（在 policy 调用前取，不能被 step 覆盖）
        u_consumed = obs_td["policy"][:, cmd_slice[0]:cmd_slice[1]].cpu().numpy().copy()
        with torch.inference_mode():
            actions = policy(obs_td)
        joint_cmd = actions.cpu().numpy().copy()
        rec = capture(unwrapped, cmd_np, u_consumed, joint_cmd, phase,
                      (step + 1) * control_dt, time.time() - wall0,
                      step * unwrapped.cfg.decimation, step,
                      foot_ids, sensor_foot_ids)
        obs_td, _, dones, _ = env.step(actions)
        if not alive:
            continue
        for k in BUFFER_KEYS:
            v = rec[k][0]
            buffers[0][k].append(v.copy() if isinstance(v, np.ndarray) else v)
        if bool(dones[0]):
            alive = False
            term = "terminated"
    return buffers, term


def main():
    plan = json.load(open(args.plan))
    os.makedirs(args.out_root, exist_ok=True)
    seed = 1000
    torch.manual_seed(seed)
    env_cfg = Go2V07EnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = seed
    env = gym.make("Isaac-Velocity-Flat-Unitree-Go2-v0", cfg=env_cfg)
    uw = env.unwrapped
    env = RslRlVecEnvWrapper(env)
    device = uw.device
    control_dt = uw.step_dt
    agent_cfg = UnitreeGo2FlatPPORunnerCfg()
    agent_cfg.device = device
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(POLICY_CKPT)
    policy = runner.get_inference_policy(device=device)

    cmd_term = uw.command_manager.get_term("base_velocity")
    om = uw.observation_manager
    terms = om.active_terms["policy"]
    dims = [int(np.prod(om.group_obs_term_dim["policy"][i])) for i in range(len(terms))]
    cum, sl = 0, {}
    for t, d in zip(terms, dims):
        sl[t] = (cum, cum + d); cum += d
    cmd_slice = sl["velocity_commands"]
    foot_ids = uw.scene["robot"].find_bodies(FOOT_EXPR)[0]
    sensor_foot_ids = uw.scene["contact_forces"].find_bodies(FOOT_EXPR)[0]

    import hashlib
    plan_hash = hashlib.sha256(open(args.plan, "rb").read()).hexdigest()
    ctrl_hash = hashlib.sha256(open(POLICY_CKPT, "rb").read()).hexdigest()[:16]

    frictions = list(plan["frictions"].items())
    groups = plan["groups"]
    if args.smoke:
        groups = groups[:3]
        frictions = frictions[:1]
        for g in groups:
            g["_scripts_subset"] = 4
    index, ep_global = [], 0
    readback_log = {}
    t_start = time.time()

    for g in groups:
        gid = g["group_id"]
        scripts = plan["scripts"][gid]
        if args.smoke:
            scripts = scripts[:g["_scripts_subset"]]
        for fname, fval in frictions:
            set_friction(uw, float(fval))
            readback_log.setdefault(fname, material_readback(uw))
            for sc in scripts:
                segs = [{"cell": s["cell"], "duration_s": s["duration_s"],
                         "cmd_values": _cell_values(s["cell"])} for s in sc["segments"]]
                torch.manual_seed(int(g["reset_seed"]))
                env.reset()
                anchors = anchor_state(uw)
                buffers, term = run_episode(env, uw, policy, cmd_term, cmd_slice, segs,
                                            plan["episodes"]["settle_s"] if "episodes" in plan
                                            else 1.0, control_dt, foot_ids, sensor_foot_ids)
                dur = len(buffers[0]["sim_time"]) * control_dt
                meta = {
                    "track_id": "cvpr_v07", "data_version": DATA_VERSION,
                    "controller_hash": ctrl_hash, "scene_id": "go2_flat_default",
                    "anchor_group_id": gid, "episode_id": ep_global,
                    "reset_seed": int(g["reset_seed"]),
                    "command_seed": int(g["reset_seed"]) + 1000,
                    "split": g["split"],
                    "coverage_regime": sc.get("regime", sc.get("level", "eval")),
                    "command_cell_ids": [s["cell"] for s in sc["segments"]],
                    "timing_template": sc["timing_template"],
                    "family_ids": sorted({_cell_type(s["cell"]) for s in sc["segments"]}),
                    "script_id": sc["script_id"], "condition": fname,
                    "friction_written_static": float(fval),
                    "friction_written_dynamic": max(0.05, 0.7 * float(fval)),
                    "friction_readback": readback_log[fname],
                    "ground_combine_mode": "multiply",
                    "termination_reason": term, "duration_s": dur,
                    "plan_hash": plan_hash, "allow_real_robot": False,
                }
                sub = os.path.join(args.out_root, g["split"], gid, fname)
                os.makedirs(sub, exist_ok=True)
                arrs = {k: np.stack(v, axis=0) for k, v in buffers[0].items()}
                arrs.update({k: v for k, v in anchors.items()})
                path = os.path.join(sub, f"ep_{ep_global:05d}.npz")
                np.savez_compressed(path, **arrs)
                with open(os.path.join(sub, f"ep_{ep_global:05d}.json"), "w") as f:
                    json.dump(meta, f, indent=2, ensure_ascii=False)
                index.append({**meta, "file": path,
                              "steps_50hz": int(len(arrs["sim_time"]))})
                ep_global += 1
            print(f"[{gid}/{fname}] {len(scripts)} scripts done ({ep_global} eps, "
                  f"{(time.time()-t_start)/60:.1f} min)", flush=True)

    with open(os.path.join(args.out_root, "index.json"), "w") as f:
        json.dump({"episodes": index, "plan": args.plan, "plan_hash": plan_hash,
                   "controller_hash": ctrl_hash, "data_version": DATA_VERSION,
                   "friction_readback": readback_log,
                   "smoke": bool(args.smoke)}, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] 完成: {ep_global} episodes -> {args.out_root}", flush=True)
    env.close()


def _cell_values(cid):
    return C.cell_values(cid)


def _cell_type(cid):
    return C.cell_type(cid)


if __name__ == "__main__":
    main()
    os._exit(0)
