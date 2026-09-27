"""V0.8 Stage B3：同步 RGB + 物理采集器（冻结 V-JEPA 2 smoke / pilot 用）。

在 V0.7 已验证采集器基础上仅增加一个固定机载相机：
  - CameraCfg：256x256、update_period=0.05（20Hz）、data_types=["rgb"]、固定内外参；
  - 相机为传感器 prim：无碰撞体、不参与质量（A/B 物理一致性由 check_camera_invariance 核验）；
  - 逐 control tick 记录 camera frame counter 与读取时刻 sim_time/wall_time —— 真实 render
    tick 与 sensor age 由记录数据计算，不伪造时间；
  - 相机世界位姿逐帧保存为 label-only 字段（label_camera_pos_w / label_camera_quat_w_ros），
    绝不进模型输入。

物理通道与 V0.7 完全同口径（同一 controller、同一计划格式、同一 50Hz 记录）。

运行（smoke ≤12 条）:
  cd ~/IsaacLab && OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 ./isaaclab.sh -p \
    ~/cvpr_embed-v08/execution_wm/v08_visual/collect_v08.py \
    --plan ~/cvpr_embed-v08/results/v0_7_composition/prereg/split_plan.json \
    --out-root /media/hdd1/yuhang/datasets/execution_wm/v0_8_smoke --smoke
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
ap.add_argument("--resume", action="store_true")
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.headless = True
simulation_app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ)
from execution_wm.composition_v07 import cells as C  # noqa: E402

import gymnasium as gym  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import (  # noqa: E402
    UnitreeGo2FlatPPORunnerCfg,
)
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from execution_wm.v08_visual.env_cfg_v08 import (  # noqa: E402
    CAM_PERIOD, CAM_POS, CAM_RES, CAM_ROT_WORLD, Go2V08CamEnvCfg,
)

FOOT_EXPR = ".*_foot"
POLICY_CKPT = os.path.expanduser(
    "~/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-09-20_22-52-26/model_299.pt")
DATA_VERSION = "v08.0"


def set_friction(env, friction):
    dyn = max(0.05, 0.7 * friction)
    view = env.scene["robot"].root_physx_view
    mats = view.get_material_properties()
    eids = torch.arange(env.num_envs, dtype=torch.long)
    mats[eids, :, 0] = friction
    mats[eids, :, 1] = dyn
    mats[eids, :, 2] = 0.0
    view.set_material_properties(mats, eids)


BUFFER_KEYS = (
    "physics_tick", "control_tick", "sim_time", "wall_time",
    "cmd_vel", "u_consumed", "joint_command",
    "base_position", "base_orientation", "base_linear_velocity_world",
    "base_linear_velocity_body", "base_angular_velocity",
    "imu_linear_acceleration", "imu_angular_velocity", "projected_gravity",
    "joint_position", "joint_velocity", "applied_torque", "feet_contact",
    "foot_velocity", "execution", "residual", "phase")

RGB_KEYS = ("rgb_frame_counter", "rgb_ctrl_tick_first_seen", "rgb_sim_time_first_seen",
            "rgb_wall_time_first_seen", "label_camera_pos_w", "label_camera_quat_w_ros")


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
    rgb_frames, rgb_meta = [], {k: [] for k in RGB_KEYS}
    cam = unwrapped.scene["camera"]
    last_frame = None
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
        u_consumed = obs_td["policy"][:, cmd_slice[0]:cmd_slice[1]].cpu().numpy().copy()
        with torch.inference_mode():
            actions = policy(obs_td)
        joint_cmd = actions.cpu().numpy().copy()
        rec = capture(unwrapped, cmd_np, u_consumed, joint_cmd, phase,
                      (step + 1) * control_dt, time.time() - wall0,
                      step * unwrapped.cfg.decimation, step,
                      foot_ids, sensor_foot_ids)
        obs_td, _, dones, _ = env.step(actions)
        # ---- RGB：step 之后读相机（渲染发生在 step 内的物理子步）。
        # 注意：Isaac Lab 相机为懒更新——必须每步访问 cam.data 才会触发 outdated 缓冲刷新；
        # 只读 cam.frame 不触发更新（曾导致整段 episode 只有 2 帧）。
        sim_t_read = (step + 1) * control_dt
        wall_t_read = time.time() - wall0
        cam_data = cam.data                              # 访问即触发懒更新
        fc = int(cam.frame[0].item())
        if last_frame is None or fc != last_frame:
            rgb = cam_data.output["rgb"][0].cpu().numpy()   # [H,W,3|4] float 0-1 或 uint8
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb[..., :3] * 255.0, 0, 255).astype(np.uint8)
            else:
                rgb = rgb[..., :3]
            rgb_frames.append(rgb)
            rgb_meta["rgb_frame_counter"].append(fc)
            rgb_meta["rgb_ctrl_tick_first_seen"].append(step)
            rgb_meta["rgb_sim_time_first_seen"].append(sim_t_read)
            rgb_meta["rgb_wall_time_first_seen"].append(wall_t_read)
            rgb_meta["label_camera_pos_w"].append(cam_data.pos_w[0].cpu().numpy().copy())
            rgb_meta["label_camera_quat_w_ros"].append(
                cam_data.quat_w_ros[0].cpu().numpy().copy())
            last_frame = fc
        if not alive:
            continue
        for k in BUFFER_KEYS:
            v = rec[k][0]
            buffers[0][k].append(v.copy() if isinstance(v, np.ndarray) else v)
        if bool(dones[0]):
            alive = False
            term = "terminated"
    return buffers, term, rgb_frames, rgb_meta


def main():
    plan = json.load(open(args.plan))
    os.makedirs(args.out_root, exist_ok=True)
    seed = 1000
    torch.manual_seed(seed)
    env_cfg = Go2V08CamEnvCfg()
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

    cam = uw.scene["camera"]
    cam_intr = cam.data.intrinsic_matrices[0].cpu().numpy()
    print(f"[cam] intrinsics:\n{cam_intr}", flush=True)

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
    index, ep_global, done = [], 0, set()
    readback_log = {}
    if args.resume:
        import glob as _glob
        for jf in sorted(_glob.glob(os.path.join(args.out_root, "*", "*", "*", "ep_*.json"))):
            m = json.load(open(jf))
            npz = jf.replace(".json", ".npz")
            if not os.path.exists(npz):
                print(f"[resume] 跳过不完整: {jf}"); continue
            n_steps = int(len(np.load(npz)["sim_time"]))
            index.append({**m, "file": npz, "steps_50hz": n_steps})
            ep_global = max(ep_global, int(m["episode_id"]) + 1)
            done.add((m["anchor_group_id"], m["condition"], m["script_id"]))
        print(f"[resume] 已有 {len(index)} episodes，跳过 {len(done)} 个", flush=True)
        for e in index:
            if "friction_readback" in e and "condition" in e:
                readback_log.setdefault(e["condition"], e["friction_readback"])
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
                if (gid, fname, sc["script_id"]) in done:
                    continue
                segs = [{"cell": s["cell"], "duration_s": s["duration_s"],
                         "cmd_values": C.cell_values(s["cell"])} for s in sc["segments"]]
                torch.manual_seed(int(g["reset_seed"]))
                env.reset()
                anchors = anchor_state(uw)
                buffers, term, rgb_frames, rgb_meta = run_episode(
                    env, uw, policy, cmd_term, cmd_slice, segs,
                    plan["episodes"]["settle_s"] if "episodes" in plan else 1.0,
                    control_dt, foot_ids, sensor_foot_ids)
                dur = len(buffers[0]["sim_time"]) * control_dt
                meta = {
                    "track_id": "cvpr_v08", "data_version": DATA_VERSION,
                    "controller_hash": ctrl_hash, "scene_id": "go2_flat_default",
                    "anchor_group_id": gid, "episode_id": ep_global,
                    "reset_seed": int(g["reset_seed"]),
                    "command_seed": int(g["reset_seed"]) + 1000,
                    "split": g["split"],
                    "coverage_regime": sc.get("regime", sc.get("level", "eval")),
                    "command_cell_ids": [s["cell"] for s in sc["segments"]],
                    "timing_template": sc["timing_template"],
                    "family_ids": sorted({C.cell_type(s["cell"]) for s in sc["segments"]}),
                    "script_id": sc["script_id"], "condition": fname,
                    "friction_written_static": float(fval),
                    "friction_written_dynamic": max(0.05, 0.7 * float(fval)),
                    "friction_readback": readback_log[fname],
                    "ground_combine_mode": "multiply",
                    "termination_reason": term, "duration_s": dur,
                    "plan_hash": plan_hash, "allow_real_robot": False,
                    "camera": {
                        "prim": "{ENV_REGEX_NS}/Robot/base/front_cam",
                        "resolution": list(CAM_RES), "update_period_s": CAM_PERIOD,
                        "offset_pos_base": list(CAM_POS),
                        "offset_rot_world_conv_wxyz": list(CAM_ROT_WORLD),
                        "intrinsic_matrix": cam_intr.tolist(),
                        "rgb_dtype": "uint8", "rgb_channels": "RGB",
                    },
                    "n_rgb_frames": len(rgb_frames),
                }
                sub = os.path.join(args.out_root, g["split"], gid, fname)
                os.makedirs(sub, exist_ok=True)
                arrs = {k: np.stack(v, axis=0) for k, v in buffers[0].items()}
                arrs.update({k: v for k, v in anchors.items()})
                arrs["rgb"] = np.stack(rgb_frames, axis=0)
                for k in RGB_KEYS:
                    arrs[k] = np.stack([np.asarray(v) for v in rgb_meta[k]], axis=0)
                path = os.path.join(sub, f"ep_{ep_global:05d}.npz")
                np.savez_compressed(path, **arrs)
                with open(os.path.join(sub, f"ep_{ep_global:05d}.json"), "w") as f:
                    json.dump(meta, f, indent=2, ensure_ascii=False)
                index.append({**meta, "file": path,
                              "steps_50hz": int(len(arrs["sim_time"]))})
                ep_global += 1
                print(f"  [{ep_global-1}] {gid}/{fname}/{sc['script_id']} "
                      f"phys={len(buffers[0]['sim_time'])} rgb={len(rgb_frames)} "
                      f"term={term}", flush=True)
            print(f"[{gid}/{fname}] done ({ep_global} eps, "
                  f"{(time.time()-t_start)/60:.1f} min)", flush=True)

    with open(os.path.join(args.out_root, "index.json"), "w") as f:
        json.dump({"episodes": index, "plan": args.plan, "plan_hash": plan_hash,
                   "controller_hash": ctrl_hash, "data_version": DATA_VERSION,
                   "friction_readback": readback_log,
                   "smoke": bool(args.smoke)}, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] 完成: {ep_global} episodes -> {args.out_root}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    os._exit(0)
