"""V0.7 Stage 0：运行时短审计（≤12 条 smoke，不做正式采集）。

覆盖任务书 §3.1–§3.3 中必须在安装环境确认的部分：
  - controller 合法命令范围（用于冻结 a_i / b_i）
  - u_requested / u_consumed / joint_command / execution 四者的时间关系（1 tick 滞后复核）
  - reset 状态记录（robot state / policy hidden state / action queue / sensor history）
  - 摩擦材质：写入=读回、static/dynamic 分别、combine mode、binding prim
  - 平面 tilt 与 ∫wz vs 净 yaw 的近似误差（为 §7 轨迹口径提供实测依据）

运行:
  cd ~/IsaacLab && OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 ./isaaclab.sh -p \
    ~/cvpr_embed-v07/execution_wm/composition_v07/stage0_runtime_audit.py \
    --out ~/cvpr_embed-v07/results/v0_7_composition/audit/time_label_feature_material_tests.json
"""
import argparse
import json
import math
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--waves", type=int, default=3)
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.headless = True
simulation_app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(PROJ))

import gymnasium as gym  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
from isaaclab.utils import configclass  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import (  # noqa: E402
    UnitreeGo2FlatPPORunnerCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (  # noqa: E402
    UnitreeGo2FlatEnvCfg,
)
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

POLICY_CKPT = os.path.expanduser(
    "~/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-09-20_22-52-26/model_299.pt")
WAVES = {
    "step_vx": [(0.5, [0, 0, 0]), (0.5, [0.65, 0, 0]), (1.0, [0.65, 0, 0]),
                (0.5, [0.0, 0, 0]), (0.5, [-0.35, 0, 0])],
    "turn_wz": [(0.5, [0, 0, 0]), (1.0, [0, 0, 0.65]), (1.0, [0, 0, -0.65]), (0.5, [0, 0, 0])],
    "combo_xw": [(0.5, [0, 0, 0]), (1.5, [0.65, 0, 0.65]), (1.0, [-0.35, 0, -0.35]),
                 (0.5, [0, 0, 0])],
}


@configclass
class V07AuditCfg(UnitreeGo2FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        for ev in ("physics_material", "add_base_mass", "base_com",
                   "base_external_force_torque", "push_robot"):
            setattr(self.events, ev, None)
        cmd = self.commands.base_velocity
        cmd.heading_command = False
        cmd.rel_heading_envs = 0.0
        cmd.rel_standing_envs = 0.0
        cmd.resampling_time_range = (1.0e6, 1.0e6)
        cmd.debug_vis = False
        self.episode_length_s = 1.0e6


def cmd_at(segs, t):
    acc = 0.0
    for dur, c in segs:
        if t < acc + dur:
            return c
        acc += dur
    return [0, 0, 0]


def main():
    device = "cuda"
    cfg = V07AuditCfg()
    cfg.scene.num_envs = 1
    cfg.seed = 1000
    env = gym.make("Isaac-Velocity-Flat-Unitree-Go2-v0", cfg=cfg)
    u = env.unwrapped
    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, {**UnitreeGo2FlatPPORunnerCfg().to_dict(), "device": device},
                            log_dir=None, device=device)
    runner.load(POLICY_CKPT)
    policy = runner.get_inference_policy(device=device)

    cmd_term = u.command_manager.get_term("base_velocity")
    r = cmd_term.cfg.ranges
    ranges = {"lin_vel_x": list(r.lin_vel_x), "lin_vel_y": list(r.lin_vel_y),
              "ang_vel_z": list(r.ang_vel_z)}
    limits = np.array([r.lin_vel_x[1], r.lin_vel_y[1], r.ang_vel_z[1]])

    om = u.observation_manager
    terms = om.active_terms["policy"]
    term_dims = [(t, int(np.prod(om.group_obs_term_dim["policy"][i])))
                 for i, t in enumerate(terms)]
    cum, sl = 0, {}
    for t, d in term_dims:
        sl[t] = (cum, cum + d); cum += d

    out = {"controller_ranges": ranges,
           "obs_terms": term_dims, "cmd_slice": sl.get("velocity_commands"),
           "obs_dim": cum,
           "policy_checkpoint": POLICY_CKPT,
           "policy_ckpt_hash": None,
           "num_actions": int(u.action_manager.total_action_dim),
           "joint_names": list(u.scene["robot"].joint_names),
           "default_joint_pos_head": u.scene["robot"].data.default_joint_pos[0, :4].tolist(),
           "control_dt": float(u.step_dt), "physics_dt": float(u.physics_dt),
           "decimation": int(u.cfg.decimation),
           "waves": {}}
    import hashlib
    out["policy_ckpt_hash"] = hashlib.sha256(open(POLICY_CKPT, "rb").read()).hexdigest()[:16]

    for wname, segs in list(WAVES.items())[:args.waves]:
        settle, total = 1.0, 1.0 + sum(d for d, _ in segs)
        n = int(math.ceil(total / u.step_dt))
        torch.manual_seed(1000)
        env.reset()
        # reset 状态记录（§3.3）
        robot = u.scene["robot"]
        reset_rec = {
            "root_pos_w": robot.data.root_pos_w[0].cpu().numpy().tolist(),
            "root_quat_w": robot.data.root_quat_w[0].cpu().numpy().tolist(),
            "root_lin_vel_w": robot.data.root_lin_vel_w[0].cpu().numpy().tolist(),
            "joint_pos_head": robot.data.joint_pos[0, :4].cpu().numpy().tolist(),
            "joint_vel_head": robot.data.joint_vel[0, :4].cpu().numpy().tolist(),
            "policy_hidden_state": "gym->rsl_rl VecEnvWrapper: policy 为 MLP（无 recurrent），"
                                   "无 hidden state；action queue 无（decimation 内重复同一 action）",
        }
        log = {"u_requested": [], "u_consumed": [], "joint_command_first6": [],
               "u_consumed_age_ticks": [], "action_l2": [], "base_lin_vel_body": [],
               "projected_gravity": [], "tilt_deg": [], "wz_body": [], "yaw_quat": [],
               "phase": [], "t": []}
        obs_td = env.get_observations()
        for step in range(n):
            t = step * u.step_dt
            cmd_np = (np.zeros((1, 3), dtype=np.float32) if t < settle else
                      np.array([cmd_at(segs, t - settle)], dtype=np.float32))
            cmd_term.vel_command_b = torch.as_tensor(cmd_np, device=device)
            obs_consumed = obs_td["policy"].clone()
            with torch.inference_mode():
                actions = policy(obs_td)
            s = sl["velocity_commands"]
            obs_td, _, _, _ = env.step(actions)
            log["u_requested"].append(cmd_np[0].tolist())
            log["u_consumed"].append(obs_consumed[0][s[0]:s[1]].cpu().numpy().tolist())
            log["joint_command_first6"].append(actions[0, :6].cpu().numpy().tolist())
            log["action_l2"].append(float(actions[0].norm().item()))
            log["base_lin_vel_body"].append(
                robot.data.root_lin_vel_b[0].cpu().numpy().tolist())
            g = robot.data.projected_gravity_b[0].cpu().numpy()
            log["projected_gravity"].append(g.tolist())
            log["tilt_deg"].append(float(np.degrees(np.arccos(np.clip(-g[2], -1, 1)))))
            log["wz_body"].append(float(robot.data.root_ang_vel_b[0, 2].cpu().numpy()))
            log["yaw_quat"].append(float(robot.data.root_quat_w[0].cpu().numpy()[3]))
            log["phase"].append(0 if t < settle else 1); log["t"].append(t)
        # u_consumed 的滞后
        ur = np.array(log["u_requested"]); uc = np.array(log["u_consumed"])
        lag = {}
        for L in (0, 1, 2):
            lag[L] = int(sum(np.allclose(uc[i], ur[i - L], atol=1e-6)
                             for i in range(L, len(ur))))
        log["consumption_lag_counts"] = {str(k): v for k, v in lag.items()}
        log["n_ticks"] = n
        out["waves"][wname] = log
        print(f"[stage0] {wname}: lag counts {lag} / {n}", flush=True)

    out["reset_state"] = reset_rec
    # 材质（写入/读回 + USD 属性）
    view = robot.root_physx_view
    m = view.get_material_properties().cpu().numpy()
    out["material_readback_initial"] = {
        "static_mean": float(m[..., 0].mean()), "dynamic_mean": float(m[..., 1].mean()),
        "n_shapes": int(m.shape[1]), "static_equals_dynamic": bool(np.allclose(m[..., 0], m[..., 1]))}
    try:
        pm = u.cfg.scene.terrain.physics_material
        out["ground_material_cfg"] = {
            "static": float(pm.static_friction), "dynamic": float(pm.dynamic_friction),
            "restitution": float(pm.restitution),
            "friction_combine_mode": str(getattr(pm, "friction_combine_mode", "UNKNOWN")),
            "restitution_combine_mode": str(getattr(pm, "restitution_combine_mode", "UNKNOWN"))}
    except Exception as e:  # noqa: BLE001
        out["ground_material_cfg"] = {"error": f"{type(e).__name__}: {e}"}
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    attrs = {}
    for p in ("/World/ground/terrain/physicsMaterial", "/physicsScene/defaultMaterial"):
        prim = stage.GetPrimAtPath(p)
        attrs[p] = ({a.GetName(): str(a.Get()) for a in prim.GetAttributes()}
                    if prim and prim.IsValid() else "MISSING")
    out["usd_material_attrs"] = attrs
    # 机器人材质绑定 prim（§3.3：必须记录绑定 prim，不只记录数值）
    binds = [str(prim.GetPath()) for prim in stage.Traverse()
             if "MaterialBindingAPI" in (prim.GetAppliedSchemas()
                                         if hasattr(prim, "GetAppliedSchemas") else [])]
    out["material_binding_prims_all"] = binds[:30]
    out["material_binding_n"] = len(binds)
    try:
        mp = robot.root_physx_view.get_material_paths()
        flat = sorted({str(p) for row in mp for p in row})
        out["robot_physx_material_paths"] = flat[:10]
        out["robot_physx_material_paths_n_unique"] = len(flat)
    except Exception as e:  # noqa: BLE001
        out["robot_physx_material_paths"] = f"unavailable: {type(e).__name__}: {e}"
    # 机器人材质的 combine mode（若可从 USD 读到）
    rb = {}
    for pth in out.get("robot_physx_material_paths", [])[:3]:
        if not isinstance(pth, str):
            continue
        pr = stage.GetPrimAtPath(pth)
        if pr and pr.IsValid():
            rb[pth] = {a.GetName(): str(a.Get()) for a in pr.GetAttributes()
                       if "CombineMode" in a.GetName() or "Friction" in a.GetName()}
    out["robot_material_usd_attrs"] = rb
    # 写入三档并读回（沿用 collector 的写入路径）
    rb_log = {}
    for name, f in (("nominal", 1.0), ("mid", 0.6), ("low", 0.3)):
        mats = view.get_material_properties()
        eids = torch.arange(u.num_envs, dtype=torch.long)
        mats[eids, :, 0] = f; mats[eids, :, 1] = max(0.05, 0.7 * f); mats[eids, :, 2] = 0.0
        view.set_material_properties(mats, eids)
        m2 = view.get_material_properties().cpu().numpy()
        rb_log[name] = {"written_static": f, "written_dynamic": max(0.05, 0.7 * f),
                        "readback_static_mean": float(m2[..., 0].mean()),
                        "readback_dynamic_mean": float(m2[..., 1].mean()),
                        "readback_static_min": float(m2[..., 0].min()),
                        "readback_static_max": float(m2[..., 0].max())}
    out["material_write_readback"] = rb_log

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1, ensure_ascii=False)
    print(f"[stage0] -> {args.out}")
    env.close()


if __name__ == "__main__":
    main()
    os._exit(0)
