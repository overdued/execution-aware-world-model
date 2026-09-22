"""V0.6.1 §3 / T10：运行时确认 policy 实际消费哪个 command（<=6 条短 smoke）。

记录每个控制 tick：u_requested（本 tick 写入）、cmd_term buffer（写后 / step 后）、
policy 输入 observation 的实际 command slice、policy 输出 action、capture tick/state。

运行:
    cd ~/IsaacLab && OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 ./isaaclab.sh -p \
        ~/cvpr_embed/execution_wm/data/policy_command_audit.py --num-waves 3
"""
import argparse
import json
import math
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--num-waves", type=int, default=3)
parser.add_argument("--out", default="/home/yuhang/cvpr_embed/results/v0_6_1_correctness/audit/policy_command_timeline.json")
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(PROJECT_ROOT))

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

POLICY_CKPT = "/home/yuhang/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-09-20_22-52-26/model_299.pt"
WAVES = {
    "Q1_step_vx": [(0.5, [0, 0, 0]), (0.5, [0.8, 0, 0]), (1.0, [0.8, 0, 0]),
                   (0.5, [0.0, 0, 0]), (0.5, [-0.4, 0, 0])],
    "Q3_turn": [(0.5, [0, 0, 0]), (1.0, [0, 0, 0.8]), (1.0, [0, 0, -0.8]), (0.5, [0, 0, 0])],
    "Q2_step_vy": [(0.5, [0, 0, 0]), (1.0, [0, 0.5, 0]), (1.0, [0, -0.5, 0]), (0.5, [0, 0, 0])],
}


@configclass
class AuditEnvCfg(UnitreeGo2FlatEnvCfg):
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


def cmd_at(segs, t):
    acc = 0.0
    for dur, c in segs:
        if t < acc + dur:
            return c
        acc += dur
    return [0, 0, 0]


def main():
    torch.manual_seed(1000)
    env_cfg = AuditEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.seed = 1000
    env = gym.make("Isaac-Velocity-Flat-Unitree-Go2-v0", cfg=env_cfg)
    unwrapped = env.unwrapped
    env = RslRlVecEnvWrapper(env)
    device = unwrapped.device
    control_dt = unwrapped.step_dt

    agent_cfg = UnitreeGo2FlatPPORunnerCfg()
    agent_cfg.device = device
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(POLICY_CKPT)
    policy = runner.get_inference_policy(device=device)

    # policy 观测组构成（用于定位 command slice）
    om = unwrapped.observation_manager
    cfg_terms = om.active_terms["policy"]
    term_dims = [(t, int(np.prod(om.group_obs_term_dim["policy"][i])))
                 for i, t in enumerate(cfg_terms)]
    obs_dim = sum(d for _, d in term_dims)
    cum, slices = 0, {}
    for t, d in term_dims:
        slices[t] = (cum, cum + d)
        cum += d

    cmd_term = unwrapped.command_manager.get_term("base_velocity")
    out = {"policy_checkpoint": POLICY_CKPT, "control_dt": control_dt,
           "obs_terms": term_dims, "obs_dim": obs_dim, "waves": {},
           "cmd_slice_from_terms": slices.get("velocity_commands")}

    for wname, segs in list(WAVES.items())[:args.num_waves]:
        settle_s, total = 1.0, 1.0 + sum(d for d, _ in segs)
        n_steps = int(math.ceil(total / control_dt))
        torch.manual_seed(1000)
        env.reset()
        # reset 后立刻记录 policy 首次看到的 obs（用于判断 step0 消费的 command）
        obs_td = env.get_observations()      # TensorDict：policy 需要它本身
        log = {"u_requested": [], "term_after_write": [], "term_after_step": [],
               "policy_obs_cmd": [], "policy_obs_poststep_cmd": [], "action_l2": [],
               "action_first3": [], "base_lin_vel_body": [], "phase": [], "t": []}
        for step in range(n_steps):
            t = step * control_dt
            if t < settle_s:
                cmd_np = np.zeros((1, 3), dtype=np.float32); phase = 0
            else:
                cmd_np = np.array([cmd_at(segs, t - settle_s)], dtype=np.float32); phase = 1
            cmd_term.vel_command_b = torch.as_tensor(cmd_np, device=device)
            term_after_write = cmd_term.vel_command_b[0].cpu().numpy().copy()
            # 关键：记录 policy **实际消费**的 obs（在 policy 调用之前，不能被 step 覆盖）
            obs_consumed_pol = obs_td["policy"].clone()
            with torch.inference_mode():
                actions = policy(obs_td)
            act = actions[0].cpu().numpy()
            term_after_step = cmd_term.vel_command_b[0].cpu().numpy().copy()
            obs_td, _, _, _ = env.step(actions)
            log["u_requested"].append(cmd_np[0].tolist())
            log["term_after_write"].append(term_after_write.tolist())
            log["term_after_step"].append(term_after_step.tolist())
            s = slices.get("velocity_commands")
            log["policy_obs_cmd"].append(
                obs_consumed_pol[0][s[0]:s[1]].cpu().numpy().tolist())
            log["policy_obs_poststep_cmd"].append(
                obs_td["policy"][0][s[0]:s[1]].cpu().numpy().tolist())
            log["obs_dim_actual"] = int(obs_consumed_pol.shape[-1])
            log["action_l2"].append(float(np.linalg.norm(act)))
            log["action_first3"].append(act[:3].tolist())
            log["base_lin_vel_body"].append(
                unwrapped.scene["robot"].data.root_lin_vel_b[0].cpu().numpy().tolist())
            log["phase"].append(phase); log["t"].append(t)
        out["waves"][wname] = log
        print(f"[wave] {wname} done ({n_steps} ticks)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[policy_command_audit] -> {args.out}")
    env.close()


if __name__ == "__main__":
    main()
    os._exit(0)
