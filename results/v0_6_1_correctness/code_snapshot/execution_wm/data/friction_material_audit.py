"""V0.6.1 §4：运行时读取摩擦 material（robot 双侧 static/dynamic + floor + combine mode）。

查不到就写 UNKNOWN，不把 hardcoded average 当实测。static/dynamic 分别报告。
"""
import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="/home/yuhang/cvpr_embed/results/v0_6_1_correctness/audit/friction_material.json")
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import gymnasium as gym  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
from isaaclab.utils import configclass  # noqa: E402
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (  # noqa: E402
    UnitreeGo2FlatEnvCfg,
)


@configclass
class MatEnvCfg(UnitreeGo2FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        for ev in ("physics_material", "add_base_mass", "base_com",
                   "base_external_force_torque", "push_robot"):
            setattr(self.events, ev, None)
        self.episode_length_s = 1.0e6


def read_material(view, n):
    out = {}
    try:
        m = view.get_material_properties().cpu().numpy()
        out["n_shapes_per_env"] = int(m.shape[1])
        for j, nm in enumerate(("static_friction", "dynamic_friction", "restitution")):
            out[f"robot_{nm}_mean"] = float(m[..., j].mean())
            out[f"robot_{nm}_min"] = float(m[..., j].min())
            out[f"robot_{nm}_max"] = float(m[..., j].max())
        out["robot_static_equals_dynamic"] = bool(
            np.allclose(m[..., 0], m[..., 1]))
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def main():
    cfg = MatEnvCfg()
    cfg.scene.num_envs = 1
    env = gym.make("Isaac-Velocity-Flat-Unitree-Go2-v0", cfg=cfg)
    u = env.unwrapped
    res = {"robot_material_write_readback": {}}

    # 写入三个档位并读回（与 collector 相同写入路径）
    view = u.scene["robot"].root_physx_view
    for name, fric in (("normal", 1.0), ("friction_mid", 0.6), ("friction_low", 0.3)):
        mats = view.get_material_properties()
        eids = torch.arange(u.num_envs, dtype=torch.long)
        mats[eids, :, 0] = fric
        mats[eids, :, 1] = max(0.05, 0.7 * fric)
        mats[eids, :, 2] = 0.0
        view.set_material_properties(mats, eids)
        res["robot_material_write_readback"][name] = {
            "written_static": fric, "written_dynamic": max(0.05, 0.7 * fric),
            **read_material(view, u.num_envs)}

    # 地面 material（cfg 级）
    gm = {}
    try:
        pm = u.cfg.scene.terrain.physics_material
        gm = {"static_friction": float(pm.static_friction),
              "dynamic_friction": float(pm.dynamic_friction),
              "restitution": float(pm.restitution),
              "friction_combine_mode": getattr(pm, "friction_combine_mode", "UNKNOWN"),
              "restitution_combine_mode": getattr(pm, "restitution_combine_mode", "UNKNOWN"),
              "source": "scene.terrain.physics_material cfg"}
    except Exception as e:  # noqa: BLE001
        gm = {"source": f"UNKNOWN: {type(e).__name__}: {e}"}
    res["ground_material_cfg"] = gm

    # 尝试从 USD/PhysX schema 读 combine mode（查不到写 UNKNOWN）
    attempts = {}
    try:
        import omni.usd
        from pxr import UsdShade
        stage = omni.usd.get_context().get_stage()
        found = []
        for prim in stage.Traverse():
            sch = prim.GetAppliedSchemas() if hasattr(prim, "GetAppliedSchemas") else []
            for s in sch:
                if "Material" in s or "material" in s:
                    found.append({"prim": str(prim.GetPath()), "schema": s})
        attempts["physx_material_schemas"] = found[:20]
        attempts["n_found"] = len(found)
    except Exception as e:  # noqa: BLE001
        attempts["error"] = f"{type(e).__name__}: {e}"
    res["combine_mode_runtime_probe"] = attempts
    res["combine_mode_status"] = "UNKNOWN" if not attempts.get("physx_material_schemas") else \
        "READ (见 combine_mode_runtime_probe)"
    res["effective_friction_if_average"] = {
        "note": "仅当 combine=average 且 ground static=dynamic=1 时成立；未从引擎读回 -> 不可当实测",
        "static_effective": {"normal": 1.0, "friction_mid": 0.8, "friction_low": 0.65},
        "dynamic_effective": {"normal": 0.85, "friction_mid": 0.71, "friction_low": 0.605},
        "ground_static": gm.get("static_friction"), "ground_dynamic": gm.get("dynamic_friction"),
    }
    res["naming_rule"] = "档位一律按 written_static / written_dynamic 命名，不使用'真实摩擦'"

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False)[:3000])
    env.close()


if __name__ == "__main__":
    main()
    os._exit(0)
