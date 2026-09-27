"""V0.8 相机环境配置（无 argparse 副作用，可被采集器/探针/测试安全 import）。"""
from isaaclab.sensors import CameraCfg, ImuCfg  # noqa: F401
from isaaclab.sim import PinholeCameraCfg  # noqa: F401
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (  # noqa: E402
    UnitreeGo2FlatEnvCfg,
)

# 固定机载相机（内外参冻结，逐 episode 落盘同一值）
CAM_POS = (0.35, 0.0, 0.10)          # base 前方 35cm、高 10cm
CAM_ROT_WORLD = (1.0, 0.0, 0.0, 0.0)  # world 约定：光轴 = base +X（前），上 = +Z
CAM_RES = (256, 256)
CAM_PERIOD = 0.05                     # 20 Hz


@configclass
class Go2V08CamEnvCfg(UnitreeGo2FlatEnvCfg):
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
        # 渲染对齐 20Hz 网格：物理 200Hz(dt=0.005)，render_interval=10 -> 每 0.05s 渲染一次，
        # 与相机 update_period=0.05 及下游 20Hz 派生网格严格对齐（帧内容时刻 = 0.05k）。
        # 渲染与控制解耦，不影响物理；A/B 物理一致性由 check_camera_invariance 核验。
        self.sim.render_interval = 10
        self.scene.imu = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/base")
        self.scene.camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
            update_period=CAM_PERIOD,
            height=CAM_RES[0], width=CAM_RES[1],
            data_types=["rgb"],
            spawn=PinholeCameraCfg(
                focal_length=2.08, focus_distance=400.0,
                horizontal_aperture=2.815,          # ≈83° HFoV，固定不扫参
                clipping_range=(0.05, 100.0),
            ),
            offset=CameraCfg.OffsetCfg(pos=CAM_POS, rot=CAM_ROT_WORLD, convention="world"),
        )
