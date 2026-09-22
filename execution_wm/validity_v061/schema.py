"""V0.6.1 B4：显式 FeatureSchema —— 禁止散落魔法索引。

proprio 输入布局（与 execution_wm/data/dataset.py:PROPRIO_KEYS 一致，dim=40）:
    [ 0: 3] base_linear_velocity_body   vx, vy, vz   (m/s,  body frame)
    [ 3: 6] base_angular_velocity       wx, wy, wz   (rad/s, body frame)
    [ 6: 9] projected_gravity
    [ 9:12] imu_linear_acceleration
    [12:24] joint_position
    [24:36] joint_velocity
    [36:40] feet_contact

execution 目标 = [vx, vy, wz] -> proprio 索引 [0, 1, 5]（旧代码用 [0,1,2] 是错的）。
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Field:
    name: str
    offset: int
    size: int
    unit: str
    frame: str
    role: str          # "deployable" | "privileged_gt" | "derived"

    @property
    def sl(self):
        return slice(self.offset, self.offset + self.size)


@dataclass(frozen=True)
class FeatureSchema:
    name: str
    dim: int
    fields: tuple
    execution_fields: tuple      # (field_name, index_within_field)
    vel_names: tuple = ("vx", "vy", "wz")

    def field(self, name):
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(name)

    def execution_indices(self):
        """execution 目标 [vx,vy,wz] 在拼接后 proprio 向量中的绝对索引。"""
        out = []
        for fname, j in self.execution_fields:
            out.append(self.field(fname).offset + j)
        return np.array(out, dtype=int)

    def extract_execution_from_proprio(self, proprio):
        """proprio [..., dim] -> execution [..., 3]。本版按 schema 对应 [0,1,5]。"""
        return proprio[..., self.execution_indices()]

    def describe(self):
        lines = [f"schema={self.name} dim={self.dim}",
                 f"execution = {[f'{f}[{j}]' for f, j in self.execution_fields]}"
                 f" -> abs idx {self.execution_indices().tolist()}"]
        for f in self.fields:
            lines.append(f"  [{f.offset:2d}:{f.offset + f.size:2d}] {f.name:28s} "
                         f"{f.unit:8s} {f.frame:12s} {f.role}")
        return "\n".join(lines)


PROPRIO_SCHEMA = FeatureSchema(
    name="go2_proprio_v1",
    dim=40,
    fields=(
        Field("base_linear_velocity_body", 0, 3, "m/s", "body", "privileged_gt"),
        Field("base_angular_velocity", 3, 3, "rad/s", "body", "privileged_gt"),
        Field("projected_gravity", 6, 3, "1", "body", "deployable"),
        Field("imu_linear_acceleration", 9, 3, "m/s^2", "body", "deployable"),
        Field("joint_position", 12, 12, "rad", "joint", "deployable"),
        Field("joint_velocity", 24, 12, "rad/s", "joint", "deployable"),
        Field("feet_contact", 36, 4, "1", "n/a", "deployable"),
    ),
    execution_fields=(("base_linear_velocity_body", 0),
                      ("base_linear_velocity_body", 1),
                      ("base_angular_velocity", 2)),
)

EXECUTION_KEYS = ("vx", "vy", "wz")


def extract_execution_from_proprio(proprio, schema=PROPRIO_SCHEMA):
    """统一入口。proprio [..., 40] -> [..., 3]。"""
    return schema.extract_execution_from_proprio(proprio)
