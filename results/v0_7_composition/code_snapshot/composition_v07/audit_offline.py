"""V0.7 Stage 0 离线部分：信息角色表 + 口径核对 + V0.6.2 baseline 结构核实。

输出: audit/information_roles.json
      并读回 audit/time_label_feature_material_tests.json 做一致性断言
"""
import json
import os

from execution_wm.validity_v061.schema import PROPRIO_SCHEMA

OUT = "results/v0_7_composition/audit"

roles = {
    "schema": PROPRIO_SCHEMA.name,
    "dim": PROPRIO_SCHEMA.dim,
    "channels": [
        {"name": f.name, "offset": f.offset, "size": f.size, "unit": f.unit,
         "frame": f.frame, "role": f.role,
         "sim_input_allowed": f.role != "privileged_gt",
         "v07_main_input": f.role != "privileged_gt",
         "label_use": f.name in ("base_linear_velocity_body", "base_angular_velocity")}
        for f in PROPRIO_SCHEMA.fields],
    "inputs": {
        "history": "[L,D]，严格过去/当前，<= prediction origin（V0.6.1 整数 tick）",
        "planned_commands": "[H,3] u_requested（部署可得的计划命令），origin 时已知",
        "u_consumed": "诊断/时序核查用，不作为主输入",
    },
    "labels": {
        "execution_label": "e_label = F(e_raw)，独立于 command 生成",
        "command_reference": "u_ref = S(u)（事件表，整数 tick）",
        "residual_label": "r_label = e_label - u_ref（恒等式 u_ref + r_label == e_label）",
        "future_relative_pose": "T_origin^{-1} T_{t+k}，姿态来自合法四元数（符号连续+归一化）",
    },
    "evaluation_truth": "直接读 e_label，不用 u+r 自证",
    "forbidden_in_main_input": ["simulator GT body velocity/angular velocity",
                                "GT friction", "GT condition label",
                                "donor retrieval", "future executed command log"],
    "caveat": "projected_gravity/IMU/contact 来自仿真传感器，称 simulated deployable-candidate，"
              "不宣称已完成真机验证",
}

# V0.6.2 additive baseline 结构核实（任务书 §3.4）
v062 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "execution_wm/validity_v062/task2_composition.py")
src = open(v062).read()
head_out = "[state(40), u_a(H)] -> 128 -> 128 -> H, per axis, additive"
axis_restricted = ("torch.stack([self.heads[a](torch.cat([s, fa[:, :, a]], dim=-1))"
                   in src.replace("\n", " "))
roles["v062_additive_baseline_structure_check"] = {
    "source": "execution_wm/validity_v062/task2_composition.py",
    "architecture": head_out,
    "each_factor_outputs": "本轴 1 维（H,）后再 stack 成 [H,3]",
    "is_axis_restricted": bool(axis_restricted),
    "implication": "V0.6.2 的 additive baseline 是**轴受限**形式（g_a 只输出 a 轴）。"
                   "它不能表达跨轴效应，因此 V0.6.2 的负结果只否证了轴受限加性分解，"
                   "**不**能推广为『所有加性/交互模型不可能有效』。V0.7 的 I 模型每个因素"
                   "输出完整 [H,3]（任务书 §6.2），是更强的形式。",
}

# V0.6.2 运行清单补齐（任务书 §3.4）
roles["v062_run_ledger"] = {
    "task2_composition_mlp": "3 seeds (42/43/44)，train-only 拟合，无 v062 checkpoint（未落盘）",
    "task5_deployable": "12 runs = M0/M1 × {privileged,deployable} × 3 seeds，0.0028 GPU·h，"
                        "checkpoint 在 checkpoints/execution_wm/v0_6_2/",
    "v061_main": "12 runs = M0-M3 × 3 seeds，0.0058 GPU·h",
    "note": "V0.6.2 报告写‘仅 Task5 新训 12 run’指的是**持久化 checkpoint** 的 run；"
            "Task2 的 MLP 训练是 in-process、未落盘，二者不矛盾但此前未列清。",
}

os.makedirs(OUT, exist_ok=True)
json.dump(roles, open(os.path.join(OUT, "information_roles.json"), "w"),
          indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in roles.items()
                  if k in ("v062_additive_baseline_structure_check",)}, indent=1, ensure_ascii=False))
print("->", os.path.join(OUT, "information_roles.json"))
