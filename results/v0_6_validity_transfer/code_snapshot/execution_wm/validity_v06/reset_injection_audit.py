"""V0.6 阶段A3：Reset / 干预应用审计（代码级 + index.json 统计）。

运行时 PhysX readback 需要起仿真——按 §1 停机规则归入阶段B smoke 验证项；
本模块给出代码路径事实与数据侧统计，并明确标注哪些是"未验证的写入路径"。

输出: audit/reset_and_injection_audit.md
用法: python -m execution_wm.validity_v06.reset_injection_audit --config execution_wm/configs/v06_validity.yaml
"""
import argparse
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
import yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out = os.path.join(cfg["out_dir"], "audit")
    os.makedirs(out, exist_ok=True)

    ep = pd.read_csv(os.path.join(cfg["out_dir"], "manifests", "episodes.csv"))
    d0 = ep[ep.dataset_version == "d0"]
    d1 = ep[ep.dataset_version == "d1"]

    term = dict(Counter(d0.termination_reason))
    # 摔倒 episode 在训练 split 中的分布
    tr = d0[d0.termination_reason == "terminated"].groupby(
        ["condition", "train_val_test_role"]).size().to_dict()
    # 每 condition 平均链长（一个 anchor 几条 episode）
    chain_len = d0.groupby("reset_anchor_group").size()
    chained = d0.groupby("reset_anchor_group").filter(lambda g: len(g) > 1)
    chained_cond = dict(Counter(chained.condition))

    L = []
    A = L.append
    A("# 阶段A3：Reset / 干预应用审计\n")
    A("方法：collector.py / collect_controlled_probes.py 代码路径审计 + index.json 统计。"
      "运行时 PhysX readback 归阶段B smoke（§1：阶段A 不采数）。\n")

    A("## 1. Episode 间 reset 状态（d0 主数据集）\n")
    A("**结论：d0 的 episode 之间不做 sim reset。** 代码证据：collector.py 主循环中 "
      "`start_episode()` 只分配新 command schedule，不调用 env.reset()；`schedule_end` 是 "
      "collector 侧逻辑结束，机器人保持当时 base pose/velocity、joint q/dq、接触状态、"
      "外力历史直接进入下一 episode。唯一触发真实 reset 的是 env 内建 done"
      "（termination=摔倒；episode_length_s=1e6 下 timeout 不发生）。\n")
    A(f"统计：d0 {len(d0)} episodes，{d0.reset_anchor_group.nunique()} 个 anchor 链；"
      f"链长分布 mean={chain_len.mean():.2f} max={chain_len.max()}；"
      f"多 episode 链的条件分布 {chained_cond}；termination: {term}。\n")
    A("含义：\n"
      "- **episode 不是独立 reset anchor**——pair 独立性分析必须以 anchor chain 为单元（A5）。\n"
      "- 摩擦干预在 condition 轮次边界写入（`set_friction`），此时机器人**保持上一轮末态步态**"
      "（mid-stride 切换摩擦）。\n"
      "- locomotion policy（RSL-RL MLP）无循环隐状态；last action 不跨 episode 传递"
      "（policy(obs) 无记忆）；command buffer 每步由 collector 覆写；"
      "但 **gait phase（隐含于关节状态）与接触历史跨 chained episode 连续**。\n")

    A("## 2. d1 controlled probes\n")
    A("每条 probe episode 前 `torch.manual_seed(seed); env.reset()`（collect_controlled_probes.py "
      "L215-221），metadata 记录 seed 与 reset_state=default_pose_zero_vel。"
      "**同一 seed 跨 condition 复现相同初始状态**；但 solver contact cache / warm start "
      "不可完整恢复——跨 condition 的同一 probe 标注为 **approximate-paired**，"
      "不使用'精确反事实'措辞。\n")

    A("## 3. 摩擦注入（未验证的写入路径）\n")
    A("代码路径：`set_friction()` 直接写 PhysX material buffer：static=friction, "
      "dynamic=max(0.05, 0.7*friction), restitution=0，**写入后无 readback 验证**。\n"
      "- 生效时刻：写入后下一步物理步应生效（未运行时验证）。\n"
      "- **combine rule 未定案**：地面对象（GroundPlane）自身 material 与 combine 规则"
      "（PhysX 默认 average）决定有效摩擦；YAML friction=0.3 不能认定足-地有效摩擦=0.3。\n"
      "- body/material 索引与 ground-foot 组合：写入覆盖 robot 全部 shapes；"
      "地面 material 未改。\n"
      "- **运行时 readback（get_material_properties after write + 地面 material + combine rule）"
      "已列入阶段B smoke 验证项。**\n"
      "- 摩擦变化不预设只能减速：解释以实际轨迹为准（§B4 同样要求）。\n")

    A("## 4. 执行器缩放与外力\n")
    A("`ActuatorScaler` 备份 effort_limit/saturation_effort 原值，每条件 orig*s 覆写"
      "（条件间隐式恢复，无累积）；`Disturber` 每步 set_external_force_and_torque"
      "（is_global=True，base body，0.5–1.0s 重采样 30–60N 水平力）。写入路径代码级正确，"
      "运行时 readback 同列 B smoke。\n")

    A("## 5. fall/termination 标签\n")
    A(f"d0 termination_reason 分布 {term}；terminated（摔倒）episode 按 train/val/test 角色："
      f"{ {str(k): v for k, v in tr.items()} }。\n"
      "- 摔倒 episode **保留并带标签**（terminated_keep_s=2.0）；V0 sanity check 曾对 terminated "
      "放宽接触阈值（0.05 vs 0.7）——该处理只影响 sanity 通过率，**不应用于把 fall 数据伪装成正常接触**；"
      "本轮所有分析可按 termination_reason 单独统计（manifests/episodes.csv 已含）。\n"
      "- 外力条件（disturbance）本轮不作为 friction 主实验，仅保留标签。\n")

    A("## 6. 对旧结果的影响范围\n")
    A("- V0 训练：episode-level split + 链内连续 -> 同 anchor 跨 split 的链 2/155（manifest_build 实测），"
      "泄漏面小但非零；所有 V0/V0.5 的 test 结论需带此注记。\n"
      "- Context Swap v2 / V0.5：pair 的 source/target episode 可能是同链连续段（同 anchor）"
      "——A5 以 anchor group 重做依赖统计。\n"
      "- 'friction=0.3' 的有效值未验证 -> 所有按档位解释的剂量效应结论暂保留为'按写入值命名'。")
    with open(os.path.join(out, "reset_and_injection_audit.md"), "w") as f:
        f.write("\n".join(L))
    print("\n".join(L[6:10]))
    print(f"[A3] -> {out}/reset_and_injection_audit.md")


if __name__ == "__main__":
    main()
