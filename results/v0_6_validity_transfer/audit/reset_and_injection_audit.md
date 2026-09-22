# 阶段A3：Reset / 干预应用审计

方法：collector.py / collect_controlled_probes.py 代码路径审计 + index.json 统计。运行时 PhysX readback 归阶段B smoke（§1：阶段A 不采数）。

## 1. Episode 间 reset 状态（d0 主数据集）

**结论：d0 的 episode 之间不做 sim reset。** 代码证据：collector.py 主循环中 `start_episode()` 只分配新 command schedule，不调用 env.reset()；`schedule_end` 是 collector 侧逻辑结束，机器人保持当时 base pose/velocity、joint q/dq、接触状态、外力历史直接进入下一 episode。唯一触发真实 reset 的是 env 内建 done（termination=摔倒；episode_length_s=1e6 下 timeout 不发生）。

统计：d0 163 episodes，155 个 anchor 链；链长分布 mean=1.05 max=3；多 episode 链的条件分布 {'normal': 14}；termination: {'terminated': 34, 'schedule_end': 129}。

含义：
- **episode 不是独立 reset anchor**——pair 独立性分析必须以 anchor chain 为单元（A5）。
- 摩擦干预在 condition 轮次边界写入（`set_friction`），此时机器人**保持上一轮末态步态**（mid-stride 切换摩擦）。
- locomotion policy（RSL-RL MLP）无循环隐状态；last action 不跨 episode 传递（policy(obs) 无记忆）；command buffer 每步由 collector 覆写；但 **gait phase（隐含于关节状态）与接触历史跨 chained episode 连续**。

## 2. d1 controlled probes

每条 probe episode 前 `torch.manual_seed(seed); env.reset()`（collect_controlled_probes.py L215-221），metadata 记录 seed 与 reset_state=default_pose_zero_vel。**同一 seed 跨 condition 复现相同初始状态**；但 solver contact cache / warm start 不可完整恢复——跨 condition 的同一 probe 标注为 **approximate-paired**，不使用'精确反事实'措辞。

## 3. 摩擦注入（未验证的写入路径）

代码路径：`set_friction()` 直接写 PhysX material buffer：static=friction, dynamic=max(0.05, 0.7*friction), restitution=0，**写入后无 readback 验证**。
- 生效时刻：写入后下一步物理步应生效（未运行时验证）。
- **combine rule 未定案**：地面对象（GroundPlane）自身 material 与 combine 规则（PhysX 默认 average）决定有效摩擦；YAML friction=0.3 不能认定足-地有效摩擦=0.3。
- body/material 索引与 ground-foot 组合：写入覆盖 robot 全部 shapes；地面 material 未改。
- **运行时 readback（get_material_properties after write + 地面 material + combine rule）已列入阶段B smoke 验证项。**
- 摩擦变化不预设只能减速：解释以实际轨迹为准（§B4 同样要求）。

## 4. 执行器缩放与外力

`ActuatorScaler` 备份 effort_limit/saturation_effort 原值，每条件 orig*s 覆写（条件间隐式恢复，无累积）；`Disturber` 每步 set_external_force_and_torque（is_global=True，base body，0.5–1.0s 重采样 30–60N 水平力）。写入路径代码级正确，运行时 readback 同列 B smoke。

## 5. fall/termination 标签

d0 termination_reason 分布 {'terminated': 34, 'schedule_end': 129}；terminated（摔倒）episode 按 train/val/test 角色：{"('actuator_low', 'test_ood')": 6, "('actuator_mid', 'train')": 1, "('disturbance', 'test_id')": 2, "('disturbance', 'train')": 19, "('disturbance', 'val')": 2, "('friction_low', 'train')": 1, "('friction_vlow', 'test_ood')": 1, "('normal', 'train')": 2}。
- 摔倒 episode **保留并带标签**（terminated_keep_s=2.0）；V0 sanity check 曾对 terminated 放宽接触阈值（0.05 vs 0.7）——该处理只影响 sanity 通过率，**不应用于把 fall 数据伪装成正常接触**；本轮所有分析可按 termination_reason 单独统计（manifests/episodes.csv 已含）。
- 外力条件（disturbance）本轮不作为 friction 主实验，仅保留标签。

## 6. 对旧结果的影响范围

- V0 训练：episode-level split + 链内连续 -> 同 anchor 跨 split 的链 2/155（manifest_build 实测），泄漏面小但非零；所有 V0/V0.5 的 test 结论需带此注记。
- Context Swap v2 / V0.5：pair 的 source/target episode 可能是同链连续段（同 anchor）——A5 以 anchor group 重做依赖统计。
- 'friction=0.3' 的有效值未验证 -> 所有按档位解释的剂量效应结论暂保留为'按写入值命名'。