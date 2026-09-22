# 偏差记录 — PRE_REGISTRATION 执行偏差（追加，不修改原文）

1. **smoke 文件数**：预登记写"12 条 smoke"，实际执行 num_envs=4 → 48 文件
   （12 个 (condition × family × anchor) 组合 × 4 并行副本）。smoke 验证 PASS
   （audit/smoke_validation.md）；smoke 数据目录已删除，不计入最终数据集。
2. **pilot 收集方式**：为严格满足"总新增 ≤300"，pilot 以 num_envs=1 逐 wave 收集，
   最终 192 query + 48 support = 240 episodes（每 wave 一个独立 reset）。
3. **摩擦有效值**：smoke readback 证实 robot material 写入=读回（1.0/0.6/0.3），
   地面 material static=dynamic=1.0，combine rule 为 PhysX 默认 average
   → 有效足-地摩擦约为 (写入值+1)/2（normal 1.0、"0.6"→0.8、"0.3"→0.65）。
   所有档位结论按写入值命名，有效值对照记录于此。
4. **anchor 复原**：同 seed 跨 condition 的 anchor 状态差 = 0.00e+00（bit 级一致）；
   仍标记 approximate-paired（solver contact cache 不可复原）。
