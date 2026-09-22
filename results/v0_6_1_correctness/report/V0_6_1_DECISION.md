# V0_6_1_DECISION — correctness patch 与有限公平重跑判定

> 2026-09-22 · 依据 `tasks/v0.6.1/02` · 全部证据在 `results/v0_6_1_correctness/`
> 数据：`/media/hdd1/yuhang/datasets/execution_wm/v0_6_1`（240 ep 重派生，未覆盖旧目录）
> 模型：`/media/hdd1/yuhang/checkpoints/execution_wm/v0_6_1`（12 run，0.0058 GPU·h）
> 旧 V0.6 结果原地保留、未删除、未改写；本轮为开发/探索，**不是**新的独立 confirmatory test

```text
LABEL_IDENTITY:            PASS（全量恒等式 max err 1.11e-16；旧管线 max err 1.120208，0/240 成立）
SUPPORT_PAIR_INTEGRITY:    PASS（新抽样 2000 次 0 违反；旧实现在同一玩具上 98.59% 错配）
POLICY_COMMAND_TIMELINE:   VERIFIED（consumed[i]=requested[i-1]，3×199 tick 精确成立；
                                     执行滞后 2 tick；旧 raw 可复用，未重采）
FEATURE_SCHEMA:            PASS（[11,12,13,21,22,23] -> [11,12,23]；persistence/k0 全量修正）
SPLIT_AND_DONOR_LINEAGE:   PASS（A/B/C 互不重叠 + supp 单列；donor 表固定且逐窗可复现）
FIXED_LEAD_EVALUATION:     PASS（lead 索引由 timestamp 推导；与 prefix 分离；toy 验证不等）
NATIVE_PREDICTION_GAIN:    INCONCLUSIVE（split 依赖：见下"关键反转"）
CONTEXT_TRANSFER:          INCONCLUSIVE（paired same-cond donor 优于 wrong-cond donor；
                                         但 donor 选择用真值 condition -> oracle 假设）
ORACLE_DIAGNOSTIC:         分支确实响应 friction（Δr̂≈r̂量级的 3%），但 M3−M1≈−0.0001
                           -> 该注入方式在当前目标下无增益；记 INCONCLUSIVE，非上界，
                           不得推导"同类数据更多无效"
STATISTICAL_SCOPE:         EXPLORATORY（held-anchor 仅 2 个 anchor group）
RECOMMENDED_NEXT_STEP:     见 report/NEXT_ACTIONS_V061.md（单条：deployable 输入 + 条件推断）
```

## 关键反转（相对 V0.6）

修复标签与配对后，**主结论方向改变**：

| 对比（split） | V0.6（旧标签/旧抽样） | V0.6.1（修复后） |
|---|---|---|
| M1 − command-copy, A | −0.0190 [−0.0271,−0.0118] 胜 | **−0.0244 [−0.0335,−0.0166] 胜** |
| M1 − command-copy, B | −0.0257 [−0.0361,−0.0148] 胜 | **−0.0002 [−0.0062,+0.0059] 跨 0** |
| M1 − M0, B/C | −0.0023 / — 判 M1 胜 | **+0.0043 / +0.0041（M0 胜）** |
| M2 − M1, A | −0.0021 判 M2 胜 | **+0.0038（M1 胜）** |
| M2 vs ridge, B/C | 更好 | B 更好 / C 跨 0 |

归一化主标量（只由 train 拟合的容差，`metrics/train_normalized_scalar.csv`，<1 优于
command-copy）：**A: 全部模型 0.70–0.75、ridge 0.71、ccopy 0.98 → 模型明确胜；
B: 模型 1.80–2.07 vs ccopy 1.81；C: 模型 1.86–2.06 vs ccopy 1.83 → 模型与 ccopy 相当或更差。**

逐轴 fixed-lead 27 个单元（split×lead×axis）中胜出次数：M0 13、M2 9、M3 5、**M1 0**。

## 一句话结论

修复使实验变得可解释、可复核；**但没有证伪研究方向，也没有确认任何模型胜出**。
V0.6 的"跨 episode context 已被证伪"与"模型分布内显著胜 command-copy"两个方向的
强结论**都不成立**：真实情况是 split 依赖——在已见命令族的 held-out anchor 上模型
明确优于 command-copy（≈0.71 vs 0.98），在未见命令族上模型与 command-copy 相当或更差，
且 Direct ≥ Context。

## 本轮成功标准（任务书 §7）自评

"成功标准是实验可解释、可复核，不是保证 M2/M1 胜出" —— 达成：
- 6 个确定性问题全部修复并有回归单测（T01–T10 全部 PASS）；
- 1 个运行时风险（§3）实测解决（VERIFIED），并因此**避免了不必要的数据重采**；
- 全部预测单次推理 + 哈希固定，bootstrap 只读同一缓存；
- 首个交付包可独立重算（含源码、12 checkpoint、240×2 数据 hash、逐窗 lineage、donor 表）。
- 未执行项如实标记：完整相对位姿 NOT_RUN；AQ5×Q4 仅单列；held-anchor 仅 2 组 → 探索性。
