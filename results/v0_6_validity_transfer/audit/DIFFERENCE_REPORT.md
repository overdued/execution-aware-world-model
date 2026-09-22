# DIFFERENCE_REPORT — 外部复核 10 条声明逐项复算

复算基于 v0.5 缓存（float32）与原始 manifest，容差 0.0005（相对）。
共 31 项，一致 31，**不一致 0**。

## 全量对照

```
 id                                      item      claimed   recomputed  match
  1             P_target normal->friction_low    -0.171499    -0.171499   True
  1             P_target friction_low->normal     0.686717     0.686717   True
  2                      low source native L2     1.349489     1.349489   True
  2                low source command-copy L2     1.308615     1.308615   True
  2                   low MAE vx command-copy     0.070384     0.070384   True
  2                        low MAE vx Context     0.068214     0.068214   True
  2                   low MAE vy command-copy     0.067221     0.067221   True
  2                        low MAE vy Context     0.086826     0.086826   True
  2                   low MAE wz command-copy     0.090306     0.090306   True
  2                        low MAE wz Context     0.129350     0.129350   True
  3          unique actual futures (all rows)   968.000000   968.000000   True
  3                N->L unique source futures   424.000000   424.000000   True
  3                N->L unique target futures   334.000000   334.000000   True
  5         N->L swap-vs-native mean distance     0.370617     0.370617   True
  5       N->L distance / native target error     0.274600     0.274635   True
  5                         N->L matched rows  5510.000000  5510.000000   True
  6 vlow/vx truth-on-pred slope (pearson^2/a)     0.993000     0.993020   True
  6                low/vx truth-on-pred slope     1.038280     1.038280   True
  6                low/wz truth-on-pred slope     0.849330     0.849325   True
  8   class_tau0.25 identical to class_tau0.5     1.000000     1.000000   True
  8        distinct per-window min-age values     3.000000     3.000000   True
  8            min-age value set == {0,1,inf}     1.000000     1.000000   True
  8                     rows with all-inf age 10682.000000 10682.000000   True
  9                        N->L D_before mean     1.216720     1.216723   True
  9   N->L ||e_B - e_A|| mean (actual-actual)     1.485315     1.485315   True
  9                       N->VL D_before mean     1.646530     1.646527   True
  9                  N->VL ||e_B - e_A|| mean     1.884600     1.884600   True
 10            N->L squared_target_gain total    -0.375238    -0.375238   True
 10             N->L per_axis_squared_gain vx     0.033757     0.033757   True
 10             N->L per_axis_squared_gain vy     0.110085     0.110085   True
 10             N->L per_axis_squared_gain wz    -0.519081    -0.519081   True
```

注：声明 4（episode-pair cluster 仍共享 episode）为概念性批评，无法由缓存复算，
在 A5 依赖统计中以 node-aware / leave-one-episode-out 处理。声明 7（affine 后 CI 跨 0）
与 V0.5 报告一致，已在本轮报告措辞中降级为 inconclusive。