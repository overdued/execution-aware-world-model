# Smoke validation — V0.6 阶段B

## 1. 摩擦注入 readback
```
normal: wrote=1.0 readback_static=1.0000 (min 1.0000 max 1.0000) dyn=0.7000 ground_static=1.0 combine=physx_default_average（代码常量注明，未从引擎枚举读回） -> OK
friction_mid: wrote=0.6 readback_static=0.6000 (min 0.6000 max 0.6000) dyn=0.4200 ground_static=1.0 combine=physx_default_average（代码常量注明，未从引擎枚举读回） -> OK
friction_low: wrote=0.3 readback_static=0.3000 (min 0.3000 max 0.3000) dyn=0.2100 ground_static=1.0 combine=physx_default_average（代码常量注明，未从引擎枚举读回） -> OK
```
注：读回验证的是 robot material 写入路径；有效足-地摩擦还受地面 material 与 average combine 影响（effective ≈ (robot+ground)/2），档位解释按写入值命名。

## 2. anchor 复原（同 seed 跨 condition）
```
AQ0/Q1_straight: max |Δanchor joint pos| vs normal = 0.00e+00
AQ0/Q3_turn: max |Δanchor joint pos| vs normal = 0.00e+00
AQ1/Q1_straight: max |Δanchor joint pos| vs normal = 0.00e+00
AQ1/Q3_turn: max |Δanchor joint pos| vs normal = 0.00e+00
```
最大 anchor 复原差 0.00e+00（approximate-paired；solver contact cache 不可复原）。

## 3. 记录完整性
```
ep0 normal/Q1_straight: T=200 dt∈[0.0200,0.0200] settle=50步 term=schedule_end OK
ep1 normal/Q1_straight: T=200 dt∈[0.0200,0.0200] settle=50步 term=schedule_end OK
ep2 normal/Q1_straight: T=200 dt∈[0.0200,0.0200] settle=50步 term=schedule_end OK
ep3 normal/Q1_straight: T=200 dt∈[0.0200,0.0200] settle=50步 term=schedule_end OK
ep4 normal/Q3_turn: T=200 dt∈[0.0200,0.0200] settle=50步 term=schedule_end OK
ep5 normal/Q3_turn: T=200 dt∈[0.0200,0.0200] settle=50步 term=schedule_end OK
ep6 normal/Q3_turn: T=200 dt∈[0.0200,0.0200] settle=50步 term=schedule_end OK
ep7 normal/Q3_turn: T=200 dt∈[0.0200,0.0200] settle=50步 term=schedule_end OK
ep8 normal/Q1_straight: T=200 dt∈[0.0200,0.0200] settle=50步 term=schedule_end OK
ep9 normal/Q1_straight: T=200 dt∈[0.0200,0.0200] settle=50步 term=schedule_end OK
ep10 normal/Q1_straight: T=200 dt∈[0.0200,0.0200] settle=50步 term=schedule_end OK
ep11 normal/Q1_straight: T=200 dt∈[0.0200,0.0200] settle=50步 term=schedule_end OK
```

## 4. 命令执行一致性（settle 段 cmd=0）
```
ep0: settle 段 |cmd|max=0.00e+00
ep1: settle 段 |cmd|max=0.00e+00
ep2: settle 段 |cmd|max=0.00e+00
ep3: settle 段 |cmd|max=0.00e+00
```

## 5. termination 分布
`{'schedule_end': 48}`

## 判定: SMOKE PASS — 可进入增量 pilot
