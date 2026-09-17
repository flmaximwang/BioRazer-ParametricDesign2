# Plan D: N 端伸长目标 CA 的语义决策

> 范围: CrickHelix.trim_or_extend N 端伸长时的目标 CA 来源
> 状态: **已按方案 1 落实 (2026-09-17)** — Crick 跟随重设计后, 新片段在
> 连接前于自身坐标系解出贴 Crick 轨迹的二面角, N 端与 C 端对称, 均精确
> 跟随 Crick 外推 (build_bb_chain_ic 理想输入 0.0000 Å)。若后续要改
> 方案 2 (化学合理延续), 只需替换 build_bb_chain_following_ca 的目标
> CA 来源 (用原结构几何外推替代 Crick 外推), 接缝/放置机制不变。

## 问题

N 端伸长时, 新残基的目标 CA 来自 Crick 拟合参数外推。
但 Crick 拟合的 N 端外推方向与原结构 (alpha-helix) 的 N 端方向不一致:

- Crick 外推 N 端目标: (-2.896, 2.211, -2.472) (螺旋 -z 端)
- 化学合理 alpha-helix 延续 (从 res1 反向生长): (1.458, 0, 0)
- 两者差 ~5.5 Å

C 端外推则一致 (0 Å), 因为 C 端方向与拟合螺旋方向相同。

## 根因 (已诊断)

1. Crick fit/generate 本身无 bug (已知参数反拟合完全还原)
2. 默认 pitch_angle=0.876 非 alpha-helix 值 (见 Plan C) — 这是主要贡献
3. 即使 pitch 修正, 等角螺旋 vs 二面角链仍有模型差异 (omega 98.87 vs 100)

## 可选方案

### 方案 1: 跟随 Crick 外推 (当前实现)
- 新残基 CA = Crick 拟合轨迹外推
- 贴合拟合螺旋, 但偏离化学合理 alpha-helix 延续 ~5.5 Å
- 连接处 backbone 几何可能扭曲

### 方案 2: 跟随原结构几何 (化学合理)
- 新残基 CA = 从原结构 N 端沿 alpha-helix 延续 (内坐标反向生长)
- 化学合理, backbone 键长键角正常
- 偏离 Crick 拟合轨迹

### 方案 3: 混合
- 连接处局部用原结构几何, 远端逐渐过渡到 Crick 轨迹
- 复杂, 需额外设计

## 任务

1. 用户决策方案 (1/2/3)
2. 按决策修改 `_extend_backbone_internal_coord` 的目标 CA 生成逻辑
   (assembly_helix.py:245-252)
3. 更新 Plan B 测试的 N 端断言以匹配决策

## 前提

Plan C (pitch 默认值) 应先行, 因为目标 CA 本身依赖正确的 Crick 参数。
