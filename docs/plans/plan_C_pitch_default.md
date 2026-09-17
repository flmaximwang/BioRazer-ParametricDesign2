# Plan C: pitch_angle 默认值核查与修正

> 范围: biorazer_prds/params/helix_cp/generate.py (+ fit.py 签名默认值)
> 状态: 用户已在原始仓库修复, 本分支需同步确认

## 问题

`generate_helix_ca_by_crick` 默认 `pitch_angle=0.876` rad (50.2°),
产生 CA-CA=6.0 Å、每残基上升 4.87 Å 的松散螺旋, **不是 alpha-helix**。

alpha-helix 的等价 Crick 参数:
- omega ≈ 100° / 残基 (3.6 残基/圈)
- pitch_angle ≈ 0.36~0.38 rad (20.4°~21.8°)
- 每残基上升 1.5 Å, 一圈 5.4 Å

## 0.876 的来源

- Initial commit (98644e8, 2025-08-14) 就在
  `biorazer_sba/utils/helix/cccp/generator.py` 和 `fitter.py`
- 注释仅写 "pitch angle in radians", 无物理来源说明
- 传承至今 (generate.py:11, fit.py:98, assembly_helix.py:99)

## 影响

- 默认参数生成的螺旋几何错误 (非 alpha-helix)
- trim_or_extend 外推的目标 CA 基于错误螺旋 -> N 端偏差 ~5.5 Å
- CCCP 束的 pitch_angles 默认 -0.2096 是另一套 (coiled-coil 缠绕角), 不受影响

## 任务

1. 确认用户已在原始仓库的修复内容 (查 BioRazer 仓库 generate.py / fit.py)
2. 本分支同步: 确认/更新默认 pitch_angle
   - 选项 a: 改为 ~0.36 rad (alpha-helix)
   - 选项 b: 保留 0.876 但明确文档注明其含义 (若 0.876 是某设计意图)
3. 检查所有依赖默认值的调用点 (from_param, trim_or_extend 等) 是否受影响
4. 确认 fit_helix_by_crick 的默认 pitch_angle 同样处理
5. 跑相关测试确认无回归

## 决策点

需要用户确认 0.876 是否有设计意图 (比如某种特定设计螺旋),
还是单纯 magic number 应改为 alpha-helix 值。

## 参考

- biorazer_prds/params/helix_cp/generate.py:11
- biorazer_prds/params/helix_cp/fit.py:98
- biorazer_prds/models/assembly_helix.py:99
- git: 98644e8 (Initial commit)
