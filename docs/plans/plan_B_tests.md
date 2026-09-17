# Plan B: trim_or_extend 完整测试套件

> 目标: 为内坐标版 trim_or_extend 写正式 pytest, 覆盖 C/N 两端、伸长/缩短、
> backbone 化学合理性、既有链不动性、CCCP 束批量场景

## 背景

CrickHelix.trim_or_extend / CCCPHelixBundle.trim_or_extend 已从
"CA + pulchra 重建" 改为 "内坐标 build_template + connect + 优化"。
当前只有 ad-hoc 验证 (见 plan 主文档), 无正式测试。

## 任务

在 tests/ 下新增测试 (建议 tests/test_trim_or_extend_ic.py):

1. **C 端伸长**: 输入理想 alpha-helix 前 7 残基 (完整 backbone),
   伸长 3 个到 C 端, 断言:
   - 新残基 CA 与理想 10-mer 的 res8-10 一致 (atol 1e-2)
   - 新残基 backbone 键长化学合理 (N-CA≈1.458, CA-C≈1.525)
   - 既有链 CA 完全不动 (atol 1e-6)
   - res_id 正确 (1..10)
2. **N 端伸长**: 伸长 2 个到 N 端, 断言:
   - 新残基 CA 与 Crick 外推目标一致
   - 既有链完全不动
   - backbone 键长化学合理
   - res_id 正确 (-1, 0, 1..7)
3. **缩短**: N/C 端缩短, res_id 数量正确
4. **resn 参数**: 单字母/三字母/大小写, 新残基 res_name 正确
5. **错误处理**: 非法 terminus / 非整数 n / 非法 resn / 缩短超过长度
6. **CCCPHelixBundle**: 批量 spec {0: (nN, nC), 1: (nN, nC)} 场景,
   单根/多根螺旋伸长, param 清空
7. **既有链不动性**: 是所有场景的硬性断言 (内坐标方案的核心承诺)

## 注意事项

- 用 /opt/envs/BioRazer/bin/python 跑 (AGENTS.md 约定)
- 若在 worktree 中: sys.path 需前置 worktree 路径 (biorazer.pth 指向 main tree)
- 测试中构造理想 alpha-helix 用 build_bb_chain_ic (不要依赖 pulchra)
- 24/7 型 Crick 螺旋 (CA-CA 6.0Å) 与 alpha-helix 几何不兼容, 新残基 CA
  会有残余偏差 — 测试断言应以 "backbone 化学合理" 为主, CA 贴合为次

## 参考

- tests/test_assembly.py (现有 trim_or_extend 测试, TestCrickTrimOrExtend /
  TestCCCPTrimOrExtend, 需要同步更新以匹配新实现的行为)
- biorazer_prds/models/assembly_helix.py
- biorazer_prds/params/util.py
