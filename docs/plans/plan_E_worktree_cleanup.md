# Plan E: 工作树收尾与提交整理

> 范围: 本 worktree (fix_trim_or_extend) 的未提交改动整理
> 状态: 已完成 (2026-09-17) — 见主文档"未提交改动整理"

## 当前未提交改动

```
 M biorazer_prds/models/assembly_helix.py     # trim_or_extend 内坐标改造
 M biorazer_prds/params/util.py               # 内坐标工具函数
 ?? biorazer_prds/scripts/demo_crick_100_phipsi.py  # demo 脚本
 ?? biorazer_prds/scripts/demo_phi_psi_fit.pdb       # demo 输出 (应 gitignore)
 ?? docs/plans/*.md                            # 本批 plan 文档
```

## 任务

1. **demo 输出文件处理**:
   - demo_phi_psi_fit.pdb 是运行产物, 应加入 .gitignore 或从 scripts/ 移出
   - 建议: .gitignore 加 `biorazer_prds/scripts/*_fit.pdb` 或让 demo 输出到临时目录

2. **按 AGENTS.md 约定逐步 commit** (每步一个 commit, 标准格式):
   - commit 1: fix(import): biorazer.database.amino_acid -> alphabet
   - commit 2: feat(params): 内坐标 backbone 工具 (build_bb_chain_ic,
     connect_ic_fragments, optimize_bb_to_ca 等)
   - commit 3: feat(models): CrickHelix.trim_or_extend 内坐标化
   - commit 4: docs: 工作要点与 plans
   - (demo 脚本视 Plan A 完成情况决定是否纳入)

3. **CCCPHelixBundle.trim_or_extend 是否同步内坐标化** (待确认):
   - 当前 CCCP 版本仍用 pulchra_fix_backbone
   - 若需要, 作为独立 commit/plan

## 验证

- 每个 commit 后跑相关测试确认无回归
- 最终 git status 干净
- 与 main 分支同步 (rebase/merge 视用户安排)

## 参考

- AGENTS.md: 每步一个 commit, 测试用 /opt/envs/BioRazer/bin/python
