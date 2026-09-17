# Plan: fix_trim_or_extend 内坐标重建 backbone

> 分支: `fix_trim_or_extend` (worktree: `.worktrees/fix_trim_or_extend`)
> 状态: 进行中 — 内坐标化主体完成并验证 (C/N 端新 CA 距 Crick 目标
> 0.11/0.21 Å, 既有链不动, 键长理想); 待 Plan A demo 收尾 / Plan B 正式
> 测试 / Plan D N 端目标语义决策

## 背景与问题

`CrickHelix.trim_or_extend` / `CCCPHelixBundle.trim_or_extend` 伸长时,
旧实现用 "CA + pulchra 重建 backbone" (pulchra_fix_backbone)。
实测发现 pulchra 对非 alpha-helix 型 Crick 轨迹 (CA-CA ≈ 6.0 Å)
重建出**物理畸变** backbone:

- N-CA 键长 3.01 Å (理想 1.458)
- CA-C 键长 2.69 Å (理想 1.525)
- N-CA-C 键角 90.1° (理想 111.2°)

用户要求改用 BioRazer 内坐标表示:
`build_template(resn, alpha_helix, canonical)` 生成新残基模板 →
`connect_internal_coords` 连接两段 → 对 backbone 的 phi/psi/omega
做分阶段优化, 使 CA 在 ss 允许范围内贴近拟合轨迹。

## 已完成并验证

### 1. 内坐标工具函数 (biorazer_prds/params/util.py)

- `_kabsch(P, Q)`: 刚体对齐 (Kabsch)
- `build_bb_chain_ic(n_res, resn, ss, start_res, chain_id)`:
  build_template 串联 n 个残基, 补连接处 psi/phi, 只保留链首 anchor
- `_place_new_fragment(ic_new, ic_old, ss, terminus)`:
  把新片段 anchor 从模板局部坐标换到实际空间坐标 (C 端对齐既有链末残基,
  N 端对齐既有链首残基), 用 Kabsch + 理想 (n+1)-mer 参考链
- `connect_ic_fragments(ic_old, ic_new, ss, terminus)`:
  连接两段, **两段 anchor 都保留**, 连接处仅靠 connect_internal_coords
  记录的 omega (不需补 psi/phi), 使 to_coords 时两段都不动
- `backbone_dihedral_quads(ic, new_atoms)`:
  只选 "旋转轴下游在新片段" 的主链二面角 (psi 第4原子/omega 第3原子/
  phi 第4原子在新片段) 作为可优化变量, 避免牵动既有链
- `optimize_bb_to_ca(ic, target_ca, ca_indices, ss, new_atoms)`:
  分阶段优化 (Stage1 只 psi/phi, Stage2 全部), lb/up 从
  `SS_BB_TORSION_ANGLE[ss]` 数据库读取

### 2. CrickHelix.trim_or_extend 改造 (biorazer_prds/models/assembly_helix.py)

- 伸长分支改用 `_extend_backbone_internal_coord(n, resn, terminus)`:
  1. 目标 CA: Crick 拟合参数外推 (generate_helix_ca_by_crick)
  2. 既有链 -> InternalCoord.from_atomarray (anchor 在链首, 实际坐标)
  3. 新片段 build_bb_chain_ic
  4. connect_ic_fragments (N/C 端)
  5. optimize_bb_to_ca
  6. to_atomarray 重建, 按 res_id 排序
- 缩短分支不变 (remove_atoms)
- 漏传 terminus 参数 bug 已修 (N 端曾走 C 端逻辑)

### 3. 验证结果 (ad-hoc, 非完整测试套件; 2026-09-17 现场重跑)

输入 = 7 残基 alpha-helix 等价 Crick 链 (omega=100°, pitch=0.378,
radius=2.26, CA 经 pulchra 重建 backbone), 链坐在自身居中 t 网格上:
- C 端伸长 3: 既有链最大位移 1.4e-15 Å (不动), 新残基 CA 距 Crick 目标
  0.11 Å, N-CA=1.458 / CA-C=1.525 (理想)
- N 端伸长 2: 既有链最大位移 1.4e-15 Å (不动), 新残基 CA 距 Crick 目标
  0.21 Å, 键长理想
- 注意: 验证输入须坐在自身长度网格上 (from_param 语义)。若输入是更长网格
  的子链 (如 10-mer 取前 7), 目标外推带网格相位差 (实测 ~2-3 Å), 非代码 bug。

### 4. 本次会话修复的两个 bug (2026-09-17)

- 接缝几何必须实测传入: `connect_internal_coords` 默认用理想 Engh & Huber
  值记录接缝; 实际摆放几何与之不符时 (非理想输入), to_coords 抛
  "Inconsistent coordinate"。已新增 `_seam_geometry` 实测四原子
  (CA, C, N, CA) 几何后显式传入。
- N 端接缝不可优化 (架构限制): 接缝 omega 的父原子是新片段末残基 (CA/C),
  其放置的既有链首残基 CA 同时在 anchor 里; 新片段任何内部二面角旋转都会
  传播到接缝父原子, to_coords 判定接缝放置与 anchor 不一致而抛错 (既有链
  又必须不动)。因此 N 端跳过优化 (实测 0.21 Å, 与 C 端优化后同量级);
  C 端可优化 (接缝父原子是既有链原子, 不动)。
- CA-only 输入 (backbone_type="CA") 保持 CA-only 延长: 无既有 backbone 可
  连接, 也不伪造 pulchra 畸变 backbone; 新 CA 由 Crick 外推生成。

## 关键发现 (影响后续设计)

### A. Crick 拟合外推的 N 端偏差 ~5.5 Å

用 res1-7 拟合 Crick, 外推 N 端 2 个 CA, 与化学合理 alpha-helix 延续
(理想 12-mer 前 2 个) 差 5.473 Å。C 端外推则 0 Å (不对称)。

排查结论 (demo + 数值验证):
- fit 本身无 bug: 用已知 Crick 参数生成的螺旋反拟合, 参数完全还原 (rmsd 4.8e-8)
- generate 本身无 bug: 同一参数下不同 residue_num 网格自洽
- **根因: 默认 pitch_angle=0.876 不是 alpha-helix 的 pitch**
  - 0.876 rad (50.2°) → 每残基上升 4.87 Å, 一圈 17.04 Å
  - alpha-helix 应约 0.36 rad (20.4°) → 每残基上升 1.5 Å
  - omega=100° + r=2.26 + pitch≈0.378 才匹配 alpha-helix (CA-CA=3.80 Å)
- 0.876 是 Initial commit (2025-08-14) 就存在的 magic number, 无文档来源

### B. N-CA-C-O 二面角 bug (database 层, 用户已修)

build_template 硬编码 carbonyl O 的 (N, CA, C, O) dihedral = 180.0
(template.py 第 318 行)。真实 alpha-helix (1MBN 统计) 该二面角主峰
120°~150°。用户指示: 正常应使用 (CA, C, O, N_{i+1}) 二面角控制 O 位置。
**用户已在原始仓库修复, 无需本分支处理。**

## 未完成 / 待办

### 1. demo 脚本收尾 (biorazer_prds/scripts/demo_crick_100_phipsi.py)

**已完成 (2026-09-17)**。修正:
- ground truth 改用正确 pitch_angle=0.378 (CA-CA=3.80 Å)
- 锚点 9 自由度 (N1/CA1/C1 坐标) + blen/bang 高权重残差 (权重 100+)
- phi/psi alpha-helix 物理范围约束 (phi∈[-85,-35], psi∈[-75,-15])
- carbonyl O 跟踪 psi: dihedral(N,CA,C,O) = psi - 180 (trans 肽平面)

验证结果: RMSD 0.0025 Å (12 残基, 逐残基 ≤0.003 Å); psi 收敛 -47.4→-38.6,
phi -58.7→-64.9 (理想 -60/-45, 带轻微漂移以贴合等角螺旋); 锚点 N-CA=1.458,
CA-C=1.525, N-CA-C=111.2°; O···N_next=2.25 Å (trans 肽平面正确值)。输出 PDB
写入 scripts/ 但已被 .gitignore 覆盖 (biorazer_prds/scripts/*_fit.pdb)。

### 2. 完整测试套件

**已完成 (2026-09-17)**: 全量 pytest 79 passed (含 8 个新 IC 测试)。
- [x] 跑 tests/ 全量 pytest (amino_acid import 已改 alphabet)
- [x] 为 trim_or_extend 内坐标实现写正式测试 (tests/test_trim_or_extend_ic.py:
      C/N 端伸长后 backbone 化学合理、CA 贴合目标 (atol 1e-3)、既有链不动
      (atol 1e-6)、缩短、resn、错误处理)
- [x] 覆盖 CCCPHelixBundle.trim_or_extend (批量 spec 场景, 见
      tests/test_assembly.py TestCCCPTrimOrExtend)
- 测试暴露并修复 bug: build_bb_chain_ic/_place_new_fragment 硬编码 4 原子/
  残基, resn≠GLY (如 ALA 带 CB) 时 C_index 取到 O 崩溃 → 改为按残基号/
  原子名查找。
- 已知限制 (不测): 顺序多次单端伸长会因居中 t 网格相位漂移导致目标错位;
  CCCP 束用"一次生成双端"规避 (见 plan 主文档关键发现)。

### 3. 未提交改动整理

**已完成 (2026-09-17)**: 按 AGENTS.md 逐步 commit —— import 修复 / params
IC 工具 / models 内坐标化 / 测试 PYTHONPATH / gitignore / docs / demo /
O 跟踪 psi / IC 测试 / 非 GLY bug 修复。
- demo_crick_100_phipsi.py 已纳入 (feat(demo))
- demo_phi_psi_fit.pdb 已加入 .gitignore
- 分支落后 main 5 个 commit (pitch 默认值修复等), 后续需 rebase/merge 同步

## 相关文件

- biorazer_prds/models/assembly_helix.py
- biorazer_prds/params/util.py
- biorazer_prds/params/helix_cp/generate.py (pitch 默认值待确认)
- biorazer_prds/params/helix_cp/fit.py
- biorazer_prds/scripts/demo_crick_100_phipsi.py
- 参考: biorazer 仓库 biorazer/database/molecule/icoor/protein/template.py
  (build_template, N-CA-C-O 硬编码 180 已由用户在原始仓库修复)
