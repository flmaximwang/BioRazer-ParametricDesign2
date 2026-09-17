# Plan A: demo 脚本收尾 — 正确 ground truth + 锚点 blen/bang 约束

> 入口: `biorazer_prds/scripts/demo_crick_100_phipsi.py`
> 状态: 已完成 (2026-09-17) — RMSD 0.0025 Å, psi/phi 收敛到 -60/-45 附近
> (带轻微漂移以贴合等角螺旋), 锚点键长/键角理想, O 已跟踪 psi

## 背景

demo 脚本验证 "Crick 等角螺旋是否能由 backbone phi/psi 重建"。
当前版本有两个缺陷:
1. ground truth 用了错误的默认 pitch_angle=0.876 (CA-CA=5.86 Å, 非 alpha-helix)
2. 锚点 6 自由度无化学约束, 优化器把 backbone 二面角扭到非物理值 (±180°)

## 已确认的事实

- alpha-helix (phi=-60/psi=-45 链) 的 CA-CA = 3.804 Å
- 匹配 alpha-helix 的 Crick 参数: omega≈100°, r=2.26, **pitch≈0.378 rad**
  (扫描验证: pitch=0.378 → CA-CA=3.80 Å, 上升 1.57 Å)
- Crick fit 给出的 pitch≈0.372 (与 0.378 一致, 说明 fit 没错)
- 默认 0.876 rad 是 Initial commit 就有的 magic number, 非 alpha-helix 值

## 任务

1. ground truth 改用正确参数:
   `generate_helix_ca_by_crick(residue_num=12, radius=2.26, omega=radians(100), pitch_angle=0.378, phi0=0)`
   (或直接用 fit_helix_by_crick 对理想 alpha-helix CA 拟合出的参数)
2. 优化变量: 锚点 3 原子坐标 (9 自由度) + 所有残基 phi/psi
   (或锚点刚体 6 自由度 + 内部 blen/bang 约束)
3. **锚点 blen/bang 化学约束** (用户指示):
   - blen: N-CA = 1.458, CA-C = 1.525 (作为高权重残差项)
   - bang: N-CA-C = 111.2° (作为高权重残差项)
   - 权重建议 100+ (相对 CA 位置残差)
4. phi/psi 加物理范围约束 (如 alpha-helix: phi ∈ [-85,-35], psi ∈ [-75,-15])
5. 报告: 每残基 phi/psi, 逐残基 CA 偏差, 总 RMSD
6. 验证: 若 Crick 等角螺旋 = alpha-helix 二面角链, 反推 phi/psi 应收敛到
   -60/-45 附近, RMSD 应 < 1 Å (理想情况)

## 输出

- 终端: 每残基 phi/psi + RMSD
- PDB: 写入临时目录 (不要留在 scripts/, 或加入 .gitignore)

## 参考代码位置

- biorazer_prds/scripts/demo_crick_100_phipsi.py (当前版本)
- biorazer_prds/params/helix_cp/generate.py
- biorazer_prds/params/helix_cp/fit.py
- biorazer_prds/params/util.py (build_bb_chain_ic, _kabsch 等)
