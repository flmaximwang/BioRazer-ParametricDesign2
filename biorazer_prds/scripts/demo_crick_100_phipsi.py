#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Demo: 用 alpha-helix 等价 Crick 参数 (omega=100°, pitch=0.378) 生成螺旋,
再逐残基优化 phi/psi (锚点 9 自由度 + blen/bang 化学约束) 反推其数值。

背景
----
trim_or_extend 用 Crick 拟合参数外推新残基 CA。本 demo 验证: alpha-helix
二面角链 (phi=-60/psi=-45) 的 CA 轨迹能否重建 Crick 等角螺旋的 CA。

关键几何 (2026-09-17 修正)
--------------------------
- ground truth 用正确 pitch_angle=0.378 (alpha-helix 等价, CA-CA=3.80 Å)。
  旧值 0.876 (CA-CA=5.86 Å) 非 alpha-helix, 任何物理 backbone 都拟合不上,
  旧版优化器收敛到全伸展链 (psi/phi≈±180°), RMSD=4.31 Å。
- 锚点 = 残基 1 的 N/CA/C 三个自由原子 (9 自由度), 加 blen (N-CA=1.458,
  CA-C=1.525) 与 bang (N-CA-C=111.2°) 高权重残差 (权重 100+), 防止优化器
  靠扭曲锚点硬凑。
- phi/psi 加 alpha-helix 物理范围约束: phi∈[-85,-35], psi∈[-75,-15]。
- carbonyl O 必须跟踪 psi: dihedral(N,CA,C,O) = psi - 180 (trans 肽平面)。
  旧版 O 固定为模板值 135°, psi 优化到 ±180° 时 O 与下一个残基 N 碰撞
  (O···N_next ≈ 0.65 Å, 应为 2.8~3.0 Å)。

输出
----
- 终端: 每残基 phi/psi, 逐残基 CA 偏差, 总 RMSD, 锚点键长/键角
- demo_phi_psi_fit.pdb: 优化后的完整 backbone (O 已跟踪 psi)

运行:
    python demo_crick_100_phipsi.py
"""

import sys
from pathlib import Path

import numpy as np

# 确保从本 worktree 导入 biorazer_prds (而非 main tree 的已安装版本)
_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from biorazer_prds.params.helix_cp.generate import generate_helix_ca_by_crick

# --------------------------------------------------------------------------- #
# 1) 用 alpha-helix 等价 Crick 参数生成 CA 螺旋 (ground truth)
# --------------------------------------------------------------------------- #
RESIDUE_NUM = 12
OMEGA_DEG = 100.0
PITCH_ANGLE = 0.378  # alpha-helix 等价 (plan A 扫描: CA-CA=3.80 Å, 上升 1.57 Å)

xyz, gen_param = generate_helix_ca_by_crick(
    residue_num=RESIDUE_NUM,
    centroid=(0.0, 0.0, 0.0),
    direction=(0.0, 0.0, 1.0),
    radius=2.26,
    omega=np.radians(OMEGA_DEG),
    pitch_angle=PITCH_ANGLE,
    phi0=0.0,
)
print(f"[1] Crick 生成 {RESIDUE_NUM} 个 CA, omega={OMEGA_DEG}°, pitch={PITCH_ANGLE} rad")
print(f"    相邻 CA 距离: {np.round(np.linalg.norm(np.diff(xyz, axis=0), axis=1), 3)}")

# --------------------------------------------------------------------------- #
# 2) 为 CA 轨迹重建 backbone, 联合优化锚点 (9 DOF + blen/bang) + 所有 phi/psi
# --------------------------------------------------------------------------- #
from biorazer.database.molecule.icoor.protein.template import build_template
from biorazer.structure.manipulation.modification import connect_internal_coords
from scipy.optimize import least_squares


def build_full_chain(n_res):
    """构建 n_res 残基的完整 backbone 链, 返回 InternalCoord。

    锚点 = 残基 1 的 N/CA/C (模板坐标), 其余原子由二面角生长。
    每残基内部: phi_i (C_{i-1},N_i,CA_i,C_i), psi_i (N_i,CA_i,C_i,N_{i+1}),
    omega_i (CA_i,C_i,N_{i+1},CA_{i+1})。变量取 phi/psi (omega 固定 trans)。
    """
    ic = build_template("GLY", "alpha-helix", "canonical")
    ic.res_id = [1] * len(ic)
    ic.chain_id = ["A"] * len(ic)
    for i in range(2, n_res + 1):
        nxt = build_template("GLY", "alpha-helix", "canonical")
        nxt.res_id = [i] * len(nxt)
        nxt.chain_id = ["A"] * len(nxt)
        merged = connect_internal_coords(ic, nxt, C_index=len(ic) - 2, N_index=0)
        off = len(ic)
        merged.anchor = {k: v for k, v in merged.anchor.items() if k < off}
        ic = merged
    return ic


ic = build_full_chain(RESIDUE_NUM)
# 识别每残基的原子下标
atom_res = np.array([a.res_id for a in ic.atoms])
res_idx = {rid: {} for rid in np.unique(atom_res)}
for idx, a in enumerate(ic.atoms):
    res_idx[a.res_id][a.name] = idx

# 变量键:
# psi_i  quad = (N_i, CA_i, C_i, N_{i+1})      i = 1..n-1
# phi_i  quad = (C_{i-1}, N_i, CA_i, C_i)      i = 2..n
# O_i    quad = (N_i, CA_i, C_i, O_i)          跟踪 psi_i (仅 1..n-1)
psi_keys = [(res_idx[i - 1]["N"], res_idx[i - 1]["CA"], res_idx[i - 1]["C"],
             res_idx[i]["N"]) for i in range(2, RESIDUE_NUM + 1)]
phi_keys = [(res_idx[i - 1]["C"], res_idx[i]["N"], res_idx[i]["CA"],
             res_idx[i]["C"]) for i in range(2, RESIDUE_NUM + 1)]
o_keys = [(res_idx[i]["N"], res_idx[i]["CA"], res_idx[i]["C"],
           res_idx[i]["O"]) for i in range(1, RESIDUE_NUM + 1)]

n_anchor = 9
n_psi = len(psi_keys)
n_phi = len(phi_keys)

# 初始: 锚点 = 模板 N1/CA1/C1 坐标; psi=-45, phi=-60
raw_anchor = ic.anchor  # {0: N, 1: CA, 2: C}
x0 = np.zeros(n_anchor + n_psi + n_phi)
x0[0:9] = np.concatenate([np.asarray(raw_anchor[k], float) for k in (0, 1, 2)])
x0[9:9 + n_psi] = -45.0
x0[9 + n_psi:] = -60.0


def unpack(x):
    anchor = x[0:9].reshape(3, 3)   # N1, CA1, C1
    psi = x[9:9 + n_psi]
    phi = x[9 + n_psi:9 + n_psi + n_phi]
    return anchor, psi, phi


def build_coords(x):
    anchor, psi, phi = unpack(x)
    ic2 = build_full_chain(RESIDUE_NUM)
    ic2.anchor = {0: tuple(anchor[0]), 1: tuple(anchor[1]), 2: tuple(anchor[2])}
    for q, v in zip(psi_keys, psi):
        ic2.dihedra[q] = v
    for q, v in zip(phi_keys, phi):
        ic2.dihedra[q] = v
    # O 跟踪 psi: trans 肽平面 dihedral(N,CA,C,O) = psi - 180
    for j in range(RESIDUE_NUM - 1):
        ic2.dihedra[o_keys[j]] = psi[j] - 180.0
    coords = ic2.to_coords()
    ca = np.array([coords[res_idx[i]["CA"]] for i in range(1, RESIDUE_NUM + 1)])
    return ca


BOND_IDEAL = {"N-CA": 1.458, "CA-C": 1.525}
ANGLE_IDEAL = 111.2
W_BLEN_BANG = 10.0  # 残差缩放 = sqrt(权重 100)


def _angle(a, b, c):
    v1 = a - b
    v2 = c - b
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def resid(x):
    ca = build_coords(x)
    r = (ca - xyz).ravel()
    # 锚点 blen/bang 化学约束 (高权重, 权重 100+)
    anchor, _, _ = unpack(x)
    N1, CA1, C1 = anchor
    r = np.concatenate([
        r,
        W_BLEN_BANG * np.array([
            np.linalg.norm(N1 - CA1) - BOND_IDEAL["N-CA"],
            np.linalg.norm(CA1 - C1) - BOND_IDEAL["CA-C"],
            _angle(N1, CA1, C1) - ANGLE_IDEAL,
        ]),
    ])
    return r


# phi/psi alpha-helix 物理范围约束 (锚点自由)
lb = np.full(n_anchor + n_psi + n_phi, -np.inf)
ub = np.full(n_anchor + n_psi + n_phi, np.inf)
lb[9:9 + n_psi] = -75.0
ub[9:9 + n_psi] = -15.0
lb[9 + n_psi:] = -85.0
ub[9 + n_psi:] = -35.0

res = least_squares(resid, x0, bounds=(lb, ub), max_nfev=20000)
ca_final = build_coords(res.x)
per_res = np.linalg.norm(ca_final - xyz, axis=1)
rmsd = np.sqrt((per_res ** 2).mean())
print(f"\n[2] 联合优化 (锚点9自由度+blen/bang + {n_psi} psi + {n_phi} phi) 完成")
print(f"    优化后 RMSD: {rmsd:.4f} A")
print(f"    逐残基 |d|: {np.round(per_res, 3)}")
print(f"    最大偏差: {per_res.max():.3f} A")

anchor, psi, phi = unpack(res.x)
print("\n    优化后 psi (逐残基):", np.round(psi, 2), " (理想 -45)")
print("    优化后 phi (逐残基):", np.round(phi, 2), " (理想 -60)")
N1, CA1, C1 = anchor
print(f"    锚点键长: N-CA={np.linalg.norm(N1-CA1):.3f} (1.458)  "
      f"CA-C={np.linalg.norm(CA1-C1):.3f} (1.525)  "
      f"N-CA-C={_angle(N1, CA1, C1):.1f}° (111.2°)")

# --------------------------------------------------------------------------- #
# 3) 输出优化后的 backbone 结构 (O 已跟踪 psi)
# --------------------------------------------------------------------------- #
ic_final = build_full_chain(RESIDUE_NUM)
ic_final.anchor = {0: tuple(anchor[0]), 1: tuple(anchor[1]), 2: tuple(anchor[2])}
for q, v in zip(psi_keys, psi):
    ic_final.dihedra[q] = v
for q, v in zip(phi_keys, phi):
    ic_final.dihedra[q] = v
for j in range(RESIDUE_NUM - 1):
    ic_final.dihedra[o_keys[j]] = psi[j] - 180.0
arr = ic_final.to_atomarray()
out = Path(__file__).with_name("demo_phi_psi_fit.pdb")
from biorazer.structure.io.protein import AtomArray_Pdb
AtomArray_Pdb(output_io=out).write(arr)
print(f"\n[3] 优化后的 backbone 已写出: {out}")
print(f"    RMSD = {rmsd:.4f} A")
