"""Crick 螺旋主链生成: InternalCoord + build_template + phi/psi fit。

替代旧 pulchra_fix_backbone 流程。核心思路:

1. 用 ``generate_helix_ca_by_crick`` 生成目标 CA 轨迹 (Crick 参数定义)。
2. 用 ``build_template(resn, ss)`` 提供每个残基的 InternalCoord 模板 (含任意侧链)。
3. 逐残基 ``connect_internal_coords`` 拼成一条链, 得到完整主链/侧链的
   可重构 InternalCoord (刚性骨架恒定, 化学键长/角来自数据库)。
4. 两阶段拟合, 使链的 CA 贴合 Crick 目标 CA:

   * 关键洞察 (见 biorazer-development skill): 残基 ``i`` 的 CA 由
     ``omega_{i-1} = (CA_{i-1}, C_{i-1}, N_i, CA_i)`` 与 (omega 固定 trans 时)
     ``psi_{i-1} = (N_{i-1}, CA_{i-1}, C_{i-1}, N_i)`` 决定; 残基自身 ``phi``
     **不影响 CA 位置**。因此 *phi 不作为自由变量*, 固定为该 ss 的均值。
   * 阶段 1: 固定 omega=trans, 仅优化 anchor 刚体 + psi;
   * 阶段 2: 放开 omega (限在 trans 平面 170-190°), 优化 anchor + psi + omega。

对非 alpha-helix (CA-CA 间距偏离 3.8 Å) 的 Crick 螺旋, 化学键长/角恒定使
CA 拟合保留残差 (无法同时满足键长角与任意 CA 轨迹), 但主链/侧链始终
化学合理 —— 相比 pulchra 拉伸键长/角的物理破坏是本质改进。
"""

from __future__ import annotations

import copy
from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

from biorazer.database.molecule.bond.dihedral.protein import (
    ALIAS_QUAD,
    SS_BB_TORSION_ANGLE,
)
from biorazer.database.molecule.icoor.protein.template import (
    build_template,
    ss_torsions,
)
from biorazer.structure.manipulation.modification import connect_internal_coords


def _residue_torsions(ss: str) -> tuple[float, float, float]:
    """返回 ss 类的 (phi, psi, omega) 均值 (度)。"""
    t = ss_torsions(ss)
    return t["phi"], t["psi"], t["omega"]


def _bounds(ss: str, alias: str) -> tuple[float, float]:
    """该 ss 类二面角的 (lb, up) (度), 来自数据库。"""
    quad = ALIAS_QUAD[alias]
    rec = SS_BB_TORSION_ANGLE[ss][quad]
    return rec["lb"], rec["up"]


def _build_chain_ic(resn: str, ss: str, n: int):
    """构建 n 残基链的 InternalCoord (完整主链 + 侧链)。

    返回 (ic, per_atom_names, phi_mean, psi_mean, omega_mean)。
    """
    tpl = build_template(resn, ss, "canonical")
    per_atom_names = [a.name for a in tpl.atoms]
    n_heavy = len(per_atom_names)
    nC = per_atom_names.index("C")
    phi_t, psi_t, om_t = _residue_torsions(ss)

    ic = build_template(resn, ss, "canonical")
    for r in range(1, n):
        nxt = build_template(resn, ss, "canonical")
        # 唯一 res_id: _residue_ca_index 依赖 (chain_id, res_id) 定位 seam CA
        for a in nxt.atoms:
            a.res_id = r + 1
        off = len(ic)
        ic = connect_internal_coords(ic, nxt, C_index=off - n_heavy + nC, N_index=0)
        # connect_internal_coords 只在 seam 加了 omega; 补 psi / phi
        seam_psi = (off - n_heavy, off - n_heavy + 1, off - n_heavy + 2, off)
        seam_phi = (off - n_heavy + 2, off, off + 1, off + 2)
        if seam_psi not in ic.dihedra:
            ic.dihedra[seam_psi] = psi_t
        if seam_phi not in ic.dihedra:
            ic.dihedra[seam_phi] = phi_t
        # 只保留首残基 anchor 为根
        ic.anchor = {k: v for k, v in ic.anchor.items() if k < off}
    return ic, per_atom_names, phi_t, psi_t, om_t


def _ca_indices(per_atom_names, n):
    ca_col = per_atom_names.index("CA")
    return [r * len(per_atom_names) + ca_col for r in range(n)]


def _kabsch(P, Q):
    """P --(R, t)--> Q, 最小化 ||R P + t - Q|| (n,3) 输入。"""
    Pc = P - P.mean(0)
    Qc = Q - Q.mean(0)
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = Q.mean(0) - R @ P.mean(0)
    return R, t


def fit_bb_to_ca(resn, target_ca: np.ndarray, ss: str = "alpha-helix",
                 max_nfev: int = 10000):
    """用 InternalCoord + 两阶段 psi/omega 拟合, 把 backbone 建到 Crick 目标 CA。

    参数
    ----
    resn : str
        三字母残基名 (含任意侧链; ``GLY``/``ALA`` 等)。
    target_ca : (n, 3) ndarray
        Crick 生成的目标 CA 轨迹。
    ss : str
        二级结构类 (决定模板初始 phi/psi/omega 与拟合 bounds)。

    返回
    ----
    (atom_array, dict): 主链 AtomArray + 拟合统计。
    """
    n = len(target_ca)
    target_ca = np.asarray(target_ca, float)
    ic, names, phi_t, psi_t, om_t = _build_chain_ic(resn, ss, n)
    nh = len(names)
    k = n - 1
    ca_idx = _ca_indices(names, n)

    # seam dihedral quads (索引全局, 每次增长一残基)
    psi_quads = [(r * nh - nh, r * nh - nh + 1, r * nh - nh + 2, r * nh)
                 for r in range(1, n)]
    om_quads = [(r * nh - nh + 1, r * nh - nh + 2, r * nh, r * nh + 1)
                for r in range(1, n)]

    # 初始 anchor 刚体: 用理想 IC 链 CA 对目标 CA 做 Kabsch
    coords = ic.to_coords()
    ideal_ca = np.array([coords[i] for i in ca_idx], float)
    R0, t0 = _kabsch(ideal_ca, target_ca)
    rv0 = Rotation.from_matrix(R0).as_rotvec()

    psi_lo, psi_hi = _bounds(ss, "psi")
    om_lo, om_hi = _bounds(ss, "omega")

    def apply(x, omega_free: bool):
        rv, t = x[:3], x[3:6]
        psi = x[6:6 + k]
        om = x[6 + k:]
        R = Rotation.from_rotvec(rv).as_matrix()
        anchor = {i_: (np.asarray(v, float) @ R.T + t)
                  for i_, v in ic.anchor.items()}
        ic2 = copy.deepcopy(ic)
        ic2.anchor = anchor
        for q, v in zip(psi_quads, psi):
            ic2.dihedra[q] = v
        if omega_free:
            for q, v in zip(om_quads, om):
                ic2.dihedra[q] = v
        return ic2

    def residual(x, omega_free: bool):
        ic2 = apply(x, omega_free)
        cd = ic2.to_coords()
        ca = np.array([cd[i] for i in ca_idx], float)
        return (ca - target_ca).ravel()

    # ---- 阶段 1: 固定 omega=trans, 只优化 anchor 刚体 + psi (phi 固定) ----
    x0 = np.concatenate([rv0, t0, np.full(k, psi_t)])
    nvar1 = 6 + k
    lb1 = np.concatenate([np.full(6, -np.inf), np.full(k, psi_lo)])
    ub1 = np.concatenate([np.full(6, np.inf), np.full(k, psi_hi)])
    out1 = least_squares(lambda x: residual(x, False), x0,
                         bounds=(lb1[:nvar1], ub1[:nvar1]),
                         max_nfev=max_nfev, xtol=1e-6, ftol=1e-4)

    # ---- 阶段 2: 放开 omega, 但限在 trans 平面 ----
    x0b = np.concatenate([out1.x, np.full(k, om_t)])
    lb2 = np.concatenate([np.full(6, -np.inf),
                          np.full(k, psi_lo), np.full(k, om_lo)])
    ub2 = np.concatenate([np.full(6, np.inf),
                          np.full(k, psi_hi), np.full(k, om_hi)])
    out2 = least_squares(lambda x: residual(x, True), x0b,
                         bounds=(lb2, ub2),
                         max_nfev=max_nfev, xtol=1e-12, ftol=1e-12)

    best = apply(out2.x, True)
    cd = best.to_coords()
    ca = np.array([cd[i] for i in ca_idx], float)
    per = np.linalg.norm(ca - target_ca, axis=1)
    rmsd = float(np.sqrt((per ** 2).mean()))

    return best.to_atomarray(), {
        "rmsd": rmsd,
        "max_per_ca": float(per.max()),
        "phi_mean": float(phi_t),
        "psi_mean": float(out2.x[6:6 + k].mean()),
        "omega_mean": float(out2.x[6 + k:].mean()),
        "coords": cd,
    }