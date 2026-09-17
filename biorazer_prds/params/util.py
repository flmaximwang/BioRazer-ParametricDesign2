import numpy as np
import biotite.structure as bio_struct


# --------------------------------------------------------------------------- #
# 内坐标 backbone 构建与优化 (trim_or_extend 伸长用)
#
# 伸长不再走 "CA + pulchra 重建" (pulchra 对非 alpha-helix 型 Crick 轨迹会
# 重建出 N-CA ~3Å 的畸变 backbone), 改为: 用 biorazer 的内坐标模板
# (build_template, 化学理想键长/键角) 构建新残基 backbone, connect 到既有链,
# 再对 backbone 的 phi/psi/omega 做分阶段优化, 使新残基 CA 在 ss 允许范围内
# 尽量贴近拟合轨迹给出的目标 CA。
# --------------------------------------------------------------------------- #

def _kabsch(P, Q):
    """把 P (n×3) 刚体对齐到 Q (n×3), 返回 (R, t): Q ≈ R @ P.T + t。"""
    p_cent = P.mean(0)
    q_cent = Q.mean(0)
    Pc = P - p_cent
    Qc = Q - q_cent
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = q_cent - R @ p_cent
    return R, t


def build_bb_chain_ic(n_res, resn="GLY", ss="alpha-helix", start_res=1,
                      chain_id="A"):
    """用 build_template 串 n_res 个残基的内坐标 backbone 链。

    每残基经 connect_internal_coords 连接, 并补上连接处的 psi (N_i,CA_i,C_i,N_j)
    与 phi (C_i,N_j,CA_j,C_j) 二面角 (connect 本身只记录 omega); 只保留链首
    残基的 anchor, 其余 anchor 删除, 使全链由 backbone 二面角唯一决定。
    """
    from biorazer.database.molecule.icoor.protein.template import build_template
    from biorazer.structure.manipulation.modification import (
        connect_internal_coords,
    )

    ic = build_template(resn, ss, "canonical")
    ic.res_id = [start_res] * len(ic)
    ic.chain_id = [chain_id] * len(ic)
    phi_t, psi_t = ic.phi, ic.psi
    for r in range(2, n_res + 1):
        nxt = build_template(resn, ss, "canonical")
        nxt.res_id = [start_res + r - 1] * len(nxt)
        nxt.chain_id = [chain_id] * len(nxt)
        # 既有链末残基的 N/CA/C (不能按固定偏移取: 非 GLY 残基带侧链原子)
        last_rid = max(a.res_id for a in ic.atoms)
        last = {}
        for i, a in enumerate(ic.atoms):
            if a.res_id == last_rid:
                last[a.name] = i
        merged = connect_internal_coords(
            ic, nxt, C_index=last["C"], N_index=0
        )
        off = len(ic)
        # 连接处: psi (N_i, CA_i, C_i, N_j), phi (C_i, N_j, CA_j, C_j)
        merged.dihedra[(last["N"], last["CA"], last["C"], off)] = psi_t
        merged.dihedra[(last["C"], off, off + 1, off + 2)] = phi_t
        # 只保留 N-terminal (既有链) 的 anchor
        merged.anchor = {k: v for k, v in merged.anchor.items() if k < off}
        ic = merged
    return ic


def build_bb_chain_following_ca(target_ca, resn="GLY", ss="alpha-helix",
                                start_res=1, chain_id="A", max_nfev=20000,
                                uniform=False, fit_target_ca=None):
    """构建 n 残基 backbone 片段, 使其 CA 尽量贴合 target_ca (Crick 轨迹)。

    uniform=False (旧行为): 逐残基 phi/psi 联合 least_squares。变量多
    (6 + 2(n-1))、解不唯一, 优化器可能找到"CA 贴住但二面角扭曲"的解
    (psi/phi 被推到边界, 片段不再是均匀螺旋) —— 放置后整条轴倾斜。

    uniform=True (推荐): 只解 **一个均匀 psi + 一个均匀 phi** (加刚体,
    共 8 变量), 片段保持均匀螺旋 —— 与拟合 Crick 螺旋的几何一致, 内部
    二面角不被全局优化扭曲; 目标轨迹与均匀螺旋几何的残余偏差 (~0.3 Å)
    由连接处 (junction) 吸收。这是"新螺旋用拟合参数生成、只修接缝"的
    实现方式。

    Parameters
    ----------
    target_ca : (n, 3) ndarray
        新片段 n 个残基 CA 的目标坐标 (Crick 外推)。
    uniform : bool
        True = 均匀二面角模式 (单 psi/phi + 刚体, 8 变量)。
    fit_target_ca : (m, 3) ndarray | None
        仅 uniform=True 有效。均匀 psi/phi 在该长轨迹 (整条拟合螺旋) 上
        解出, 而不是在短小的 target_ca 上 —— 短片段 (n 很小) 欠定会使
        均匀解漂移 (n=2 时偏差 ~0.08 Å); 用整条轨迹约束后解稳定, 再套
        回 n 残基片段。None = 直接在 target_ca 上解。
    """
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation as R

    target_ca = np.asarray(target_ca, float)
    n = len(target_ca)

    def _chain_keys(chain):
        """(psi_keys, phi_keys, o_keys, ca_idx) for a uniform-residue chain."""
        res_idx = {}
        for i, a in enumerate(chain.atoms):
            res_idx.setdefault(int(a.res_id), {})[a.name] = i
        rids = sorted(res_idx)
        n_res = len(rids)
        psi_keys = [(res_idx[rids[i - 1]]["N"], res_idx[rids[i - 1]]["CA"],
                     res_idx[rids[i - 1]]["C"], res_idx[rids[i]]["N"])
                    for i in range(1, n_res)]
        phi_keys = [(res_idx[rids[i - 1]]["C"], res_idx[rids[i]]["N"],
                     res_idx[rids[i]]["CA"], res_idx[rids[i]]["C"])
                    for i in range(1, n_res)]
        o_keys = [(res_idx[rids[i]]["N"], res_idx[rids[i]]["CA"],
                   res_idx[rids[i]]["C"], res_idx[rids[i]]["O"])
                  for i in range(n_res)]
        ca_idx = [res_idx[rids[i]]["CA"] for i in range(n_res)]
        return psi_keys, phi_keys, o_keys, ca_idx

    # 返回片段: n 残基, 二面角由下面求解决定; anchor 留在模板帧
    # (由 _place_new_fragment 放置)。
    ic_ret = build_bb_chain_ic(n, resn=resn, ss=ss, start_res=start_res,
                               chain_id=chain_id)
    psi_keys, phi_keys, o_keys, ca_idx = _chain_keys(ic_ret)
    raw_anchor = dict(ic_ret.anchor)
    n_psi, n_phi = len(psi_keys), len(phi_keys)
    REG_W = 1e-3  # 弱正则权重: 打破刚体自由度简并 (见 resid)

    if uniform:
        # 均匀模式: 8 变量 (rotvec 3 + t 3 + 单个 psi + 单个 phi)。所有残基
        # 共用同一 psi/phi —— 片段保持均匀螺旋, 二面角不会被全局优化扭曲。
        # 均匀 psi/phi 在 fit_target_ca (整条拟合轨迹) 上解出: 短片段目标
        # (n 很小) 欠定, 直接解会漂移 (n=2 偏差 ~0.08 Å)。
        fit_ca = (np.asarray(fit_target_ca, float)
                  if fit_target_ca is not None else target_ca)
        m = len(fit_ca)
        ic_fit = build_bb_chain_ic(m, resn=resn, ss=ss, start_res=start_res,
                                   chain_id=chain_id)
        psi_keys_f, phi_keys_f, o_keys_f, ca_idx_f = _chain_keys(ic_fit)
        raw_anchor_f = dict(ic_fit.anchor)

        x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -45.0, -60.0])

        def build_coords_fit(x):
            rotvec, t = x[0:3], x[3:6]
            psi, phi = x[6], x[7]
            ic2 = build_bb_chain_ic(m, resn=resn, ss=ss, start_res=start_res,
                                    chain_id=chain_id)
            rmat = R.from_rotvec(rotvec).as_matrix()
            ic2.anchor = {k: tuple(rmat @ np.asarray(v, float) + t)
                          for k, v in raw_anchor_f.items()}
            for q in psi_keys_f:
                ic2.dihedra[q] = psi
            for q in phi_keys_f:
                ic2.dihedra[q] = phi
            for j in range(m - 1):
                ic2.dihedra[o_keys_f[j]] = psi - 180.0
            coords = ic2.to_coords()
            return np.array([coords[i] for i in ca_idx_f], float)

        def resid_uniform(x):
            return (build_coords_fit(x) - fit_ca).ravel()

        lb = np.array([-np.inf] * 6 + [-75.0, -85.0])
        ub = np.array([np.inf] * 6 + [-15.0, -35.0])
        res = least_squares(resid_uniform, x0, bounds=(lb, ub),
                            max_nfev=max_nfev)
        psi_val, phi_val = float(res.x[6]), float(res.x[7])
        for q in psi_keys:
            ic_ret.dihedra[q] = psi_val
        for q in phi_keys:
            ic_ret.dihedra[q] = phi_val
        for j in range(n - 1):
            ic_ret.dihedra[o_keys[j]] = psi_val - 180.0
        return ic_ret

    # 非均匀 (旧) 模式: 逐残基 phi/psi 联合 least_squares
    x0 = np.zeros(6 + n_psi + n_phi)
    x0[6:6 + n_psi] = -45.0
    x0[6 + n_psi:] = -60.0

    def build_coords(x):
        rotvec, t = x[0:3], x[3:6]
        psi = x[6:6 + n_psi]
        phi = x[6 + n_psi:]
        ic2 = build_bb_chain_ic(n, resn=resn, ss=ss, start_res=start_res,
                                chain_id=chain_id)
        rmat = R.from_rotvec(rotvec).as_matrix()
        ic2.anchor = {k: tuple(rmat @ np.asarray(v, float) + t)
                      for k, v in raw_anchor.items()}
        for q, v in zip(psi_keys, psi):
            ic2.dihedra[q] = v
        for q, v in zip(phi_keys, phi):
            ic2.dihedra[q] = v
        for j in range(n - 1):
            ic2.dihedra[o_keys[j]] = psi[j] - 180.0
        coords = ic2.to_coords()
        return np.array([coords[i] for i in ca_idx], float)

    def resid(x):
        ca_r = (build_coords(x) - target_ca).ravel()
        # 打破刚体自由度冗余: 短片段 (n+1 残基) 变量数可能多于 CA 约束,
        # 解不唯一, 优化器可能挑到非自然二面角使残基框架旋转 (N 端放置后
        # CA 漂移)。加弱正则把二面角拉向自然值 (-45/-60), 只破简并、不
        # 干扰真实的 CA 拟合。
        psi = x[6:6 + n_psi]
        phi = x[6 + n_psi:]
        reg = REG_W * np.concatenate([psi - (-45.0), phi - (-60.0)])
        return np.concatenate([ca_r, reg])

    # phi/psi 物理范围 (alpha-helix): 目标若为非 alpha-helix Crick 轨迹,
    # 在物理范围内取最优 (新残基 CA 允许残余偏差, 保证键长键角合理)
    lb = np.full(6 + n_psi + n_phi, -np.inf)
    ub = np.full(6 + n_psi + n_phi, np.inf)
    lb[6:6 + n_psi] = -75.0
    ub[6:6 + n_psi] = -15.0
    lb[6 + n_psi:] = -85.0
    ub[6 + n_psi:] = -35.0

    res = least_squares(resid, x0, bounds=(lb, ub), max_nfev=max_nfev)
    psi = res.x[6:6 + n_psi]
    phi = res.x[6 + n_phi:]

    for q, v in zip(psi_keys, psi):
        ic_ret.dihedra[q] = v
    for q, v in zip(phi_keys, phi):
        ic_ret.dihedra[q] = v
    for j in range(n - 1):
        ic_ret.dihedra[o_keys[j]] = psi[j] - 180.0
    return ic_ret


def _sync_fragment_dihedrals(ic_new, ref_ic, n, terminus):
    """把参考链中属于新片段残基的二面角同步到 ic_new (按残基名映射)。

    ic_new 的原子/键长/键角/anchor 由模板脚手架提供, 只有二面角需要来自
    参考链 (Crick 跟随解)—— 保证片段形状与 _place_new_fragment 的放置参考
    完全一致 (避免两个独立 least_squares 因刚体自由度解不唯一而失配)。
    """
    ref_idx = {(int(a.res_id), a.name): i for i, a in enumerate(ref_ic.atoms)}
    new_idx = {(int(a.res_id), a.name): i for i, a in enumerate(ic_new.atoms)}
    first_new = int(ic_new.atoms[0].res_id)
    if terminus == "C":
        frag_ref = range(2, n + 2)          # 参考链残基 2..n+1
        rid_map = {r: first_new + (r - 2) for r in frag_ref}
    else:
        frag_ref = range(1, n + 1)          # 参考链残基 1..n
        rid_map = {r: first_new + (r - 1) for r in frag_ref}
    for quad, val in ref_ic.dihedra.items():
        rids = {int(ref_ic.atoms[i].res_id) for i in quad}
        if not rids <= set(frag_ref):
            continue
        new_quad = tuple(
            new_idx[(rid_map[int(ref_ic.atoms[i].res_id)], ref_ic.atoms[i].name)]
            for i in quad
        )
        ic_new.dihedra[new_quad] = val
    return ic_new


def _place_new_fragment(ic_new, ic_old, ss="alpha-helix", terminus="C",
                        ref_ic=None):
    """把 ic_new (build_template 串出的新片段) 放到既有链末端之后的实际位置。

    build_template 的 anchor 是模板局部坐标 (N 原点, CA +x), 不是实际空间
    位置。伸长时新片段必须接在既有链的 N 或 C 端之外, 但 IC 只能正向
    生长 —— 所以用理想 alpha-helix 几何把新片段显式"摆"到既有链末端之外
    的实际坐标, 并写入 ic_new 的 anchor (0, 1, 2):

    1. 构造一条 (n_new+1) 残基的理想 alpha-helix 链 (从链首 anchor 生长,
       残基 1..n_new = 新片段, 残基 n_new+1 = 紧挨既有链的那个);
    2. Kabsch 对齐: C 端时参考链首残基 N/CA/C -> 既有链末残基 N/CA/C;
       N 端时参考链末残基 N/CA/C -> 既有链首残基 N/CA/C;
    3. 变换后参考链的对应残基坐标即新片段的实际位置, 写入 ic_new
       anchor (0, 1, 2)。

    Parameters
    ----------
    ic_new : InternalCoord
        新片段 (build_bb_chain_ic 产物, anchor 目前是模板坐标)。
    ic_old : InternalCoord
        既有链 (anchor 是实际坐标)。
    ss : str
        二级结构类。
    terminus : str
        "C" / "N"。
    """
    new_res = {}
    for i, a in enumerate(ic_new.atoms):
        new_res.setdefault(a.res_id, {})[a.name] = i
    n_new = len(new_res)
    n_atoms_new = len(ic_new)

    old_res = {}
    for i, a in enumerate(ic_old.atoms):
        old_res.setdefault(a.res_id, {})[a.name] = i
    # 既有链目标残基 (C 端 = 末残基, N 端 = 首残基) 的实际坐标:
    # from_atomarray 的 anchor 是链首残基, 但 C 端要对齐末残基,
    # 所以统一用 to_coords() 取目标残基 N/CA/C 的实测坐标。
    coords_old = ic_old.to_coords()

    # 参考链 (n_new+1) 残基: 默认模板 alpha-helix; 传 ref_ic (Crick 跟随
    # 片段) 时用其几何做放置 —— 参考链残基 2..n+1 (C 端) / 1..n (N 端) 即
    # 新片段, 使放置后片段 CA 精确贴住目标轨迹
    if ref_ic is None:
        ref_ic = build_bb_chain_ic(n_new + 1, resn=ic_new.atoms[0].res_name,
                                   ss=ss, start_res=1, chain_id="A")
    ref_coords = ref_ic.to_coords()
    ref_rids = np.array([a.res_id for a in ref_ic.atoms])
    last_rid_ref = n_new + 1
    last_ref = {}
    for i, a in enumerate(ref_ic.atoms):
        if a.res_id == last_rid_ref:
            last_ref[a.name] = i

    if terminus == "C":
        # 参考链首残基对齐到既有链末残基; 新片段 = 参考链残基 2..n_new+1
        rid_old = max(old_res)
        N_i = old_res[rid_old]["N"]
        CA_i = old_res[rid_old]["CA"]
        C_i = old_res[rid_old]["C"]
        P = np.stack([
            np.asarray(ref_coords[0], float),
            np.asarray(ref_coords[1], float),
            np.asarray(ref_coords[2], float),
        ])
        Q = np.stack([
            np.asarray(coords_old[N_i], float),
            np.asarray(coords_old[CA_i], float),
            np.asarray(coords_old[C_i], float),
        ])
        R, t = _kabsch(P, Q)
        # 新片段原子 = 参考链中非首残基的原子 (残基 2..n_new+1)
        sel = np.where(ref_rids != 1)[0][:n_atoms_new]
    else:
        # 参考链末残基对齐到既有链首残基; 新片段 = 参考链残基 1..n_new
        rid_old = min(old_res)
        N_i = old_res[rid_old]["N"]
        CA_i = old_res[rid_old]["CA"]
        C_i = old_res[rid_old]["C"]
        P = np.stack([
            np.asarray(ref_coords[last_ref["N"]], float),
            np.asarray(ref_coords[last_ref["CA"]], float),
            np.asarray(ref_coords[last_ref["C"]], float),
        ])
        Q = np.stack([
            np.asarray(coords_old[N_i], float),
            np.asarray(coords_old[CA_i], float),
            np.asarray(coords_old[C_i], float),
        ])
        R, t = _kabsch(P, Q)
        # 新片段原子 = 参考链中非末残基的原子 (残基 1..n_new)
        sel = np.where(ref_rids != last_rid_ref)[0][:n_atoms_new]

    new_xyz = np.array(
        [R @ np.asarray(ref_coords[i], float) + t for i in sel]
    )
    ic_new.anchor[0] = tuple(new_xyz[0])
    ic_new.anchor[1] = tuple(new_xyz[1])
    ic_new.anchor[2] = tuple(new_xyz[2])


def _seam_geometry(ca_n, c, n, ca_c):
    """由接缝四原子 (CA_n, C, N, CA_c) 的实测坐标求 peptide seam 几何。

    供 connect_ic_fragments 传给 connect_internal_coords: 该函数默认用
    理想 Engh & Huber 值记录接缝, 若实际摆放的几何与之不符, to_coords 生长
    时接缝放置的原子会与另一片段 anchor 冲突而抛 "Inconsistent
    coordinate"。显式传实测值后, 合并 IC 对任意输入几何都自洽。
    """
    ca_n = np.asarray(ca_n, float)
    c = np.asarray(c, float)
    n = np.asarray(n, float)
    ca_c = np.asarray(ca_c, float)

    blen = float(np.linalg.norm(c - n))

    def _angle(a, b, c_):
        v1 = a - b
        v2 = c_ - b
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

    ang_c = _angle(ca_n, c, n)   # 角在 C: CA-C-N
    ang_n = _angle(c, n, ca_c)   # 角在 N: C-N-CA

    b0 = ca_n - c
    b1 = n - c
    b2 = ca_c - n
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    omega = float(np.degrees(np.arctan2(y, x)))
    return blen, ang_c, ang_n, omega


def connect_ic_fragments(ic_old, ic_new, ss="alpha-helix", terminus="C",
                         ref_ic=None):
    """把 ic_new (新片段) 连接到 ic_old 的 N 或 C 端。

    两段 fragment 的 anchor **都保留** (各自实际坐标), 连接只靠
    ``connect_internal_coords`` 记录的 omega (CA, C, N, CA) —— 新片段
    首残基的 N/CA/C 是它自己的 anchor, 不需要 psi/phi 定位。这样
    to_coords 生长时两段都不动、几何连续。

    Parameters
    ----------
    ic_old : InternalCoord
        既有链。
    ic_new : InternalCoord
        新片段。
    ss : str
        二级结构类 (预留, 暂未使用; 连接处 omega 用默认 trans)。
    terminus : str
        "C" = 新片段接在既有链 C 端之后 (merged = [既有链][新片段]);
        "N" = 新片段接在既有链 N 端之前 (merged = [新片段][既有链])。
    ref_ic : InternalCoord | None
        放置参考链 (含对齐残基 + 新片段, Crick 跟随); None = 模板
        alpha-helix 参考链。

    Returns
    -------
    InternalCoord
        合并后的内坐标, 两段 anchor 均保留, 连接处仅 omega 为实测值。
    """
    from biorazer.structure.manipulation.modification import (
        connect_internal_coords,
    )

    if terminus == "C":
        # 新片段在后段
        old_res = {}
        for i, a in enumerate(ic_old.atoms):
            old_res.setdefault(a.res_id, {})[a.name] = i
        last_rid = max(old_res)
        C_idx = old_res[last_rid]["C"]

        # 新片段 anchor 从模板坐标换到既有链 C 端之后的实际位置
        _place_new_fragment(ic_new, ic_old, ss=ss, terminus="C", ref_ic=ref_ic)
        coords_old = ic_old.to_coords()
        coords_new = ic_new.to_coords()
        blen, ang_c, ang_n, omega = _seam_geometry(
            coords_old[old_res[last_rid]["CA"]], coords_old[C_idx],
            coords_new[0], coords_new[1],
        )
        merged = connect_internal_coords(
            ic_old, ic_new, C_index=C_idx, N_index=0,
            bond_length=blen, angle_CA_C_N=ang_c, angle_C_N_CA=ang_n,
            omega=omega,
        )
    elif terminus == "N":
        # 新片段在前段: N_terminal = ic_new (末残基 C), C_terminal = ic_old
        # (首残基 N)。新片段的 anchor 必须从模板局部坐标换成实际空间坐标
        # (既有链 N 端之前), 见 _place_new_fragment_at_nterm。
        new_res = {}
        for i, a in enumerate(ic_new.atoms):
            new_res.setdefault(a.res_id, {})[a.name] = i
        last_rid_new = max(new_res)
        C_new = new_res[last_rid_new]["C"]

        old_res = {}
        for i, a in enumerate(ic_old.atoms):
            old_res.setdefault(a.res_id, {})[a.name] = i
        first_rid = min(old_res)
        N_i = old_res[first_rid]["N"]

        _place_new_fragment(ic_new, ic_old, ss=ss, terminus="N", ref_ic=ref_ic)
        coords_new = ic_new.to_coords()
        coords_old = ic_old.to_coords()
        blen, ang_c, ang_n, omega = _seam_geometry(
            coords_new[new_res[last_rid_new]["CA"]], coords_new[C_new],
            coords_old[N_i], coords_old[old_res[first_rid]["CA"]],
        )
        merged = connect_internal_coords(
            ic_new, ic_old, C_index=C_new, N_index=N_i,
            bond_length=blen, angle_CA_C_N=ang_c, angle_C_N_CA=ang_n,
            omega=omega,
        )
    else:
        raise ValueError(f"terminus 仅支持 'N'/'C', 得到 {terminus!r}")

    # 两段 anchor 都保留 (各自实际坐标)
    return merged


def backbone_dihedral_quads(ic, new_atoms):
    """返回 ic 中可安全优化的主链二面角 (phi/psi/omega)。

    只选"旋转轴下游在新片段"的二面角 —— 优化它只会移动新片段,
    不会牵动既有链 (锚定的观测结构):

    * psi   (N, CA, C, N): 旋转 C-N 键下游, 要求 N(第 4 原子)在新片段;
    * phi   (C, N, CA, C): 旋转 N-CA 键下游, 要求 C(第 4 原子)在新片段;
    * omega (CA, C, N, CA): 旋转 C-N 键下游, 要求 N(第 3 原子)在新片段。

    Parameters
    ----------
    new_atoms : set[int]
        新片段原子的下标集合。

    Returns
    -------
    list[tuple]
        去重保序的 (i,j,k,l) 二面角列表。
    """
    psi_q, phi_q, omega_q = [], [], []
    for quad in ic.dihedra:
        names = tuple(ic.atoms[i].name for i in quad)
        if names == ("N", "CA", "C", "N"):
            if quad[3] in new_atoms:
                psi_q.append(quad)
        elif names == ("CA", "C", "N", "CA"):
            if quad[2] in new_atoms:
                omega_q.append(quad)
        elif names == ("C", "N", "CA", "C"):
            if quad[3] in new_atoms:
                phi_q.append(quad)
    seen = set()
    out = []
    anchored = set(ic.anchor)
    for q in psi_q + phi_q + omega_q:
        if q[3] in anchored:
            # 放置原子被 anchor 钉死 (如连接接缝的 omega: 它放置新片段
            # 首残基/既有链首残基的 CA, 而该原子同时在 anchor 里); 优化
            # 它会与 anchor 坐标冲突, to_coords 抛 "Inconsistent
            # coordinate"。
            continue
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out


def optimize_bb_to_ca(ic, target_ca, ca_indices, ss="alpha-helix",
                      new_atoms=None, max_nfev=5000):
    """分阶段优化 ic 的 backbone 二面角, 使 ca_indices 的 CA 贴近 target_ca。

    两阶段 (与 CCCP fit 的分段策略一致):
      Stage 1: 只优化 psi/phi (omega 固定), 把新片段拉到目标附近;
      Stage 2: 全部 (psi/phi/omega) 一起精调。
    每个二面角的 lb/up 直接读 SS_BB_TORSION_ANGLE[ss] 的数据库值。

    Parameters
    ----------
    ic : InternalCoord
        待优化对象 (原地修改其 dihedra)。
    target_ca : (n, 3) ndarray
        新残基 CA 的目标坐标 (拟合轨迹外推)。
    ca_indices : list[int]
        ic 中新残基 CA 的原子下标 (顺序与 target_ca 对应)。
    ss : str
        二级结构类, 决定各二面角的 lb/up 与初值。
    new_atoms : set[int] | None
        新片段原子下标集合; 主链二面角只要含其中任一原子即参与优化。
        None = 全部主链二面角 (默认, 兼容旧调用)。
    """
    from scipy.optimize import least_squares

    from biorazer.database.molecule.bond.dihedral.protein import (
        SS_BB_TORSION_ANGLE,
    )

    bb = SS_BB_TORSION_ANGLE[ss]
    if new_atoms is None:
        new_atoms = set(range(len(ic)))
    quads = backbone_dihedral_quads(ic, set(new_atoms))
    target_ca = np.asarray(target_ca, float)

    def _resid(x, quads_):
        for q, v in zip(quads_, x):
            ic.dihedra[q] = v
        coords = ic.to_coords()
        ca = np.array([coords[i] for i in ca_indices], float)
        return (ca - target_ca).ravel()

    def _bounds(quads_):
        lb, ub = [], []
        for q in quads_:
            names = tuple(ic.atoms[i].name for i in q)
            entry = bb[names]
            lb.append(entry["lb"])
            ub.append(entry["up"])
        return np.array(lb, float), np.array(ub, float)

    # Stage 1: psi/phi only
    pp = [q for q in quads
          if tuple(ic.atoms[i].name for i in q) != ("CA", "C", "N", "CA")]
    if pp:
        x0 = np.array([ic.dihedra[q] for q in pp], float)
        lb, ub = _bounds(pp)
        # 初值已在范围内 (模板 = ss 均值), 但保险起见夹一下
        x0 = np.clip(x0, lb, ub)
        least_squares(
            lambda x: _resid(x, pp), x0, bounds=(lb, ub),
            max_nfev=max_nfev,
        )

    # Stage 2: 全部 (psi/phi/omega)
    if quads:
        x0 = np.array([ic.dihedra[q] for q in quads], float)
        lb, ub = _bounds(quads)
        x0 = np.clip(x0, lb, ub)
        least_squares(
            lambda x: _resid(x, quads), x0, bounds=(lb, ub),
            max_nfev=max_nfev,
        )

    # O 跟踪 psi: 优化改动了 psi, 但 carbonyl O 的二面角 (N,CA,C,O) 停在
    # 模板值, 需同步为 psi - 180 (trans 肽平面), 否则 O 相对 N_next 漂移
    # (demo 中 psi→±180° 时 O 曾与 N_next 碰撞, O···N ≈ 0.65 Å)。O 是分支
    # 原子, 不影响 CA 拟合, 故在优化结束后统一更新。
    by_res = {}
    for i, a in enumerate(ic.atoms):
        by_res.setdefault(a.res_id, {})[a.name] = i
    for q in quads:
        names = tuple(ic.atoms[i].name for i in q)
        if names == ("N", "CA", "C", "N"):
            o_idx = by_res.get(ic.atoms[q[0]].res_id, {}).get("O")
            if o_idx is not None:
                ic.dihedra[(q[0], q[1], q[2], o_idx)] = ic.dihedra[q] - 180.0

    return ic


def _construct_param_vector(param_dict: dict, param_names: list):
    """
    Construct a parameter vector from the parameter dictionary based on the specified names.

    Parameters
    ----------
    param_dict : dict
        The dictionary containing parameter names and their values which will be picked by param_names.
    param_names : list
        The list of parameter names to include in the vector. Order of names determines order in vector.

    Returns
    -------
    param_vector : np.ndarray
        The constructed parameter vector.
    parse_dict : dict
        A dictionary mapping parameter names to their indices or slices in the param_vector.
    """
    param_vector = []
    for name in param_names:
        if isinstance(param_dict[name], (int, float)):
            param_vector.append(param_dict[name])
        else:
            param_vector.extend(param_dict[name])
    param_vector = np.array(param_vector)

    parse_dict = {}
    param_index = 0
    for name in param_names:
        if isinstance(param_dict[name], (int, float)):
            parse_dict[name] = param_index
            param_index += 1
        else:
            parse_dict[name] = slice(param_index, param_index + len(param_dict[name]))
            param_index += len(param_dict[name])

    return param_vector, parse_dict


def _construct_param_dict(params: np.ndarray, parse_dict: dict):
    """
    Construct a parameter dictionary from the parameter vector based on the parse dictionary.

    Parameters
    ----------
    params : np.ndarray
        The flattened parameter vector.
    parse_dict : dict
        A dictionary mapping parameter names to their indices or slices in the params vector.

    Returns
    -------
    param_dict : dict
        The constructed parameter dictionary.
    """
    param_dict = {}
    for key, value in parse_dict.items():
        param_dict[key] = params[value]
    return param_dict


def pulchra_fix_backbone(structure, app_bin="pulchra"):
    """
    Rebuild full backbone atoms (N, CA, C, O) from a CA-only structure by
    calling the external ``pulchra`` binary directly (no wrapper package).

    Each chain is processed separately because pulchra output loses chain IDs;
    the original chain IDs are restored afterwards. Pulchra ATOM records also
    lack occupancy/B-factor columns, so they are padded before biotite parsing.
    """
    import subprocess
    import tempfile
    from pathlib import Path

    from biorazer.structure.io.protein import AtomArray_Pdb, Pdb_AtomArray

    chain_ids = sorted(set(structure.chain_id))
    rebuilt_chains = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        for chain_id in chain_ids:
            chain_arr = structure[structure.chain_id == chain_id]
            input_file = tmp_path / "input.pdb"
            rebuilt_file = tmp_path / "input.rebuilt.pdb"
            fixed_file = tmp_path / "fixed.pdb"
            AtomArray_Pdb(output_io=input_file).write(chain_arr)
            proc = subprocess.run(
                [app_bin, input_file.name],
                cwd=tmp_path,
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"pulchra failed (exit {proc.returncode}) on chain {chain_id!r}:\n"
                    f"{proc.stderr.strip() or proc.stdout.strip()}"
                )
            # pulchra 的 ATOM 行没有 occupancy/B-factor 列, 补上以便 biotite 按列解析
            with open(rebuilt_file) as f_in, open(fixed_file, "w") as f_out:
                for line in f_in:
                    if line.startswith("ATOM"):
                        f_out.write(line.rstrip("\n") + "  1.00  0.00\n")
                    else:
                        f_out.write(line)
            rebuilt = Pdb_AtomArray(input_io=fixed_file).read()
            rebuilt.chain_id = np.array([chain_id] * len(rebuilt))
            rebuilt_chains.append(rebuilt)
    return bio_struct.concatenate(rebuilt_chains)


def ca_xyz_to_atom_array(xyz, chain_id_i="A", res_name="GLY"):

    if len(xyz.shape) == 3:
        helix_len = xyz.shape[1]
        length = xyz.shape[0] * xyz.shape[1]
    elif len(xyz.shape) == 2:
        helix_len = xyz.shape[0]
        length = xyz.shape[0]
    else:
        raise ValueError(
            "xyz must be of shape (helix_num, residue_num, 3) or (residue_num, 3)"
        )
    xyz = np.reshape(xyz, (-1, 3))
    structure = bio_struct.AtomArray(length=length)
    structure.atom_name = np.array(["CA"] * length)
    structure.chain_id = np.array(
        [chr(ord(chain_id_i) + i // helix_len) for i in range(length)]
    )
    structure.res_id = np.array(list(range(1, helix_len + 1)) * (length // helix_len))
    structure.element = np.array(["C"] * length)
    structure.res_name = np.array([res_name] * length)
    structure.coord = xyz
    return structure
