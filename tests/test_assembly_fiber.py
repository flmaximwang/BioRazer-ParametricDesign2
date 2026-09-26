"""AssemblyFiber (screw 纤维) 与 AssemblyCn (纯转动 Cn) 的 fit / generate_parts。

三层覆盖:
1. 合成 part + 已知 screw —— 精确可验 (不需要外部数据):
   fit 必须还原 omega / rise / 轴, generate_parts 必须还原原结构;
   Cn 另验 rise ≡ 0、环闭合 (含末→首步)。
2. 6vy1 (双层螺旋纤维, part = 2 条链 G + g) —— 真实数据:
   沿纤维 (质心 z) 排序后第一个 part 就是 G+g, 拟合出的 screw 约 -48.92 deg /
   +18.33 A, 用它生成的纤维贴住沉积结构 ~1 A 以内; 同一结构的 G/g 两条链是一对
   **垂直纤维轴** 的 2 重轴 → 用 AssemblyCn 拟合 (Ω ≈ 179.7 deg, rise ≡ 0)。
3. part 标识 / 导出: ``chain_id`` 保持原样, part 序号写 ``ins_code`` (1 字符
   ``0``-``9``/``A``-``Z``/``a``-``z``), key = ``链名:ins_code`` —— PDB 列 27 与
   mmCIF ``pdbx_PDB_ins_code`` 都自动读写, 不需要 extra_fields。

6vy1 坐标: 仓库内压缩的 ``tests/data/6vy1.cif.gz`` 直接在内存里解压读入 (不落
解压文件); ``BIORAZER_6VY1_CIF`` 可指向未压缩的 cif 覆盖它。
"""

import gzip
import os

import numpy as np
import pytest
import biotite.structure as bt_struct
import biotite.structure.io.pdbx as pdbx
from scipy.spatial.transform import Rotation as R

from biorazer_prds.models import Assembly, AssemblyFiber, AssemblyCn
from biorazer_prds.models.assembly_fiber import PART_CODES

CIF_GZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "6vy1.cif.gz")

# 6vy1 已知几何 (实测, 见 run 记录): 每步绕纤维轴 -48.92 deg, 沿轴 +18.33 A
VY1_OMEGA_DEG = -48.922
VY1_RISE = 18.327


# ----------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------

def _atoms(coords, chain="A", res_id=None, res_name="ALA"):
    """由坐标造一个全 CA 的 AtomArray (每个原子一个残基)。"""
    coords = np.asarray(coords, dtype=float)
    arr = bt_struct.AtomArray(len(coords))
    arr.coord = coords
    arr.atom_name = np.array(["CA"] * len(coords))
    arr.element = np.array(["C"] * len(coords))
    arr.chain_id = np.array([chain] * len(coords))
    arr.res_id = np.arange(1, len(coords) + 1) if res_id is None else np.asarray(res_id)
    arr.res_name = np.array([res_name] * len(coords))
    return arr


def _screw(omega_deg, rise, axis_point=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0)):
    """4x4 screw: 绕过 axis_point 的 axis 转 omega_deg, 再沿 axis 平移 rise。"""
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    rotation = R.from_rotvec(axis * np.deg2rad(omega_deg))
    point = np.asarray(axis_point, float)
    T = np.eye(4)
    T[:3, :3] = rotation.as_matrix()
    T[:3, 3] = point - rotation.apply(point) + rise * axis
    return T


def _fiber_from_prototype(prototype, n, transformation, key="A", cls=AssemblyFiber):
    """由原型 part 施加 transformation k 次造 n 个 part 的纤维 (观测面)。

    part 的 chain_id 保持原样 (与真实 PDB 一样是单字符) —— ``generate_parts``
    的序号是加在它们之上的。``cls=AssemblyCn`` 可造环状 (Cn) 的观测面。
    """
    fiber = cls()
    for k in range(n):
        part = cls.apply_transformation(prototype, transformation, k)
        fiber.append_part(f"{key}{k:02d}", Assembly(structure=part))
    fiber.merge_up()
    return fiber


@pytest.fixture(scope="module")
def vy1():
    """6vy1: 清掉各链缺原子的位点, 返回 (STRUCT, pairs, order)。

    坐标从仓库内的 ``tests/data/6vy1.cif.gz`` 在内存里解压读入 —— 不写解压文件,
    也就没有要清理的残留; ``BIORAZER_6VY1_CIF`` 指向未压缩的 cif 时优先用它。
    """
    env_cif = os.environ.get("BIORAZER_6VY1_CIF")
    path = env_cif if env_cif else CIF_GZ
    if not os.path.exists(path):
        pytest.skip(f"6vy1 数据缺失: {path} (BIORAZER_6VY1_CIF 或 {CIF_GZ})")
    opener = open if env_cif else gzip.open
    with opener(path, "rt") as fh:
        arr = pdbx.get_structure(pdbx.CIFFile.read(fh), model=1)
    chains = sorted(set(arr.chain_id), key=lambda c: (c.islower(), c))
    assert len(chains) == 14, f"6vy1 应有 14 条链, 得到 {sorted(chains)}"
    # 链 a 缺 res 8 的 N: 只保留 14 条链共有的 atoms, 保证 part 完全同构
    per_chain = [
        set(zip(arr[arr.chain_id == c].res_id.tolist(),
                arr[arr.chain_id == c].atom_name.tolist()))
        for c in chains
    ]
    common = set.intersection(*per_chain)
    keep = np.array([(r, a) in common
                     for r, a in zip(arr.res_id.tolist(), arr.atom_name.tolist())])
    arr = arr[keep]

    pairs = {}
    for letter in "ABCDEFG":
        sel = (arr.chain_id == letter) | (arr.chain_id == letter.lower())
        pairs[letter] = arr[sel]
    # 沿纤维排序: part 的 CA 质心 z 升序
    centroid_z = {
        L: float(bt_struct.centroid(pairs[L][pairs[L].atom_name == "CA"])[2])
        for L in pairs
    }
    order = sorted(pairs, key=centroid_z.get)
    struct = bt_struct.concatenate([pairs[L] for L in order])
    return struct, pairs, order


def _vy1_fiber(struct, pairs, order):
    fiber = AssemblyFiber(structure=struct)
    for letter in order:
        fiber.append_part(f"{letter}{letter.lower()}", Assembly(structure=pairs[letter]))
    return fiber


# ----------------------------------------------------------------------
# 1. 合成数据: 精确可验
# ----------------------------------------------------------------------

PROTOTYPE_COORD = [[3.0, 0.0, 0.0], [3.6, 1.1, 0.2], [4.2, 0.3, 0.9],
                   [2.8, 2.0, 1.4], [4.6, 1.7, 2.1]]
SYNTH_OMEGA_DEG = -100.0
SYNTH_RISE = 5.0
SYNTH_AXIS_POINT = [1.0, -2.0, 0.5]


def test_fit_recovers_synthetic_screw():
    proto = _atoms(PROTOTYPE_COORD)
    T = _screw(SYNTH_OMEGA_DEG, SYNTH_RISE, SYNTH_AXIS_POINT)
    fiber = _fiber_from_prototype(proto, 4, T)
    fiber.fit()

    assert np.degrees(fiber.param["omega"]) == pytest.approx(SYNTH_OMEGA_DEG, abs=1e-4)
    assert fiber.param["rise"] == pytest.approx(SYNTH_RISE, abs=1e-5)
    np.testing.assert_allclose(fiber.xyz[2], [0, 0, 1], atol=1e-6)
    assert fiber.rmsd < 1e-4                      # 严格 screw, 模型精确
    # ref_structure = parts[0] 经 transformation 生成的理想 fiber (此处与观测重合)
    assert len(fiber.ref_structure) == 4 * len(proto)
    np.testing.assert_allclose(
        fiber.ref_structure.coord, fiber.structure.coord, atol=1e-5
    )
    np.testing.assert_allclose(fiber.extra_param["part_rmsd"], 0, atol=1e-5)
    assert fiber.extra_param["axis_deviation"] < 1e-4
    # transformation 本身也应是同一个 screw (绕轴旋转 + 沿轴平移)
    np.testing.assert_allclose(fiber.transformation, T, atol=1e-5)
    # 原点在轴上: 与构造时的 axis_point 只差一个沿 z 的分量 (同一条轴)
    offset = fiber.param["axis_point"] - np.array(SYNTH_AXIS_POINT)
    np.testing.assert_allclose(offset[:2], 0, atol=1e-5)


def test_frame_x_points_from_axis_to_part0_centroid():
    proto = _atoms(PROTOTYPE_COORD)
    fiber = _fiber_from_prototype(proto, 3, _screw(SYNTH_OMEGA_DEG, SYNTH_RISE))
    fiber.fit()
    x, y, z = fiber.xyz

    # 正交右手帧
    np.testing.assert_allclose([np.dot(x, y), np.dot(x, z), np.dot(y, z)], 0, atol=1e-12)
    assert np.linalg.det(np.vstack((x, y, z))) == pytest.approx(1.0)

    part0 = list(fiber.parts.values())[0]
    d = part0.centroid - fiber.centroid        # 轴 → part[0] 质心
    assert abs(float(np.dot(d, z))) < 1e-12    # 原点与质心同高 → d 垂直于 z
    np.testing.assert_allclose(d / np.linalg.norm(d), x, atol=1e-5)
    # 具体数值: 质心 (3.84, 1.02, 0.92), 轴过原点 → x = 质心的 xy 分量归一化
    c0 = np.mean(PROTOTYPE_COORD, axis=0)
    expect = np.array([c0[0], c0[1], 0.0])
    np.testing.assert_allclose(x, expect / np.linalg.norm(expect), atol=1e-5)


def test_generate_parts_naming_and_geometry():
    proto = _atoms(PROTOTYPE_COORD)
    T = _screw(SYNTH_OMEGA_DEG, SYNTH_RISE, SYNTH_AXIS_POINT)
    fiber = _fiber_from_prototype(proto, 4, T)
    fiber.fit()

    observed = [p.structure.coord.copy() for p in fiber.parts.values()]
    fiber.generate_parts(6, verbose=False)

    keys = list(fiber.parts)
    assert keys == [f"A:{c}" for c in "012345"]
    assert len(fiber.structure) == 6 * len(proto)
    # 链名按 part 序号递增
    for k, key in enumerate(keys):
        assert set(fiber.parts[key].structure.chain_id) == {"A"}           # chain_id 不动
        assert set(fiber.parts[key].structure.ins_code) == {"012345"[k]}  # part 序号在 ins_code
    # part 0 = 原型本身; 每个 part 贴住原设计坐标 (未再叠合)
    np.testing.assert_allclose(fiber.parts["A:0"].structure.coord, observed[0], atol=1e-6)
    for k in range(4):
        np.testing.assert_allclose(fiber.parts[keys[k]].structure.coord, observed[k],
                                   atol=1e-6)
    # 第 4、5 个 part 是 fit 出来的 screw 的外推 (合成数据应精确)
    for k in (4, 5):
        expected = AssemblyFiber.apply_transformation(proto, T, k).coord
        np.testing.assert_allclose(fiber.parts[keys[k]].structure.coord, expected, atol=1e-5)
    # mask 与 structure 同源 (树不变式)
    for key in keys:
        assert fiber.mask[key].sum() == len(proto)
        assert len(fiber.mask[key]) == len(fiber.structure)
    # structure = parts 拼接 (顺序一致)
    np.testing.assert_allclose(fiber.structure.coord[: len(proto)], observed[0], atol=1e-6)


def test_transformation_is_derived_from_param():
    """screw 参数就是 ``param``: 直接给 param (不经过 fit) 也能 generate_parts。"""
    proto = _atoms(PROTOTYPE_COORD)
    T = _screw(SYNTH_OMEGA_DEG, SYNTH_RISE, SYNTH_AXIS_POINT)

    fiber = AssemblyFiber()
    fiber.append_part("A", Assembly(structure=proto))
    assert fiber.transformation is None                     # 无参数 → 无变换

    fiber.param = {
        "omega": np.deg2rad(SYNTH_OMEGA_DEG),
        "rise": SYNTH_RISE,
        "axis_direction": np.array([0.0, 0.0, 1.0]),
        "axis_point": np.array(SYNTH_AXIS_POINT),
    }
    np.testing.assert_allclose(fiber.transformation, T, atol=1e-12)

    fiber.generate_parts(3)
    assert list(fiber.parts) == ["A:0", "A:1", "A:2"]
    for k in range(3):
        expected = AssemblyFiber.apply_transformation(proto, T, k)
        np.testing.assert_allclose(
            fiber.parts[f"A:{k}"].structure.coord, expected.coord, atol=1e-12
        )


def test_validation_errors():
    proto = _atoms(PROTOTYPE_COORD)
    fiber = _fiber_from_prototype(proto, 3, _screw(SYNTH_OMEGA_DEG, SYNTH_RISE))

    # 未 fit / 未给 param 里的 screw 参数
    with pytest.raises(ValueError, match="screw 参数"):
        fiber.generate_parts(3)

    # 只有 1 个 part 定不出 screw
    one = _fiber_from_prototype(proto, 1, _screw(SYNTH_OMEGA_DEG, SYNTH_RISE))
    with pytest.raises(ValueError, match="至少需要 2 个 part"):
        one.fit()

    # part 之间 atoms 不一致 → 拒绝对齐 (改子节点后 merge_up, 见树不变式)
    bad = _fiber_from_prototype(proto, 3, _screw(SYNTH_OMEGA_DEG, SYNTH_RISE))
    keys = list(bad.parts)
    bad.parts[keys[1]] = Assembly(structure=bad.parts[keys[1]].structure[:-1])
    bad.merge_up()
    with pytest.raises(ValueError, match="原子数不同"):
        bad.fit()

    mixed = _fiber_from_prototype(proto, 3, _screw(SYNTH_OMEGA_DEG, SYNTH_RISE))
    keys = list(mixed.parts)
    other = mixed.parts[keys[1]].copy()
    other.structure.res_id = other.structure.res_id + 10   # 同长但残基号不同
    mixed.parts[keys[1]] = other
    mixed.merge_up()
    with pytest.raises(ValueError, match="不完全相同"):
        mixed.fit()

    fiber.fit()
    with pytest.raises(TypeError, match="整数"):
        fiber.generate_parts(2.5)
    with pytest.raises(ValueError, match=">= 1"):
        fiber.generate_parts(0)


def test_fit_is_invariant_under_rigid_motion():
    """整体旋转 + 平移纤维: screw 参数不变, 轴随之旋转。"""
    proto = _atoms(PROTOTYPE_COORD)
    T = _screw(SYNTH_OMEGA_DEG, SYNTH_RISE, SYNTH_AXIS_POINT)
    fiber = _fiber_from_prototype(proto, 4, T)
    fiber.fit()
    z_before = fiber.xyz[2].copy()
    omega_before = fiber.param["omega"]

    rigid = R.from_euler("xyz", [25.0, -30.0, 55.0], degrees=True)
    moved = fiber.copy()
    moved.structure.coord = rigid.apply(moved.structure.coord) + np.array([5.0, 6.0, -7.0])
    moved.fit()

    assert moved.param["omega"] == pytest.approx(omega_before, abs=1e-5)
    assert moved.param["rise"] == pytest.approx(SYNTH_RISE, abs=1e-5)
    np.testing.assert_allclose(moved.xyz[2], rigid.apply(z_before), atol=1e-5)
    assert moved.rmsd < 1e-4


def test_parent_structure_is_authoritative_when_masks_exist():
    """有 mask 时 fit 从父 structure 切片 (与 CCCPHelixBundle.fit 同一约定):

    只改子节点而不 ``merge_up()``, fit 看不到 (父 structure 是权威);
    ``merge_up()`` 之后子节点成为权威 —— 一个横向错位的 part 不在 screw 上,
    模型拟合残差 ``rmsd`` 随之明显变大。
    """
    proto = _atoms(PROTOTYPE_COORD)
    fiber = _fiber_from_prototype(proto, 3, _screw(SYNTH_OMEGA_DEG, SYNTH_RISE))
    fiber.fit()
    rmsd_before = fiber.rmsd
    assert rmsd_before < 1e-4
    keys = list(fiber.parts)

    # 只挪第二个子节点: 父 structure 未变 → fit 结果不变
    fiber.parts[keys[1]].structure.coord = (
        fiber.parts[keys[1]].structure.coord + np.array([3.0, 0.0, 4.0])
    )
    fiber.fit()
    assert fiber.rmsd == pytest.approx(rmsd_before, abs=1e-9)

    # merge_up() 后子节点成为权威 → 错位的 part 让模型残差爆炸 (screw 表达不了它)
    fiber.merge_up()
    fiber.fit()
    assert fiber.rmsd > 1.0


# ----------------------------------------------------------------------
# 2. 6vy1 真实数据
# ----------------------------------------------------------------------

def test_vy1_first_part_is_G_plus_g(vy1):
    """沿纤维 (质心 z) 排序, 第一个 part 是 G+g —— 用户指定的重复单元。"""
    _, pairs, order = vy1
    assert order[0] == "G"
    assert len(order) == 7
    for letter in order:
        sel = pairs[letter]
        assert set(sel.chain_id) == {letter, letter.lower()}


def test_vy1_fit_screw_and_frame(vy1):
    struct, pairs, order = vy1
    fiber = _vy1_fiber(struct, pairs, order)
    fiber.fit()

    # 纤维轴 ≈ 晶体 c 轴; 每步 -48.92 deg / +18.33 A
    assert abs(float(np.dot(fiber.xyz[2], [0, 0, 1]))) > 0.999
    assert np.degrees(fiber.param["omega"]) == pytest.approx(VY1_OMEGA_DEG, abs=0.3)
    assert fiber.param["rise"] == pytest.approx(VY1_RISE, abs=0.3)
    assert fiber.param["rise"] > 0                      # z 指向 part 序号递增一侧
    assert fiber.extra_param["axis_deviation"] < 1.0    # 各步旋转轴确实共轴
    # 每步的转角/升高彼此一致 (part 顺序正确)
    step_omega = np.degrees(fiber.extra_param["step_omega"])
    assert np.ptp(step_omega) < 1.0
    assert np.ptp(fiber.extra_param["step_rise"]) < 0.5
    assert fiber.rmsd < 1.2

    # x = 由轴指向 parts[0] (G+g) 质心, 且与 z 垂直
    x, y, z = fiber.xyz
    assert abs(float(np.dot(x, z))) < 1e-9
    assert np.linalg.det(np.vstack((x, y, z))) == pytest.approx(1.0)
    part0 = fiber.parts[list(fiber.parts)[0]]
    d = part0.centroid - fiber.centroid
    np.testing.assert_allclose(d / np.linalg.norm(d), x, atol=1e-9)


def test_vy1_generate_parts_reproduces_deposited_fiber(vy1):
    struct, pairs, order = vy1
    fiber = _vy1_fiber(struct, pairs, order)
    fiber.fit()
    observed = [pairs[L] for L in order]

    fiber.generate_parts(7)
    keys = list(fiber.parts)
    assert keys == [f"G:{k}+g:{k}" for k in range(7)]
    assert len(fiber.structure) == len(struct)
    for k, key in enumerate(keys):
        generated = fiber.parts[key]
        assert set(generated.structure.chain_id) == {"G", "g"}             # chain_id 保持原样
        assert set(generated.structure.ins_code) == {str(k)}             # part 序号在 ins_code
        # 逐原子 RMSD (原始坐标, 不再叠合): 生成纤维贴住沉积结构
        rmsd = bt_struct.rmsd(generated.structure.coord, observed[k].coord)
        assert rmsd < 1.5, f"part {key} 偏离观测 {rmsd:.3f} A"
    # 第一个 part 就是原型 (G+g)
    np.testing.assert_allclose(
        fiber.parts["G:0+g:0"].structure.coord, pairs[order[0]].coord, atol=1e-6
    )


def test_ref_structure_is_the_generated_ideal_fiber(vy1):
    """ref_structure = parts[0] 经 transformation 生成的理想 fiber, rmsd 是它与观测的 RMSD。"""
    struct, pairs, order = vy1
    fiber = _vy1_fiber(struct, pairs, order)
    fiber.fit()

    # 长度与观测纤维一致 (同样的 part 数)
    assert len(fiber.ref_structure) == len(struct)
    # rmsd 就是 ref_structure 与观测结构的逐原子 RMSD
    np.testing.assert_allclose(
        bt_struct.rmsd(fiber.ref_structure.coord, struct.coord), fiber.rmsd, atol=1e-6
    )
    assert fiber.rmsd == pytest.approx(0.9996, abs=0.05)   # 6vy1 实测值
    # 逐 part rmsd: 第一个 part 就是原型 (G+g) → 0; 其余各 ~1 A
    part_rmsd = fiber.extra_param["part_rmsd"]
    assert len(part_rmsd) == len(order)
    assert part_rmsd[0] == 0
    offset = 0
    for k, letter in enumerate(order):
        n = len(pairs[letter])
        rmsd = bt_struct.rmsd(fiber.ref_structure.coord[offset:offset + n],
                              pairs[letter].coord)
        assert rmsd == pytest.approx(part_rmsd[k], abs=1e-6)
        offset += n
    # 除原型外每个 part 的模型残差都在 1.5 A 以内 (构象差异, 非几何错误)
    assert part_rmsd[1:].max() < 1.5


def test_vy1_mask_path_matches_leaf_path(vy1):
    """from_atomarray(mask=...) 建树 (fit 从父 structure 切片) → 同一组参数。"""
    struct, pairs, order = vy1
    leaf = _vy1_fiber(struct, pairs, order)
    leaf.fit()

    masks, offset = {}, 0
    for letter in order:
        n = len(pairs[letter])
        mask = np.zeros(len(struct), dtype=bool)
        mask[offset:offset + n] = True
        masks[f"{letter}{letter.lower()}"] = mask
        offset += n
    mask_fiber = AssemblyFiber.from_atomarray(structure=struct, mask=masks)
    mask_fiber.fit()

    assert mask_fiber.param["omega"] == pytest.approx(leaf.param["omega"], abs=1e-9)
    assert mask_fiber.param["rise"] == pytest.approx(leaf.param["rise"], abs=1e-9)
    assert mask_fiber.rmsd == pytest.approx(leaf.rmsd, abs=1e-9)


# ----------------------------------------------------------------------
# 3. from_param (参数 → 坐标) 与命名 / 导出限制
# ----------------------------------------------------------------------

def test_from_param_builds_fiber_without_observation():
    """from_param 直接由参数造纤维 (AtomArray 也行), 参数 → 坐标 → 参数自洽。"""
    proto = _atoms(PROTOTYPE_COORD)
    T = _screw(SYNTH_OMEGA_DEG, SYNTH_RISE, SYNTH_AXIS_POINT)
    fiber = AssemblyFiber.from_param(
        proto, np.deg2rad(SYNTH_OMEGA_DEG), SYNTH_RISE, part_num=4,
        axis_point=SYNTH_AXIS_POINT,
    )

    assert list(fiber.parts) == [f"A:{k}" for k in range(4)]
    assert fiber.param["part_num"] == 4 and fiber.param["atom_num"] == len(proto)
    np.testing.assert_allclose(fiber.transformation, T, atol=1e-12)
    for k in range(4):
        expected = AssemblyFiber.apply_transformation(proto, T, k)
        np.testing.assert_allclose(
            fiber.parts[f"A:{k}"].structure.coord, expected.coord, atol=1e-12
        )
    # 局部帧: z = axis_direction, 原点在给定轴上 (只可能沿轴偏移)
    np.testing.assert_allclose(fiber.xyz[2], [0, 0, 1], atol=1e-6)
    np.testing.assert_allclose(
        np.cross(fiber.centroid - np.array(SYNTH_AXIS_POINT), [0, 0, 1]), 0, atol=1e-6
    )
    # 生成的纤维再 fit → 回到原来的参数
    fiber.fit()
    assert np.degrees(fiber.param["omega"]) == pytest.approx(SYNTH_OMEGA_DEG, abs=1e-4)
    assert fiber.param["rise"] == pytest.approx(SYNTH_RISE, abs=1e-5)
    assert fiber.rmsd < 1e-4


def test_part_identity_is_chain_plus_ins_code(tmp_path):
    """part 标识 = chain + ins_code: chain_id 不动, 序号进 ins_code (两种格式都自动读写)。

    ins_code 是 PDB 列 27 / mmCIF ``pdbx_PDB_ins_code`` —— 写回文件再读进来, 按 ins_code
    就能把各 part 分回来 (seg_id 做不到: 写 CIF 要 extra_fields, 读 PDB 会丢)。
    """
    from biorazer.structure.io.protein import AtomArray_Pdb, Pdb_AtomArray, AtomArray_Cif, Cif_AtomArray

    proto = _atoms(PROTOTYPE_COORD)
    fiber = AssemblyFiber.from_param(
        proto, np.deg2rad(SYNTH_OMEGA_DEG), SYNTH_RISE, part_num=3
    )
    assert list(fiber.parts) == ["A:0", "A:1", "A:2"]
    assert set(fiber.structure.chain_id) == {"A"}          # chain_id 从头到尾没被改
    for index, key in enumerate(fiber.parts):
        assert set(fiber.parts[key].structure.ins_code) == {PART_CODES[index]}
    n = len(proto)

    # PDB: chain 名合法 + ins_code 在列 27
    pdb_path = tmp_path / "fiber.pdb"
    AtomArray_Pdb(output_io=str(pdb_path)).write(fiber.structure)
    atoms = [l for l in pdb_path.read_text().splitlines() if l.startswith("ATOM")]
    assert [l[21] for l in atoms] == ["A"] * 3 * n
    assert [l[26] for l in atoms] == [PART_CODES[k] for k in range(3) for _ in range(n)]
    back = Pdb_AtomArray(input_io=str(pdb_path)).read()
    assert list(back.ins_code) == list(fiber.structure.ins_code)   # 读回来分得出 part

    # mmCIF: 同样的 ins_code, 不需要任何 extra_fields
    cif_path = tmp_path / "fiber.cif"
    AtomArray_Cif(output_io=str(cif_path)).write(fiber.structure)
    assert "pdbx_PDB_ins_code" in cif_path.read_text()
    back_cif = Cif_AtomArray(input_io=str(cif_path)).read()
    assert list(back_cif.ins_code) == list(fiber.structure.ins_code)
    # 按 (chain, ins_code) 分组就是 3 个 part, 每个 n 个原子
    keys = list(zip(back_cif.chain_id.tolist(), back_cif.ins_code.tolist()))
    assert [keys.count(("A", c)) for c in PART_CODES[:3]] == [n] * 3

    # 序号超过 ins_code 的 1 字符范围时显式报错
    with pytest.raises(ValueError, match="ins_code 只有 1 列"):
        fiber.generate_parts(len(PART_CODES) + 1)


# ----------------------------------------------------------------------
# 4. AssemblyCn: 纯转动 + 环闭合
# ----------------------------------------------------------------------

def test_cn_fit_recovers_rotation_only_ring():
    """合成 C3: rise 必须为 0, omega = 120 deg, step 覆盖 (2 -> 0) 闭合步。"""
    proto = _atoms(PROTOTYPE_COORD)
    T = _screw(-120.0, 0.0, SYNTH_AXIS_POINT)
    cn = _fiber_from_prototype(proto, 3, T, cls=AssemblyCn)
    cn.fit()

    assert cn.param["rise"] == 0.0                       # 纯转动, 不是 "≈0"
    assert cn.param["omega"] == 2 * np.pi / 3            # 转角由重数钉死 (精确)
    assert np.degrees(cn.extra_param["measured_omega"]) == pytest.approx(120.0, abs=1e-3)
    assert np.degrees(cn.extra_param["measured_omega"]) > 0   # 正向 = part 序号递增一侧
    assert cn.extra_param["implied_n"] == pytest.approx(3.0, abs=1e-3)
    assert cn.rmsd < 1e-4
    # 环状步: 3 个 (纤维只有 n-1 = 2 个), 且三步转角一致
    assert len(cn.extra_param["step_rmsd"]) == 3
    assert np.ptp(np.degrees(cn.extra_param["step_omega"])) < 1e-4   # biotite 内部 float32
    # transformation = 构造的纯旋转 (z 反向 + omega 反向 = 同一个刚体运动)
    np.testing.assert_allclose(cn.transformation, T, atol=1e-6)
    axis_point = cn.param["axis_point"]
    np.testing.assert_allclose(
        cn.transformation[:3, :3] @ axis_point + cn.transformation[:3, 3],
        axis_point, atol=1e-6,                             # 轴上点不动 → 无平移
    )
    # 转 n 次回到原处 (环闭合)
    np.testing.assert_allclose(
        AssemblyFiber.apply_transformation(proto, T, 3).coord, proto.coord, atol=1e-5
    )


def test_cn_ring_average_dilutes_a_pure_screw():
    """把 screw 观测面当 Cn 拟合: 闭合步 (3 -> 0) 要求 -3 步, 平均 omega 被稀释。

    4 个 part 各差 +100 deg / +5 A。Cn 的环形步里有一步是 "part3 -> part0" =
    -300 deg (≡ +60 deg), 于是平均 omega = (100+100+100+60)/4 = 90 deg, rmsd 崩掉
    —— 纯转动模型不适用于 screw 数据的直接证据 (该用 AssemblyFiber)。
    """
    proto = _atoms(PROTOTYPE_COORD)
    T = _screw(SYNTH_OMEGA_DEG, SYNTH_RISE)               # rise = 5 A ≠ 0
    cn = _fiber_from_prototype(proto, 4, T, cls=AssemblyCn)
    cn.fit()
    assert cn.param["rise"] == 0.0                        # 仍然只给纯转动
    assert cn.param["omega"] == 2 * np.pi / 4             # 钉死 = 360/4
    # 数据里量到的是被稀释的 90 deg (含 (3 -> 0) 那一步的 +60 deg)
    assert np.degrees(cn.extra_param["measured_omega"]) == pytest.approx(90.0, abs=1e-3)
    assert cn.extra_param["implied_n"] == pytest.approx(4.0, abs=1e-3)
    assert cn.rmsd > 1.0
    # 同一份数据用 AssemblyFiber (screw) 是精确的
    fiber = _fiber_from_prototype(proto, 4, T)
    fiber.fit()
    assert fiber.rmsd < 1e-4


def test_cn_from_param_makes_closed_ring():
    """from_param(n=4): 4 个 part, 每步 +90 deg, rise 0, 转 4 次回到原处。"""
    proto = _atoms(PROTOTYPE_COORD)
    cn = AssemblyCn.from_param(proto, n=4)
    assert list(cn.parts) == [f"A:{k}" for k in range(4)]
    assert cn.param["omega"] == pytest.approx(2 * np.pi / 4)
    assert cn.param["rise"] == 0.0
    np.testing.assert_allclose(
        AssemblyFiber.apply_transformation(proto, cn.transformation, 4).coord,
        proto.coord, atol=1e-9,
    )
    cn.fit()
    assert cn.param["omega"] == 2 * np.pi / 4             # 钉死, 与构造值一致
    assert np.degrees(cn.extra_param["measured_omega"]) == pytest.approx(90.0, abs=1e-6)
    assert cn.extra_param["implied_n"] == pytest.approx(4.0, abs=1e-6)
    assert cn.rmsd < 1e-6


def test_vy1_G_plus_g_is_a_C2(vy1):
    """6vy1 的 G/g 两条链是一对 C2 (垂直纤维轴), rise ≡ 0 —— AssemblyCn 能表达。"""
    struct, pairs, order = vy1
    fiber = _vy1_fiber(struct, pairs, order)
    fiber.fit()

    cn = AssemblyCn(
        structure=bt_struct.concatenate([pairs["G"], pairs["G"][pairs["G"].chain_id == "g"]])
    )
    for letter in ("G", "g"):
        chain = pairs["G"][pairs["G"].chain_id == letter]
        cn.append_part(letter, Assembly(structure=chain))
    cn.fit()

    # G → g: 2 重轴, 转角钉死 180 deg; 数据里量到 179.72 deg
    assert cn.param["omega"] == np.pi
    assert np.degrees(cn.extra_param["measured_omega"]) == pytest.approx(179.722, abs=0.05)
    assert cn.param["rise"] == 0.0
    assert cn.extra_param["implied_n"] == pytest.approx(2.0031, abs=0.002)
    # 叠合残差 1.268 A (G 与 g 的构象差异); rmsd 是含 part_0 (残差恒为 0) 的合并值
    assert cn.extra_param["step_rmsd"].max() == pytest.approx(1.268, abs=0.02)
    assert cn.rmsd == pytest.approx(cn.extra_param["part_rmsd"][1] / np.sqrt(2), abs=1e-9)
    # 2 重轴 ⟂ 纤维轴, 且两轴近乎相交 (两条直线的距离 < 1 A)。注意不能直接比两个
    # "轴上一点" —— pinv 给的是各自最近世界原点的点, 同一条轴上也会差很远。
    assert abs(float(np.dot(cn.param["axis_direction"], fiber.xyz[2]))) < 0.02
    cross = np.cross(fiber.xyz[2], cn.param["axis_direction"])
    gap = abs(float(np.dot(
        cn.param["axis_point"] - fiber.param["axis_point"], cross
    ))) / np.linalg.norm(cross)
    assert gap < 1.0                                      # 实测 0.34 A
    # 链对 A/a 是同一几何、构象差异更小的一对
    cn_a = AssemblyCn(structure=pairs["A"].copy())
    for letter in ("A", "a"):
        cn_a.append_part(letter, Assembly(structure=pairs["A"][pairs["A"].chain_id == letter]))
    cn_a.fit()
    assert cn_a.param["omega"] == np.pi
    assert np.degrees(cn_a.extra_param["measured_omega"]) == pytest.approx(179.925, abs=0.05)
    assert cn_a.extra_param["step_rmsd"].max() == pytest.approx(0.625, abs=0.02)
    assert cn_a.rmsd < cn.rmsd
