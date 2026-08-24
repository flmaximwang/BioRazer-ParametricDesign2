"""统一 Assembly 递归节点的树结构传播测试 (push_down / merge_up)。

覆盖:
- merge_up: 子 structure 按插入顺序拼接为父 structure, 且重建每个子节点的 mask
  为拼接数组上的连续布尔区间 (mask 不失效)。
- push_down: 用父 structure + mask 切片出子 structure, 逐层递归到叶。
- 两者严格互逆: merge_up 后 push_down 能还原出完全一致的子结构。
- 嵌套树 (内部节点还有子节点) 的多层往返一致。
"""

import numpy as np
import pytest
import biotite.structure as bt_struct

from biorazer_prds.models.assembly import Assembly


def _atoms(n, offset, chain):
    """构造 n 个原子的 AtomArray, 坐标为 (offset+i, 0, 0)。"""
    a = bt_struct.AtomArray(n)
    a.coord = np.array([[offset + i, 0, 0] for i in range(n)], dtype=float)
    a.atom_name = np.array(["CA"] * n)
    a.res_id = np.arange(1, n + 1)
    a.chain_id = np.array([chain] * n)
    a.element = np.array(["C"] * n)
    return a


def _coords(arr):
    return np.asarray(arr.coord)


class TestMergeUp:
    def test_concatenates_children_in_insertion_order(self):
        parent = Assembly()
        parent.append_part("a", Assembly(structure=_atoms(3, 0, "A")))
        parent.append_part("b", Assembly(structure=_atoms(2, 10, "B")))
        parent.merge_up()

        assert len(parent.structure) == 5
        # 拼接顺序 = parts 插入顺序: 先 a (3 个), 后 b (2 个)
        np.testing.assert_array_equal(
            _coords(parent.structure),
            np.concatenate([_coords(_atoms(3, 0, "A")), _coords(_atoms(2, 10, "B"))]),
        )

    def test_rebuilds_masks_as_contiguous_ranges(self):
        parent = Assembly()
        parent.append_part("a", Assembly(structure=_atoms(3, 0, "A")))
        parent.append_part("b", Assembly(structure=_atoms(2, 10, "B")))
        parent.merge_up()

        # mask 与拼接后的父 structure 同源: 长度一致, 且为连续 True 区间
        assert len(parent.mask["a"]) == len(parent.structure)
        assert len(parent.mask["b"]) == len(parent.structure)
        assert parent.mask["a"].sum() == 3
        assert parent.mask["b"].sum() == 2
        np.testing.assert_array_equal(
            parent.mask["a"], [True, True, True, False, False]
        )
        np.testing.assert_array_equal(
            parent.mask["b"], [False, False, False, True, True]
        )

    def test_leaf_merge_up_is_noop(self):
        leaf = Assembly(structure=_atoms(4, 0, "A"))
        before = _coords(leaf.structure).copy()
        leaf.merge_up()
        assert len(leaf.structure) == 4
        np.testing.assert_array_equal(_coords(leaf.structure), before)


class TestPushDown:
    def test_slices_children_from_parent(self):
        parent = Assembly(structure=_atoms(5, 0, "A"))
        parent.append_part("a", Assembly())
        parent.append_part("b", Assembly())
        parent.mask = {
            "a": np.array([True, True, True, False, False]),
            "b": np.array([False, False, False, True, True]),
        }
        parent.push_down()

        assert len(parent["a"].structure) == 3
        assert len(parent["b"].structure) == 2
        # 切片出来的子结构坐标 = 父结构中对应区间
        np.testing.assert_array_equal(
            _coords(parent["a"].structure), _coords(_atoms(3, 0, "A"))
        )
        np.testing.assert_array_equal(
            _coords(parent["b"].structure), _coords(_atoms(2, 3, "A"))
        )

    def test_missing_mask_raises(self):
        parent = Assembly(structure=_atoms(3, 0, "A"))
        parent.append_part("a", Assembly())
        parent.mask = {}  # 没有给 'a' 提供 mask
        with pytest.raises(ValueError, match="缺少 mask"):
            parent.push_down()


class TestRoundTrip:
    def test_merge_then_push_restores_children(self):
        child_a = Assembly(structure=_atoms(3, 0, "A"))
        child_b = Assembly(structure=_atoms(2, 10, "B"))
        parent = Assembly()
        parent.append_part("a", child_a).append_part("b", child_b)
        parent.merge_up()
        parent.push_down()

        assert len(parent["a"].structure) == 3
        assert len(parent["b"].structure) == 2
        np.testing.assert_array_equal(
            _coords(parent["a"].structure), _coords(_atoms(3, 0, "A"))
        )
        np.testing.assert_array_equal(
            _coords(parent["b"].structure), _coords(_atoms(2, 10, "B"))
        )

    def test_nested_tree_round_trip(self):
        # root -> grandA(internal: leafA1, leafA2) + leafB
        leaf_a1 = Assembly(structure=_atoms(2, 0, "A"))
        leaf_a2 = Assembly(structure=_atoms(3, 20, "B"))
        grand_a = Assembly()
        grand_a.append_part("a1", leaf_a1).append_part("a2", leaf_a2)
        leaf_b = Assembly(structure=_atoms(4, 100, "C"))
        root = Assembly()
        root.append_part("grandA", grand_a).append_part("leafB", leaf_b)

        # 自底向上合并到根
        root.merge_up()
        assert len(root.structure) == 2 + 3 + 4
        assert len(grand_a.structure) == 5

        # 自顶向下推回叶, 各级一致
        root.push_down()
        assert len(root["grandA"].structure) == 5
        assert len(root["grandA"]["a1"].structure) == 2
        assert len(root["grandA"]["a2"].structure) == 3
        assert len(root["leafB"].structure) == 4
        np.testing.assert_array_equal(
            _coords(root["grandA"]["a1"].structure), _coords(_atoms(2, 0, "A"))
        )
        np.testing.assert_array_equal(
            _coords(root["leafB"].structure), _coords(_atoms(4, 100, "C"))
        )

        # 每层 mask 都与各自父 structure 同长且为连续区间
        assert len(root.mask["grandA"]) == len(root.structure)
        assert len(root.mask["leafB"]) == len(root.structure)
        assert len(grand_a.mask["a1"]) == len(grand_a.structure)
        assert len(grand_a.mask["a2"]) == len(grand_a.structure)


class _AxisAssembly(Assembly):
    """提供局部坐标轴的 Assembly 子类 (用于 center / between 测试)。"""

    def __init__(self, structure, x=(1.0, 0, 0), y=(0.0, 1, 0), z=(0.0, 0, 1)):
        super().__init__(structure=structure)
        self._x = np.asarray(x, float)
        self._y = np.asarray(y, float)
        self._z = np.asarray(z, float)

    @property
    def xyz(self):
        return self._x, self._y, self._z


class TestCenterThenMergeUp:
    """center_part 已移除; 正确顺序 = 子节点 center() + 父节点 merge_up()。"""

    def test_center_child_then_merge_up_recomputes_parent(self):
        child_a = _AxisAssembly(_atoms(3, 0, "A"))  # 规范轴, center 只做平移
        child_b = _AxisAssembly(_atoms(2, 10, "B"))
        parent = Assembly()
        parent.append_part("a", child_a).append_part("b", child_b)

        child_a.center(max_try=10)
        # 平移后 a 的质心回到原点
        np.testing.assert_allclose(child_a.centroid, [0, 0, 0], atol=1e-6)

        parent.merge_up()
        assert len(parent.structure) == 5
        # a 居中后, 父结构里 a 区间的质心也应为 0
        a_coords = parent.structure.coord[parent.mask["a"]]
        np.testing.assert_allclose(a_coords.mean(axis=0), [0, 0, 0], atol=1e-6)
        # mask 与父结构同源
        assert parent.mask["a"].sum() == 3
        assert parent.mask["b"].sum() == 2

    def test_center_part_is_removed(self):
        parent = Assembly()
        assert not hasattr(parent, "center_part")


class TestStaticBetween:
    """calculate_*_between 为 staticmethod, 可直接对任意两个 Assembly 计算。"""

    def test_staticmethod(self):
        import inspect

        # 声明为 staticmethod (类属性是 staticmethod 对象, 而非普通函数/绑定方法)
        assert isinstance(
            inspect.getattr_static(Assembly, "calculate_rotation_between"),
            staticmethod,
        )
        assert isinstance(
            inspect.getattr_static(Assembly, "calculate_translation_between"),
            staticmethod,
        )

    def test_translation_between(self):
        a = _AxisAssembly(_atoms(3, 0, "A"))
        b = _AxisAssembly(_atoms(2, 10, "B"))
        t = Assembly.calculate_translation_between(a, b)
        np.testing.assert_allclose(t, b.centroid - a.centroid)

    def test_rotation_between_identity_for_canonical(self):
        a = _AxisAssembly(_atoms(3, 0, "A"))  # 规范轴
        b = _AxisAssembly(_atoms(2, 10, "B"))  # 规范轴
        rot = Assembly.calculate_rotation_between(a, b)
        np.testing.assert_allclose(rot.as_matrix(), np.eye(3), atol=1e-6)

    def test_quat_and_euler_consistent(self):
        a = _AxisAssembly(_atoms(3, 0, "A"))
        b = _AxisAssembly(_atoms(2, 10, "B"))
        q = Assembly.calculate_quat_between(a, b)
        e = Assembly.calculate_euler_between(a, b, "xyz")
        from scipy.spatial.transform import Rotation as R

        np.testing.assert_allclose(
            R.from_quat(q).as_euler("xyz"), e, atol=1e-6
        )


class TestFromAtomArray:
    """Assembly.from_atomarray: mask='all' 为叶; mask=dict 构建树 (含嵌套投影)。"""

    def test_leaf_default(self):
        leaf = Assembly.from_atomarray(structure=_atoms(4, 0, "A"))
        assert leaf.parts == {}
        assert leaf.mask == {}
        assert len(leaf.structure) == 4

    def test_flat_tree(self):
        m1 = np.array([True, True, True, False, False], dtype=bool)
        m2 = np.array([False, False, False, True, True], dtype=bool)
        root = Assembly.from_atomarray(
            structure=_atoms(5, 0, "A"), mask={"a": m1, "b": m2}
        )
        assert set(root.parts) == {"a", "b"}
        assert len(root.parts["a"].structure) == 3
        assert len(root.parts["b"].structure) == 2
        # 根 mask 与输入等长 (指向顶层结构)
        assert len(root.mask["a"]) == 5 and root.mask["a"].sum() == 3
        # 子节点是叶 (mask='all' => 无 parts / 空 mask)
        assert root.parts["a"].parts == {}
        assert root.parts["a"].mask == {}

    def test_nested_projection(self):
        # 输入所有 mask 等长于顶层 structure; dict 嵌套即树拓扑
        m11 = np.array([True, True, False, False, False, False], dtype=bool)   # 2
        m12 = np.array([False, False, True, False, False, False], dtype=bool)  # 1
        m2 = np.array([False, False, False, True, True, True], dtype=bool)     # 3
        nested = Assembly.from_atomarray(
            structure=_atoms(6, 0, "A"),
            mask={"grp": {"x": m11, "y": m12}, "b": m2},
        )
        assert set(nested.parts) == {"grp", "b"}
        assert set(nested.parts["grp"].parts) == {"x", "y"}
        assert len(nested.parts["grp"].structure) == 3  # x(2)+y(1)
        assert len(nested.parts["grp"].parts["x"].structure) == 2
        assert len(nested.parts["b"].structure) == 3
        # grp 在顶层结构上的掩码 = x|y (等长 6)
        assert len(nested.mask["grp"]) == 6 and nested.mask["grp"].sum() == 3
        # x 在 grp structure 上的掩码被投影为长度 3 (≠ 输入长度 6)
        xm = nested.parts["grp"].mask["x"]
        assert len(xm) == 3 and xm.sum() == 2
        np.testing.assert_array_equal(xm, [True, True, False])

    def test_nested_merge_push_round_trip(self):
        m11 = np.array([True, True, False, False, False, False], dtype=bool)
        m12 = np.array([False, False, True, False, False, False], dtype=bool)
        m2 = np.array([False, False, False, True, True, True], dtype=bool)
        nested = Assembly.from_atomarray(
            structure=_atoms(6, 0, "A"),
            mask={"grp": {"x": m11, "y": m12}, "b": m2},
        )
        nested.merge_up()
        assert len(nested.structure) == 6
        assert len(nested.parts["grp"].structure) == 3
        # 每层 mask 与各自父 structure 同长
        assert len(nested.mask["grp"]) == 6
        assert len(nested.parts["grp"].mask["x"]) == 3
        nested.push_down()
        assert len(nested.parts["grp"].parts["x"].structure) == 2

    def test_ref_structure(self):
        rs = _atoms(3, 50, "R")
        r = Assembly.from_atomarray(structure=_atoms(4, 0, "A"), ref_structure=rs)
        assert r.ref_structure is rs

    def test_uncovered_atoms_raise(self):
        # 所有 mask 取 or 后仍留有 False => 未被任何子 assembly 覆盖 => 抛错
        m1 = np.array([True, True, True, False, False], dtype=bool)
        m2 = np.array([False, False, False, True, False], dtype=bool)  # 第 5 原子未覆盖
        with pytest.raises(ValueError, match="未完全覆盖"):
            Assembly.from_atomarray(structure=_atoms(5, 0, "A"), mask={"a": m1, "b": m2})

    def test_nested_full_coverage_passes(self):
        # 嵌套 dict 递归展开后 or 完全覆盖 (含跨层互补) 则正常构建
        m11 = np.array([True, True, False, False, False], dtype=bool)
        m12 = np.array([False, False, True, False, False], dtype=bool)
        m2 = np.array([False, False, False, True, True], dtype=bool)
        root = Assembly.from_atomarray(
            structure=_atoms(5, 0, "A"),
            mask={"grp": {"x": m11, "y": m12}, "b": m2},
        )
        assert set(root.parts) == {"grp", "b"}
        assert len(root.parts["grp"].structure) == 3
        assert len(root.parts["b"].structure) == 2


class TestSplit:
    """split(): 把已构造的叶节点继续拆分为子树。"""

    def test_split_leaf_into_children(self):
        leaf = Assembly.from_atomarray(structure=_atoms(5, 0, "A"))
        assert leaf.parts == {} and leaf.mask == {}
        m1 = np.array([True, True, True, False, False], dtype=bool)
        m2 = np.array([False, False, False, True, True], dtype=bool)
        leaf.split({"a": m1, "b": m2})
        assert set(leaf.parts) == {"a", "b"}
        assert len(leaf.parts["a"].structure) == 3
        assert len(leaf.parts["b"].structure) == 2
        assert leaf.mask["a"].sum() == 3 and len(leaf.mask["a"]) == 5

    def test_split_nested_projection(self):
        leaf = Assembly.from_atomarray(structure=_atoms(6, 0, "A"))
        m11 = np.array([True, True, False, False, False, False], dtype=bool)
        m12 = np.array([False, False, True, False, False, False], dtype=bool)
        m2 = np.array([False, False, False, True, True, True], dtype=bool)
        leaf.split({"grp": {"x": m11, "y": m12}, "b": m2})
        assert len(leaf.parts["grp"].structure) == 3
        assert len(leaf.parts["grp"].parts["x"].structure) == 2
        # 子节点 mask 被投影到其自身 structure 上 (长度 ≠ 输入)
        assert len(leaf.parts["grp"].mask["x"]) == 3

    def test_split_then_merge_push_round_trip(self):
        leaf = Assembly.from_atomarray(structure=_atoms(6, 0, "A"))
        m11 = np.array([True, True, False, False, False, False], dtype=bool)
        m12 = np.array([False, False, True, False, False, False], dtype=bool)
        m2 = np.array([False, False, False, True, True, True], dtype=bool)
        leaf.split({"grp": {"x": m11, "y": m12}, "b": m2})
        leaf.merge_up()
        assert len(leaf.structure) == 6
        leaf.push_down()
        assert len(leaf.parts["grp"].parts["x"].structure) == 2

    def test_split_twice_raises(self):
        leaf = Assembly.from_atomarray(structure=_atoms(4, 0, "A"))
        m = {f"x": np.array([True] * 4, dtype=bool)}
        leaf.split(m)
        with pytest.raises(ValueError, match="已有子节点"):
            leaf.split({"y": np.array([True] * 4, dtype=bool)})


class TestSetType:
    """set_type(): 修改自身类型, 保留 structure/parts/mask (mask 属结构, 不由类型指定)。"""

    def test_set_type_changes_class_preserves_structure(self):
        leaf = Assembly.from_atomarray(structure=_atoms(5, 0, "A"))
        from biorazer_prds.models.assembly_helix import CrickHelix

        before = _coords(leaf.structure).copy()
        leaf.set_type(CrickHelix)
        assert isinstance(leaf, CrickHelix)
        assert len(leaf.structure) == 5
        np.testing.assert_array_equal(_coords(leaf.structure), before)
        # _parent 保留
        assert leaf._parent is None

    def test_set_type_internal_node_to_bundle(self):
        from biorazer_prds.models.assembly_helix import CCCPHelixBundle

        # 先定义结构: 内部节点 'bundle' 有两个螺旋子节点
        S = _atoms(14, 0, "A")
        m_helix1 = np.zeros(14, bool); m_helix1[:7] = True
        m_helix2 = np.zeros(14, bool); m_helix2[7:] = True
        tree = Assembly.from_atomarray(
            structure=S, mask={"bundle": {"helix_1": m_helix1, "helix_2": m_helix2}}
        )
        node = tree["bundle"]
        assert not isinstance(node, CCCPHelixBundle)

        # 再指定类型
        node.set_type(CCCPHelixBundle)
        assert isinstance(node, CCCPHelixBundle)
        # 子节点 (螺旋) 与 mask 保留
        assert set(node.parts) == {"helix_1", "helix_2"}
        assert len(node.parts["helix_1"].structure) == 7
        assert node.helix_num == 2
        # 父节点对 'bundle' 的引用仍有效, mask 一致
        assert tree["bundle"] is node
        assert tree.mask["bundle"].sum() == 14


class TestRotateRotvec:
    """rotate_rotvec: 绕过给定点的轴旋转, 检验轴上点不动、整圈还原、
    与 scipy 绕点旋转一致、角度单位 (rad/deg) 一致、零向量报错。"""

    @staticmethod
    def _make():
        return Assembly(structure=_atoms(7, 0, "A"))

    def test_point_on_axis_is_invariant(self):
        a = self._make()
        before = _coords(a.structure).copy()
        # 轴过 (3,0,0) 且方向 [1,0,0]; 原子索引 3 坐标恰为 (3,0,0) 落在轴上
        a.rotate_rotvec([3, 0, 0], [1, 0, 0], np.pi / 3)
        np.testing.assert_allclose(
            np.asarray(a.structure.coord[3]), [3, 0, 0], atol=1e-12
        )
        # 每个原子到轴上点 (3,0,0) 的距离在旋转前后不变
        for i in range(len(before)):
            d_before = np.linalg.norm(before[i] - [3, 0, 0])
            d_after = np.linalg.norm(a.structure.coord[i] - [3, 0, 0])
            np.testing.assert_allclose(d_after, d_before, atol=1e-12)

    def test_full_turn_restores_original(self):
        a = self._make()
        before = _coords(a.structure).copy()
        a.rotate_rotvec([1, 2, 3], [0, 0, 1], 2 * np.pi)
        np.testing.assert_allclose(_coords(a.structure), before, atol=1e-9)

    def test_matches_scipy_rotvec_about_point(self):
        from scipy.spatial.transform import Rotation as R
        a = self._make()
        p = np.array([2.0, -1.0, 4.0])
        axis = np.array([1.0, 2.0, -1.0])
        axis = axis / np.linalg.norm(axis)
        ang = 0.7
        expected = R.from_rotvec(axis * ang).apply(
            np.asarray(a.structure.coord) - p
        ) + p
        a.rotate_rotvec(p, axis, ang)
        np.testing.assert_allclose(_coords(a.structure), expected, atol=1e-12)

    def test_composition_equals_sum_of_angles(self):
        a = self._make()
        b = a.copy()
        p, axis = [[0, 0, 5], [0, 1, 0]]
        a.rotate_rotvec(p, axis, 1.0)
        b.rotate_rotvec(p, axis, 0.5)
        b.rotate_rotvec(p, axis, 0.5)
        np.testing.assert_allclose(
            _coords(a.structure), _coords(b.structure), atol=1e-12
        )

    def test_degrees_flag_consistent(self):
        from scipy.spatial.transform import Rotation as R
        a = self._make()
        b = self._make()
        a.rotate_rotvec([0, 0, 0], [0, 0, 1], 90, degrees=True)
        b.rotate_rotvec([0, 0, 0], [0, 0, 1], np.pi / 2)
        np.testing.assert_allclose(
            _coords(a.structure), _coords(b.structure), atol=1e-12
        )

    def test_zero_axis_raises(self):
        a = self._make()
        with pytest.raises(ValueError, match="non-zero"):
            a.rotate_rotvec([0, 0, 0], [0, 0, 0], 1.0)


class _CnProbe(Assembly):
    """calculate_Cn_among 的最小具体探针: centroid=CA 质心, xyz=自定义正交帧。

    C3+ 的轴方向由质心 SVD 求得 (不需 xyz); xyz 仅在 C2 时兜底使用。因此这里
    xyz 存一个固定帧即可, 无需依赖 helix 拟合。
    """

    def __init__(self, structure, axes=None):
        super().__init__(structure=structure)
        self._axes = np.eye(3) if axes is None else np.asarray(axes, float)

    @property
    def xyz(self):
        return self._axes


def _cn_generator(n, coords, axis_point, axis_dir):
    """生成 n 个绕 axis (axis_point, axis_dir) 相隔 2π/n 旋转的对称副本。"""
    base = _CnProbe(_atoms_from_coords(coords))
    out = []
    for k in range(n):
        c = base.copy()
        c.rotate_rotvec(axis_point, axis_dir, k * 2 * np.pi / n)
        out.append(c)
    return out


def _atoms_from_coords(coords):
    a = bt_struct.AtomArray(len(coords))
    a.coord = np.asarray(coords, dtype=float)
    a.atom_name = np.array(["CA"] * len(coords))
    a.res_id = np.arange(1, len(coords) + 1)
    a.chain_id = np.array(["A"] * len(coords))
    a.element = np.array(["C"] * len(coords))
    return a


_CN_COORDS = [[0, 0, 0], [2, 1, 0], [0, 2, 1], [1, 1, 2], [-2, 0, 3]]
_CN_AXIS_POINT = np.array([3.0, -2.0, 5.0])
_CN_AXIS_DIR = np.array([1.0, 1.0, 1.0])
_CN_AXIS_DIR = _CN_AXIS_DIR / np.linalg.norm(_CN_AXIS_DIR)


class TestCalculateCnAmong:
    """calculate_Cn_among: 骨架/全原子 RMSD, 总原子数不同的兜底与报错。"""

    def test_c3_exact_symmetry_recovers_axis_and_zero_rmsd(self):
        copies = _cn_generator(3, _CN_COORDS, _CN_AXIS_POINT, _CN_AXIS_DIR)
        res = Assembly.calculate_Cn_among(copies, bb_rmsd=True)
        # 轴方向恢复 (误差 < 1e-3)
        assert abs(float(np.dot(res["axis_direction"], _CN_AXIS_DIR))) > 1 - 1e-3
        # 严格对称: 旋转后与目标一致, RMSD ≈ 0
        np.testing.assert_allclose(res["rmsd"], [0, 0], atol=1e-4)

    def test_bb_rmsd_false_also_recovers_same_atom_count(self):
        copies = _cn_generator(3, _CN_COORDS, _CN_AXIS_POINT, _CN_AXIS_DIR)
        res = Assembly.calculate_Cn_among(copies, bb_rmsd=False)
        np.testing.assert_allclose(res["rmsd"], [0, 0], atol=1e-4)

    def test_bb_rmsd_true_tolerates_differing_total_atom_count(self):
        """总原子数不同 (多一个侧链 H) 但骨架一致: bb_rmsd=True 不报错, RMSD≈0。"""
        copies = _cn_generator(3, _CN_COORDS, _CN_AXIS_POINT, _CN_AXIS_DIR)
        extra = bt_struct.AtomArray(1)
        extra.coord = np.array([[1.0, 1.0, 1.0]])
        extra.atom_name = np.array(["HE1"])
        extra.res_id = np.array([1])
        extra.chain_id = np.array(["A"])
        extra.element = np.array(["H"])
        mixed = copies[0].copy()
        mixed2 = copies[1].copy()
        mixed2.structure = bt_struct.concatenate([mixed2.structure, extra])
        mixed3 = copies[2].copy()
        res = Assembly.calculate_Cn_among([mixed, mixed2, mixed3], bb_rmsd=True)
        np.testing.assert_allclose(res["rmsd"], [0, 0], atol=1e-4)

    def test_bb_rmsd_false_raises_on_differing_atom_count(self):
        """全原子 (bb_rmsd=False) 且原子数不同 → 显式报错, 而非广播得出错误 RMSD。"""
        copies = _cn_generator(3, _CN_COORDS, _CN_AXIS_POINT, _CN_AXIS_DIR)
        extra = bt_struct.AtomArray(1)
        extra.coord = np.array([[1.0, 1.0, 1.0]])
        extra.atom_name = np.array(["HE1"])
        extra.res_id = np.array([1])
        extra.chain_id = np.array(["A"])
        extra.element = np.array(["H"])
        mixed = copies[0].copy()
        mixed2 = copies[1].copy()
        mixed2.structure = bt_struct.concatenate([mixed2.structure, extra])
        mixed3 = copies[2].copy()
        with pytest.raises(ValueError, match="原子数不一致"):
            Assembly.calculate_Cn_among([mixed, mixed2, mixed3], bb_rmsd=False)


class TestCCCPCenterConvergence:
    """CCCPHelixBundle.center() 在含子节点的束上应能收敛。

    回归: fit() 必须读 ``self.structure`` + mask 而非子节点 structure。否则
    ``center()`` 的 rotate/translate 只更新父 structure, 不更新子节点; 若 fit
    读子节点, 每轮重 fit 都看到未变坐标, 返回同一帧 → 死循环 → TimeoutError。
    """

    @staticmethod
    def _make_rotated_bundle():
        from biorazer_prds.models.assembly_helix import CCCPHelixBundle
        from scipy.spatial.transform import Rotation as R

        base = CCCPHelixBundle.from_param(
            helix_num=2, residue_num=12, centroid=[0, 0, 0],
            y_prototype=[0, 1, 0], z=[0, 0, 1], backbone_type="CA",
        )
        S = base.structure
        n = len(S) // 2
        m1 = np.zeros(len(S), bool); m1[:n] = True
        m2 = np.zeros(len(S), bool); m2[n:] = True
        bundle = CCCPHelixBundle.from_atomarray(
            structure=S, mask={"h1": m1, "h2": m2}
        )
        rot = R.from_euler("xyz", [40, -25, 60], degrees=True)
        bundle.structure.coord = rot.apply(bundle.structure.coord) + np.array([30.0, -20.0, 10.0])
        return bundle

    def test_center_converges_on_rotated_bundle(self):
        bundle = self._make_rotated_bundle()
        bundle.center(max_try=60, atol_rot=1e-3, atol_trans=1e-3)
        # 收敛后: 束质心回到原点, 束轴 (z) 对齐规范轴
        np.testing.assert_allclose(np.asarray(bundle.centroid), [0, 0, 0], atol=1e-2)
        np.testing.assert_allclose(np.asarray(bundle.xyz[2]), [0, 0, 1], atol=1e-2)

    def test_fit_reflects_structure_rotation(self):
        """fit() 读到的是被旋转过的 self.structure (而非未更新的子节点)。"""
        from scipy.spatial.transform import Rotation as R

        bundle = self._make_rotated_bundle()
        before = bundle.xyz[2].copy()
        # 把束绕 x 轴再转 90°, 重新 fit 后 z 轴应随之旋转 (不再等于原 z)
        extra = R.from_euler("x", 90, degrees=True)
        bundle.structure.coord = extra.apply(bundle.structure.coord)
        bundle._xyz = None
        bundle._centroid = None
        bundle.fit()
        assert not np.allclose(np.asarray(bundle.xyz[2]), before, atol=1e-2)


class TestCCCPTrimOrExtend:
    """CCCPHelixBundle.trim_or_extend: 用束参数伸长/缩短单根螺旋, 重建束结构。"""

    @staticmethod
    def _make_bundle(residue_num=7):
        from biorazer_prds.models.assembly_helix import CCCPHelixBundle

        base = CCCPHelixBundle.from_param(
            helix_num=2, residue_num=residue_num, centroid=[0, 0, 0],
            y_prototype=[0, 1, 0], z=[0, 0, 1], backbone_type="CA",
        )
        S = base.structure
        n = len(S) // 2
        m1 = np.zeros(len(S), bool); m1[:n] = True
        m2 = np.zeros(len(S), bool); m2[n:] = True
        bundle = CCCPHelixBundle.from_atomarray(structure=S, mask={"h1": m1, "h2": m2})
        bundle.fit()  # 填充束参数 (伸长几何由这些拟合参数生成)
        return bundle

    def test_extend_helix0_nterm_uses_bundle_param(self):
        bundle = self._make_bundle()
        h0_before = bundle.parts["h1"].structure.coord.copy()
        h1_before = bundle.parts["h2"].structure.coord.copy()
        new_len = len(np.unique(bundle.parts["h1"].structure.res_id)) + 2

        result = bundle.trim_or_extend(0, 2, "N")

        # 束结构被重建, helix 0 变长 2, helix 1 不变
        assert result is bundle.parts["h1"]
        assert len(np.unique(bundle.parts["h1"].structure.res_id)) == new_len
        assert len(np.unique(bundle.parts["h2"].structure.res_id)) == new_len - 2
        np.testing.assert_allclose(
            bundle.parts["h2"].structure.coord, h1_before, atol=1e-12
        )
        # 新增 N 端残基在质心 z 更负的一端 (沿 +z 前进, N 端在 -z)
        z_min_old = h0_before[:, 2].min()
        assert bundle.parts["h1"].structure.coord[:, 2].min() < z_min_old

    def test_extend_helix1_cterm(self):
        bundle = self._make_bundle()
        h0_before = bundle.parts["h1"].structure.coord.copy()
        new_len = len(np.unique(bundle.parts["h2"].structure.res_id)) + 3

        bundle.trim_or_extend(1, 3, "C")
        assert len(np.unique(bundle.parts["h2"].structure.res_id)) == new_len
        # helix 0 不受影响
        np.testing.assert_allclose(
            bundle.parts["h1"].structure.coord, h0_before, atol=1e-12
        )
        # C 端新残基在 +z 更远端
        z_max_old = h0_before[:, 2].max()
        assert bundle.parts["h2"].structure.coord[:, 2].max() >= z_max_old

    def test_trim_helix(self):
        bundle = self._make_bundle()
        h1_before = bundle.parts["h2"].structure.coord.copy()
        new_len = len(np.unique(bundle.parts["h1"].structure.res_id)) - 2

        bundle.trim_or_extend(0, -2, "C")
        assert len(np.unique(bundle.parts["h1"].structure.res_id)) == new_len
        np.testing.assert_allclose(
            bundle.parts["h2"].structure.coord, h1_before, atol=1e-12
        )

    def test_validation_errors(self):
        bundle = self._make_bundle()
        with pytest.raises(IndexError, match="越界"):
            bundle.trim_or_extend(5, 1, "N")
        with pytest.raises(ValueError, match="terminus"):
            bundle.trim_or_extend(0, 1, "X")
        with pytest.raises(TypeError, match="整数"):
            bundle.trim_or_extend(0, 1.5, "N")
        # 缩短到不足 1 个残基
        with pytest.raises(ValueError, match="无法缩短"):
            bundle.trim_or_extend(0, -7, "N")

    def test_extend_missing_params_raises(self):
        bundle = self._make_bundle()
        bundle.param = {"residue_num": 7}  # 破坏参数完整性
        with pytest.raises(ValueError, match="缺少"):
            bundle.trim_or_extend(0, 2, "N")

    def test_leaf_bundle_raises(self):
        from biorazer_prds.models.assembly_helix import CCCPHelixBundle

        leaf = CCCPHelixBundle.from_param(helix_num=2, residue_num=7)
        assert not leaf.parts
        with pytest.raises(ValueError, match="子螺旋"):
            leaf.trim_or_extend(0, 1, "N")


class TestReplaceWith:
    """replace_with(): 用新块替换本节点, 重建本节点及以上所有祖先的 structure/mask。"""

    def test_replace_leaf_rebuilds_ancestors(self):
        # 三层树: root -> mid -> leaf ; 替换 leaf 为原子数不同的块
        S = _atoms(12, 0, "A")
        m_leaf = np.zeros(12, bool); m_leaf[:3] = True
        m_mid = np.zeros(12, bool); m_mid[:3] = True
        m_other = np.zeros(12, bool); m_other[3:] = True
        root = Assembly.from_atomarray(
            structure=S, mask={"mid": {"leaf": m_leaf}, "other": m_other}
        )
        leaf = root["mid"]["leaf"]
        assert len(leaf.structure) == 3

        # 替换成 5 原子的块 (原子数不同)
        new_chunk = Assembly.from_atomarray(structure=_atoms(5, 100, "A"))
        new_leaf = leaf.replace_with(new_chunk)

        # 新块挂上, 旧块脱离
        assert root["mid"]["leaf"] is new_leaf
        assert len(new_leaf.structure) == 5
        # 祖先 structure/mask 重建
        assert len(root["mid"].structure) == 5           # mid = leaf(5)
        assert root["mid"].mask["leaf"].sum() == 5
        assert len(root.structure) == 5 + 9              # root = mid(5) + other(9)
        assert root.mask["mid"].sum() == 5
        assert root.mask["other"].sum() == 9
        # mask 与各自 structure 同长
        assert len(root.mask["mid"]) == 14
        assert len(root["mid"].mask["leaf"]) == 5

    def test_replace_with_preserves_sibling(self):
        S = _atoms(10, 0, "A")
        m_a = np.zeros(10, bool); m_a[:4] = True
        m_b = np.zeros(10, bool); m_b[4:] = True
        root = Assembly.from_atomarray(structure=S, mask={"a": m_a, "b": m_b})
        root["a"].replace_with(Assembly.from_atomarray(structure=_atoms(6, 200, "A")))
        assert len(root.structure) == 6 + 6
        assert root.mask["a"].sum() == 6
        assert root.mask["b"].sum() == 6
