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


class TestReplacePart:
    """replace_part(): 把节点重构成特定子类 (如 CCCPHelixBundle) 后替换。"""

    def test_replace_leaf_with_bundle(self):
        from biorazer_prds.models.assembly_helix import CCCPHelixBundle

        # 构造一个 2 螺旋束 (14 原子, 2 个 CrickHelix 子节点)
        bun = CCCPHelixBundle.from_param(
            helix_num=2, residue_num=7, backbone_type="CA"
        )
        n = len(bun.structure)
        m1 = np.zeros(n, bool); m1[:7] = True
        m2 = np.zeros(n, bool); m2[7:] = True
        bundle = CCCPHelixBundle.from_atomarray(
            structure=bun.structure, mask={"helix_1": m1, "helix_2": m2}
        )

        # 父树里一个 14 原子的叶节点 'node'
        parent = Assembly.from_atomarray(structure=_atoms(20, 0, "A"))
        m_left = np.zeros(20, bool); m_left[:14] = True
        m_right = np.zeros(20, bool); m_right[14:] = True
        parent.split({"node": m_left, "other": m_right})
        assert parent.parts["node"].parts == {}

        # 重构成 CCCPHelixBundle 并替换
        parent.replace_part("node", bundle)
        assert parent.parts["node"] is bundle
        assert isinstance(parent.parts["node"], CCCPHelixBundle)
        # 父 mask 保持有效 (选中同样 14 原子区间)
        assert parent.mask["node"].sum() == 14 and len(parent.mask["node"]) == 20

        # 束可拟合; 父结构仍一致
        bundle.fit(verbose=False)
        assert bundle.rmsd is not None
        parent.merge_up()
        assert len(parent.structure) == 20
        assert parent.mask["node"].sum() == 14 and parent.mask["other"].sum() == 6

    def test_replace_atom_count_mismatch_raises(self):
        parent = Assembly.from_atomarray(structure=_atoms(6, 0, "A"))
        m = np.zeros(6, bool); m[:4] = True
        parent.split({"a": m})
        with pytest.raises(ValueError, match="原子数不一致"):
            parent.replace_part("a", Assembly(structure=_atoms(99, 0, "A")))
