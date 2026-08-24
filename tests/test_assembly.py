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
