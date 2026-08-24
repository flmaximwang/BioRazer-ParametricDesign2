"""CCCP 束拟合的 RMSD 报告与 ref_structure 标注复制测试。

覆盖两个回归:
1. RMSD 报告: fit_cc_by_cccp 之前把 ``result.fun`` (已是每原子 RMSD) 再按
   ``helix_num * residue_num`` 归一化, 导致报告值被低估 ``sqrt(helix_num*residue_num)``
   倍 (真实 ~5A 报成 ~0.5A)。修复后应直接报告每原子 RMSD。
2. ref_structure: generate_cc_ca_by_cccp 新增 ``ref_structure`` 参数, 生成时
   复制对应 residue CA 的 chain_id / res_id / res_name; 为 None 时回退到
   默认标注 (按索引 A,B,C,... 与 res_id 1..N)。CCCPHelixBundle.fit 借此让
   ref_structure 保留原始链标注, 避免按索引重排导致的链标签错位。
"""

import numpy as np
import pytest
import biotite.structure as bt_struct

from biorazer_prds.params.cccp.fit import fit_cc_by_cccp
from biorazer_prds.params.cccp.generate import generate_cc_ca_by_cccp
from biorazer_prds.models.assembly_helix import CCCPHelixBundle


def _chain_order(arr):
    seen = []
    for c in arr.chain_id:
        if not seen or c != seen[-1]:
            seen.append(c)
    return seen


class TestFitRmsdReporting:
    def test_rmsd_equals_true_per_atom_rmsd(self):
        rng = np.random.default_rng(0)
        # 理想 2 螺旋束 + 噪声 -> 拟合非完美, 使 RMSD 有真实大小
        base, _, _ = generate_cc_ca_by_cccp(
            helix_num=2, residue_num=12, centroid=[0, 0, 0]
        )
        obs = base + rng.normal(0, 0.4, base.shape)
        params, rmsd, xyz, structure_list = fit_cc_by_cccp(obs)
        # 每原子 RMSD = sqrt(mean over atoms of |dx|^2)
        true = np.sqrt(np.mean(np.sum((xyz - obs) ** 2, axis=2)))
        np.testing.assert_allclose(rmsd, true, rtol=1e-6)
        # 必须是真实大小 (旧 bug 会低估 sqrt(N_residues) 倍, 接近 0)
        assert rmsd > 0.1

    def test_bundle_fit_rmsd_matches_ref_structure(self):
        # 端到端: assembly.rmsd 应等于 ref_structure 与观测 CA 的每原子 RMSD
        base, _, _ = generate_cc_ca_by_cccp(
            helix_num=2, residue_num=10, centroid=[0, 0, 0]
        )
        aa = bt_struct.AtomArray(20)
        aa.coord = base.reshape(-1, 3)
        aa.atom_name = np.array(["CA"] * 20)
        aa.element = np.array(["C"] * 20)
        aa.chain_id = np.array(["X"] * 10 + ["Y"] * 10)
        aa.res_id = np.array(list(range(1, 11)) * 2)
        aa.res_name = np.array(["GLY"] * 20)
        m1 = np.zeros(20, bool); m1[:10] = True
        m2 = np.zeros(20, bool); m2[10:] = True
        bundle = CCCPHelixBundle.from_atomarray(structure=aa, mask={"h1": m1, "h2": m2})
        bundle.fit()
        obs = aa.coord
        fit_coord = bundle.ref_structure.coord
        true = np.sqrt(np.mean(np.sum((obs - fit_coord) ** 2, axis=1)))
        np.testing.assert_allclose(bundle.rmsd, true, rtol=1e-4)


class TestFitTrajectory:
    """fit_cc_by_cccp 返回每一步 optimize 生成的 structure 列表 (debug 用)。"""

    def test_returns_one_structure_per_optimize_step(self):
        rng = np.random.default_rng(0)
        base, _, _ = generate_cc_ca_by_cccp(
            helix_num=2, residue_num=12, centroid=[0, 0, 0]
        )
        obs = base + rng.normal(0, 0.2, base.shape)
        params, rmsd, xyz, structure_list = fit_cc_by_cccp(obs)
        # Stage 0 + 10 轮 * 3 个 stage = 31 步
        assert len(structure_list) == 1 + 3 * 10
        for arr in structure_list:
            assert isinstance(arr, bt_struct.AtomArray)
            assert len(arr) == 2 * 12  # helix_num * residue_num
        # 最后一个 structure 坐标应等于最终拟合坐标
        np.testing.assert_allclose(
            structure_list[-1].coord, xyz.reshape(-1, 3), atol=1e-9
        )

    def test_bundle_fit_returns_trajectory(self):
        base, _, _ = generate_cc_ca_by_cccp(
            helix_num=2, residue_num=10, centroid=[0, 0, 0]
        )
        aa = bt_struct.AtomArray(20)
        aa.coord = base.reshape(-1, 3)
        aa.atom_name = np.array(["CA"] * 20)
        aa.element = np.array(["C"] * 20)
        aa.chain_id = np.array(["X"] * 10 + ["Y"] * 10)
        aa.res_id = np.array(list(range(1, 11)) * 2)
        aa.res_name = np.array(["GLY"] * 20)
        m1 = np.zeros(20, bool); m1[:10] = True
        m2 = np.zeros(20, bool); m2[10:] = True
        bundle = CCCPHelixBundle.from_atomarray(structure=aa, mask={"h1": m1, "h2": m2})
        trajectory = bundle.fit()
        assert len(trajectory) == 1 + 3 * 10
        for arr in trajectory:
            assert isinstance(arr, bt_struct.AtomArray)
            assert len(arr) == 20
        # 不记录到属性
        assert not hasattr(bundle, "fit_trajectory")


class TestGenerateRefStructure:
    def test_none_uses_default_labeling(self):
        _, _, aa = generate_cc_ca_by_cccp(
            helix_num=2, residue_num=5, ref_structure=None
        )
        assert _chain_order(aa) == ["A", "B"]
        assert aa.res_id.tolist() == [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
        assert set(aa.res_name) == {"GLY"}
        assert set(aa.atom_name) == {"CA"}

    def test_copies_chain_residue_attributes(self):
        ref = bt_struct.AtomArray(10)
        ref.atom_name = np.array(["CA"] * 10)
        ref.element = np.array(["C"] * 10)
        ref.chain_id = np.array(["P"] * 5 + ["Q"] * 5)
        ref.res_id = np.array(list(range(11, 16)) * 2)
        ref.res_name = np.array(["LEU"] * 10)
        _, _, aa = generate_cc_ca_by_cccp(
            helix_num=2, residue_num=5, ref_structure=ref
        )
        np.testing.assert_array_equal(aa.chain_id, ref.chain_id)
        np.testing.assert_array_equal(aa.res_id, ref.res_id)
        np.testing.assert_array_equal(aa.res_name, ref.res_name)
        # 坐标仍为生成的拟合坐标 (非 ref 坐标)
        assert aa.coord.shape == (10, 3)

    def test_wrong_length_ref_structure_raises(self):
        ref = bt_struct.AtomArray(3)  # 期望 2*5=10
        ref.atom_name = np.array(["CA"] * 3)
        ref.chain_id = np.array(["A"] * 3)
        ref.res_id = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="ref_structure"):
            generate_cc_ca_by_cccp(helix_num=2, residue_num=5, ref_structure=ref)


class TestBundleFitPreservesChainLabels:
    def test_fit_ref_structure_keeps_original_chain_order(self):
        base, _, _ = generate_cc_ca_by_cccp(
            helix_num=2, residue_num=10, centroid=[0, 0, 0]
        )
        aa = bt_struct.AtomArray(20)
        aa.coord = base.reshape(-1, 3)
        aa.atom_name = np.array(["CA"] * 20)
        aa.element = np.array(["C"] * 20)
        # 两条螺旋用非常规链标签 X / Y
        aa.chain_id = np.array(["X"] * 10 + ["Y"] * 10)
        aa.res_id = np.array(list(range(1, 11)) * 2)
        aa.res_name = np.array(["GLY"] * 20)
        m1 = np.zeros(20, bool); m1[:10] = True
        m2 = np.zeros(20, bool); m2[10:] = True
        bundle = CCCPHelixBundle.from_atomarray(structure=aa, mask={"h1": m1, "h2": m2})
        bundle.fit()
        # ref_structure 应按 helix 顺序保留原始链标签 X, Y
        assert _chain_order(bundle.ref_structure) == ["X", "Y"]
        assert bundle.ref_structure.res_id.tolist() == list(range(1, 11)) * 2
