"""CrickHelix.trim_or_extend 内坐标 (IC) 实现正式测试。

覆盖完整 backbone 输入 (build_bb_chain_ic 理想 alpha-helix) 下:
- 既有链完全不动 (IC 方案核心承诺, atol 1e-6 硬断言)
- 新残基 CA 贴合 Crick 外推目标 / 理想 alpha-helix 延续 (atol 1e-3)
- 新残基 backbone 化学合理 (键长)
- 缩短 / resn / 错误处理

注意:
- 单次调用只处理一端 (docstring: 不支持两端)。顺序多次单端伸长会因居中
  t 网格相位漂移导致目标错位 (CCCP 束用"一次生成双端"规避, 见
  TestCCCPTrimOrExtend::test_extend_both_ends_single_regeneration_matches_ideal)。
- CA-only 输入的语义 (保持 CA-only) 由 tests/test_assembly.py
  TestCrickTrimOrExtend 覆盖。
"""

import numpy as np
import pytest

from biorazer_prds.models.assembly_helix import CrickHelix
from biorazer_prds.params.helix_cp.fit import fit_helix_by_crick
from biorazer_prds.params.helix_cp.generate import generate_helix_ca_by_crick
from biorazer_prds.params.util import build_bb_chain_ic


def _make_helix7():
    """理想 alpha-helix 7-mer (完整 backbone), 并拟合 Crick 参数。"""
    mer7 = build_bb_chain_ic(
        7, resn="GLY", ss="alpha-helix", start_res=1, chain_id="A"
    ).to_atomarray()
    ca7 = mer7.coord[mer7.atom_name == "CA"]
    r = fit_helix_by_crick(ca7, residue_num=7)
    p = r[0] if isinstance(r, tuple) else r
    h = CrickHelix.from_atomarray(mer7)
    h.param = {**p, "residue_num": 7}
    return h, mer7


def _by_atom(S):
    """{(res_id, atom_name): coord}, 对原子顺序不敏感。"""
    return {
        (int(r), str(a)): np.asarray(c, float)
        for r, a, c in zip(S.res_id, S.atom_name, S.coord)
    }


def _crick_target(param, n, terminus, current_residue_num=7):
    """与 _extend_backbone_internal_coord 相同的目标 CA 生成逻辑。"""
    kwargs = {**param, "residue_num": current_residue_num + 2 * n}
    helix_ca, _ = generate_helix_ca_by_crick(**kwargs)
    return helix_ca[:n] if terminus == "N" else helix_ca[-n:]


def _assert_existing_chain_untouched(S, mer7_before):
    before = _by_atom(mer7_before)
    after = _by_atom(S)
    assert set(before) <= set(after), "伸长后既有链原子丢失"
    for key in before:
        np.testing.assert_allclose(after[key], before[key], atol=1e-6)


def _assert_bonds_ideal(S, res_ids):
    for r in res_ids:
        A = {
            a: c
            for a, c in zip(S.atom_name[S.res_id == r], S.coord[S.res_id == r])
        }
        assert np.linalg.norm(A["N"] - A["CA"]) == pytest.approx(1.458, abs=0.05)
        assert np.linalg.norm(A["CA"] - A["C"]) == pytest.approx(1.525, abs=0.05)


class TestCrickTrimOrExtendIC:
    """CrickHelix.trim_or_extend 内坐标伸长 (完整 backbone 输入)。"""

    def test_extend_cterm_matches_ideal(self):
        h, mer7 = _make_helix7()
        h.trim_or_extend(3, "C")
        S = h.structure
        assert np.unique(S.res_id).tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        _assert_existing_chain_untouched(S, mer7)
        # 新残基 CA == Crick 外推目标 == 理想 alpha-helix 延续
        new_ca = S.coord[(S.atom_name == "CA") & (S.res_id > 7)]
        np.testing.assert_allclose(new_ca, _crick_target(h.param, 3, "C"), atol=1e-3)
        mer10 = build_bb_chain_ic(
            10, resn="GLY", ss="alpha-helix", start_res=1, chain_id="A"
        ).to_atomarray()
        ref10 = mer10.coord[(mer10.atom_name == "CA") & (mer10.res_id > 7)]
        np.testing.assert_allclose(new_ca, ref10, atol=1e-3)
        _assert_bonds_ideal(S, [8, 9, 10])

    def test_extend_nterm_matches_ideal(self):
        h, mer7 = _make_helix7()
        h.trim_or_extend(2, "N")
        S = h.structure
        assert np.unique(S.res_id).tolist() == [-1, 0, 1, 2, 3, 4, 5, 6, 7]
        _assert_existing_chain_untouched(S, mer7)
        new_ca = S.coord[(S.atom_name == "CA") & (S.res_id < 1)]
        np.testing.assert_allclose(new_ca, _crick_target(h.param, 2, "N"), atol=1e-3)
        _assert_bonds_ideal(S, [-1, 0])

    def test_trim_nterm(self):
        h, _ = _make_helix7()
        h.trim_or_extend(-2, "N")
        assert np.unique(h.structure.res_id).tolist() == [3, 4, 5, 6, 7]

    def test_trim_cterm(self):
        h, _ = _make_helix7()
        h.trim_or_extend(-3, "C")
        assert np.unique(h.structure.res_id).tolist() == [1, 2, 3, 4]

    @pytest.mark.parametrize("resn_in", ["ala", "A", "ALA"])
    def test_extend_resn(self, resn_in):
        h, _ = _make_helix7()
        h.trim_or_extend(1, "C", resn=resn_in)
        S = h.structure
        assert np.unique(S.res_id)[-1] == 8
        r8 = S[S.res_id == 8]
        assert set(r8.res_name) == {"ALA"}
        # ALA 模板带 CB 侧链原子
        assert "CB" in set(r8.atom_name)
        _assert_bonds_ideal(S, [8])

    def test_validation(self):
        h, _ = _make_helix7()
        with pytest.raises(ValueError, match="terminus"):
            h.trim_or_extend(1, "X")
        with pytest.raises(TypeError, match="整数"):
            h.trim_or_extend(1.5, "N")
        with pytest.raises(ValueError, match="无法缩短"):
            h.trim_or_extend(-7, "N")
        with pytest.raises(ValueError, match="残基"):
            h.trim_or_extend(1, "C", resn="XYZ")
