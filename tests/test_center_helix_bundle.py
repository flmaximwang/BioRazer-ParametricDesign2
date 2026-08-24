"""center_helix_bundle.py 脚本端到端测试。

覆盖重构后的行为: 输入含 loop 的结构, 输出为完整居中结构 (helix + loop),
不丢弃 loop; helix 束被居中到原点。
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import biotite.structure as bt_struct
from scipy.spatial.transform import Rotation as R

from biorazer_prds.models.assembly_helix import CCCPHelixBundle
from biorazer.structure.io.protein import Pdb_AtomArray, AtomArray_Pdb

SCRIPT = Path(__file__).resolve().parent.parent / "biorazer_prds/scripts/center_helix_bundle.py"
REPO_ROOT = SCRIPT.parent.parent


def _make_rotated_bundle_with_loop():
    """2-helix CCCP 束 (规范朝向) + 8 残基 loop, 整体施加已知刚体变换。"""
    base = CCCPHelixBundle.from_param(
        helix_num=2, residue_num=12, centroid=[0, 0, 0],
        y_prototype=[0, 1, 0], z=[0, 0, 1], backbone_type="CA",
    )
    loop = bt_struct.AtomArray(length=8)
    loop.atom_name = np.array(["CA"] * 8)
    loop.element = np.array(["C"] * 8)
    loop.chain_id = np.array(["C"] * 8)
    loop.res_id = np.arange(1, 9)
    loop.res_name = np.array(["GLY"] * 8)
    loop.coord = np.array([[5.0, 5.0, 5.0]] * 8) + np.arange(8)[:, None] * np.array(
        [0.5, 0.3, 0.1]
    )
    merged = bt_struct.concatenate([base.structure, loop])
    rot = R.from_euler("xyz", [40, -25, 60], degrees=True)
    merged.coord = rot.apply(merged.coord) + np.array([30.0, -20.0, 10.0])
    return merged


class TestCenterHelixBundleScript:
    def test_outputs_full_centered_structure(self, tmp_path):
        inp = tmp_path / "in.pdb"
        out = tmp_path / "out.pdb"
        AtomArray_Pdb(output_io=str(inp)).write(_make_rotated_bundle_with_loop())

        result = subprocess.run(
            [sys.executable, str(SCRIPT), str(inp),
             "--helix", "A:1-12", "--helix", "B:1-12",
             "-o", str(out), "--atol-rot", "1e-3", "--atol-trans", "1e-3"],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0, result.stderr + result.stdout

        centered = Pdb_AtomArray(input_io=str(out)).read()
        # 完整 32 原子: 24 helix + 8 loop, loop 不再被丢弃
        assert len(centered) == 32
        helix = (centered.chain_id == "A") | (centered.chain_id == "B")
        assert helix.sum() == 24
        assert (centered.chain_id == "C").sum() == 8
        # 束被居中到原点
        np.testing.assert_allclose(
            centered.coord[helix].mean(axis=0), [0, 0, 0], atol=0.1
        )