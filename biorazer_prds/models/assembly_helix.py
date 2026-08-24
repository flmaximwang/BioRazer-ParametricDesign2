from typing import Iterable, Self
from dataclasses import dataclass, field

import numpy as np
import biotite.structure as bt_struct

import biorazer.structure.io as br_struct_io

from .assembly_para_ref import AssemblyParaRef
from ..params.helix_cp.generate import generate_helix_ca_by_crick
from ..params.helix_cp.fit import fit_helix_by_crick
from ..params.cccp.generate import generate_cc_ca_by_cccp
from ..params.cccp.fit import fit_cc_by_cccp
from ..params.util import ca_xyz_to_atom_array, pulchra_fix_backbone


@dataclass
class CrickHelix(AssemblyParaRef):
    """单条 Crick 螺旋的 Assembly (叶节点, 继承 AssemblyParaRef)。

    Params
    ------
    direction: np.ndarray
        螺旋方向的归一化向量。
    centroid: np.ndarray
        螺旋质心。
    radius: float
        螺旋半径。
    pitch: float
        螺距。
    phase: float
        螺旋相位。
    """

    @property
    def xyz(self):
        if self._xyz is None:
            self.fit()
            self._xyz = np.vstack(
                (self.extra_param["x"], self.extra_param["y"], self.extra_param["z"])
            )
        return self._xyz

    @staticmethod
    def calculate_helix_type(omega):
        omega_str_list = [
            "10/3",
            "27/8",
            "17/5",
            "24/7",
            "7/2",
            "25/7",
            "18/5",
            "29/8",
            "11/3",
            "26/7",
            "15/4",
            "19/5",
            "23/6",
            "27/7",
        ]
        omega_list = list(map(lambda x: 2 * np.pi / eval(x), omega_str_list))
        if omega > omega_list[0] + (omega_list[0] - omega_list[1]) / 2:
            return None
        if omega < omega_list[-1] - (omega_list[-2] - omega_list[-1]) / 2:
            return None
        for i in range(len(omega_list) - 1):
            omega_upper = omega_list[i]
            omega_lower = omega_list[i + 1]
            if omega < omega_lower:
                continue
            elif omega < (omega_lower + omega_upper) / 2:
                return omega_str_list[i]
            elif omega < omega_upper:
                return omega_str_list[i + 1]
            else:
                continue

    @classmethod
    def from_param(
        cls,
        residue_num: int = 7,
        centroid: Iterable[float] = (0, 0, 0),
        direction: Iterable[float] = (0, 0, 1),
        radius: float = 2.26,
        omega: float = 4 * np.pi / 7,
        pitch_angle: float = 0.876,
        phi0: float = 0.0,
        backbone_type: str = "Gly",
    ):
        """Parameters
        ----------
        backbone_type : str
            - "CA": only CA atoms
            - "Gly": Glycine backbone atoms (N, CA, C, O)
            - "Ala": Alanine backbone atoms (N, CA, C, O, CB)
        """
        helix = cls()
        xyz, param = generate_helix_ca_by_crick(
            residue_num=residue_num,
            centroid=centroid,
            direction=direction,
            radius=radius,
            omega=omega,
            pitch_angle=pitch_angle,
            phi0=phi0,
        )
        if backbone_type == "CA":
            structure = ca_xyz_to_atom_array(xyz)
        elif backbone_type == "Gly":
            structure = ca_xyz_to_atom_array(xyz)
            structure = pulchra_fix_backbone(structure)
        else:
            raise ValueError(f"Unsupported backbone_type: {backbone_type}")
        helix.structure = structure
        helix.param = param
        return helix

    def fit(self, verbose: bool = False):
        """把观测 CA 坐标拟合成单条 Crick 螺旋模型。

        叶节点: structure 即整条螺旋, 直接用其 CA 原子拟合。
        """

        def _log(message: str):
            if verbose:
                print(f"[CrickHelix.fit] {message}")

        _log("Preparing CA coordinates from helix structure")
        atom_array = self.structure
        ca_mask = atom_array.atom_name == "CA"
        ca_atoms = atom_array[ca_mask]
        ca_coord = ca_atoms.coord

        _log(f"Running Crick fitting on {ca_coord.shape[0]} CA atoms")
        param, rmsd, fitted_coord = fit_helix_by_crick(ca_coord, verbose=verbose)
        self.param = param

        _log("Constructing local orthonormal frame")
        z = self.param["direction"]
        x_prototype = ca_atoms.coord[0] - self.param["centroid"]
        y = np.cross(z, x_prototype)
        y /= np.linalg.norm(y)
        x = np.cross(y, z)
        self.extra_param["x"] = x
        self.extra_param["y"] = y
        self.extra_param["z"] = z
        self.rmsd = rmsd

        fitted_structure = bt_struct.AtomArray(length=ca_coord.shape[0])
        fitted_structure.atom_name = np.array(["CA"] * ca_coord.shape[0])
        fitted_structure.element = np.array(["C"] * ca_coord.shape[0])
        fitted_structure.chain_id = ca_atoms.chain_id
        fitted_structure.res_id = ca_atoms.res_id
        fitted_structure.res_name = ca_atoms.res_name
        fitted_structure.coord = fitted_coord
        self.ref_structure = fitted_structure
        self.extra_param["helix_type"] = self.calculate_helix_type(self.param["omega"])
        _log(f"Completed fit, RMSD={self.rmsd:.4f}")

    def elongate_with_gly(self, length: int, terminus: str = "C"):
        """通过添加甘氨酸残基伸长螺旋。

        Parameters
        ----------
        length : int
            要添加的残基数 (正整数)。
        terminus : str
            "N" 为 N 端, "C" 为 C 端, "B" 为两端。
        """
        assert (
            isinstance(length, int) and length > 0
        ), "Length must be a positive integer."

        kwargs = self.param.copy()
        kwargs["residue_num"] += length * 2

        helix_ca, _ = generate_helix_ca_by_crick(**kwargs)
        N_ca = helix_ca[:length]
        C_ca = helix_ca[-length:]
        n_terminal_res_id = min(self.structure.res_id)
        c_terminal_res_id = max(self.structure.res_id)
        if not terminus in ["N", "C", "B"]:
            raise ValueError(f"Unsupported terminus: {terminus}")
        if terminus in ["N", "B"]:
            new_structure = ca_xyz_to_atom_array(
                N_ca, chain_id_i=self.structure.chain_id[0]
            )
            n_terminal_res_id = min(self.structure.res_id)
            new_structure.res_id += n_terminal_res_id - 1 - length
            self.structure = bt_struct.concatenate([new_structure, self.structure])
        if terminus in ["C", "B"]:
            new_structure = ca_xyz_to_atom_array(
                C_ca, chain_id_i=self.structure.chain_id[0]
            )
            c_terminal_res_id = max(self.structure.res_id)
            new_structure.res_id += c_terminal_res_id
            self.structure = bt_struct.concatenate([self.structure, new_structure])
        self.structure = pulchra_fix_backbone(self.structure)


@dataclass
class CCCPHelixBundle(AssemblyParaRef):
    """多螺旋 CCCP 束的 Assembly (继承 AssemblyParaRef)。

    structure 为束结构 (仅含螺旋, 无连接区); 每条螺旋是一个子节点, 由基类
    ``from_atomarray`` 按有序 mask 构建 (全覆盖不变式: 束内原子必须被所有
    螺旋掩码完全覆盖, 因此束内不允许出现非螺旋的连接区)。螺旋按子节点顺序
    识别, 与 key 名无关。

    mask: dict[str, np.ndarray]
        每条螺旋在束 structure 上的布尔掩码 (有序, 按顺序对应各螺旋)。
    parts: dict[str, CCCPHelixBundle]
        每条螺旋是一个叶子节点 (按顺序识别, 与 key 名无关)。
    """

    param: dict = field(
        default_factory=lambda: {
            "helix_num": None,
            "residue_num": None,
            "senses": None,
            "centroid": None,
            "y_prototype": None,
            "z": None,
            "r0": None,
            "w0": None,
            "phi0": None,
            "r1s": None,
            "w1s": None,
            "phi1s": None,
            "pitch_angles": None,
            "dphi0s": None,
            "z_offsets": None,
        }
    )

    @property
    def centroid(self):
        if self._centroid is None:
            self.fit()
        return self._centroid

    @property
    def xyz(self):
        if self._xyz is None:
            self.fit()
        return self._xyz

    @property
    def helix_num(self):
        """螺旋条数 = 有序子节点数。

        sub-assembly 互不交集且恰好构成束, 因此按子节点顺序计数, 与 mask
        key 名无关。叶节点 (如 ``from_param`` 生成的束) 无子节点时回退到
        ``param[\"helix_num\"]``。
        """
        if self.parts:
            return len(self.parts)
        return self.param.get("helix_num")

    @classmethod
    def from_mask(cls, structure: bt_struct.AtomArray, mask: dict[str, np.ndarray]):
        return cls.from_atomarray(structure=structure, mask=mask)

    @classmethod
    def from_param(
        cls,
        helix_num: int = 2,
        residue_num: int = 7,
        senses: Iterable[int] = None,
        centroid: Iterable[float] = [0.0, 0.0, 0.0],
        y_prototype: Iterable[float] = [0.0, 1.0, 0.0],
        z: Iterable[float] = [0.0, 0.0, 1.0],
        r0: float = 5.0,
        w0: float = -2 * np.pi / 100,
        phi0: float = 0.0,
        r1s: Iterable[float] | float = 2.26,
        w1s: Iterable[float] | float = 4 * np.pi / 7,
        phi1s: Iterable[float] | float = -np.pi / 20,
        pitch_angles: Iterable[float] | float = -0.2096,
        dphi0s: Iterable[float] = None,
        z_offsets: Iterable[float] | float = 0.0,
        backbone_type: str = "Gly",
    ):
        """按参数生成束结构 (CA/Gly 主链); 不构建子节点 (mask 未提供)。"""
        res_obj = cls()
        res_obj.param["helix_num"] = helix_num
        xyz, param = generate_cc_ca_by_cccp(
            helix_num=helix_num,
            residue_num=residue_num,
            senses=senses,
            centroid=centroid,
            y_prototype=y_prototype,
            z=z,
            r0=r0,
            w0=w0,
            phi0=phi0,
            r1s=r1s,
            w1s=w1s,
            phi1s=phi1s,
            pitch_angles=pitch_angles,
            dphi0s=dphi0s,
            z_offsets=z_offsets,
        )
        if backbone_type == "CA":
            structure = ca_xyz_to_atom_array(xyz)
        elif backbone_type == "Gly":
            structure = ca_xyz_to_atom_array(xyz)
            structure = pulchra_fix_backbone(structure)
        else:
            raise ValueError(f"Unsupported backbone_type: {backbone_type}")
        res_obj.structure = structure
        return res_obj

    def fit(self, verbose: bool = False):
        """把多螺旋束拟合成 CCCP 参数化模型。

        每条螺旋由 ``self.mask`` 在 ``self.structure`` 上按有序掩码切片识别
        (与 key 名无关)。直接读 ``self.structure`` + mask 而非子节点 structure:
        这样 ``center()`` 等对父 structure 施加的刚体变换能被 fit 感知
        (rotate/translate 不更新子节点, 若读子节点会导致 center 死循环)。
        """

        def _log(message: str):
            if verbose:
                print(f"[CCCPHelixBundle.fit] {message}")

        _log(f"Validating CA lengths for {self.helix_num} helices")
        helix_names = list(self.parts)
        helix_lens = [
            np.sum(self.structure[self.mask[name]].atom_name == "CA")
            for name in helix_names
        ]

        assert (
            len(set(helix_lens)) == 1
        ), f"All helices must have the same length to fit a CCCP model. Current lengths: {helix_lens}"
        self.initial_param["helix_num"] = len(helix_lens)
        self.initial_param["residue_num"] = helix_lens[0]

        _log(
            f"Collecting observed CA coordinates (helix_num={self.initial_param['helix_num']}, "
            f"residue_num={self.initial_param['residue_num']})"
        )
        ca_coord_obs = np.zeros(
            shape=(self.initial_param["helix_num"], helix_lens[0], 3)
        )
        for i, name in enumerate(helix_names):
            seg = self.structure[self.mask[name]]
            ca_mask = seg.atom_name == "CA"
            ca_coord_obs[i] = seg[ca_mask].coord

        _log("Running staged CCCP bundle optimization")
        param, rmsd, ca_coord_fitted = fit_cc_by_cccp(
            ca_coord_obs,
            params_not_to_fit=self.params_not_to_fit,
            verbose=verbose,
            **self.initial_param,
        )

        self.param = param
        z = param["z"]
        y_prototype = param["y_prototype"]

        _log("Building fitted bundle local frame")
        x = np.cross(y_prototype, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        self.extra_param["x"] = x
        self.extra_param["y"] = y
        self.extra_param["z"] = z

        self.rmsd = rmsd
        ca_coord_fitted = np.reshape(
            ca_coord_fitted,
            shape=(ca_coord_fitted.shape[0] * ca_coord_fitted.shape[1], 3),
        )
        array_length = ca_coord_fitted.shape[0]

        fitted_structure = bt_struct.AtomArray(length=array_length)
        fitted_structure.res_name = np.array(["GLY"] * array_length)
        fitted_structure.element = np.array(["C"] * array_length)
        fitted_structure.atom_name = np.array(["CA"] * array_length)
        fitted_structure.chain_id = np.array(
            [chr(ord("A") + i // helix_lens[0]) for i in range(array_length)]
        )
        fitted_structure.res_id = np.array(
            list(range(1, helix_lens[0] + 1)) * self.helix_num
        )
        fitted_structure.coord = ca_coord_fitted
        self.ref_structure = fitted_structure

        self._xyz = np.vstack(
            (self.extra_param["x"], self.extra_param["y"], self.extra_param["z"])
        )
        self._centroid = self.param["centroid"]
        _log(f"Completed fit, RMSD={self.rmsd:.4f}")
