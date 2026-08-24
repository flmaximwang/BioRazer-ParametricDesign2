import copy
from typing import Iterable, Self
from dataclasses import dataclass, field

import numpy as np
import biotite.structure as bt_struct

import biorazer.structure.io as br_struct_io
from biorazer.database.amino_acid import (
    AMINO_ACIDS_1LETTER,
    AMINO_ACIDS_1TO3_UPPER,
    AMINO_ACIDS_3TO1_UPPER,
)

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

    def trim_or_extend(self, n: int, terminus: str, resn: str = "GLY"):
        """在螺旋一端伸长或缩短若干残基 (仅 backbone)。

        Parameters
        ----------
        n : int
            有符号残基数: n > 0 伸长 n 个残基, n < 0 缩短 |n| 个残基。
        terminus : str
            "N" 或 "C" (不支持两端)。
        resn : str
            新增残基名称 (仅影响伸长部分; 缩短无新增)。单字母或三字母,
            大小写不敏感; 统一转大写。

        仅支持叶节点; 修改后经 ``replace_with`` 重建祖先 structure/mask。
        """
        if self.parts:
            raise ValueError("trim_or_extend 仅支持叶节点")
        if terminus not in ("N", "C"):
            raise ValueError(f"terminus 仅支持 'N'/'C', 得到 {terminus!r}")
        if not isinstance(n, int):
            raise TypeError(f"n 必须为整数, 得到 {type(n).__name__}")
        if n == 0:
            return self

        if len(resn) == 1:
            if resn.upper() not in AMINO_ACIDS_1LETTER:
                raise ValueError(f"无效单字母残基: {resn!r}")
            resn = AMINO_ACIDS_1TO3_UPPER[resn.upper()]
        elif len(resn) == 3:
            if resn not in AMINO_ACIDS_3TO1_UPPER:
                raise ValueError(f"无效三字母残基: {resn!r}")
            resn = resn.upper()
        else:
            raise ValueError(f"resn 必须为单字母或三字母残基名, 得到 {resn!r}")

        new_part = copy.copy(self)
        if n > 0:
            new_part.add_atoms(self._extend_fragment(n, resn, terminus))
            # 补全 backbone, 使新残基与既有链的 junction 正确; pulchra 保留
            # res_id, 再按 res_id 稳定排序使 N 端新增残基回到链首。
            new_part.structure = pulchra_fix_backbone(new_part.structure)
            order = np.argsort(new_part.structure.res_id, kind="stable")
            new_part.structure = new_part.structure[order]
        else:
            new_part.remove_atoms(self._trim_mask(-n, terminus))
        new_part.param = {
            **self.param,
            "residue_num": len(np.unique(new_part.structure.res_id)),
        }
        new_part.ref_structure = None
        return self.replace_with(new_part)

    def _extend_fragment(self, n: int, resn: str, terminus: str):
        """生成 terminus 端 n 个新残基的 CA (理想 Crick 延伸), 供 add_atoms 追加。"""
        required = {
            "residue_num",
            "centroid",
            "direction",
            "radius",
            "omega",
            "pitch_angle",
            "phi0",
        }
        missing = required - set(self.param)
        if missing:
            raise ValueError(
                "trim_or_extend 伸长需要完整 Crick 参数, 缺少 "
                f"{sorted(missing)}; 请先调用 fit() 拟合参数"
            )
        residue_num = len(np.unique(self.structure.res_id))
        kwargs = {**self.param, "residue_num": residue_num + 2 * n}
        helix_ca, _ = generate_helix_ca_by_crick(**kwargs)
        new_ca = helix_ca[:n] if terminus == "N" else helix_ca[-n:]
        new_structure = ca_xyz_to_atom_array(
            new_ca, chain_id_i=self.structure.chain_id[0], res_name=resn
        )
        res_ids = np.unique(self.structure.res_id)
        offset = min(res_ids) - 1 - n if terminus == "N" else max(res_ids)
        new_structure.res_id += offset
        return new_structure

    def _trim_mask(self, n: int, terminus: str):
        """返回要删除的末端 n 个残基的布尔掩码, 供 remove_atoms 使用。"""
        res_ids = self.structure.res_id
        unique_res = np.unique(res_ids)
        if n >= len(unique_res):
            raise ValueError(f"无法缩短 {n} 个残基: 螺旋仅 {len(unique_res)} 个残基")
        remove = unique_res[:n] if terminus == "N" else unique_res[-n:]
        return np.isin(res_ids, remove)


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
        xyz, param, _ = generate_cc_ca_by_cccp(
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
        # 每条螺旋的 CA 原子 (helix-major 顺序), 供生成 ref_structure 时复制
        # 链/残基属性, 使输出保留原始 chain_id / res_id 标注。
        helix_ca_list = []
        for i, name in enumerate(helix_names):
            seg = self.structure[self.mask[name]]
            ca_mask = seg.atom_name == "CA"
            ca_coord_obs[i] = seg[ca_mask].coord
            helix_ca_list.append(seg[ca_mask])
        ref_ca = bt_struct.concatenate(helix_ca_list)

        _log("Running staged CCCP bundle optimization")
        param, rmsd, _ = fit_cc_by_cccp(
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
        # 用拟合参数重新生成 CA, 并从观测 CA (ref_ca) 复制链/残基属性,
        # 使 ref_structure 保留原始 chain_id / res_id 标注。
        _, _, atom_array = generate_cc_ca_by_cccp(**param, ref_structure=ref_ca)
        self.ref_structure = atom_array

        self._xyz = np.vstack(
            (self.extra_param["x"], self.extra_param["y"], self.extra_param["z"])
        )
        self._centroid = self.param["centroid"]
        _log(f"Completed fit, RMSD={self.rmsd:.4f}")

    def trim_or_extend(
        self, helix_index: int, n: int, terminus: str, resn: str = "GLY"
    ):
        """延长/缩短束中某一根螺旋的一端 (仅 backbone)。

        仿照 ``CrickHelix.trim_or_extend``, 但伸长几何由 **束参数** 生成:
        束中每根螺旋是超螺旋 (superhelical) 弯曲轨迹, 用 ``self.param`` 的 CCCP
        参数重新生成该螺旋理想轨迹, 取末端新增残基, 再接回原螺旋并
        ``pulchra_fix_backbone``。缩短则直接删除该螺旋末端残基。

        Parameters
        ----------
        helix_index : int
            要操作的螺旋在 ``self.parts`` 中的顺序下标 (束内螺旋按子节点顺序
            识别, 与 key 名无关)。
        n : int
            有符号残基数: n > 0 伸长 n 个残基, n < 0 缩短 |n| 个残基。
        terminus : str
            "N" 或 "C" (不支持两端)。
        resn : str
            新增残基名称 (仅影响伸长部分; 缩短无新增)。单字母或三字母,
            大小写不敏感; 统一转大写。

        Notes
        -----
        只改该螺旋的几何, 不保持"束内各螺旋等长"不变式: 伸长/缩短单根螺旋后,
        束参数 ``residue_num`` 不再代表所有螺旋, 对修改后的束再 ``fit()`` 会因
        长度不一致报错 (等长断言)。
        """
        if not self.parts:
            raise ValueError("trim_or_extend 需要子螺旋 (请用 from_atomarray 构建束)")
        if not isinstance(helix_index, int):
            raise TypeError(
                f"helix_index 必须为整数, 得到 {type(helix_index).__name__}"
            )
        if helix_index < 0 or helix_index >= len(self.parts):
            raise IndexError(
                f"helix_index {helix_index} 越界: 束共 {len(self.parts)} 根螺旋"
            )
        if terminus not in ("N", "C"):
            raise ValueError(f"terminus 仅支持 'N'/'C', 得到 {terminus!r}")
        if not isinstance(n, int):
            raise TypeError(f"n 必须为整数, 得到 {type(n).__name__}")
        if n == 0:
            return self

        if len(resn) == 1:
            if resn.upper() not in AMINO_ACIDS_1LETTER:
                raise ValueError(f"无效单字母残基: {resn!r}")
            resn = AMINO_ACIDS_1TO3_UPPER[resn.upper()]
        elif len(resn) == 3:
            if resn not in AMINO_ACIDS_3TO1_UPPER:
                raise ValueError(f"无效三字母残基: {resn!r}")
            resn = resn.upper()
        else:
            raise ValueError(f"resn 必须为单字母或三字母残基名, 得到 {resn!r}")

        key = list(self.parts)[helix_index]
        helix_part = self.parts[key]
        new_part = copy.copy(helix_part)
        if n > 0:
            new_part.add_atoms(self._extend_helix_fragment(helix_index, n, resn, terminus))
            # 补全 backbone, 使新残基与既有螺旋的 junction 正确; pulchra 保留
            # res_id, 再按 res_id 稳定排序使 N 端新增残基回到链首。
            new_part.structure = pulchra_fix_backbone(new_part.structure)
            order = np.argsort(new_part.structure.res_id, kind="stable")
            new_part.structure = new_part.structure[order]
        else:
            new_part.remove_atoms(self._trim_helix_mask(helix_index, -n, terminus))
        new_part.param = {
            **helix_part.param,
            "residue_num": len(np.unique(new_part.structure.res_id)),
        }
        new_part.ref_structure = None
        return helix_part.replace_with(new_part)

    def _extend_helix_fragment(self, helix_index: int, n: int, resn: str, terminus: str):
        """用束参数生成 helix_index 螺旋末端 n 个新残基的 CA, 供 add_atoms 追加。"""
        required = {
            "helix_num",
            "residue_num",
            "senses",
            "centroid",
            "y_prototype",
            "z",
            "r0",
            "w0",
            "phi0",
            "r1s",
            "w1s",
            "phi1s",
            "pitch_angles",
            "dphi0s",
            "z_offsets",
        }
        missing = {
            k for k in required if k not in self.param or self.param.get(k) is None
        }
        if missing:
            raise ValueError(
                "trim_or_extend 伸长需要完整 CCCP 束参数, 缺少 "
                f"{sorted(missing)}; 请先调用 fit() 拟合参数"
            )
        helix_part = self.parts[list(self.parts)[helix_index]]
        residue_num = len(np.unique(helix_part.structure.res_id))
        # residue_t 关于 0.5 中心对称: 以 N+2n 重新生成后, 原 N 个残基仍居中,
        # 取新轨迹最前/最后 n 个即为该螺旋末端的理想超螺旋延伸。
        kwargs = {**self.param, "residue_num": residue_num + 2 * n}
        coords, _, _ = generate_cc_ca_by_cccp(**kwargs)
        new_ca = coords[helix_index][:n] if terminus == "N" else coords[helix_index][-n:]
        new_structure = ca_xyz_to_atom_array(
            new_ca,
            chain_id_i=helix_part.structure.chain_id[0],
            res_name=resn,
        )
        res_ids = np.unique(helix_part.structure.res_id)
        offset = min(res_ids) - 1 - n if terminus == "N" else max(res_ids)
        new_structure.res_id += offset
        return new_structure

    def _trim_helix_mask(self, helix_index: int, n: int, terminus: str):
        """返回 helix_index 螺旋末端要删除的 n 个残基的布尔掩码, 供 remove_atoms 使用。"""
        helix_part = self.parts[list(self.parts)[helix_index]]
        res_ids = helix_part.structure.res_id
        unique_res = np.unique(res_ids)
        if n >= len(unique_res):
            raise ValueError(
                f"无法缩短 {n} 个残基: 螺旋仅 {len(unique_res)} 个残基"
            )
        remove = unique_res[:n] if terminus == "N" else unique_res[-n:]
        return np.isin(res_ids, remove)
