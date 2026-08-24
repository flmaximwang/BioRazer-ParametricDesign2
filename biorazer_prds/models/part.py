"""统一的递归 Part 节点。

将旧模型的 ``Assembly`` / ``AssemblyPart`` / ``component`` 三层概念合并为单个
可递归的 ``Part``: 一个 Part 既是"树中的一个节点", 又是"以该节点为根的子树"。

- 叶节点: 只携带 ``structure`` (真实原子结构) 与自身的 param/ref 语义。
- 内部节点: 通过 ``parts`` 持有子节点, 通过 ``mask`` 记录每个子节点在父
  ``structure`` 上的原子掩码。

核心不变式 (用户确立, 2026-08):
  内部节点的 ``structure`` 永远 = 子节点按 ``parts`` 插入顺序拼接;
  每个子节点的 ``mask`` 永远从同一次拼接派生 (连续布尔区间)。
  因此 ``push_down`` 与 ``merge_up`` 严格互逆, 来回切换不会让 mask 失效;
  排序只允许在导出边界 (``to_pdb`` / ``to_cif``) 做。
"""

from dataclasses import dataclass, field

import numpy as np
import biotite.structure as bt_struct
from scipy.spatial.transform import Rotation as R

from ..util.alignment import calculate_rotation
import biorazer.structure.io as br_struct_io


@dataclass
class Part:
    """一个可递归的 Part 节点。

    Properties
    ----------
    structure : bt_struct.AtomArray
        真实原子结构。叶节点权威; 内部节点由子节点 ``merge_up`` 派生。
    parts : dict[str, Part]
        子 Part 的有序字典 (替代旧 ``component`` 字典与旧 ``Assembly.parts`` 列表)。
    mask : dict[str, np.ndarray]
        每个子节点在 ``structure`` 上的布尔原子掩码。
        仅供 ``push_down`` (自顶向下) 使用; ``merge_up`` 后会重建为连续区间。
    """

    structure: bt_struct.AtomArray = None
    parts: dict[str, "Part"] = field(default_factory=dict)
    mask: dict[str, np.ndarray] = field(default_factory=dict)

    _centroid: np.ndarray = None
    _xyz: np.ndarray = None

    # ------------------------------------------------------------------
    # 树结构传播 (本次重构核心)
    # ------------------------------------------------------------------

    def push_down(self):
        """自顶向下: 用父 structure + 每个子 mask 切片出子 structure, 逐层递归到叶。

        父 structure 是权威; 每个子节点先被赋值为 ``structure[self.mask[name]]``,
        再对自身递归 ``push_down`` (若它还有子节点)。
        """
        for name, child in self.parts.items():
            mask = self.mask.get(name)
            if mask is None:
                raise ValueError(
                    f"Part '{name}' 缺少 mask, 无法向下 push_down (已知 masks: "
                    f"{list(self.mask)})"
                )
            child.structure = self.structure[mask]
            child.push_down()

    def merge_up(self):
        """自底向上: 先递归各子节点, 再把子 structure 按插入顺序拼接为父 structure,
        并重建每个子节点的 mask 为拼接数组上的连续布尔区间。

        不变式: 拼接后 mask 与 structure 同源, 不会失效。
        叶节点 (无子) 的 structure 即权威, 无需改动。
        """
        if not self.parts:
            return
        for child in self.parts.values():
            child.merge_up()
        self.structure = bt_struct.concatenate(
            [child.structure for child in self.parts.values()]
        )
        offset = 0
        for name, child in self.parts.items():
            n = len(child.structure)
            mask = np.zeros(len(self.structure), dtype=bool)
            mask[offset : offset + n] = True
            self.mask[name] = mask
            offset += n

    # ------------------------------------------------------------------
    # 子节点管理 (旧 Assembly 的 parts 列表操作, 改按 name)
    # ------------------------------------------------------------------

    def append_part(self, name, part):
        """追加一个子 Part。子节点顺序即 ``merge_up`` 的拼接顺序。"""
        self.parts[name] = part
        return self

    def check_part_name(self, name):
        """校验子 Part 名是否存在。"""
        if name not in self.parts:
            raise IndexError(
                f"Part '{name}' not found in parts: {list(self.parts)}"
            )

    def __getitem__(self, name):
        """按子 Part 名取出对应的子节点。"""
        self.check_part_name(name)
        return self.parts[name]

    # ------------------------------------------------------------------
    # 坐标访问 (旧 AssemblyPart)
    # ------------------------------------------------------------------

    @property
    def centroid(self):
        """结构质心 (默认用 CA 原子)。self._centroid 可缓存。"""
        if self._centroid is None:
            ca_atoms = self.structure[self.structure.atom_name == "CA"]
            self._centroid = bt_struct.centroid(ca_atoms)
        return self._centroid

    @property
    def xyz(self):
        """局部坐标轴 (x, y, z)。任意 Part 无定义, 由子类实现。"""
        raise TypeError("An arbitrary Part has no definition of local xyz!")

    @property
    def coord(self):
        return self.structure.coord

    @coord.setter
    def coord(self, new_coord):
        self.structure.coord = new_coord

    # ------------------------------------------------------------------
    # 刚体变换 (旧 AssemblyPart; 只改坐标, 不改原子数, 故 mask 仍有效)
    # ------------------------------------------------------------------

    def translate(self, x, y, z):
        self.structure = bt_struct.translate(self.structure, [x, y, z])
        self._centroid = None

    def rotate(self, rotation: R, centroid_to_origin=True, XYZ_to_xyz=True):
        """对当前结构施加旋转。

        - ``centroid_to_origin=True``: 先把质心移到原点, 旋转后移回 (绕质心旋转)。
        - ``XYZ_to_xyz=True``: 先把局部轴 ``self.xyz`` 对齐到规范轴, 旋转后还原
          (在结构自身坐标系里旋转)。启用时强制 ``centroid_to_origin=True``。
        """
        if XYZ_to_xyz:
            centroid_to_origin = True

        if centroid_to_origin:
            center_translation = self.calculate_center_translation()
            self.coord += center_translation
        if XYZ_to_xyz:
            center_rotation = self.calculate_center_rotation()
            self.coord = center_rotation.apply(self.structure.coord)

        self.coord = rotation.apply(self.structure.coord)

        if XYZ_to_xyz:
            inv_center_rotation = center_rotation.inv()
            self.coord = inv_center_rotation.apply(self.structure.coord)
        if centroid_to_origin:
            inv_center_translation = -center_translation
            self.coord += inv_center_translation

        if not centroid_to_origin:
            self._centroid = None
        self._xyz = None

    def rotate_euler(
        self, axis_spec, a, b, c, degrees=False, centroid_to_origin=True,
        XYZ_to_xyz=True,
    ):
        rotation = R.from_euler(axis_spec, [a, b, c], degrees=degrees)
        self.rotate(rotation, centroid_to_origin=centroid_to_origin,
                    XYZ_to_xyz=XYZ_to_xyz)

    def rotate_quat(self, x, y, z, w, centroid_to_origin=True, XYZ_to_xyz=True):
        rotation = R.from_quat([x, y, z, w])
        self.rotate(rotation, centroid_to_origin=centroid_to_origin,
                    XYZ_to_xyz=XYZ_to_xyz)

    def center(self, max_try=10, atol_rot=1e-5, atol_trans=1e-5, verbose=False):
        """迭代地把自身居中: 旋转使局部轴对齐规范轴, 平移到质心=原点。"""
        def _log(message: str):
            if verbose:
                print(f"[Part.center] {message}")

        if max_try <= 0:
            raise ValueError("max_try must be a positive integer")
        if atol_rot < 0 or atol_trans < 0:
            raise ValueError("atol_rot and atol_trans must be non-negative")

        counter = 0
        while True:
            counter += 1
            rotation = self.calculate_center_rotation()
            self.rotate(rotation, centroid_to_origin=True, XYZ_to_xyz=False)
            translation = self.calculate_center_translation()
            self.translate(*translation)

            euler_angles = self.calculate_center_rotation().as_euler(
                "xyz", degrees=False
            )
            translation = self.calculate_center_translation()
            _log(
                f"{counter}/{max_try}: euler(rad)={np.array2string(euler_angles, precision=4)}, "
                f"translation={np.array2string(translation, precision=4)}"
            )
            if np.allclose(euler_angles, [0, 0, 0], atol=atol_rot) and np.allclose(
                translation, [0, 0, 0], atol=atol_trans
            ):
                _log(f"Converged in {counter} iterations")
                break
            if counter >= max_try:
                raise TimeoutError(
                    f"Failed to center the part after {max_try} attempts. "
                    f"Thresholds: atol_rot={atol_rot}, atol_trans={atol_trans}."
                )

    def calculate_center_rotation(self):
        """把自身局部轴对齐到规范轴所需的旋转。"""
        x, y, z = self.xyz
        return calculate_rotation(x, y, z).inv()

    def calculate_center_translation(self):
        """把自身质心移到原点所需的平移。"""
        return -self.centroid

    def check_axes_aligned(self, atol=1e-3):
        """检查结构是否与规范 X/Z 轴对齐。"""
        x, y, z = self.xyz
        flags = [
            np.allclose(x, [1, 0, 0], atol=atol),
            np.allclose(y, [0, 1, 0], atol=atol),
            np.allclose(z, [0, 0, 1], atol=atol),
        ]
        if not np.all(flags):
            raise ValueError(
                "Structure must be aligned with X and Z axes before ZXZ rotation\n"
                f"Current x: {x}\nCurrent y: {y}\nCurrent z: {z}"
            )

    # ------------------------------------------------------------------
    # 子 Part 之间的变换 (旧 Assembly, 改按 name)
    # ------------------------------------------------------------------

    def center_part(self, name, max_try=10, atol_rot=1e-5, atol_trans=1e-5,
                    verbose=False):
        """居中指定子 Part, 并对所有子 Part 施加同一个刚体变换。

        这是旧 ``Assembly.center(part_index)`` 的语义: 以子节点 ``name`` 为锚,
        计算使其居中的旋转+平移, 然后作用到全部子节点。
        """
        self.check_part_name(name)

        def _log(message: str):
            if verbose:
                print(f"[Part.center_part({name})] {message}")

        if max_try <= 0:
            raise ValueError("max_try must be a positive integer")
        if atol_rot < 0 or atol_trans < 0:
            raise ValueError("atol_rot and atol_trans must be non-negative")

        counter = 0
        while True:
            counter += 1
            center_part = self.parts[name]
            center_translation = center_part.calculate_center_translation()
            center_rotation = center_part.calculate_center_rotation()

            for part in self.parts.values():
                part.translate(*center_translation)
                part.rotate(
                    center_rotation, centroid_to_origin=False, XYZ_to_xyz=False
                )

            euler_angles = self.parts[name].calculate_center_rotation().as_euler(
                "xyz", degrees=False
            )
            translation = self.parts[name].calculate_center_translation()
            _log(
                f"{counter}/{max_try}: euler(rad)={np.array2string(euler_angles, precision=4)}, "
                f"translation={np.array2string(translation, precision=4)}"
            )

            if np.allclose(euler_angles, [0, 0, 0], atol=atol_rot) and np.allclose(
                translation, [0, 0, 0], atol=atol_trans
            ):
                _log(f"Converged in {counter} iterations")
                break
            if counter >= max_try:
                raise TimeoutError(
                    f"Failed to center part[{name}] after {max_try} attempts."
                )

    def calculate_rotation_between(self, name_1, name_2, atol=1e-3):
        """计算把子 Part name_1 对齐到 name_2 的旋转 (要求 name_1 已对齐规范轴)。"""
        self.check_part_name(name_1)
        self.check_part_name(name_2)
        part_1 = self.parts[name_1]
        x, y, z = part_1.xyz
        flag = (
            np.allclose(x, [1, 0, 0], atol=atol)
            and np.allclose(y, [0, 1, 0], atol=atol)
            and np.allclose(z, [0, 0, 1], atol=atol)
        )
        if not flag:
            raise ValueError(
                f"Part[{name_1}] is not aligned with the reference axes within "
                f"atol={atol}."
            )
        part_2 = self.parts[name_2]
        x, y, z = part_2.xyz
        return calculate_rotation(x, y, z)

    def calculate_quat_between(self, name_1, name_2, atol=1e-3):
        rotation = self.calculate_rotation_between(name_1, name_2, atol=atol)
        return rotation.as_quat(scalar_first=False, canonical=True)

    def calculate_euler_between(self, name_1, name_2, axis_spec, degrees=False,
                                atol=1e-3):
        rotation = self.calculate_rotation_between(name_1, name_2, atol=atol)
        return rotation.as_euler(axis_spec, degrees=degrees)

    def calculate_translation_between(self, name_1, name_2):
        """计算把子 Part name_1 平移到 name_2 的平移向量。"""
        self.check_part_name(name_1)
        self.check_part_name(name_2)
        return self.parts[name_2].centroid - self.parts[name_1].centroid

    # ------------------------------------------------------------------
    # IO
    # ------------------------------------------------------------------

    @classmethod
    def from_structure(cls, *, structure: bt_struct.AtomArray):
        """从真实结构构建一个叶 Part。"""
        return cls(structure=structure)

    def to_pdb(self, pdb_file):
        """导出结构到 PDB 文件。"""
        br_struct_io.protein.STRUCT2PDB("", pdb_file).write(self.structure)

    def to_cif(self, cif_file):
        """导出结构到 CIF 文件。"""
        br_struct_io.protein.STRUCT2CIF("", cif_file).write(self.structure)
