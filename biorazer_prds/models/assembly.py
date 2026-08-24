"""统一的递归 Assembly 节点。

将旧模型的 ``Assembly`` (顶层容器) / ``AssemblyPart`` (部件) / ``component``
(组件) 三层概念合并为单个可递归的 ``Assembly``: 一个 Assembly 既是"树中的一个
节点", 又是"以该节点为根的子树"。**每个节点都叫 Assembly**, part / component
的概念被消除。

- 叶节点: 只携带 ``structure`` (真实原子结构) 与自身的 param/ref 语义。
- 内部节点: 通过 ``parts`` 持有子节点 (每个子节点也是 Assembly),
  通过 ``mask`` 记录每个子节点在父 ``structure`` 上的原子掩码。

核心不变式 (用户确立, 2026-08):
  内部节点的 ``structure`` 永远 = 子节点按 ``parts`` 插入顺序拼接;
  每个子节点的 ``mask`` 永远从同一次拼接派生 (连续布尔区间)。
  因此 ``push_down`` 与 ``merge_up`` 严格互逆, 来回切换不会让 mask 失效;
  排序只允许在导出边界 (``to_pdb`` / ``to_cif``) 做。
"""

from dataclasses import dataclass, field, fields, MISSING
from copy import deepcopy
from typing import Self

import numpy as np
import biotite.structure as bt_struct
from scipy.spatial.transform import Rotation as R

from ..util.alignment import calculate_rotation
import biorazer.structure.io as br_struct_io


@dataclass
class Assembly:
    """一个可递归的 Assembly 节点。

    Properties
    ----------
    structure : bt_struct.AtomArray
        真实原子结构。叶节点权威; 内部节点由子节点 ``merge_up`` 派生。
    parts : dict[str, Assembly]
        子 Assembly 的有序字典 (替代旧 ``component`` 字典与旧顶层 ``parts`` 列表)。
    mask : dict[str, np.ndarray]
        每个子节点在 ``structure`` 上的布尔原子掩码。
        仅供 ``push_down`` (自顶向下) 使用; ``merge_up`` 后会重建为连续区间。
    """

    structure: bt_struct.AtomArray = None
    ref_structure: bt_struct.AtomArray = None
    parts: dict[str, "Assembly"] = field(default_factory=dict)
    mask: dict[str, np.ndarray] = field(default_factory=dict)

    _centroid: np.ndarray = None
    _xyz: np.ndarray = None
    _parent: "Assembly" = field(default=None, init=False, repr=False, compare=False)

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
                    f"Assembly '{name}' 缺少 mask, 无法向下 push_down (已知 masks: "
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
        self._recompute_from_children()

    def _recompute_from_children(self):
        """用子节点 structure 重建本节点 structure 与 mask (同 merge_up 的合成部分)。

        只合成本节点一层, 不递归; 用于局部替换后沿祖先向上重建。
        """
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

    def split(self, mask: dict):
        """把一个尚无子节点的节点按 mask 拆分成子树 (叶 → 内部节点)。

        与 ``from_atomarray`` 的 mask 语义一致: mask 的每个值要么是等长于
        本节点 ``structure`` 的布尔数组, 要么是嵌套 dict (子树); 构建出的
        每个子节点所存 mask 会被投影到其自身 structure 上 (长度 ≠ 输入)。

        Parameters
        ----------
        mask : dict
            拆分用的掩码字典 (等长于本节点 structure)。
        """
        if self.parts:
            raise ValueError(
                f"该节点已有子节点 ({list(self.parts)}), 不能重复拆分。"
            )
        self._build_subtree(mask, np.arange(len(self.structure)))
        return self

    # ------------------------------------------------------------------
    # 子节点管理
    # ------------------------------------------------------------------

    def append_part(self, name, part):
        """追加一个子 Assembly。子节点顺序即 ``merge_up`` 的拼接顺序。"""
        self.parts[name] = part
        part._parent = self
        return self

    def check_part_name(self, name):
        """校验子 Assembly 名是否存在。"""
        if name not in self.parts:
            raise IndexError(
                f"Assembly '{name}' not found in parts: {list(self.parts)}"
            )

    def __getitem__(self, name):
        """按子名取出对应的子 Assembly (树导航)。"""
        self.check_part_name(name)
        return self.parts[name]

    def set_type(self, new_type):
        """修改自身类型为 ``new_type``, 保留 ``structure`` / ``ref_structure`` /
        ``parts`` / ``mask``。

        mask 属于树结构, 不由类型指定; 类型只改变节点的拟合/参考行为。
        就地修改 (``self.__class__``), 树中对该节点的引用无需改动。不递归
        (子节点类型需单独设置)。

        典型用法: 先 ``from_atomarray`` / ``split`` 定义结构, 再对节点
        ``set_type(CCCPHelixBundle)`` 指定其拟合类型。
        """
        if self.__class__ is new_type:
            return self
        keep = {f.name: getattr(self, f.name) for f in fields(self)}
        self.__class__ = new_type
        for f in fields(new_type):
            if f.name in keep:
                setattr(self, f.name, keep[f.name])
            elif f.default is not MISSING:
                setattr(self, f.name, f.default)
            elif f.default_factory is not MISSING:
                setattr(self, f.name, f.default_factory())
            else:
                setattr(self, f.name, None)
        return self

    def replace_with(self, new_part):
        """用 ``new_part`` 替换本节点所在的这一块, 并重建本节点及以上所有祖先节点
        的 ``structure`` 与 ``mask``。

        原子数可不同 (这是一次真正的结构替换)。本节点作为其父节点的子节点被
        ``new_part`` 替换; 随后从父节点沿 ``_parent`` 向上逐层重建
        (structure = 子节点拼接, mask = 连续区间)。返回新挂上的节点。
        """
        parent = self._parent
        if parent is None:
            # 根节点: 采用 new_part 的内容作为新的根。
            self.structure = new_part.structure
            self.ref_structure = new_part.ref_structure
            self.parts = new_part.parts
            self.mask = new_part.mask
            for child in self.parts.values():
                child._parent = self
            return self
        name = None
        for k, v in parent.parts.items():
            if v is self:
                name = k
                break
        if name is None:
            raise ValueError("本节点未挂在其父节点的 parts 中")
        parent.parts[name] = new_part
        new_part._parent = parent
        node = parent
        while node is not None:
            node._recompute_from_children()
            node = node._parent
        return new_part

    # ------------------------------------------------------------------
    # 坐标访问
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
        """局部坐标轴 (x, y, z)。任意 Assembly 无定义, 由子类实现。"""
        raise TypeError("An arbitrary Assembly has no definition of local xyz!")

    @property
    def coord(self):
        return self.structure.coord

    @coord.setter
    def coord(self, new_coord):
        self.structure.coord = new_coord

    def atoms(self, name):
        """按掩码名取回父 structure 中对应的原子子集 (旧 ``__getitem__`` 语义)。

        名须存在于 ``mask`` (不要求是子节点; 叶节点也可用, 如 CrickHelix 的
        ``\"helix\"``)。
        """
        if name not in self.mask:
            raise KeyError(f"Mask '{name}' not found in {list(self.mask)}")
        return self.structure[self.mask[name]]

    # ------------------------------------------------------------------
    # 刚体变换 (只改坐标, 不改原子数, 故 mask 仍有效)
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
                print(f"[Assembly.center] {message}")

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
                    f"Failed to center the assembly after {max_try} attempts. "
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
    # 任意两个 Assembly 之间的变换 (static; 两参数都需定义 centroid 与 xyz)
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_rotation_between(part_1, part_2, atol=1e-3):
        """计算把 part_1 的局部坐标轴对齐到 part_2 的旋转。

        要求 part_1 已对齐规范轴 (与旧实现语义一致)。part_1 / part_2 需实现
        ``centroid`` 与 ``xyz`` (如螺旋/束这类具体 Assembly)。
        """
        x, y, z = part_1.xyz
        flag = (
            np.allclose(x, [1, 0, 0], atol=atol)
            and np.allclose(y, [0, 1, 0], atol=atol)
            and np.allclose(z, [0, 0, 1], atol=atol)
        )
        if not flag:
            raise ValueError(
                f"part_1 is not aligned with the reference axes within atol={atol}."
            )
        x, y, z = part_2.xyz
        return calculate_rotation(x, y, z)

    @staticmethod
    def calculate_quat_between(part_1, part_2, atol=1e-3):
        """计算把 part_1 对齐到 part_2 的旋转的四元数 (x, y, z, w)。"""
        rotation = Assembly.calculate_rotation_between(part_1, part_2, atol=atol)
        return rotation.as_quat(scalar_first=False, canonical=True)

    @staticmethod
    def calculate_euler_between(part_1, part_2, axis_spec, degrees=False,
                                atol=1e-3):
        """计算把 part_1 对齐到 part_2 的旋转的欧拉角。"""
        rotation = Assembly.calculate_rotation_between(part_1, part_2, atol=atol)
        return rotation.as_euler(axis_spec, degrees=degrees)

    @staticmethod
    def calculate_translation_between(part_1, part_2):
        """计算把 part_1 平移到 part_2 的平移向量
        (part_2.centroid - part_1.centroid)。"""
        return part_2.centroid - part_1.centroid

    @staticmethod
    def calculate_transformation_between(part_1, part_2):
        """计算把 part_1 对齐到 part_2 的变换 (translation, rotation)。

        先平移使质心对齐, 再旋转使 part_1 的局部轴对齐 part_2 (要求 part_1
        已对齐规范轴)。两参数都需定义 ``centroid`` 与 ``xyz``。
        """
        translation = part_2.centroid - part_1.centroid

        part_1_center_rotation = part_1.calculate_center_rotation()
        part_2_copy = deepcopy(part_2)
        part_2_copy.rotate(
            part_1_center_rotation, centroid_to_origin=False, XYZ_to_xyz=False
        )
        x, y, z = part_2_copy.xyz
        rotation = calculate_rotation(x, y, z)

        return translation, rotation

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    def to_pymol_axes(self, prefix="default", length=5.0):
        """把该 Assembly 的 x/y/z 局部轴导出为 PyMOL 可视化命令。"""
        print("This method requires Biorazer-PyMOL to visualize the axes in PyMOL.")
        x, y, z = self.xyz
        centroid = self.centroid
        print(
            f"arrow_pass {centroid[0]},{centroid[1]},{centroid[2]},{x[0]},{x[1]},{x[2]}, r_color=1, g_color=0, b_color=0, name={prefix}_x, length={length}"
        )
        print(
            f"arrow_pass {centroid[0]},{centroid[1]},{centroid[2]},{y[0]},{y[1]},{y[2]}, r_color=0, g_color=1, b_color=0, name={prefix}_y, length={length}"
        )
        print(
            f"arrow_pass {centroid[0]},{centroid[1]},{centroid[2]},{z[0]},{z[1]},{z[2]}, r_color=0, g_color=0, b_color=1, name={prefix}_z, length={length}"
        )

    def copy(self):
        """返回对象的深拷贝。"""
        return deepcopy(self)

    # ------------------------------------------------------------------
    # IO
    # ------------------------------------------------------------------

    @classmethod
    def from_atomarray(cls, structure: bt_struct.AtomArray,
                       ref_structure: bt_struct.AtomArray = None,
                       mask: "str | dict" = "all") -> Self:
        """从 AtomArray 构建 Assembly。

        Parameters
        ----------
        structure : bt_struct.AtomArray
            真实原子结构 (必填)。
        ref_structure : bt_struct.AtomArray, optional
            参考结构 (参考 Assembly 携带)。
        mask : "all" | dict
            - ``"all"`` (默认): 该节点是叶节点, 直接包裹 ``structure``, 不构建子节点。
            - ``dict``: 按 mask 的结构构建树。mask 的每个值要么是布尔数组
              (等长于顶层 ``structure``, 指向顶层原子), 要么是嵌套 dict (子树)。
              子节点的 mask 会被投影到该子节点自身的 structure 上, 因此
              **长度不一定等于输入的 mask** (见核心不变式)。

        Notes
        -----
        输入 ``mask`` 中所有布尔数组都等长于顶层 ``structure``; dict 的嵌套
        即树的拓扑。构建出的每个节点所存 ``mask`` 是其自身 structure 上的
        掩码 (长度 = 该节点原子数)。
        """
        obj = cls(structure=structure, ref_structure=ref_structure)
        if mask != "all":
            indices = np.arange(len(structure))
            obj._build_subtree(mask, indices)
        return obj

    def _build_subtree(self, mask: dict, indices: np.ndarray):
        """按 mask 字典递归构建子树。

        Parameters
        ----------
        mask : dict
            name -> 顶层布尔数组 或 嵌套 dict (子树)。
        indices : np.ndarray
            本节点原子对应的顶层结构索引 (用于把顶层掩码投影到本节点)。
        """
        for name, m in mask.items():
            if isinstance(m, dict):
                # 子树: 先取该子树全部叶子掩码在本节点上的并集, 作为本节点对
                # 该子树的掩码; 再递归构建子树 (索引投影到子树原子的顶层位置)。
                union = np.zeros(len(indices), dtype=bool)
                for sub in m.values():
                    union |= sub[indices]
                self.mask[name] = union
                child = type(self)(
                    structure=self.structure[union], ref_structure=self.ref_structure
                )
                self.parts[name] = child
                child._parent = self
                child._build_subtree(m, indices[union])
            else:
                # 叶: m 是等长于顶层 structure 的掩码, 投影到本节点即为其子掩码。
                local = m[indices]
                self.mask[name] = local
                self.parts[name] = type(self)(
                    structure=self.structure[local], ref_structure=self.ref_structure
                )
                self.parts[name]._parent = self
