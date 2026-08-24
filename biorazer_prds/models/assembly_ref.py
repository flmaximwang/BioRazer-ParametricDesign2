"""参考 Assembly: AssemblyRef (基类) + AssemblyRealRef (刚性真实参考)。

核心语义 (用户在 48bc6e8 重构中确立):
- ``Assembly`` 每个节点都携带真实原子 ``structure``。
- "参考" 的区分不在 ``structure``, 而在 ``ref_structure``:
  - AssemblyParaRef (见 assembly_parametric.py): ref_structure 虚拟 (由参数生成)。
  - AssemblyRealRef (本文件): ref_structure 真实 (如 RCSB HEM)。
"""

from dataclasses import dataclass
from copy import deepcopy

import biotite.structure as bt_struct

from .assembly import Assembly


@dataclass
class AssemblyRef(Assembly):
    """参考 Assembly: 在真实 ``structure`` 之外携带 ``ref_structure`` (继承自基类)。

    ``ref_structure`` 是用于放置/拟合的参考几何, 其内容由子类决定:
    - AssemblyParaRef : ref_structure 虚拟 (由参数生成, 理想轨迹)
    - AssemblyRealRef : ref_structure 真实 (加载, 如 RCSB HEM)

    构建统一走基类的 ``Assembly.from_atomarray(structure, ref_structure=..., mask=...)``。
    """

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


@dataclass
class AssemblyRealRef(AssemblyRef):
    """刚性真实参考 Assembly (如 HEM 血红素辅基, 从 RCSB 加载)。

    无参数拟合 (没有 ``param`` / ``fit`` / ``modify``); 是刚性真实参考。
    ``xyz`` 需要一个显式的 frame 约定 (惯性主轴 / 显式方向 / 用户 frame),
    目前为 NotImplementedError 占位。
    """

    @property
    def xyz(self):
        raise NotImplementedError(
            "AssemblyRealRef.xyz: define a frame convention for the real "
            "reference (inertia axes / explicit direction / user frame)."
        )
