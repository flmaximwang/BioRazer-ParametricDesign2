"""刚性真实参考 Assembly: AssemblyRealRef。

ref_structure 为真实结构 (如 RCSB HEM)。无参数拟合 (没有 ``param`` /
``fit`` / ``modify``); 是刚性真实参考。
"""

from dataclasses import dataclass

from .assembly import Assembly


@dataclass
class AssemblyRealRef(Assembly):
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

    def fit(self):
        """把 ``ref_structure`` 对齐到真实结构 (``structure``), 从而可计算
        真实结构的 centroid / xyz 等参数。

        真实 ``structure`` 固定为权威; ``ref_structure`` 是可动的, 被配准到
        真实结构上。对齐后, 用配准后的 ref 的 frame 计算真实结构的
        centroid / xyz 等参数。目前为占位, 待定义 frame 约定后实现。
        """
        raise NotImplementedError(
            "AssemblyRealRef.fit: define how to align ref_structure onto the "
            "real structure and derive its frame (centroid/xyz)."
        )
