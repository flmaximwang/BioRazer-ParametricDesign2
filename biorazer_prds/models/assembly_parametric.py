"""参数化参考 Assembly: AssemblyParaRef。

ref_structure 是 *虚拟* 的: 由拟合参数生成的理想轨迹, 作为放置/注册的参考几何。
携带参数拟合机制 (``param`` / ``initial_param`` / ``extra_param`` /
``params_not_to_fit`` / ``fitted_structure`` / ``rmsd``) 与
``fit`` / ``modify`` / ``from_params``。
"""

from abc import abstractmethod
from dataclasses import dataclass, field

import biotite.structure as bt_struct

from .assembly import Assembly


@dataclass
class AssemblyParaRef(Assembly):
    """参数化参考 Assembly: ref_structure 虚拟, 由拟合参数生成。

    Properties
    ----------
    param : dict
        拟合得到的参数。
    initial_param : dict
        拟合的初始参数 (键不应超出 ``param``)。
    extra_param : dict
        不参与拟合的额外参数。
    params_not_to_fit : list[str]
        拟合期间保持固定的参数名。
    fitted_structure : bt_struct.AtomArray
        由拟合参数生成的虚拟参考结构。
    rmsd : float
        拟合模型的均方根偏差。
    """

    param: dict = field(default_factory=dict)
    initial_param: dict = field(default_factory=dict)
    extra_param: dict = field(default_factory=dict)
    params_not_to_fit: list[str] = field(default_factory=list)

    fitted_structure: bt_struct.AtomArray = None
    rmsd: float = None

    @classmethod
    def from_params(cls, *, params: dict, **kwargs):
        """从给定参数加载结构; 其余属性根据参数自动生成。"""
        raise NotImplementedError("from_params method is not implemented")

    @abstractmethod
    def fit(self, verbose: bool = False):
        """用给定坐标拟合参数, 并把参数/rmsd/拟合坐标存入对象。

        这是"坐标 → 参数"的拟合 (参数化参考的 fit)。

        ``initial_param`` 提供初始猜测; ``params_not_to_fit`` 指定固定参数;
        ``verbose=True`` 打印拟合过程。
        """

    @abstractmethod
    def modify(self, method, *args, **kwargs):
        """用给定方法与参数修改结构; 依赖对象内存储的参数。"""
