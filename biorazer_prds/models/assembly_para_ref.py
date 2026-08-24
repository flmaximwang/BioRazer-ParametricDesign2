"""参数化参考 Assembly: AssemblyParaRef。

ref_structure 是 *虚拟* 的: 由拟合参数生成的理想轨迹, 作为放置/注册的参考几何。
携带参数拟合机制 (``param`` / ``initial_param`` / ``extra_param`` /
``params_not_to_fit`` / ``rmsd``) 与 ``fit`` / ``from_params``。
拟合得到的虚拟结构直接存入 ``ref_structure``。
"""

from abc import abstractmethod
from dataclasses import dataclass, field

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
    rmsd : float
        拟合模型的均方根偏差。
    """

    param: dict = field(default_factory=dict)
    initial_param: dict = field(default_factory=dict)
    extra_param: dict = field(default_factory=dict)
    params_not_to_fit: list[str] = field(default_factory=list)

    rmsd: float = None

    @classmethod
    def from_param(cls, *, params: dict, **kwargs):
        """从给定参数加载结构; 其余属性根据参数自动生成。"""
        raise NotImplementedError("from_params method is not implemented")

    @abstractmethod
    def fit(self, verbose: bool = False):
        """用给定坐标拟合参数; 把参数/rmsd 存入对象, 拟合的虚拟结构存入
        ``ref_structure``。

        这是"坐标 → 参数"的拟合 (参数化参考的 fit)。

        ``initial_param`` 提供初始猜测; ``params_not_to_fit`` 指定固定参数;
        ``verbose=True`` 打印拟合过程。
        """
