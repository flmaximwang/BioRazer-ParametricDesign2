"""参数化参考 Assembly: AssemblyParaRef。

ref_structure 是 *虚拟* 的: 由拟合参数生成的理想轨迹, 作为放置/注册的参考几何。
携带参数拟合机制 (``params`` / ``fixed_params`` / ``rmsd``) 与 ``fit`` / ``from_param``。
拟合得到的虚拟结构直接存入 ``ref_structure``。

``params`` 是唯一的参数字典: 给定的键值即拟合初值, 拟合结果原地写回; 派生量
(局部坐标帧 ``x``/``y``/``z``、``helix_type``、``fit_stats``) 也存在这里。
``fixed_params`` 列出 ``params`` 中拟合期间保持不动的键名。

注意: ``params`` 里混有派生量, 而 generate_*/fit_* 都是显式签名 (没有
``**kwargs``), 所以 splat 前必须按目标函数的签名过滤 (见
``assembly_helix._fit_kwargs``)。
"""

from abc import abstractmethod
from dataclasses import dataclass, field

from .assembly import Assembly


@dataclass
class AssemblyParaRef(Assembly):
    """参数化参考 Assembly: ref_structure 虚拟, 由拟合参数生成。

    Properties
    ----------
    params : dict
        唯一的参数字典: 拟合初值 + 拟合结果 + 派生量 (x/y/z、helix_type、fit_stats)。
    fixed_params : list[str]
        拟合期间保持固定的参数名 (须是 ``params`` 的键)。
    rmsd : float
        拟合模型的均方根偏差。
    """

    params: dict = field(default_factory=dict)
    fixed_params: list[str] = field(default_factory=list)

    rmsd: float = None

    @classmethod
    def from_param(cls, *, params: dict, **kwargs):
        """从给定参数加载结构; 其余属性根据参数自动生成。"""
        raise NotImplementedError("from_param method is not implemented")

    @abstractmethod
    def fit(self, verbose: bool = False):
        """用给定坐标拟合参数; 把参数/rmsd 存入对象, 拟合的虚拟结构存入
        ``ref_structure``。

        这是"坐标 → 参数"的拟合 (参数化参考的 fit)。

        ``params`` 中已有的值提供初始猜测 (原地写回拟合结果);
        ``fixed_params`` 指定固定参数; ``verbose=True`` 打印拟合过程。
        """
