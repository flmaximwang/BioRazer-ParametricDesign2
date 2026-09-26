"""螺旋对称纤维 Assembly: AssemblyFiber (screw) 与 AssemblyCn (纯转动 Cn)。

``part`` 是沿纤维重复的刚体单元 (可含多条链, 如 6vy1 把 G + g 两条链当 1 个 part):
相邻 part 之间是 screw (螺旋对称) 关系 —— 绕纤维轴 (局部 ``z``) 旋转 ``omega``,
同时沿 ``z`` 平移 ``rise``。参数全部存在 ``params`` 里 (与 CrickHelix /
CCCPHelixBundle 一致); world-frame 的 4x4 变换是它们的派生形式, 见
:attr:`AssemblyFiber.transformation`。``rise = 0`` 时 screw 退化为纯转动, 即
n 重旋转对称 :class:`AssemblyCn` (零件成环, 含末→首的闭合步)。

局部帧约定 (与 CrickHelix 一致):
- ``z``   = 纤维轴 / Cn 轴, 符号取使 part 序号递增的一侧 (``rise > 0``, 或
  Cn 的 ``omega > 0``);
- ``x``   = 由轴指向 ``parts[0]`` 质心的方向 (垂直于 z);
- ``y``   = z × x;
- 原点    = 纤维轴上与 ``parts[0]`` 质心同高的点。

不变式: 所有 part 必须具有**完全相同的 atoms** (残基号 / 残基名 / 原子名按序一一
对应), 否则无法逐原子 Kabsch 对齐 —— ``fit`` 显式报错, 不给出看似合理的坏结果。

chain_id / ins_code 命名 (``generate_parts``): ``chain_id`` **保持原样**(如 ``G`` / ``g``),
part 序号写进 ``ins_code`` (1 字符: ``0``-``9`` → ``A``-``Z`` → ``a``-``z``, 见 ``PART_CODES``);
part 的 key = ``链名:ins_code`` 用 ``+`` 连接 (6vy1 的 G+g 得 ``G:0+g:0``)。``ins_code`` 是
PDB (列 27) 与 mmCIF (``_atom_site.pdbx_PDB_ins_code``) **都自动读写**的字段 —— 也正好是
"同一条链、同样残基号、不同拷贝" 的格式语义; ``seg_id`` 写 CIF 要 ``extra_fields=['seg_id']``
才带得上, 读 PDB 时 biotite 又会丢掉它, 所以不用它。
"""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import biotite.structure as bt_struct
from scipy.spatial.transform import Rotation as R

from .assembly import Assembly
from .assembly_para_ref import AssemblyParaRef


#: part 序号的 1 字符编码 —— ``ins_code`` 在 PDB 里只有 1 列 (mmCIF 同), 最多 62 个 part
PART_CODES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


@dataclass
class AssemblyFiber(AssemblyParaRef):
    """沿纤维轴 (z) 螺旋重复的 Assembly (内部节点, 每个 part 是一个子节点)。

    全部参数都在 ``params`` 里 (与 CrickHelix / CCCPHelixBundle 一致):
    ``omega`` (绕轴转角, rad) / ``rise`` (沿轴升高, A) / ``axis_direction``
    (纤维轴) / ``axis_point`` (轴上一点, 即局部原点), 再加 ``part_num`` /
    ``atom_num``。:attr:`transformation` 是它们的 world-frame 4x4 形式,
    由 ``params`` 派生 (不是独立字段), ``generate_parts`` 反复施加它。
    """

    #: 子类钩子 —— ``True`` 时 screw 退化为纯转动 (``rise ≡ 0``), 见 AssemblyCn
    _rotation_only = False

    # ------------------------------------------------------------------
    # 变换 (由 params 派生)
    # ------------------------------------------------------------------

    @property
    def transformation(self):
        """``part_i -> part_{i+1}`` 的 4x4 刚体变换 (world frame 齐次矩阵)。

        = 绕纤维轴 (过 ``params['axis_point']``, 方向 ``params['axis_direction']``)
        转 ``params['omega']``, 再沿轴平移 ``params['rise']``。
        ``params`` 里没有完整 screw 参数时返回 ``None``。
        """
        required = ("omega", "rise", "axis_point", "axis_direction")
        if any(self.params.get(key) is None for key in required):
            return None
        axis = np.asarray(self.params["axis_direction"], dtype=float)
        axis = axis / np.linalg.norm(axis)
        rotation = R.from_rotvec(axis * float(self.params["omega"]))
        point = np.asarray(self.params["axis_point"], dtype=float)
        transformation = np.eye(4)
        transformation[:3, :3] = rotation.as_matrix()
        transformation[:3, 3] = point - rotation.apply(point) \
            + float(self.params["rise"]) * axis
        return transformation

    # ------------------------------------------------------------------
    # 局部帧 (懒 fit, 与 CrickHelix / CCCPHelixBundle 一致)
    # ------------------------------------------------------------------

    @property
    def xyz(self):
        """局部正交帧 (x, y, z): z = 纤维轴, x = 轴 → ``parts[0]`` 质心。"""
        if self._xyz is None:
            self.fit()
        return self._xyz

    @property
    def centroid(self):
        """局部原点: 纤维轴上与 ``parts[0]`` 质心同高的点。"""
        if self._centroid is None:
            self.fit()
        return self._centroid

    # ------------------------------------------------------------------
    # 变换施加
    # ------------------------------------------------------------------

    @staticmethod
    def apply_transformation(structure, transformation, n: int = 1):
        """把 4x4 齐次变换施加 ``n`` 次到结构, 返回**新** AtomArray (不改原结构)。

        ``n`` 可正可负 (负 = 逆变换), ``n = 0`` 返回原样拷贝。
        """
        M = np.linalg.matrix_power(np.asarray(transformation, dtype=float), n)
        out = structure.copy()
        out.coord = np.asarray(out.coord, dtype=float) @ M[:3, :3].T + M[:3, 3]
        return out

    # ------------------------------------------------------------------
    # 拟合 (坐标 → 参数)
    # ------------------------------------------------------------------

    @staticmethod
    def check_parts_alignable(segments):
        """校验所有 part 具有完全相同的 atoms (一一对应), 否则无法 Kabsch 对齐。"""
        def _key(seg):
            return list(zip(seg.res_id.tolist(), seg.res_name.tolist(),
                            seg.atom_name.tolist()))

        ref = _key(segments[0])
        for i, seg in enumerate(segments[1:], start=1):
            key = _key(seg)
            if len(key) != len(ref):
                raise ValueError(
                    f"part {i} 与 part 0 的原子数不同 ({len(key)} vs {len(ref)}), "
                    "无法逐原子对齐"
                )
            if key != ref:
                bad = [j for j, (a, b) in enumerate(zip(key, ref)) if a != b]
                raise ValueError(
                    f"part {i} 与 part 0 的 atoms 不完全相同 ({len(bad)} 个原子"
                    f"不匹配, 首个: {key[bad[0]]} vs {ref[bad[0]]}), 无法逐原子对齐"
                )

    def _part_segments(self):
        """按 ``parts`` 顺序取每个 part 的原子。

        有与 ``structure`` 等长的 mask 时从父 structure 切片 —— ``rotate`` /
        ``center`` 这类刚体操作只改父 structure, 读子节点会看不到它们 (与
        CCCPHelixBundle.fit 同一教训); 否则退回子节点自身的 ``structure``。
        """
        segments = []
        for name, part in self.parts.items():
            mask = self.mask.get(name)
            if mask is not None and self.structure is not None \
                    and len(mask) == len(self.structure):
                segments.append(self.structure[mask])
            else:
                segments.append(part.structure)
        return segments

    def _step_pairs(self, segments):
        """相邻 part 对 ``(part_i, part_{i+1})`` (子类可改成环状, 见 AssemblyCn)。"""
        return list(zip(segments[:-1], segments[1:]))

    def fit(self, verbose: bool = False):
        """按 ``structure`` 与 ``parts[0], parts[1], ...`` 拟合 screw 与局部帧。

        步骤:
        1. 校验所有 part 具有完全相同的 atoms (可逐原子对齐);
        2. 逐对相邻 part 做 Kabsch 叠合 → 每步的刚体变换 ``(R_i, t_i)``;
        3. 各步旋转轴方向平均 = 纤维轴 ``z`` (符号取 ``rise > 0`` 一侧);
           逐步有符号转角 / 沿轴升高的平均 = ``omega`` / ``rise``;
        4. 各步螺旋轴的共同点 = 局部原点; ``x`` = 轴 → ``parts[0]`` 质心;
        5. screw 参数进 ``params`` (``transformation`` 由它们派生);
           ``ref_structure`` = 用 ``parts[0]`` + 该 screw 生成的理想 fiber,
           ``rmsd`` = ``ref_structure`` 与观测 parts 的逐原子 RMSD (逐 part 的
           ``rmsd`` 另存 ``params['part_rmsd']``)。

        注意: parts 必须是**沿纤维顺序**排的相邻重复单元 (每对相差一步);
        顺序错时各步的转角/升高彼此不一致, ``rmsd`` 会明显变大 —— 用
        ``params['step_omega'] / ['step_rise'] / ['axis_deviation']`` 诊断。
        """
        def _log(message):
            if verbose:
                print(f"[AssemblyFiber.fit] {message}")

        segments = self._part_segments()
        if len(segments) < 2:
            raise ValueError(
                f"fit 至少需要 2 个 part 才能定出 screw (parts 共 {len(segments)} 个)"
            )
        self.check_parts_alignable(segments)

        steps = []
        for i, (mobile, fixed) in enumerate(self._step_pairs(segments)):
            # superimpose(fixed, mobile): 把 mobile 叠到 fixed → 变换 = part_i → part_{i+1}
            fixed_coord = np.asarray(fixed.coord, dtype=float)
            mobile_coord = np.asarray(mobile.coord, dtype=float)
            fitted, affine = bt_struct.superimpose(fixed_coord, mobile_coord)
            # as_matrix() 带一个 (1, 4, 4) 的前导维 (stack 兼容); 单结构取 [0]
            T = np.asarray(affine.as_matrix(), dtype=float).reshape(-1, 4, 4)[0]
            rmsd = float(bt_struct.rmsd(fitted, fixed_coord))
            steps.append((R.from_matrix(T[:3, :3]), np.asarray(T[:3, 3], dtype=float),
                          rmsd))
            _log(f"step {i} -> {i + 1}: 叠合 rmsd {rmsd:.4f} A")

        # 1) 纤维轴: 各步旋转轴 (转角 ≈ 0 的步不提供方向) 平均。先把各轴对齐到第一个
        #    非退化步的轴 —— 180 度旋转的转轴符号在 scipy 里是任意的, 不对齐会把
        #    同一个 C2 的 +180 与 -180 平均成 0。
        axes = []
        for rot, _, _ in steps:
            rotvec = rot.as_rotvec()
            if np.linalg.norm(rotvec) > 1e-6:
                axes.append(rotvec / np.linalg.norm(rotvec))
        if not axes:
            raise ValueError(
                "各步转角均 ≈ 0: 纯平移纤维没有唯一旋转轴, 定不出 z "
                "(这种情况用普通 Assembly 表达即可)"
            )
        ref_axis = axes[0]
        axes = [a if np.dot(a, ref_axis) >= 0 else -a for a in axes]
        z = np.mean(axes, axis=0)
        z /= np.linalg.norm(z)
        axis_deviation = max(
            float(np.degrees(np.arccos(np.clip(abs(np.dot(a, z)), -1.0, 1.0))))
            for a in axes
        )

        # 2) 逐步有符号转角 / 升高, 以及各步螺旋轴上的一点
        eye = np.eye(3)
        step_omega, step_rise, axis_points = [], [], []
        for rot, t, _ in steps:
            rotvec = rot.as_rotvec()
            angle = float(np.linalg.norm(rotvec))
            # 只翻轴 (不翻角): 180 度步的符号歧义由 "与参考步同向" 消掉, 转角的
            # 符号仍由轴相对 z 的方向给出
            axis = rotvec / angle if angle > 1e-6 else z
            if np.dot(axis, ref_axis) < 0:
                axis = -axis
            theta = angle * float(np.dot(axis, z))
            rise = float(np.dot(t, z))
            step_omega.append(theta)
            step_rise.append(rise)
            # 螺旋轴过 p: S(x) = R (x - p) + p + rise * z  =>  (I - R) p = t - rise * z
            # (I - R) 沿轴奇异, pinv 给垂直于轴的最小范数解 —— 同一条轴上,
            # 各步应给出同一个点。
            axis_points.append(np.linalg.pinv(eye - rot.as_matrix()) @ (t - rise * z))
        # z 的符号: 纤维取 "part 序号递增 = 沿 +z 前进 (rise > 0)";
        # Cn 没有 rise, 取 "part 序号递增 = 绕 +z 正向转 (omega > 0)"
        forward = np.mean(step_omega) if self._rotation_only else np.median(step_rise)
        if forward < 0:
            z = -z
            step_omega = [-theta for theta in step_omega]
            step_rise = [-rise for rise in step_rise]
        measured_omega = float(np.mean(step_omega))
        # Cn: 转角由重数**钉死** (omega = 360/n), 数据里量到的值只作诊断
        omega = 2 * np.pi / len(segments) if self._rotation_only else measured_omega
        rise = 0.0 if self._rotation_only else float(np.mean(step_rise))
        axis_point = np.mean(axis_points, axis=0)

        # 3) 局部帧: 原点 = 轴上与 parts[0] 质心同高的点, x = 轴 → parts[0] 质心
        ca0 = segments[0][segments[0].atom_name == "CA"]
        if len(ca0) == 0:
            raise ValueError(
                "parts[0] 没有 CA 原子: x 定义为 '由轴指向 parts[0] 质心', 需要 CA 求质心"
            )
        centroid0 = bt_struct.centroid(ca0)
        origin = axis_point + float(np.dot(centroid0 - axis_point, z)) * z
        x = centroid0 - origin
        if np.linalg.norm(x) < 1e-6:
            raise ValueError("parts[0] 质心落在纤维轴上: x 无定义 (需要偏离轴的 part)")
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        self._centroid = origin
        self._xyz = np.vstack((x, y, z))
        self.params["x"], self.params["y"], self.params["z"] = x, y, z

        # 4) 参数进 params —— transformation 由这些参数派生 (见 transformation 属性)
        self.params = {
            "part_num": len(segments),
            "atom_num": len(segments[0]),
            "omega": omega,                 # rad, 绕 z (右手, 绕 +z)
            "rise": rise,                   # A, 沿 +z
            "axis_direction": z,
            "axis_point": origin,           # 局部原点 (轴上, 与 parts[0] 质心同高)
        }
        self.params["step_omega"] = np.array(step_omega)
        self.params["step_rise"] = np.array(step_rise)
        self.params["step_rmsd"] = np.array([step[2] for step in steps])
        self.params["axis_deviation"] = axis_deviation
        if self._rotation_only:
            # 诊断: 数据里量到的每步转角 (未钉死) → 与 360/n 的差 = Cn 假设的成立程度
            implied_n = 360.0 / abs(np.degrees(measured_omega)) if measured_omega else 0.0
            self.params["implied_n"] = implied_n
            self.params["measured_omega"] = measured_omega
            _log(f"纯转动: 数据每步 {np.degrees(measured_omega):.4f} deg "
                 f"(反推 n = {implied_n:.4f}, 零件 {len(segments)} 个) "
                 f"→ omega 钉死为 {np.degrees(omega):.4f} deg")

        # 5) ref_structure = 用 parts[0] + transformation 生成的理想 fiber;
        #    rmsd = 它与观测 parts 的逐原子 RMSD (逐 part 的 rmsd 也存下来)
        transformation = self.transformation
        residuals, model_parts = [], []
        for k, segment in enumerate(segments):
            model = self.apply_transformation(segments[0], transformation, k)
            model_parts.append(model)
            residuals.append(np.sum(
                (np.asarray(model.coord, dtype=float)
                 - np.asarray(segment.coord, dtype=float)) ** 2, axis=1
            ))
        part_rmsd = np.array([float(np.sqrt(residual.mean())) for residual in residuals])
        self.rmsd = float(np.sqrt(np.concatenate(residuals).mean()))
        self.ref_structure = bt_struct.concatenate(model_parts)
        self.params["part_rmsd"] = part_rmsd

        _log(
            f"omega {np.degrees(omega):.3f} deg / rise {rise:.3f} A "
            f"/ 每步 rmsd {np.array2string(np.array([s[2] for s in steps]), precision=3)} "
            f"/ 逐 part rmsd {np.array2string(part_rmsd, precision=3)} "
            f"/ 纤维 rmsd {self.rmsd:.4f} A / 轴偏离 {axis_deviation:.3f} deg"
        )
        return self

    # ------------------------------------------------------------------
    # 由参数生成 part
    # ------------------------------------------------------------------

    @classmethod
    def _label_part(cls, node, index: int):
        """把 part 序号写进节点 (含其所有子节点) 的 ``ins_code``; ``chain_id`` 不动。

        part 序号 → 1 字符: ``0``-``9`` / ``A``-``Z`` / ``a``-``z`` (``PART_CODES``)。
        ``ins_code`` 是 PDB 列 27 与 mmCIF ``pdbx_PDB_ins_code`` 都自动读写的字段 ——
        同一 chain + 同一 res_id 的不同拷贝只能靠它区分 (chain 名会被各 part 共用)。
        """
        if node.structure is not None and len(node.structure):
            if not isinstance(index, int) or not 0 <= index < len(PART_CODES):
                raise ValueError(
                    f"part 序号 {index} 超出 ins_code 的 1 字符范围 "
                    f"(0..{len(PART_CODES) - 1})"
                )
            node.structure.ins_code = np.array(
                [PART_CODES[index]] * len(node.structure)
            )
        for child in node.parts.values():
            cls._label_part(child, index)

    @staticmethod
    def _part_key(structure) -> str:
        """part 的 key = ``链名:ins_code`` 用 ``+`` 连接 (6vy1 的 G+g 得 ``G:0+g:0``)。

        还没贴 ins_code 时 (从文件读进来的观测结构, ins_code 常为空) 退化成只有链名。
        """
        chains = list(dict.fromkeys(str(chain) for chain in structure.chain_id))
        ins_code = str(structure.ins_code[0]) if len(structure) else ""
        if not ins_code:
            return "+".join(chains)
        return "+".join(f"{chain}:{ins_code}" for chain in chains)

    @classmethod
    def from_param(cls, part, omega, rise, part_num=1,
                   axis_point=(0.0, 0.0, 0.0), axis_direction=(0.0, 0.0, 1.0)):
        """由参数生成纤维 (不需要观测结构): ``part`` 施加 screw 共 ``part_num`` 次。

        Parameters
        ----------
        part : Assembly or AtomArray
            重复单元 (如 6vy1 的 G+g 两条链)。
        omega : float
            每步绕轴转角, rad。
        rise : float
            每步沿轴升高, A。
        part_num : int, default 1
            part 数; 第 k 个 part = ``part`` 绕轴 (过 ``axis_point``, 方向
            ``axis_direction``) 转 ``k * omega`` 并沿轴升 ``k * rise``。
        axis_point : sequence of 3 float, default (0, 0, 0)
            轴上一点 (即局部原点)。
        axis_direction : sequence of 3 float, default (0, 0, 1)
            纤维轴方向 (即局部 z)。

        Returns
        -------
        AssemblyFiber
            ``params`` 已填好, ``parts`` / ``structure`` 由 :meth:`generate_parts`
            生成 (part 序号写进 ``ins_code``, 见 :data:`PART_CODES`)。
        """
        obj = cls()
        if isinstance(part, bt_struct.AtomArray):
            part = Assembly(structure=part)
        obj.params = {
            "part_num": part_num,
            "atom_num": len(part.structure),
            "omega": omega,                                 # rad
            "rise": rise,                                   # A
            "axis_direction": np.asarray(axis_direction, dtype=float),
            "axis_point": np.asarray(axis_point, dtype=float),
        }
        obj.append_part(cls._part_key(part.structure), deepcopy(part))
        obj.generate_parts(part_num)
        return obj

    def generate_parts(self, n: int, verbose: bool = False):
        """按 ``parts`` 的第一个 part + ``params`` 的 screw 生成 n 个 part。

        第 k 个 part (k = 0 ... n-1) = 第一个 part 施加 ``transformation`` k 次
        (``transformation`` 由 ``params`` 派生: 绕轴转 ``omega`` + 沿轴升 ``rise``)。
        ``chain_id`` **保持原样** (G 还是 G), part 序号写进 ``ins_code`` (1 字符:
        ``0``-``9`` → ``A``-``Z`` → ``a``-``z``, 见 ``PART_CODES``); part 的 key =
        ``链名:ins_code`` 用 ``+`` 连接 (单链 part 即 ``A:0``; 6vy1 的 G+g 得 ``G:0+g:0``)。
        原位替换 ``self.parts``, 并把 ``self.structure`` / ``self.mask`` 重建为这些
        part 的拼接。

        为什么用 ``ins_code``: 它是 PDB (列 27) 与 mmCIF (``_atom_site.pdbx_PDB_ins_code``)
        **都自动读写**的字段, 也正好是格式里 "同一 chain、同一 res_id、不同拷贝" 的表达;
        写回文件再读进来, 按 ``ins_code`` 就能把各 part 分回来。``seg_id``/SEGID 列相反:
        写 mmCIF 要 ``extra_fields=['seg_id']`` 才带得上, 读 PDB 时 biotite 又直接丢掉。

        注意: ``ins_code`` 只有 1 列 → 最多 ``len(PART_CODES)`` = 62 个 part。
        """
        if self.transformation is None:
            raise ValueError(
                "generate_parts 需要 params 里的 screw 参数 "
                f"{['omega', 'rise', 'axis_direction', 'axis_point']}: "
                "请先 fit() 或直接给 self.params"
            )
        if isinstance(n, bool) or not isinstance(n, int):
            raise TypeError(f"n 必须为整数, 得到 {type(n).__name__}")
        if n < 1:
            raise ValueError(f"n 必须 >= 1, 得到 {n}")
        if n > len(PART_CODES):
            raise ValueError(
                f"ins_code 只有 1 列, 最多 {len(PART_CODES)} 个 part, 得到 {n} "
                "(再长就要靠 chain 名或 SEGID 列另立方案)"
            )
        if not self.parts:
            raise ValueError("generate_parts 需要 parts[0] 作为重复单元")

        template = self.parts[list(self.parts)[0]]
        new_parts = {}
        for k in range(n):
            part = deepcopy(template)
            part.structure = self.apply_transformation(
                part.structure, self.transformation, k
            )
            self._label_part(part, k)
            new_parts[self._part_key(part.structure)] = part
        self.parts = new_parts
        for part in self.parts.values():
            part._parent = self
        self._xyz = None
        self._centroid = None
        self._recompute_from_children()
        if verbose:
            print(
                f"[AssemblyFiber.generate_parts] {n} 个 part: "
                f"{list(self.parts)[0]} ... {list(self.parts)[-1]}, "
                f"{len(self.structure)} 个原子"
            )
        return self


@dataclass
class AssemblyCn(AssemblyFiber):
    """n 重旋转对称 (Cn) 的 Assembly: part 之间只差绕 z 的 360/n 旋转, 无平移。

    与 :class:`AssemblyFiber` (screw) 的差别 (其余全部继承):

    1. **纯转动** —— ``params['rise']`` 恒为 0, 转角**钉死**为 ``omega = 360 / n``
       (n = part 数; ``_rotation_only``)。数据里量到的每步转角存在
       ``params['measured_omega']``, 它与 360/n 的差 = "Cn 假设成立程度",
       由 ``rmsd`` 与 ``params['implied_n']`` (= 360 / measured_omega) 反映;
    2. **成环** —— fit 的相邻对含末→首的闭合步 ``(part_{n-1} -> part_0)``
       (``_step_pairs``), 所以 ``step_omega`` / ``step_rmsd`` 覆盖"转一圈是否闭合";
       screw 的 fit 只看 n-1 步。

    例: 6vy1 每个 part 内部的 G/g 两条链是一对 **垂直纤维轴** 的 2 重轴
    (数据每步 179.72 deg, 沿轴平移 0.03 A, 轴距纤维轴 0.34 A) —— ``parts = [G, g]``
    的 ``AssemblyCn`` 拟合出这根 C2 轴, omega 钉死 180 deg 后的 ``rmsd`` 就是
    "它到底有多 C2" 的答案。

    ``xyz`` 的 z = Cn 轴 (取 part 序号递增为正向旋转的一侧), x = 轴 →
    ``parts[0]`` 质心。
    """

    _rotation_only = True

    def _step_pairs(self, segments):
        """环状相邻对: ``(0,1) ... (n-2,n-1), (n-1,0)``。"""
        return list(zip(segments, segments[1:] + segments[:1]))

    @classmethod
    def from_param(cls, part, n=2, axis_point=(0.0, 0.0, 0.0),
                   axis_direction=(0.0, 0.0, 1.0)):
        """由参数生成 Cn: 单体 ``part`` 绕 n 重轴 (过 ``axis_point``, 方向
        ``axis_direction``) 转 ``360 / n`` 度, 共 n 个 part。

        Parameters
        ----------
        part : Assembly or AtomArray
            单体 (Cn 的不对称单元)。
        n : int, default 2
            重数 (2 = C2), 也是 part 数。
        axis_point : sequence of 3 float, default (0, 0, 0)
            轴上一点 (即局部原点)。
        axis_direction : sequence of 3 float, default (0, 0, 1)
            Cn 轴方向 (即局部 z)。
        """
        return super().from_param(
            part, omega=2 * np.pi / n, rise=0.0, part_num=n,
            axis_point=axis_point, axis_direction=axis_direction,
        )
