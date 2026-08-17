#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 helix bundle 结构居中并输出, 同时生成 PyMOL 绘图脚本 (.pml)。

输入一个 PDB/CIF, 解析为 CCCPHelixBundle: 每个 helix 的范围由用户指定,
格式 ``chain:start-end`` (如 ``A:1-30``; 无链号的结构可直接写 ``start-end``)。
然后调用 ``bundle.center()`` 把 bundle 平移至原点并让轴向 (CCCP 拟合的
x/y/z) 与笛卡尔坐标轴对齐, 最后:

- 按 ``-o`` 的后缀写出结构文件 (.pdb / .cif)
- 在同目录生成同名 (仅后缀不同) 的 .pml: 载入结构并绘制 x/y/z 三轴,
  优先用 biorazer_pymol.mark.arrow 的 ``arrow_pass`` (/opt/envs/pymol 已装),
  不可用时回退为内置 CGO 绘制。

示例:
    python center_helix_bundle.py input.pdb --helix A:1-30 --helix B:40-70 \\
        -o centered.pdb
    python center_helix_bundle.py input.cif --helix A:1-30 -o centered.cif \\
        --verbose
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

from biorazer.structure.io.protein import (
    Pdb_AtomArray,
    Cif_AtomArray,
    AtomArray_Pdb,
    AtomArray_Cif,
)
from biorazer_parametric_design.models.helix import CCCPHelixBundle
from biorazer_parametric_design.params.basic import FitError

PDB_SUFFIXES = (".pdb", ".ent")
CIF_SUFFIXES = (".cif", ".mmcif")

# 每个 helix 至少需要的 CA 数 (CCCP 拟合的最低残基数)
MIN_CA_PER_HELIX = 4


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", metavar="INPUT",
                   help="输入结构文件 (PDB: .pdb/.ent; CIF: .cif/.mmcif)")
    p.add_argument("--helix", action="append", metavar="CHAIN:START-END", default=None,
                   help="helix 范围, 可重复 (如 --helix A:1-30 --helix B:40-70); "
                        "不提供时逐个交互式输入")
    p.add_argument("-o", "--output", default=None, metavar="PDB/CIF",
                   help="输出结构文件, 后缀决定格式 (.pdb/.cif); "
                        "默认 <输入名>_centered.<输入后缀>; 同时生成同前缀 .pml")
    p.add_argument("--max-try", type=int, default=30, metavar="N",
                   help="center() 最大迭代次数")
    p.add_argument("--atol-rot", type=float, default=1e-5, metavar="RAD",
                   help="旋转收敛容差 (弧度)")
    p.add_argument("--atol-trans", type=float, default=1e-4, metavar="ANGSTROM",
                   help="平移收敛容差 (Å)")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="打印拟合/居中过程")
    return p.parse_args(argv)


def read_structure(path):
    """按后缀读取结构, 返回 biotite AtomArray。"""
    suffix = Path(path).suffix.lower()
    if suffix in PDB_SUFFIXES:
        return Pdb_AtomArray(input_io=path).read()
    if suffix in CIF_SUFFIXES:
        return Cif_AtomArray(input_io=path).read()
    sys.exit(f"error: 无法识别输入格式 (后缀 {suffix!r}), 支持: "
             + " / ".join(PDB_SUFFIXES + CIF_SUFFIXES))


def write_structure(structure, path):
    """按 -o 后缀写出结构。"""
    suffix = Path(path).suffix.lower()
    if suffix in PDB_SUFFIXES:
        AtomArray_Pdb(output_io=path).write(structure)
    elif suffix in CIF_SUFFIXES:
        AtomArray_Cif(output_io=path).write(structure)
    else:
        sys.exit(f"error: 无法根据 -o 后缀 {suffix!r} 判断输出格式, "
                 "请以 .pdb 或 .cif 结尾")


def parse_helix_spec(spec):
    """解析 'chain:start-end' (无链号时直接 'start-end'), 返回 (chain, start, end)。"""
    spec = spec.strip()
    if ":" in spec:
        m = re.fullmatch(r"([^:]*):(\d+)-(\d+)", spec)
        if not m:
            sys.exit(f"error: 无法解析 helix 范围 {spec!r}, "
                     "格式应为 chain:start-end (如 A:1-30)")
        chain, start, end = m.group(1), int(m.group(2)), int(m.group(3))
    else:
        m = re.fullmatch(r"(\d+)-(\d+)", spec)
        if not m:
            sys.exit(f"error: 无法解析 helix 范围 {spec!r}, "
                     "格式应为 chain:start-end (如 A:1-30), "
                     "无链号结构可直接写 start-end")
        chain, start, end = "", int(m.group(1)), int(m.group(2))
    if start > end:
        sys.exit(f"error: {spec!r} 的起始残基 {start} 大于结束残基 {end}")
    return chain, start, end


def prompt_helix_specs():
    """交互式逐个询问 helix 范围, 空行结束。"""
    print("未提供 --helix, 请逐个输入 helix 范围 (格式 chain:start-end, 如 A:1-30;")
    print("无链号结构可直接写 start-end)。直接回车结束输入。")
    specs = []
    i = 1
    while True:
        try:
            line = input(f"helix_{i} 范围: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            break
        specs.append(line)
        i += 1
    if not specs:
        sys.exit("error: 未输入任何 helix 范围")
    return specs


def build_masks(structure, specs):
    """把 helix 范围列表转成 {helix_i: bool mask}, 并做重叠/空范围校验。"""
    masks = {}
    for i, spec in enumerate(specs, 1):
        chain, start, end = parse_helix_spec(spec)
        key = f"helix_{i}"
        mask = (
            (structure.chain_id == chain)
            & (structure.res_id >= start)
            & (structure.res_id <= end)
        )
        if mask.sum() == 0:
            sys.exit(f"error: {key} 范围 {spec!r} 没有匹配到任何原子 "
                     f"(chain={chain!r}, res {start}-{end})")
        n_ca = np.sum(mask & (structure.atom_name == "CA"))
        if n_ca < MIN_CA_PER_HELIX:
            sys.exit(f"error: {key} 范围 {spec!r} 只有 {n_ca} 个 CA 原子, "
                     f"至少需要 {MIN_CA_PER_HELIX} 个")
        masks[key] = mask
    # 重叠校验
    keys = list(masks)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            n_overlap = np.sum(masks[keys[i]] & masks[keys[j]])
            if n_overlap:
                sys.exit(f"error: {keys[i]} ({specs[i]!r}) 与 {keys[j]} "
                         f"({specs[j]!r}) 重叠了 {n_overlap} 个原子")
    return masks


def write_pml(pml_path, structure_name, axis_length):
    """生成 PyMOL 脚本: 载入结构 + 绘制 x/y/z 三轴。

    注意: pml 由 PyMOL 自带的解析器处理, 顶层 Python 会被逐行执行
    (try:/def/多行 for 都会碎), 因此所有 Python 代码必须放进
    ``python ... python end`` 块; 注释保持纯 ASCII 且不含 ';'。
    """
    pml = f'''# PyMOL script: centered helix bundle with x/y/z axes
# Generated by center_helix_bundle.py
# Structure is translated to origin and aligned to Cartesian axes
# Usage: run from this directory:  pymol -q {pml_path.name}
# Axes are drawn with biorazer_pymol.mark.arrow.arrow_pass when importable
# (installed in /opt/envs/pymol), otherwise a built-in CGO fallback is used

python
import os, sys
# __file__ 指向 pymol 模块本身而非脚本, 改用 sys.argv 找 pml 路径,
# 切到其所在目录, 这样从任意目录运行都能按相对文件名载入结构
for _a in reversed(sys.argv):
    if _a.lower().endswith(".pml") and os.path.isfile(_a):
        os.chdir(os.path.dirname(os.path.abspath(_a)))
        break
python end

load "{structure_name}", bundle
hide everything
show cartoon
set cartoon_smooth_loops, 0

python
AXIS_LENGTH = {axis_length}

def _draw_axis_cgo(name, vec, rgb, length):
    from pymol import cgo
    v = [c * length for c in vec]
    cmd.load_cgo([
        cgo.CYLINDER, -v[0], -v[1], -v[2], v[0], v[1], v[2], 0.08,
        rgb[0], rgb[1], rgb[2], rgb[0], rgb[1], rgb[2],
        cgo.CONE, v[0], v[1], v[2], v[0] * 1.2, v[1] * 1.2, v[2] * 1.2, 0.32,
        0.0, rgb[0], rgb[1], rgb[2], rgb[0], rgb[1], rgb[2], 1.0, 1.0,
    ], name)

try:
    from biorazer_pymol.mark.arrow import arrow_pass
    arrow_pass(0, 0, 0, 1, 0, 0, length=AXIS_LENGTH, name="axis_x",
               cylinder_radius=0.08, r_color=1.0, g_color=0.2, b_color=0.2)
    arrow_pass(0, 0, 0, 0, 1, 0, length=AXIS_LENGTH, name="axis_y",
               cylinder_radius=0.08, r_color=0.2, g_color=1.0, b_color=0.2)
    arrow_pass(0, 0, 0, 0, 0, 1, length=AXIS_LENGTH, name="axis_z",
               cylinder_radius=0.08, r_color=0.2, g_color=0.4, b_color=1.0)
except ImportError:
    _draw_axis_cgo("axis_x", (1, 0, 0), (1.0, 0.2, 0.2), AXIS_LENGTH)
    _draw_axis_cgo("axis_y", (0, 1, 0), (0.2, 1.0, 0.2), AXIS_LENGTH)
    _draw_axis_cgo("axis_z", (0, 0, 1), (0.2, 0.4, 1.0), AXIS_LENGTH)
python end

python
# Axis labels
for _lab, _pos, _col in (
    ("lab_x", (AXIS_LENGTH * 1.15, 0.0, 0.0), "red"),
    ("lab_y", (0.0, AXIS_LENGTH * 1.15, 0.0), "green"),
    ("lab_z", (0.0, 0.0, AXIS_LENGTH * 1.15), "blue"),
):
    cmd.pseudoatom(_lab, pos=_pos)
    cmd.label(_lab, '"%s"' % _lab[-1])
    cmd.set("label_color", _col, _lab)
python end

zoom bundle
'''
    pml_path.write_text(pml, encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)

    structure = read_structure(args.input)
    print(f"读取 {args.input}: {len(structure)} 原子")

    specs = list(args.helix) if args.helix else prompt_helix_specs()
    if len(specs) < 2:
        sys.exit("error: helix bundle 至少需要 2 条 helix, "
                 f"当前只提供了 {len(specs)} 个范围")
    masks = build_masks(structure, specs)
    for key, spec in zip(masks, specs):
        print(f"  {key}: {spec}  ({masks[key].sum()} 原子)")

    bundle = CCCPHelixBundle.from_structure(structure=structure, mask=masks)
    try:
        bundle.center(
            max_try=args.max_try,
            atol_rot=args.atol_rot,
            atol_trans=args.atol_trans,
            verbose=args.verbose,
        )
        bundle.fit(verbose=args.verbose)
    except AssertionError as e:
        sys.exit(f"error: CCCP 拟合失败: {e}\n"
                 "提示: CCCP 参数化要求所有 helix 的 CA 数 (残基数) 相同, "
                 "请检查 --helix 范围是否等长")
    except FitError as e:
        sys.exit(f"error: CCCP 拟合失败: {e}\n"
                 "提示: 请检查 helix 范围是否准确 (是否包含了非螺旋片段/拐弯处), "
                 "以及各 helix 的残基数是否相同")
    except TimeoutError as e:
        sys.exit(f"error: 居中未收敛: {e}")

    x, y, z = bundle.xyz
    print(f"拟合 RMSD: {bundle.rmsd:.4f} Å")
    print(f"质心: {np.asarray(bundle.centroid).round(4)}")
    print(f"x 轴: {x.round(4)}")
    print(f"y 轴: {y.round(4)}")
    print(f"z 轴: {z.round(4)}")

    if args.output is None:
        args.output = str(
            Path(args.input).with_name(
                Path(args.input).stem + "_centered" + Path(args.input).suffix
            )
        )
    write_structure(bundle.structure, args.output)

    # 轴长按结构尺寸缩放 (覆盖结构外延), 最小 10 Å
    axis_length = max(10.0, round(float(np.abs(bundle.structure.coord).max()) * 1.2, 1))
    pml_path = Path(args.output).with_suffix(".pml")
    write_pml(pml_path, Path(args.output).name, axis_length)

    print(f"结构已写出: {args.output}")
    print(f"PyMOL 脚本已写出: {pml_path}")
    parent = pml_path.parent
    print(f"运行: {'cd ' + str(parent) + ' && ' if str(parent) != '.' else ''}pymol -q {pml_path.name}")


if __name__ == "__main__":
    main()
