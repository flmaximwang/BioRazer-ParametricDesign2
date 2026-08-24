"""模型包导入冒烟测试。

保证 ``biorazer_prds.models`` 及其公共导出 (Assembly / AssemblyPart / 各 ref 变体)
在真实 BioRazer 环境下可导入。这是重构前后的行为基线。
"""

import importlib


def test_models_package_importable():
    """models 包本身可导入。"""
    models = importlib.import_module("biorazer_prds.models")
    assert models is not None


def test_public_exports_exist():
    """当前公共导出应包含 Assembly 与 AssemblyPart。"""
    from biorazer_prds.models import Assembly, AssemblyPart

    assert Assembly is not None
    assert AssemblyPart is not None
