"""模型包导入冒烟测试。

保证 ``biorazer_prds.models`` 及其公共导出在真实 BioRazer 环境下可导入。
这是重构前后的行为基线。
"""

import importlib

from biorazer_prds.models import (
    Assembly,
    AssemblyParaRef,
    AssemblyRealRef,
    Helix,
    CrickHelix,
    CCCPHelixBundle,
)


def test_models_package_importable():
    """models 包本身可导入。"""
    models = importlib.import_module("biorazer_prds.models")
    assert models is not None


def test_public_exports_exist():
    """公共导出应包含统一 Assembly 及各 ref/螺旋子类。"""
    for cls in (
        Assembly,
        AssemblyParaRef,
        AssemblyRealRef,
        Helix,
        CrickHelix,
        CCCPHelixBundle,
    ):
        assert cls is not None


def test_hierarchy():
    """ref 变体直接继承 Assembly; 螺旋类挂在 AssemblyParaRef 之下。"""
    assert issubclass(AssemblyParaRef, Assembly)
    assert issubclass(AssemblyRealRef, Assembly)
    assert issubclass(CrickHelix, AssemblyParaRef)
    assert issubclass(CCCPHelixBundle, AssemblyParaRef)
