"""
CCCP (Coiled-Coil Crick Parameterization) - Python Package

A Python implementation of the CCCP suite for fitting optimal Crick parameters
to protein backbones and generating ideal coiled-coil backbones from Crick parameters.

Original MATLAB version: Copyright (C) 2011-2017 Gevorg Grigoryan
Python implementation: 2024

If used in a scientific publication, please cite:
G. Grigoryan, W. F. DeGrado, "Probing Designability via a Generalized
Model of Helical Bundle Geometry", J. Mol. Biol., 405(4): 1079-1100 (2011)

Web site: http://www.grigoryanlab.org/cccp/
"""

__version__ = "1.0.0"
__author__ = "Gevorg Grigoryan (original), Python port 2024"
__email__ = "gevorg.grigoryan@dartmouth.edu"

# from .scan_crick import fit_crick_scan
# from .fit_crick import fit_crick
# from .generate_crick_bb import generate_crick_bb
# from .read_ca import read_ca

__all__ = ["scan_crick", "fit_crick", "generate_crick_bb", "read_ca"]
