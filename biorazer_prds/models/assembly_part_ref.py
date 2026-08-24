from copy import deepcopy

from biorazer_prds.models.assembly_part import AssemblyPart

import biotite.structure as bt_struct

from dataclasses import dataclass


@dataclass
class AssemblyPartRefProperty(AssemblyPart):
    """
    Properties of an AssemblyPart that serves as a reference in an assembly.

    A Part's ``structure`` is always the real atomic structure. The
    ``ref_structure`` is the reference geometry used for placement / fitting;
    its content is decided by the subclass:

    - AssemblyPartParaRef : ref_structure is virtual (generated from params)
    - AssemblyPartRealRef : ref_structure is real (loaded, e.g. HEM from RCSB)
    """

    ref_structure: bt_struct.AtomArray = None


@dataclass
class AssemblyPartRefIO(AssemblyPartRefProperty):
    """Input/Output operations for a reference AssemblyPart."""

    @classmethod
    def from_structure(cls, *, structure: bt_struct.AtomArray):
        """Load the real structure from a given structure."""
        res_obj = cls(structure=structure)
        return res_obj


@dataclass
class AssemblyPartRefOperation(AssemblyPartRefProperty):
    """Operations common to reference AssemblyParts."""


@dataclass
class AssemblyPartRef(AssemblyPartRefIO, AssemblyPartRefOperation):
    """
    A reference part of an assembly: carries a real ``structure`` plus a
    ``ref_structure`` used for placement / registration.
    """

    def to_pymol_axes(self, prefix="default", length=5.0):
        """
        Export the axes of all parts in the assembly to a format that can be
        visualized in PyMOL.

        Returns a list of dictionaries, each containing the part index and its
        x, y, z axes.
        """
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
        """Return a deep copy of the object."""
        return deepcopy(self)


@dataclass
class AssemblyPartRealRefProperty(AssemblyPartRef):
    """
    A reference part whose ``ref_structure`` is a real structure (e.g. a HEM
    heme cofactor loaded from RCSB). No parametric fitting.
    """

    @property
    def xyz(self):
        # A real reference has no intrinsic biological axes; a frame convention
        # must be defined (e.g. inertia principal axes, an explicit direction,
        # or a user-provided frame).
        raise NotImplementedError(
            "AssemblyPartRealRef.xyz: define a frame convention for the real "
            "reference (inertia axes / explicit direction / user frame)."
        )


@dataclass
class AssemblyPartRealRefIO(AssemblyPartRealRefProperty):
    pass


@dataclass
class AssemblyPartRealRefOperation(AssemblyPartRealRefProperty):
    pass


@dataclass
class AssemblyPartRealRef(AssemblyPartRealRefIO, AssemblyPartRealRefOperation):
    """
    A reference part defined by a fixed real structure (e.g. a HEM heme
    cofactor loaded from RCSB). It is a rigid real reference: no parametric
    fitting, no ``param`` / ``fit`` / ``modify``.
    """
