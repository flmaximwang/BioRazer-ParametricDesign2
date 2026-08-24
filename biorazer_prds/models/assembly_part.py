from abc import abstractmethod
from dataclasses import dataclass, field
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING
import biotite.structure as bt_struct
from biotite.structure.io import pdb
import numpy as np
from scipy.spatial.transform import Rotation as R

from ..util.alignment import calculate_rotation, calculate_euler_ZXZ
import biorazer.structure.io as br_struct_io

if TYPE_CHECKING:
    from .assembly_part_parametric import AssemblyPartParaRef


@dataclass
class AssemblyPartProperty:
    """
    Properties of an AssemblyPart.

    Properties
    ----------
    structure : bt_struct.AtomArray
        The atomic structure of the part.
    mask : dict[str, np.ndarray]
        A dictionary mapping part names to boolean masks for selecting atoms in the structure.
    """

    structure: bt_struct.AtomArray = None
    component: dict[str, "AssemblyPartParaRef"] = field(default_factory=dict)
    mask: dict[str, np.ndarray] = field(default_factory=dict)

    _centroid: np.ndarray = None
    _xyz: np.ndarray = None

    @abstractmethod
    def update_component(self):
        """
        Update the component dictionary based on the current structure and mask.
        Call this method before accessing the component property.
        """

    @property
    def centroid(self):
        """
        Implementation of this method should return the centroid of the structure.
        self._centroid can be used to cache the result.

        By default, the centroid is calculated using CA atoms.
        """
        if self._centroid is None:
            ca_atoms = self.structure[
                self.structure.get_annotation("atom_name") == "CA"
            ]
            self._centroid = bt_struct.centroid(ca_atoms)
        return self._centroid

    @property
    def xyz(self):
        """
        Implementation of this method should return the x, y, z directions of the structure.
        This is a placeholder method and should be implemented in subclasses.
        self._xyz can be used to cache the result.

        Returns
        -------
        x, y, z directions as numpy arrays.
        """
        raise TypeError("An arbitrary AssemblyPart has no definition of local xyz!")

    @property
    def coord(self):
        coord: np.ndarray = self.structure.coord
        return coord

    @coord.setter
    def coord(self, new_coord: np.ndarray):
        self.structure.coord = new_coord

    def __getitem__(self, mask_name: str):
        """
        Get part of the atomarray based on the key-value map in self.masks
        """
        assert (
            mask_name in self.mask
        ), f"Mask {mask_name} not found in {self.mask.keys()}"
        assert self.mask[mask_name] is not None, f"Mask {mask_name} is not set"
        return self.structure[self.mask[mask_name]]


@dataclass
class AssemblyPartOperation(AssemblyPartProperty):
    """
    Operations on an AssemblyPart.
    """

    def translate(self, x, y, z):
        self.structure = bt_struct.translate(self.structure, [x, y, z])
        self._centroid = None

    def rotate(self, rotation: R, centroid_to_origin=True, XYZ_to_xyz=True):
        """
        Apply a rotation to the current structure.

        This method supports two optional preprocessing steps:

        1. Move the structure centroid to the origin.
        2. Re-orient the structure's local principal directions
           ``(X, Y, Z) = self.xyz`` onto the canonical Cartesian axes
           ``(x, y, z)``.

        After those optional steps, the input ``rotation`` is applied in the
        temporary coordinate system. The method then restores the original
        reference frame in reverse order.

        In practice, this gives three common behaviors:

        - ``centroid_to_origin=False, XYZ_to_xyz=False``:
          rotate the current coordinates directly in the global frame.
        - ``centroid_to_origin=True, XYZ_to_xyz=False``:
          rotate around the structure centroid, but still in the current frame.
        - ``XYZ_to_xyz=True``:
          rotate in the structure's own local frame. In this case centering at
          the origin is mandatory, so ``centroid_to_origin`` is implicitly
          treated as ``True``.

        Parameters
        ----------
        rotation : R
            Rotation to apply. The rotation is interpreted in the temporary
            frame defined by ``centroid_to_origin`` and ``XYZ_to_xyz``.
        centroid_to_origin : bool
            If True, translate the structure so that its centroid is located at
            the origin before the rotation, then translate it back afterwards.
            This makes the rotation happen "around the centroid" instead of
            around the global origin.
        XYZ_to_xyz : bool
            If True, first align the structure's local axes ``self.xyz`` to the
            canonical basis before applying ``rotation``, then transform back.
            This is useful when the caller wants to express the rotation in the
            part's intrinsic coordinate system rather than the current global
            coordinate system.

            When this option is enabled, ``centroid_to_origin`` is forced to
            ``True`` because axis alignment is defined around the centered
            structure.
        """

        if XYZ_to_xyz:
            centroid_to_origin = True

        if centroid_to_origin:
            # Move the current centroid to the origin so subsequent rotation is
            # performed around the part center instead of the world origin.
            center_translation = self.calculate_center_translation()
            self.coord += center_translation
        if XYZ_to_xyz:
            # Align the part-local coordinate frame (self.xyz) with the
            # canonical xyz frame, so `rotation` can be interpreted in the
            # part's own reference system.
            center_rotation = self.calculate_center_rotation()
            self.coord = center_rotation.apply(self.structure.coord)

        # Apply the user-provided rotation in the prepared coordinate frame.
        self.coord = rotation.apply(self.structure.coord)

        if XYZ_to_xyz:
            # Restore the original local-frame orientation.
            inv_center_rotation = center_rotation.inv()
            self.coord = inv_center_rotation.apply(self.coord)
        if centroid_to_origin:
            # Restore the original centroid position.
            inv_center_translation = -center_translation
            self.coord += inv_center_translation

        if not centroid_to_origin:
            # If we rotated in-place without centering, the centroid may have
            # changed relative to the global frame and must be recomputed.
            self._centroid = None
        # Any rotation invalidates the cached local axes.
        self._xyz = None

    def rotate_euler(
        self,
        axis_spec,
        a,
        b,
        c,
        degrees=False,
        centroid_to_origin=True,
        XYZ_to_xyz=True,
    ):
        """
        Rotate the structure using Euler angles.

        This is a convenience wrapper around :meth:`rotate`. It first builds a
        :class:`scipy.spatial.transform.Rotation` instance from Euler angles via
        :meth:`Rotation.from_euler`, then delegates the actual coordinate
        transformation to :meth:`rotate`.

        Parameters
        ----------
        axis_spec : str
            Euler rotation sequence accepted by
            :meth:`scipy.spatial.transform.Rotation.from_euler`, such as
            ``"xyz"``, ``"zyx"``, or ``"ZXZ"``.
        a, b, c : float
            The three Euler angles corresponding to ``axis_spec``.
        degrees : bool
            If True, interpret ``a``, ``b``, ``c`` in degrees. Otherwise they
            are interpreted in radians.
        centroid_to_origin : bool
            Passed through to :meth:`rotate`.
        XYZ_to_xyz : bool
            Passed through to :meth:`rotate`.
        """
        rotation = R.from_euler(axis_spec, [a, b, c], degrees=degrees)
        self.rotate(
            rotation, centroid_to_origin=centroid_to_origin, XYZ_to_xyz=XYZ_to_xyz
        )

    def rotate_quat(self, x, y, z, w, centroid_to_origin=True, XYZ_to_xyz=True):
        """
        Rotate the structure using a quaternion.

        This is a convenience wrapper around :meth:`rotate`. It constructs a
        :class:`scipy.spatial.transform.Rotation` object from the quaternion
        components in SciPy's expected order ``[x, y, z, w]``, then applies the
        same centering and frame-alignment logic implemented by :meth:`rotate`.

        Parameters
        ----------
        x, y, z, w : float
            Quaternion components in SciPy order ``[x, y, z, w]``.
            Note that ``w`` is the scalar term.
        centroid_to_origin : bool
            Passed through to :meth:`rotate`.
        XYZ_to_xyz : bool
            Passed through to :meth:`rotate`.
        """
        rotation = R.from_quat([x, y, z, w])
        self.rotate(
            rotation, centroid_to_origin=centroid_to_origin, XYZ_to_xyz=XYZ_to_xyz
        )

    def center(
        self,
        max_try=10,
        atol_rot: float = 1e-5,
        atol_trans: float = 1e-5,
        verbose: bool = False,
    ):
        """
        Perform centering of the structure by iteratively applying rotation and translation
        until the structure is aligned with canonical axes and translated to origin.

        Parameters
        ----------
        max_try : int
            Maximum number of centering iterations.
        atol_rot : float
            Absolute tolerance (radian) for rotation convergence.
        atol_trans : float
            Absolute tolerance for translation convergence.
        verbose : bool
            Print per-iteration progress if True.

        As long as self.center and self.xyz are properly implemented, this method should always converge.
        """

        def _log(message: str):
            if verbose:
                print(f"[AssemblyPart.center] {message}")

        if max_try <= 0:
            raise ValueError("max_try must be a positive integer")
        if atol_rot < 0 or atol_trans < 0:
            raise ValueError("atol_rot and atol_trans must be non-negative")

        counter = 0
        _log(
            f"Start centering with max_try={max_try}, "
            f"atol_rot={atol_rot}, atol_trans={atol_trans}"
        )
        while True:
            counter += 1
            rotation = self.calculate_center_rotation()
            self.rotate(rotation, centroid_to_origin=True, XYZ_to_xyz=False)
            translation = self.calculate_center_translation()
            self.translate(*translation)

            euler_angles = self.calculate_center_rotation().as_euler(
                "xyz", degrees=False
            )
            translation = self.calculate_center_translation()
            _log(
                "Iteration "
                f"{counter}/{max_try}: "
                f"euler(rad)={np.array2string(euler_angles, precision=4)}, "
                f"translation={np.array2string(translation, precision=4)}"
            )
            if np.allclose(euler_angles, [0, 0, 0], atol=atol_rot) and np.allclose(
                translation, [0, 0, 0], atol=atol_trans
            ):
                _log(f"Converged in {counter} iterations")
                break
            if counter >= max_try:
                raise TimeoutError(
                    f"Failed to center the part after {max_try} attempts. "
                    f"Thresholds: atol_rot={atol_rot}, atol_trans={atol_trans}. "
                    f"Last euler(rad)={np.array2string(euler_angles, precision=4)}, "
                    f"last translation={np.array2string(translation, precision=4)}. "
                    "Please check the structure and alignment."
                )

    def calculate_center_rotation(self):
        """
        Calculate the rotation that aligns the structure with its own X, Y, Z axes to the canonical x, y, z axes.

        Returns
        -------
        A scipy.spatial.transform.Rotation object representing the rotation.
        """
        x, y, z = self.xyz
        return calculate_rotation(x, y, z).inv()

    def calculate_center_translation(self):
        """
        Calculate the translation that moves the structure to the origin.

        Returns
        -------
        A np.ndarray that represents the translation vector.
        """
        centroid = self.centroid
        return -centroid

    @staticmethod
    def calculate_transformation_between(
        part_1: "AssemblyPart", part_2: "AssemblyPart"
    ):
        """
        Calculate the transformation that aligns part_1 to part_2.

        Returns
        -------
        translation : np.ndarray
            A numpy array representing the translation vector.
        rotation : R
            A scipy.spatial.transform.Rotation object representing the rotation.
        """
        translation = part_2.centroid - part_1.centroid

        part_1_center_rotation = part_1.calculate_center_rotation()
        part_2_copy = deepcopy(part_2)
        part_2_copy.rotate(
            part_1_center_rotation, centroid_to_origin=False, XYZ_to_xyz=False
        )
        x, y, z = part_2_copy.xyz
        rotation = calculate_rotation(x, y, z)

        return translation, rotation

    def check_axes_aligned(self, atol=1e-3):
        """
        Check if the structure is aligned with the X and Z axes.
        Returns True if aligned, False otherwise.
        """
        x, y, z = self.xyz
        flags = [
            np.allclose(x, [1, 0, 0], atol=atol),
            np.allclose(y, [0, 1, 0], atol=atol),
            np.allclose(z, [0, 0, 1], atol=atol),
        ]
        if not np.all(flags):
            raise ValueError(
                "Structure must be aligned with X and Z axes before ZXZ rotation\n"
                f"Current x: {x}\n"
                f"Current y: {y}\n"
                f"Current z: {z}"
            )


@dataclass
class AssemblyPartIO(AssemblyPartProperty):
    """
    Input/Output operations for an AssemblyPart.
    """

    @classmethod
    @abstractmethod
    def from_mask(cls, structure: bt_struct.AtomArray, mask: np.ndarray):
        """
        Load the structure from a mask on a given structure.
        Other properties, including mask, component, etc.,  will be generated automatically based on the structure.
        """

    @classmethod
    @abstractmethod
    def from_component(cls, structure: bt_struct.AtomArray, component: dict):
        """
        Load the structure from a component dictionary on a given structure.
        Other properties, including mask, component, etc.,  will be generated automatically based on the structure.
        """

    def to_pdb(self, pdb_file):
        """Export the structure to a PDB file."""
        br_struct_io.protein.STRUCT2PDB("", pdb_file).write(self.structure)

    def to_cif(self, cif_file):
        """Export the structure to a CIF file."""
        br_struct_io.protein.STRUCT2CIF("", cif_file).write(self.structure)


@dataclass
class AssemblyPart(AssemblyPartOperation, AssemblyPartIO):
    """
    An AssemblyPart is a part of an assembly that can contain one or more AssemblyComponents.
    """


