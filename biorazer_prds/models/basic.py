from abc import abstractmethod
from dataclasses import dataclass, field
from copy import deepcopy
from pathlib import Path
import biotite.structure as bt_struct
from biotite.structure.io import pdb
from biotite.structure.info import standardize_order
import numpy as np
from scipy.spatial.transform import Rotation as R
from ..util.alignment import calculate_rotation, calculate_euler_ZXZ
import biorazer.structure.io as br_struct_io


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
    component: dict[str, "AssemblyPartParametric"] = field(default_factory=dict)
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
    @abstractmethod
    def xyz(self):
        """
        Implementation of this method should return the x, y, z directions of the structure.
        This is a placeholder method and should be implemented in subclasses.
        self._xyz can be used to cache the result.

        Returns
        -------
        x, y, z directions as numpy arrays.
        """

    @property
    def coord(self):
        coord: np.ndarray = self.structure.coord
        return coord

    @coord.setter
    def coord(self, new_coord: np.ndarray):
        self.structure.coord = new_coord

    def __getitem__(self, mask_name: str):
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


@dataclass
class AssemblyPartParametricProperty(AssemblyPart):
    """
    Properties of an AssemblyPart that is parametrically defined.

    Properties
    ----------
    structure : bt_struct.AtomArray
        The atomic structure of the part.
    mask : dict[str, np.ndarray]
        A dictionary mapping part names to boolean masks for selecting atoms in the structure.
    params : dict
        A dictionary to store parameters of the fitted model.
    initial_param : dict
        A dictionary to store initial parameters for fitting. You should have no other keys than those in params.
    extra_param : dict
        A dictionary to store extra parameters that should not be used in fitting
    rmsd : float
        The root mean square deviation of the fitted model.
    fitted_coord : np.ndarray
        The coordinates of the fitted model.
    ref_structure : biotite.structure.AtomArray
        Used with fit_with_ref to provide a reference structure for fitting.
    initial_param : dict
        A dictionary to store initial parameters for fitting.
    """

    param: dict = field(default_factory=dict)
    initial_param: dict = field(default_factory=dict)
    extra_param: dict = field(default_factory=dict)
    params_not_to_fit: list[str] = field(default_factory=list)

    ref_structure: bt_struct.AtomArray = None
    fitted_structure: bt_struct.AtomArray = None
    rmsd: float = None


@dataclass
class AssemblyPartParametricIO(AssemblyPartParametricProperty):
    """
    Input/Output operations for an AssemblyPart that is parametrically defined.
    """

    @classmethod
    def from_structure(cls, *, structure: bt_struct.AtomArray):
        """Load the structure from a given structure. Other properties will be generated automatically based on the structure."""
        res_obj = cls(structure=structure)
        return res_obj

    @classmethod
    def from_params(cls, *, params: dict, **kwargs):
        """Load the structure from a given set of parameters. Other properties will be generated automatically based on the parameters."""
        raise NotImplementedError("from_params method is not implemented")


@dataclass
class AssemblyPartParametricOperation(AssemblyPartParametricProperty):
    """
    Operations on an AssemblyPart that is parametrically defined.
    """

    @abstractmethod
    def fit(self, verbose: bool = False):
        """
        Fit with the given coordinates and store the parameters, rmsd and fitted coordinates in the object.

        self.initial_param can be used to provide initial guesses for the fitting.
        self.params_not_to_fit can be used to specify parameters that should not be fitted.
        Set verbose=True to print fitting progress information.
        """

    @abstractmethod
    def fit_with_ref(self):
        """
        Fit with the given coordinates to the reference structure and store the parameters, rmsd and fitted coordinates in the object.
        - Ref is mobile, the original structure is fixed.
        """

    @abstractmethod
    def modify(self, method, *args, **kwargs):
        """
        Modify the structure with the given method and arguments.
        The method should be a string that specifies the modification method.
        The args and kwargs are the arguments for the modification method.
        Modification relies on the params stored in the object.
        """


@dataclass
class AssemblyPartParametric(
    AssemblyPartParametricIO,
    AssemblyPartParametricOperation,
):
    """
    An AssemblyComponent is a model that is parametrically defined and can be fitted to a set of coordinates.
    """

    def to_pymol_axes(self, prefix="default", length=5.0):
        """
        Export the axes of all parts in the assembly to a format that can be visualized in PyMOL.
        Returns a list of dictionaries, each containing the part index and its x, y, z axes.
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
class AssemblyProperty:
    parts: list[AssemblyPart] = None

    def append(self, new_part):
        """Append a new part to the assembly."""
        self.parts.append(new_part)

    def __getitem__(self, index):
        """Get a specific part by index."""
        if isinstance(index, slice):
            return type(self)(self.parts[index])
        return self.parts[index]

    def check_part_index(self, part_index):
        """Check if the part index is valid."""
        if part_index < 0 or part_index >= len(self.parts):
            raise IndexError("Part index out of range")


@dataclass
class AssemblyIO(AssemblyProperty):
    """
    An Assembly can be broken down into multiple parts, each represented by an AssemblyPart. Operations can be performed on the whole assembly or individual parts.
    """

    def merge_structures(self):
        """Merge all parts into a single structure and reorder atoms for PDB export."""
        merged_structure = bt_struct.concatenate(
            [part.structure for part in self.parts]
        )

        residue_starts = bt_struct.get_residue_starts(
            merged_structure, add_exclusive_stop=True
        )
        residue_order = np.lexsort(
            (
                np.arange(len(residue_starts) - 1),
                merged_structure.res_name[residue_starts[:-1]],
                merged_structure.ins_code[residue_starts[:-1]],
                merged_structure.res_id[residue_starts[:-1]],
                merged_structure.chain_id[residue_starts[:-1]],
            )
        )
        residue_sorted_indices = np.concatenate(
            [np.arange(residue_starts[i], residue_starts[i + 1]) for i in residue_order]
        )
        merged_structure = merged_structure[residue_sorted_indices]

        atom_order = standardize_order(merged_structure)
        return merged_structure[atom_order]


class AssemblyOperation(AssemblyProperty):

    def center(
        self,
        part_index,
        max_try=10,
        atol_rot: float = 1e-5,
        atol_trans: float = 1e-5,
        verbose: bool = False,
    ):
        """Center one part and apply the same rigid transform to the whole assembly."""

        self.check_part_index(part_index)
        if max_try <= 0:
            raise ValueError("max_try must be a positive integer")
        if atol_rot < 0 or atol_trans < 0:
            raise ValueError("atol_rot and atol_trans must be non-negative")

        def _log(message: str):
            if verbose:
                print(f"[Assembly.center] {message}")

        counter = 0
        _log(
            f"Start centering part[{part_index}] with max_try={max_try}, "
            f"atol_rot={atol_rot}, atol_trans={atol_trans}"
        )
        while True:
            counter += 1
            center_part: AssemblyPart = self.parts[part_index]
            center_translation = center_part.calculate_center_translation()
            center_rotation = center_part.calculate_center_rotation()

            for part in self.parts:
                part.translate(*center_translation)
                part.rotate(
                    center_rotation,
                    centroid_to_origin=False,
                    XYZ_to_xyz=False,
                )

            euler_angles = center_part.calculate_center_rotation().as_euler(
                "xyz", degrees=False
            )
            translation = center_part.calculate_center_translation()
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
                    f"Failed to center part[{part_index}] after {max_try} attempts. "
                    f"Thresholds: atol_rot={atol_rot}, atol_trans={atol_trans}. "
                    f"Last euler(rad)={np.array2string(euler_angles, precision=4)}, "
                    f"last translation={np.array2string(translation, precision=4)}. "
                    "Please check the structure and alignment."
                )

    def calculate_rotation_between(self, part_index_1, part_index_2, atol=1e-3):
        """
        Calculate the rotation that aligns part_index_1 with part_index_2.
        Returns the Rotation object representing the rotation.
        """
        self.check_part_index(part_index_1)
        self.check_part_index(part_index_2)
        part_1: AssemblyPart = self.parts[part_index_1]
        x, y, z = part_1.xyz
        flag = (
            np.allclose(x, [1, 0, 0], atol=atol)
            and np.allclose(y, [0, 1, 0], atol=atol)
            and np.allclose(z, [0, 0, 1], atol=atol)
        )
        if not flag:
            raise ValueError(
                f"Part[{part_index_1}] is not aligned with the reference axes within atol={atol}. Current axes:\n"
                f"x={x}, y={y}, z={z}"
            )
        part_2: AssemblyPart = self.parts[part_index_2]
        x, y, z = part_2.xyz
        return calculate_rotation(x, y, z)

    # def calculate_ZXZ_euler_between_old(
    #     self, part_index_1, part_index_2, degrees=False
    # ):
    #     """
    #     Calculate the ZXZ rotation that aligns part_index_1 with part_index_2.
    #     Returns the angles (a, b, c) in degrees.
    #     """
    #     rotation = self.calculate_rotation_between(part_index_1, part_index_2)
    #     euler_angles = rotation.as_euler("ZXZ", degrees=degrees)
    #     return euler_angles

    def calculate_quat_between(
        self, part_index_1, part_index_2, atol=1e-3, scaler_first=False, canonical=True
    ):
        """
        Calculate the quaternion that aligns part_index_1 with part_index_2.
        Returns the quaternion (x, y, z, w).
        """
        rotation = self.calculate_rotation_between(
            part_index_1, part_index_2, atol=atol
        )
        return rotation.as_quat(scalar_first=False, canonical=True)

    def calculate_euler_between(
        self, part_index_1, part_index_2, axis_spec, degrees=False, atol=1e-3
    ):
        """
        Calculate the Euler angles that align part_index_1 with part_index_2.
        Returns the angles (a, b, c) in degrees.
        """
        rotation = self.calculate_rotation_between(
            part_index_1, part_index_2, atol=atol
        )
        euler_angles = rotation.as_euler(axis_spec, degrees=degrees)
        return euler_angles

    def calculate_translation_between(self, part_index_1, part_index_2):
        """
        Calculate the translation that aligns part_index_1 with part_index_2.
        Returns the translation vector as a numpy array.
        """
        self.check_part_index(part_index_1)
        self.check_part_index(part_index_2)
        part_1: AssemblyPart = self.parts[part_index_1]
        part_2: AssemblyPart = self.parts[part_index_2]
        translation = part_2.centroid - part_1.centroid
        return translation


@dataclass
class Assembly(AssemblyIO, AssemblyOperation):
    """
    An Assembly can be broken down into multiple parts, each represented by an AssemblyPart. Operations can be performed on the whole assembly or individual parts.
    """
