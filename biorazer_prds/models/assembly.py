import biotite.structure as bt_struct
import numpy as np
from biotite.structure.info import standardize_order

from biorazer_prds.models.assembly_part import AssemblyPart


from dataclasses import dataclass

from biorazer_prds.util.alignment import calculate_rotation


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