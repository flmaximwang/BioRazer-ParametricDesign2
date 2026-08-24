import re
from typing import Iterable
from dataclasses import dataclass, field
from .assembly_part import *
import biorazer.structure.io as br_struct_io
from ..params.helix_cp.generate import generate_helix_ca_by_crick
from ..params.helix_cp.fit import fit_helix_by_crick
from ..params.cccp.generate import generate_cc_ca_by_cccp
from ..params.cccp.fit import fit_cc_by_cccp
from ..params.util import ca_xyz_to_atom_array, pulchra_fix_backbone
from .assembly_part_parametric import AssemblyPartParaRef


@dataclass
class HelixProperty(AssemblyPartParaRef):
    mask: dict[str, np.ndarray] = field(default_factory=lambda: {"helix": None})


@dataclass
class HelixIO(HelixProperty):
    pass


@dataclass
class HelixOperation(HelixProperty):
    pass


@dataclass
class Helix(HelixIO, HelixOperation):
    """
    A class representing a part of an assembly that is a pure helix.
    It inherits from AssemblyPart and provides specific functionality for helix parts.
    """


@dataclass
class CrickHelixProperty(Helix):
    """
    Params
    ------
    direction: np.ndarray
        A normalized vector representing the direction of the helix.
    centroid: np.ndarray
        The centroid of the helix.
    radius: float
        The radius of the helix.
    pitch: float
        The pitch of the helix.
    phase: float
        The phase of the helix.
    """

    @property
    def xyz(self):
        if not self._xyz:
            self.fit()
            self._xyz = np.vstack(
                self.extra_param["x"], self.extra_param["y"], self.extra_param["z"]
            )
        return self._xyz

    @staticmethod
    def calculate_helix_type(omega):
        omega_str_list = [
            "10/3",
            "27/8",
            "17/5",
            "24/7",
            "7/2",
            "25/7",
            "18/5",
            "29/8",
            "11/3",
            "26/7",
            "15/4",
            "19/5",
            "23/6",
            "27/7",
        ]
        omega_list = list(map(lambda x: 2 * np.pi / eval(x), omega_str_list))
        if omega > omega_list[0] + (omega_list[0] - omega_list[1]) / 2:
            return None
        if omega < omega_list[-1] - (omega_list[-2] - omega_list[-1]) / 2:
            return None
        for i in range(len(omega_list) - 1):
            omega_upper = omega_list[i]
            omega_lower = omega_list[i + 1]
            if omega < omega_lower:
                continue
            elif omega < (omega_lower + omega_upper) / 2:
                return omega_str_list[i]
            elif omega < omega_upper:
                return omega_str_list[i + 1]
            else:
                continue


@dataclass
class CrickHelixIO(CrickHelixProperty):

    @staticmethod
    def _validate_helix_mask(mask: dict[str, np.ndarray]):
        if "helix" not in mask:
            raise ValueError("Mask must contain a key 'helix' for CrickHelix.")
        if mask["helix"] is None:
            raise ValueError("Mask for 'helix' cannot be None.")
        return True

    @classmethod
    def from_structure(cls, *, structure, mask: dict[str, np.ndarray]):
        res_obj = super().from_structure(structure=structure)
        cls._validate_helix_mask(mask)
        res_obj.mask = mask
        return res_obj

    @classmethod
    def from_param(
        cls,
        residue_num: int = 7,
        centroid: Iterable[float] = (0, 0, 0),
        direction: Iterable[float] = (0, 0, 1),
        radius: float = 2.26,
        omega: float = 4 * np.pi / 7,
        pitch_angle: float = 0.876,
        phi0: float = 0.0,
        backbone_type: str = "Gly",
    ):
        """
        Parameters
        ----------
        backbone_type : str
            - "CA": only CA atoms
            - "Gly": Glycine backbone atoms (N, CA, C, O)
            - "Ala": Alanine backbone atoms (N, CA, C, O, CB)
        """
        helix = cls()
        xyz, param = generate_helix_ca_by_crick(
            residue_num=residue_num,
            centroid=centroid,
            direction=direction,
            radius=radius,
            omega=omega,
            pitch_angle=pitch_angle,
            phi0=phi0,
        )
        if backbone_type == "CA":
            structure = ca_xyz_to_atom_array(xyz)
        elif backbone_type == "Gly":
            structure = ca_xyz_to_atom_array(xyz)
            structure = pulchra_fix_backbone(structure)
        else:
            raise ValueError(f"Unsupported backbone_type: {backbone_type}")
        helix.structure = structure
        helix.param = param
        return helix


@dataclass
class CrickHelixOperation(CrickHelixProperty):

    def fit(self, verbose: bool = False):
        """
        Fit observed CA coordinates to a single Crick helix model.

        Workflow
        --------
        1. Extract CA atoms from the masked helix structure.
        2. Run non-linear fitting (`fit_helix_by_crick`) to obtain helix parameters.
        3. Build a local orthonormal frame (`x`, `y`, `z`) for downstream transforms.
        4. Store RMSD and fitted CA-only structure for diagnostics/visualization.
        5. Infer a discrete helix type label from fitted omega.
        """

        def _log(message: str):
            if verbose:
                print(f"[CrickHelix.fit] {message}")

        _log("Preparing CA coordinates from helix mask")
        atom_array = self["helix"]
        ca_mask = atom_array.atom_name == "CA"
        ca_atoms = atom_array[ca_mask]
        ca_coord = ca_atoms.coord

        # Fit Crick parameters from observed CA trace.
        _log(f"Running Crick fitting on {ca_coord.shape[0]} CA atoms")
        param, rmsd, fitted_coord = fit_helix_by_crick(ca_coord, verbose=verbose)
        self.param = param

        # Construct right-handed local axes with z as helix direction.
        _log("Constructing local orthonormal frame")
        z = self.param["direction"]
        x_prototype = ca_atoms.coord[0] - self.param["centroid"]
        y = np.cross(z, x_prototype)
        y /= np.linalg.norm(y)
        x = np.cross(y, z)
        self.extra_param["x"] = x
        self.extra_param["y"] = y
        self.extra_param["z"] = z
        self.rmsd = rmsd

        # Keep a fitted CA-only AtomArray aligned with source chain/residue metadata.
        fitted_structure = bt_struct.AtomArray(length=ca_coord.shape[0])
        fitted_structure.atom_name = np.array(["CA"] * ca_coord.shape[0])
        fitted_structure.element = np.array(["C"] * ca_coord.shape[0])
        fitted_structure.chain_id = ca_atoms.chain_id
        fitted_structure.res_id = ca_atoms.res_id
        fitted_structure.res_name = ca_atoms.res_name
        fitted_structure.coord = fitted_coord
        self.fitted_structure = fitted_structure
        self.extra_param["helix_type"] = self.calculate_helix_type(self.param["omega"])
        _log(f"Completed fit, RMSD={self.rmsd:.4f}")

    def modify(self, method, *args, **kwargs):
        """

        Methods
        -------
        elongate_with_gly : elongate the helix by adding glycine residues
            - length : int, number of residues to add
            - terminus : str, "N" for N-terminus, "C" for C-terminus, "B" for both termini
        """
        if method == "elongate_with_gly":
            length = kwargs.get("length", 1)
            terminus = kwargs.get("terminus", "C")
            self._modify_elongate_with_gly(length, terminus)
        else:
            raise ValueError(f"Unsupported modification method: {method}")

    def _modify_elongate_with_gly(self, length: int, terminus: str = "C"):
        """
        Add glycine residues to the helix to elongate it.

        Parameters
        ----------
        length : int
            Number of residues to add.
        terminus : str
            "N" for N-terminus, "C" for C-terminus, "B" for both termini.
        """
        assert (
            isinstance(length, int) and length > 0
        ), "Length must be a positive integer."

        kwargs = self.param.copy()
        kwargs["residue_num"] += length * 2

        helix_ca, _ = generate_helix_ca_by_crick(**kwargs)
        N_ca = helix_ca[:length]
        C_ca = helix_ca[-length:]
        n_terminal_res_id = min(self.structure.res_id)
        c_terminal_res_id = max(self.structure.res_id)
        if not terminus in ["N", "C", "B"]:
            raise ValueError(f"Unsupported terminus: {terminus}")
        if terminus in ["N", "B"]:
            new_structure = ca_xyz_to_atom_array(
                N_ca, chain_id_i=self.structure.chain_id[0]
            )
            n_terminal_res_id = min(self.structure.res_id)
            new_structure.res_id += n_terminal_res_id - 1 - length
            self.structure = bt_struct.concatenate([new_structure, self.structure])
        if terminus in ["C", "B"]:
            new_structure = ca_xyz_to_atom_array(
                C_ca, chain_id_i=self.structure.chain_id[0]
            )
            c_terminal_res_id = max(self.structure.res_id)
            new_structure.res_id += c_terminal_res_id
            self.structure = bt_struct.concatenate([self.structure, new_structure])
        self.structure = pulchra_fix_backbone(self.structure)


@dataclass
class CrickHelix(CrickHelixIO, CrickHelixOperation):
    """
    A class representing a part of an assembly that is a pure helix.
    It inherits from AssemblyPart and provides specific functionality for helix parts.
    """


@dataclass
class CCCPHelixBundleProperty(AssemblyPartParaRef):

    mask: dict[str, np.ndarray] = field(
        default_factory=lambda: {f"helix_{i+1}": None for i in range(2)}
    )
    component: dict[str, CrickHelix] = field(
        default_factory=lambda: {f"helix_{i+1}": CrickHelix() for i in range(2)}
    )
    param: dict = field(
        default_factory=lambda: {
            "helix_num": None,
            "residue_num": None,
            "senses": None,
            "centroid": None,
            "y_prototype": None,
            "z": None,
            "r0": None,
            "w0": None,
            "phi0": None,
            "r1s": None,
            "w1s": None,
            "phi1s": None,
            "pitch_angles": None,
            "dphi0s": None,
            "z_offsets": None,
        }
    )

    def update_component(self):
        for i in range(self.helix_num):
            key = f"helix_{i+1}"
            self.component[key] = CrickHelix.from_structure(
                structure=self[key], mask={"helix": self.mask[key]}
            )

    @property
    def centroid(self):
        if self._centroid is None:
            self.fit()
        return self._centroid

    @property
    def xyz(self):
        if self._xyz is None:
            self.fit()
        return self._xyz

    @property
    def helix_num(self):
        if "helix_num" in self.param and self.param["helix_num"] is not None:
            pass
        else:
            counter = 0
            for key in self.mask:
                if re.match(r"helix_\d+", key):
                    counter += 1
            self.param["helix_num"] = counter
        return self.param["helix_num"]

    @helix_num.setter
    def helix_num(self, value):
        self.param["helix_num"] = value


class CCCPHelixBundleIO(CCCPHelixBundleProperty):

    @staticmethod
    def _validate_helix_keys(mask: dict[str, np.ndarray]):
        helix_keys = [key for key in mask if re.match(r"helix_\d+", key)]
        if not helix_keys:
            raise ValueError(
                "No valid helix keys found in mask. Expected keys like 'helix_1', 'helix_2', etc."
            )
        helix_nums = sorted([int(key.split("_")[1]) for key in helix_keys])
        if helix_nums != list(range(1, max(helix_nums) + 1)):
            raise ValueError(
                f"Helix keys must be consecutive and start from 1. Found helix numbers: {helix_nums}"
            )
        return len(helix_keys)

    @classmethod
    def from_structure(
        cls, structure: bt_struct.AtomArray, mask: dict[str, np.ndarray]
    ):
        res_obj = super().from_structure(structure=structure)
        helix_num = cls._validate_helix_keys(mask)
        res_obj.mask = mask
        res_obj.helix_num = helix_num
        res_obj.update_component()
        return res_obj

    @classmethod
    def from_helix_num(cls, helix_num: int):
        res_obj = cls()
        res_obj.param["helix_num"] = helix_num
        for i in range(helix_num):
            key = f"helix_{i+1}"
            res_obj.mask[key] = None
            res_obj.component[key] = CrickHelix()
        return res_obj

    @classmethod
    def from_mask(cls, structure: bt_struct.AtomArray, mask: dict[str, np.ndarray]):
        helix_num = 0
        for key in mask:
            if re.match(r"helix_\d+", key):
                helix_num += 1
            else:
                raise ValueError(
                    f"Invalid key in mask: {key}. Expected format: 'helix_<number>'"
                )
        for i in range(helix_num):
            expected_key = f"helix_{i+1}"
            if expected_key not in mask:
                raise ValueError(
                    f"Missing expected key in mask: {expected_key}. Keys must be consecutive."
                )
        res_obj = cls(structure=structure)
        res_obj.mask = mask
        res_obj.component = {
            key: CrickHelix(
                structure=structure[mask[key]], mask={"helix": mask[key][mask[key]]}
            )
            for key in mask
        }
        return res_obj

    @classmethod
    def from_param(
        cls,
        helix_num: int = 2,
        residue_num: int = 7,
        senses: Iterable[int] = None,  # Sense of each helix
        centroid: Iterable[float] = [0.0, 0.0, 0.0],
        y_prototype: Iterable[float] = [0.0, 1.0, 0.0],
        z: Iterable[float] = [0.0, 0.0, 1.0],
        r0: float = 5.0,  # Radius of the coiled bundle
        w0: float = -2 * np.pi / 100,  # Frequency of the coiled bundle
        phi0: float = 0.0,  # Phase shift of the coiled bundle
        r1s: Iterable[float] | float = 2.26,  # Radius of each helix
        w1s: Iterable[float] | float = 4 * np.pi / 7,  # Frequency of each helix
        phi1s: Iterable[float] | float = -np.pi / 20,  # Phase shift of each helix
        pitch_angles: Iterable[float] | float = -0.2096,  # Pitch angle of each helix
        dphi0s: Iterable[float] = None,  # Separation angle between helices
        z_offsets: Iterable[float] | float = 0.0,  # Z-offsets for each helix
        backbone_type: str = "Gly",
    ):
        res_obj = cls.from_helix_num(helix_num)
        xyz, param = generate_cc_ca_by_cccp(
            helix_num=helix_num,
            residue_num=residue_num,
            senses=senses,
            centroid=centroid,
            y_prototype=y_prototype,
            z=z,
            r0=r0,
            w0=w0,
            phi0=phi0,
            r1s=r1s,
            w1s=w1s,
            phi1s=phi1s,
            pitch_angles=pitch_angles,
            dphi0s=dphi0s,
            z_offsets=z_offsets,
        )
        if backbone_type == "CA":
            structure = ca_xyz_to_atom_array(xyz)
        elif backbone_type == "Gly":
            structure = ca_xyz_to_atom_array(xyz)
            structure = pulchra_fix_backbone(structure)
        else:
            raise ValueError(f"Unsupported backbone_type: {backbone_type}")
        res_obj.structure = structure
        return res_obj


@dataclass
class CCCPHelixBundleOperation(CCCPHelixBundleProperty):

    def fit(self, verbose: bool = False):
        """
        Fit a multi-helix bundle to the CCCP parameterization.

        Assumptions
        -----------
        - Every helix in the bundle contributes CA atoms only.
        - All helices must have identical CA length for joint fitting.

        Workflow
        --------
        1. Validate per-helix CA lengths and set inferred initial dimensions.
        2. Assemble observed coordinates into shape `(helix_num, residue_num, 3)`.
        3. Run `fit_cc_by_cccp` to estimate bundle-level parameters.
        4. Build a bundle local frame from fitted `z` and `y_prototype`.
        5. Save fitted coordinates as a synthetic CA-only structure for inspection.
        """

        def _log(message: str):
            if verbose:
                print(f"[CCCPHelixBundle.fit] {message}")

        _log(f"Validating CA lengths for {self.helix_num} helices")
        helix_lens = []
        for i in range(self.helix_num):
            key = f"helix_{i+1}"
            helix = self[key]
            helix_lens.append(np.sum(helix.atom_name == "CA"))

        # CCCP fitting requires all helices to have the same residue count.
        assert (
            len(set(helix_lens)) == 1
        ), f"All helices must have the same length to fit a CCCP model. Current lengths: {helix_lens}"
        self.initial_param["helix_num"] = len(helix_lens)
        self.initial_param["residue_num"] = helix_lens[0]

        _log(
            f"Collecting observed CA coordinates (helix_num={self.initial_param['helix_num']}, "
            f"residue_num={self.initial_param['residue_num']})"
        )
        ca_coord_obs = np.zeros(
            shape=(self.initial_param["helix_num"], helix_lens[0], 3)
        )
        for i in range(self.initial_param["helix_num"]):
            key = f"helix_{i+1}"
            helix = self[key]
            ca_mask = helix.atom_name == "CA"
            ca_atoms = helix[ca_mask]
            ca_coord_obs[i] = ca_atoms.coord
            helix_component: CrickHelix = self.component[key]
            helix_component.structure = helix
            # helix_component.fit()

        # Jointly fit all helices into a single CCCP bundle model.
        _log("Running staged CCCP bundle optimization")
        param, rmsd, ca_coord_fitted = fit_cc_by_cccp(
            ca_coord_obs,
            params_not_to_fit=self.params_not_to_fit,
            verbose=verbose,
            **self.initial_param,
        )

        self.param = param
        z = param["z"]
        y_prototype = param["y_prototype"]

        # Build a right-handed orthonormal frame for the fitted helix bundle.
        # - z is the fitted bundle axis returned by CCCP.
        # - y_prototype is an in-plane reference vector used to fix the
        #   rotation around z; in the generator it points from the bundle
        #   centroid toward helix_1.
        # - x is defined as y_prototype x z, so it is perpendicular to the
        #   bundle axis and the reference radial direction.
        # - y is then reconstructed as z x x, yielding the final orthonormal
        #   basis (x, y, z) used by self.xyz.
        _log("Building fitted bundle local frame")
        x = np.cross(y_prototype, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        self.extra_param["x"] = x
        self.extra_param["y"] = y
        self.extra_param["z"] = z

        self.rmsd = rmsd
        ca_coord_fitted = np.reshape(
            ca_coord_fitted,
            shape=(ca_coord_fitted.shape[0] * ca_coord_fitted.shape[1], 3),
        )
        array_length = ca_coord_fitted.shape[0]

        # Materialize fitted CA coordinates as a synthetic AtomArray.
        fitted_structure = bt_struct.AtomArray(length=array_length)
        fitted_structure.res_name = np.array(["GLY"] * array_length)
        fitted_structure.element = np.array(["C"] * array_length)
        fitted_structure.atom_name = np.array(["CA"] * array_length)
        fitted_structure.chain_id = np.array(
            [chr(ord("A") + i // helix_lens[0]) for i in range(array_length)]
        )
        fitted_structure.res_id = np.array(
            list(range(1, helix_lens[0] + 1)) * self.helix_num
        )
        fitted_structure.coord = ca_coord_fitted
        self.fitted_structure = fitted_structure

        self._xyz = np.vstack(
            (self.extra_param["x"], self.extra_param["y"], self.extra_param["z"])
        )
        self._centroid = self.param["centroid"]
        _log(f"Completed fit, RMSD={self.rmsd:.4f}")


@dataclass
class CCCPHelixBundle(CCCPHelixBundleIO, CCCPHelixBundleOperation):
    """
    A class representing a part of an assembly that is a pure helix.
    It inherits from AssemblyPart and provides specific functionality for helix parts.

    mask: dict[str, np.ndarray] = field(default_factory=lambda: {})
        Like {"helix_1": None, "helix_2": None, ...}
    """
