import numpy as np
from typing import Iterable
from scipy.spatial.transform import Rotation as R
from .geometry import crick_eq


def generate_helix_ca_by_crick(
    residue_num: int = 7,
    centroid: Iterable[float] = (0, 0, 0),
    direction: Iterable[float] = (0, 0, 1),
    radius: float = 2.26,
    omega: float = 4 * np.pi / 7,
    pitch_angle: float = 0.876,
    phi0: float = 0.0,
):
    """
    Generate a straight helix of CA atoms in a Crick-like configuration.

    Parameters:
    -----------------
    - centroid: np.ndarray - The centroid of the helix.
    - direction: np.ndarray - The direction vector of the helix (z-axis).
    - residue_num: int - The number of residues in the helix.
    - radius: float - The radius of the helix.
    - omega: float - The angle between adjacent residues in radians.
    - pitch angle: float - The angle of the helix pitch in radians.
    - phi: float - The phase shift of the helix. If phi=0, then CA 1 is on x axis.

    Returns:
    -----------------
    - xyz: np.ndarray - The coordinates of the CA atoms in the helix.

    """

    xyz = np.zeros((residue_num, 3))
    z_base = np.array(direction)
    z_base /= np.linalg.norm(z_base)
    y_base = np.cross(z_base, np.array([1.0, 0.0, 0.0]))
    y_base /= np.linalg.norm(y_base)
    x_base = np.cross(y_base, z_base)
    for residue_i, residue_t in enumerate(
        np.arange(0.5 - residue_num / 2, 0.5 + residue_num / 2)
    ):
        angle = omega * residue_t + phi0
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = radius * angle * np.tan(pitch_angle)
        xyz[residue_i] = x * x_base + y * y_base + z * z_base
    xyz += centroid  # Translate back to the original centroid position
    params = dict(
        residue_num=residue_num,
        centroid=np.array(centroid),
        direction=np.array(direction),
        radius=radius,
        omega=omega,
        pitch_angle=pitch_angle,
        phi0=0.0,
    )

    return xyz, params


def generate_cc_ca_by_crick(
    helix_num: int = 2,
    residue_num: int = 7,
    senses: Iterable[int] = None,  # Sense of each helix
    centroid: Iterable[float] = [0.0, 0.0, 0.0],
    direction: Iterable[float] = [0.0, 0.0, 1.0],
    r0: float = 5.0,  # Radius of the coiled bundle
    w0: float = -2 * np.pi / 100,  # Frequency of the coiled bundle
    phi0: float = 0.0,  # Phase shift of the coiled bundle
    r1s: Iterable[float] | float = 2.26,  # Radius of each helix
    w1s: Iterable[float] | float = 4 * np.pi / 7,  # Frequency of each helix
    phi1s: Iterable[float] | float = np.pi / 20,  # Phase shift of each helix
    pitch_angles: Iterable[float] | float = -0.2096,  # Pitch angle of each helix
    dphi0s: Iterable[float] = None,  # Separation angle between helices
    z_offsets: Iterable[float] | float = 0.0,  # Z-offsets for each helix
):
    """
    Generate a coiled-coil helix structure based on the Coiled-Coil Crick model.

    Parameter:
    -----------------
    - residue_num: The number of residues in each helix.
    - helix_num: The number of helices in the coiled-coil structure.
    - centroid: The centroid of the coiled-coil structure.
    - direction: The direction vector of the coiled-coil structure.
    - senses: The sense of each helix (optional).
        - If None, each helix pair will have opposite senses.
    - r0: The radius of the coiled bundle.
    - w0: The frequency of the coiled bundle.
    - a0: The pitch angle of the coiled bundle (angle between the helix axis and the coiled bundle axis).
    - phi0: The phase shift of the coiled bundle
    - r1: The radius of each helix.
        - If a single float is provided, all helices will have the same radius.
        - If an iterable is provided, it should have length helix_num, representing the radius of each helix.
    - w1: The frequency of each helix.
        - If a single float is provided, all helices will have the same frequency.
        - If an iterable is provided, it should have length helix_num, representing the frequency of each helix.
    - dphi1: The phase shift of each helix.
        - If a single float is provided, all helices will have the same phase shift.
        - If an iterable is provided, it should have length helix_num, representing the phase shift of each helix.
    - pitch_angles: The pitch angle of each helix (optional), which is the angle between the helix axis and the coiled bundle axis.
        - If a single float is provided, all helices will have the same pitch angle.
        - If an iterable is provided, it should have length helix_num, representing the pitch angle of each helix.
    - z_offsets: The z-offsets for each helix (optional). This is the vertical offset in terms of the axis of the coiled bundle.
        - If a single float is provided, all helices will have the same z-offset.
        - If an iterable is provided, it should have length helix_num, representing the z-offset of each helix.
    - dphi0s: The beginning angle for each helix (optional).
        - If None, helices are evenly spaced.
        - If set, it should have length helix_num, representing the angles of each helix compared to phi0.

    Returns:
    -----------------
    - xyz: np.ndarray - The coordinates of the CA atoms in the coiled-coil structure
        - shape: (helix_num, residue_num, 3)
    """

    xyz = np.zeros((helix_num, residue_num, 3))
    centroid = np.array(centroid, dtype=float)
    direction = np.array(direction, dtype=float)
    if dphi0s is None:
        dphi0s = np.linspace(0, 2 * np.pi, helix_num, endpoint=False)
    if senses is None:
        senses = np.array([1 if i % 2 == 0 else -1 for i in range(helix_num)])
    else:
        senses = np.array(senses)
    if isinstance(r1s, (int, float)):
        r1s = np.full(helix_num, r1s, dtype=float)
    if isinstance(w1s, (int, float)):
        w1s = np.full(helix_num, w1s, dtype=float)
    if isinstance(phi1s, (int, float)):
        phi1s = np.full(helix_num, phi1s, dtype=float)
    if isinstance(pitch_angles, (int, float)):
        pitch_angles = np.full(helix_num, pitch_angles, dtype=float)
    if isinstance(z_offsets, (int, float)):
        z_offsets = np.full(helix_num, z_offsets, dtype=float)
    if isinstance(dphi0s, (int, float)):
        dphi0s = np.full(helix_num, dphi0s, dtype=float)
    assert len(r1s) == helix_num, "Length of r1 must match helix_num"
    assert len(w1s) == helix_num, "Length of w1 must match helix_num"
    assert len(pitch_angles) == helix_num, "Length of pitch_angles must match helix_num"
    assert len(z_offsets) == helix_num, "Length of z_offsets must match helix_num"
    assert len(dphi0s) == helix_num, "Length of dphi0s must match helix_num"

    z_base = np.array(direction)
    z_base /= np.linalg.norm(z_base)
    y_base = np.cross(z_base, np.array([1.0, 0.0, 0.0]))
    y_base /= np.linalg.norm(y_base)
    x_base = np.cross(y_base, z_base)

    for helix_i in range(helix_num):
        for residue_i, residue_t in enumerate(
            np.arange(0.5 - residue_num / 2, 0.5 + residue_num / 2)
        ):
            angle_0 = (
                w0 * residue_t * senses[helix_i]
                + phi0
                + dphi0s[helix_i]
                + z_offsets[helix_i]
                * senses[helix_i]
                * np.tan(pitch_angles[helix_i])
                / r0
            )
            angle_1 = w1s[helix_i] * residue_t + phi1s[helix_i]
            angle_1 *= senses[helix_i]
            x = (
                r0 * np.cos(angle_0)
                + r1s[helix_i] * np.cos(angle_0) * np.cos(angle_1)
                - r1s[helix_i]
                * np.cos(pitch_angles[helix_i])
                * np.sin(angle_0)
                * np.sin(angle_1)
            )
            y = (
                r0 * np.sin(angle_0)
                + r1s[helix_i] * np.sin(angle_0) * np.cos(angle_1)
                + r1s[helix_i]
                * np.cos(pitch_angles[helix_i])
                * np.cos(angle_0)
                * np.sin(angle_1)
            )
            z = (
                r0 * w0 * residue_t * senses[helix_i] / np.tan(pitch_angles[helix_i])
                - r1s[helix_i] * np.sin(angle_1) * np.sin(pitch_angles[helix_i])
                + z_offsets[helix_i] * senses[helix_i]
            )
            xyz_ii = x * x_base + y * y_base + z * z_base
            xyz[helix_i, residue_i] = xyz_ii
    c_centroid = np.mean(xyz.reshape(-1, 3), axis=0)
    xyz -= c_centroid  # Translate to origin
    xyz += centroid  # Translate back to the original centroid position
    params = dict(
        helix_num=helix_num,
        residue_num=residue_num,
        senses=senses,
        centroid=centroid,
        direction=direction,
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

    return xyz, params
