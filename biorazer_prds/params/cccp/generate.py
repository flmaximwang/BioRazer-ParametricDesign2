import numpy as np
from typing import Iterable
from numbers import Number
from scipy.spatial.transform import Rotation as R
from .geometry import crick_eq


def generate_cc_ca_by_cccp(
    helix_num: int = 2,
    residue_num: int = 7,
    senses: Iterable[int] = None,  # Sense of each helix
    centroid: Iterable[float] = [0.0, 0.0, 0.0],
    y_prototype: Iterable[float] = [0.0, 1.0, 0.0],
    z: Iterable[float] = [0.0, 0.0, 1.0],
    r0: float = 5.0,  # Radius of the coiled bundle
    w0: float = -2 * np.pi / 100,  # Frequency of the coiled bundle
    phi0: float = 0.0,  # Phase shift of the coiled bundle
    dphi0s: Iterable[float] = None,  # Separation angle between helices
    r1s: Iterable[float] | float = 2.26,  # Radius of each helix
    w1s: Iterable[float] | float = 4 * np.pi / 7,  # Frequency of each helix
    phi1s: Iterable[float] | float = np.pi / 20,  # Phase shift of each helix
    pitch_angles: Iterable[float] | float = -0.2096,  # Pitch angle of each helix
    z_offsets: Iterable[float] | float = 0.0,  # Z-offsets for each helix
):
    """
    Generate a coiled-coil helix structure based on the Coiled-Coil Crick model.

    Parameter:
    -----------------
    - residue_num: The number of residues in each helix.
    - helix_num: The number of helices in the coiled-coil structure.
    - centroid: The centroid of the coiled-coil structure.
    - x: The vector from centroid of the bundle to the first helix (optional, only for validation).
    - y: The vector from centroid of the bundle to the first helix.
    - z: The direction vector of the coiled-coil structure.
    - senses: The sense of each helix (optional).
        - If None, each helix pair will have opposite senses.
    - r0: The radius of the coiled bundle.
    - w0: The frequency of the coiled bundle.
    - a0: The pitch angle of the coiled bundle (angle between the helix axis and the coiled bundle axis).
    - phi0: The phase shift of the coiled bundle
    - dphi0s: The beginning angle for each helix (optional).
        - If None, helices are evenly spaced.
        - If set, it should have length helix_num, representing the angles of each helix compared to phi0.
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
    - z_offsets: The z-offsets for each helix (optional). This is the vertical offset *in terms of the axis of the coiled bundle*, not the helix axis.
        - If a single float is provided, all helices will have the same z-offset.
        - If an iterable is provided, it should have length helix_num, representing the z-offset of each helix.

    Returns:
    -----------------
    - xyz: np.ndarray - The coordinates of the CA atoms in the coiled-coil structure
        - shape: (helix_num, residue_num, 3)
    """

    z = np.array(z, dtype=float)
    z /= np.linalg.norm(z)
    x = np.cross(y_prototype, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)

    coords = np.zeros((helix_num, residue_num, 3))
    centroid = np.array(centroid, dtype=float)
    z = np.array(z, dtype=float)
    if dphi0s is None:
        dphi0s = np.linspace(0, 2 * np.pi, helix_num, endpoint=False)
    assert len(dphi0s) == helix_num, "Length of dphi0s must match helix_num"
    if senses is None:
        senses = np.array([1 if i % 2 == 0 else -1 for i in range(helix_num)])
    else:
        senses = np.array(senses)
    assert len(senses) == helix_num, "Length of senses must match helix_num"
    if isinstance(r1s, Number):
        r1s = np.full(helix_num, r1s, dtype=float)
    assert len(r1s) == helix_num, "Length of r1 must match helix_num"
    if isinstance(w1s, Number):
        w1s = np.full(helix_num, w1s, dtype=float)
    assert len(w1s) == helix_num, "Length of w1 must match helix_num"
    if isinstance(phi1s, Number):
        phi1s = np.full(helix_num, phi1s, dtype=float)
    if isinstance(pitch_angles, Number):
        pitch_angles = np.full(helix_num, pitch_angles, dtype=float)
    assert len(pitch_angles) == helix_num, "Length of pitch_angles must match helix_num"
    if isinstance(z_offsets, Number):
        z_offsets = np.full(helix_num, z_offsets, dtype=float)
    assert len(z_offsets) == helix_num, "Length of z_offsets must match helix_num"
    if isinstance(dphi0s, Number):
        dphi0s = np.full(helix_num, dphi0s, dtype=float)
    assert len(z_offsets) == helix_num, "Length of z_offsets must match helix_num"

    x_base = x
    y_base = y
    z_base = z

    for helix_i in range(helix_num):
        residue_t = np.arange(0.5 - residue_num / 2, 0.5 + residue_num / 2)
        angle_0 = (
            w0 * residue_t * senses[helix_i]
            + phi0
            + dphi0s[helix_i]
            + z_offsets[helix_i] * senses[helix_i] * np.tan(pitch_angles[helix_i]) / r0
        )
        angle_1 = (w1s[helix_i] * residue_t + phi1s[helix_i]) * senses[helix_i]
        xt = (
            r0 * np.cos(angle_0)
            + r1s[helix_i] * np.cos(angle_0) * np.cos(angle_1)
            - r1s[helix_i]
            * np.cos(pitch_angles[helix_i])
            * np.sin(angle_0)
            * np.sin(angle_1)
        )
        yt = (
            r0 * np.sin(angle_0)
            + r1s[helix_i] * np.sin(angle_0) * np.cos(angle_1)
            + r1s[helix_i]
            * np.cos(pitch_angles[helix_i])
            * np.cos(angle_0)
            * np.sin(angle_1)
        )
        zt = (
            r0 * w0 * residue_t * senses[helix_i] / np.tan(pitch_angles[helix_i])
            - r1s[helix_i]
            * np.sin(angle_1)
            * np.sin(pitch_angles[helix_i])  # senses lie in angle_1
            + z_offsets[helix_i] * senses[helix_i]
        )
        coords[helix_i] = (
            xt[:, np.newaxis] * x_base
            + yt[:, np.newaxis] * y_base
            + zt[:, np.newaxis] * z_base
        )

    coords += centroid  # Translate back to the original centroid position
    params = dict(
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

    return coords, params
