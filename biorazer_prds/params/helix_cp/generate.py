from typing import Iterable
import numpy as np


def generate_helix_ca_by_crick(
    residue_num: int = 7,
    centroid: Iterable[float] = (0.0, 0.0, 0.0),
    direction: Iterable[float] = (0.0, 0.0, 1.0),
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
    centroid = np.array(centroid, dtype=float)
    direction = np.array(direction, dtype=float)

    z_base = direction
    z_base /= np.linalg.norm(z_base)
    y_base = np.cross(z_base, np.array([1.0, 0.0, 0.0]))
    y_base /= np.linalg.norm(y_base)
    x_base = np.cross(y_base, z_base)

    residue_t = np.arange(0.5 - residue_num / 2, 0.5 + residue_num / 2, 1)
    angle = omega * residue_t + phi0
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    z = radius * angle * np.tan(pitch_angle)
    xyz: np.ndarray = (
        x[:, np.newaxis] * x_base
        + y[:, np.newaxis] * y_base
        + z[:, np.newaxis] * z_base
    )
    xyz += centroid  # Translate back to the original centroid position

    params = dict(
        residue_num=residue_num,
        centroid=centroid,
        direction=direction,
        radius=radius,
        omega=omega,
        pitch_angle=pitch_angle,
        phi0=phi0,
    )

    return xyz, params
