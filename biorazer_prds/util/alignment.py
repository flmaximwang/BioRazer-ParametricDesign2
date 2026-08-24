import numpy as np
from scipy.spatial.transform import Rotation as R


def calculate_rotation(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """
    Calculate the rotation that rotate the canonical xyz axes to the current x, y, z axes.
    Returns the angles (a, b, c) in degrees.
    """
    return R.from_matrix(np.column_stack((x, y, z)))


def calculate_euler_ZXZ(x: np.ndarray, y: np.ndarray, z: np.ndarray, degrees=False):
    """Calculate euler angles in ZXZ convention from x, y, z vectors."""
    rotation = calculate_rotation(x, y, z)
    return rotation.as_euler("ZXZ", degrees=degrees)


def fit_plane_norm(points: np.ndarray):
    """
    Fit a plane to a set of points and return the normal vector.

    :param points: An Nx3 array of points.
    :return: A normalized vector representing the normal of the fitted plane.
    """
    assert points.shape[1] == 3, "Points must be in 3D space."
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    _, _, vh = np.linalg.svd(centered_points)
    normal = vh[2]
    return normal / np.linalg.norm(normal)
