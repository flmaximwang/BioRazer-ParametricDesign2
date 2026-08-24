"""
Geometric utility functions for CCCP package.
"""

import numpy as np


def dihe(p1, p2, p3, p4):
    """
    Calculate dihedral angle given four points.

    Parameters
    ----------
    p1, p2, p3, p4 : array_like
        3D coordinates of four points

    Returns
    -------
    float
        Dihedral angle in radians
    """
    p1, p2, p3, p4 = np.array(p1), np.array(p2), np.array(p3), np.array(p4)

    v12 = p1 - p2
    v23 = p2 - p3
    v43 = p4 - p3

    px1 = np.cross(v12, v23)
    px1 = px1 / np.linalg.norm(px1)

    px2 = np.cross(v43, v23)
    px2 = px2 / np.linalg.norm(px2)

    dp12 = np.dot(px1, px2)
    sin2 = 1 - dp12**2

    d = np.pi / 2.0 - np.arctan(dp12 / np.sqrt(sin2))

    px3 = np.cross(px1, px2)
    if np.dot(px3, v23) > 0:
        d = -d

    return d


def superimpose(X, Y):
    """
    Superimpose two sets of 3D coordinates using SVD method.

    Implements the SVD method by Kabsh et al. (Acta Crystallogr 1976, A32, 922)
    and taken from Coutsias et al. J Comp Chem 2004, 25(15) 1849.

    Parameters
    ----------
    X, Y : array_like
        3×N matrices of coordinates

    Returns
    -------
    rmsd : float
        Root mean square deviation
    M : ndarray
        Optimal rotation matrix
    residuals : ndarray, optional
        Array of residual distances
    """
    X, Y = np.array(X), np.array(Y)

    if X.shape[0] != 3 or Y.shape[0] != 3:
        raise ValueError("Matrices X and Y must be 3×N")
    if X.shape != Y.shape:
        raise ValueError("Matrices X and Y must be of the same size")

    N = X.shape[1]

    # Center the coordinates
    X = X - np.mean(X, axis=1, keepdims=True)
    Y = Y - np.mean(Y, axis=1, keepdims=True)

    R = X @ Y.T
    U, S, Vt = np.linalg.svd(R)

    I = np.eye(3)
    sgn = np.sign(np.linalg.det(R))
    I[2, 2] = sgn

    M = Vt.T @ I @ U.T
    rmsd = np.sqrt((np.sum(X**2) + np.sum(Y**2) - 2 * (S[0] + S[1] + sgn * S[2])) / N)

    residuals = np.linalg.norm(Y - M @ X, axis=0)

    return rmsd, M, residuals


def crossing_angle(A, B, pap):
    """
    Compute crossing angle between two helical chains.

    Parameters
    ----------
    A, B : array_like
        N×3 matrices of coordinates for two chains
    pap : int
        Parallel (1) or antiparallel (0) orientation

    Returns
    -------
    float
        Crossing angle in radians
    """
    A, B = np.array(A), np.array(B)

    if A.shape[1] != 3 or B.shape[1] != 3:
        raise ValueError("A and B must be N×3 matrices")

    if pap == 0:
        B = B[::-1, :]  # reverse for antiparallel

    if min(A.shape[0], B.shape[0]) < 3:
        return 0.0

    # Find helical axes
    axsA = helical_axis_points(A)
    axsB = helical_axis_points(B)

    if len(axsA) == 0 or len(axsB) == 0:
        return 0.0

    return -dihe(axsA[0], axsA[-1], axsB[-1], axsB[0])


def helical_axis_points(H):
    """
    Compute helical axis points for a helix.

    Parameters
    ----------
    H : array_like
        N×3 matrix of helix coordinates

    Returns
    -------
    ndarray
        (N-2)×3 matrix of axis points
    """
    H = np.array(H)
    if H.shape[0] < 3:
        return np.array([])

    axs = []
    for i in range(1, H.shape[0] - 1):
        r = (H[i - 1] - H[i]) + (H[i + 1] - H[i])
        r = 2.26 * r / np.linalg.norm(r)
        axs.append(2.26 * r / np.linalg.norm(r) + H[i])

    return np.array(axs)


def crick_eq(r0, r1, w0, w1, a, ph0, ph1, t):
    """
    Crick coiled-coil equations.

    Parameters
    ----------
    r0, r1 : float
        Superhelical and helical radii
    w0, w1 : float
        Superhelical and helical frequencies
    a : float
        Pitch angle
    ph0, ph1 : float
        Superhelical and helical phase offsets
    t : array_like
        Parametric values

    Returns
    -------
    x, y, z : ndarray
        3D coordinates
    """
    t = np.array(t)

    x = (
        r0 * np.cos(w0 * t + ph0)
        + r1 * np.cos(w0 * t + ph0) * np.cos(w1 * t + ph1)
        - r1 * np.cos(a) * np.sin(w0 * t + ph0) * np.sin(w1 * t + ph1)
    )

    y = (
        r0 * np.sin(w0 * t + ph0)
        + r1 * np.sin(w0 * t + ph0) * np.cos(w1 * t + ph1)
        + r1 * np.cos(a) * np.cos(w0 * t + ph0) * np.sin(w1 * t + ph1)
    )

    z = w0 * t * r0 / np.tan(a) - r1 * np.sin(a) * np.sin(w1 * t + ph1)

    return x, y, z


def absolute_to_register_zoff(zoff, R0, w0, a, w1, ph1_1, ph1_2, p_ap):
    """
    Convert from absolute Z offset to register Z offset.

    Parameters
    ----------
    zoff : float
        Absolute Z offset
    R0 : float
        Superhelical radius
    w0 : float
        Superhelical frequency
    a : float
        Pitch angle
    w1 : float
        Helical frequency
    ph1_1, ph1_2 : float
        Helical phases for chains 1 and 2
    p_ap : int
        Parallel (1) or antiparallel (0)

    Returns
    -------
    float
        Register Z offset
    """
    assert p_ap in [0, 1], "p_ap flag expected to be either 0 or 1"

    aa1 = 2 * np.pi / w1
    b1 = (np.pi - ph1_1) / w1
    z1 = (R0 * w0 / np.tan(a)) * (aa1 * 1 + b1)

    if p_ap == 0:
        # for antiparallel chain
        aa2 = 2 * np.pi / (-w1)
        b2 = (np.pi + ph1_2) / (-w1)
        n = ((z1 - zoff) / (-R0 * w0 / np.tan(a)) - b2) / aa2
        dz = -(R0 * w0 / np.tan(a)) * (aa2 * np.floor(n) + b2) + zoff - z1
        dz1 = -(R0 * w0 / np.tan(a)) * (aa2 * np.ceil(n) + b2) + zoff - z1
    else:
        b2 = (np.pi - ph1_2) / w1
        n = ((z1 - zoff) / (R0 * w0 / np.tan(a)) - b2) / aa1
        dz = (R0 * w0 / np.tan(a)) * (aa1 * np.floor(n) + b2) + zoff - z1
        dz1 = (R0 * w0 / np.tan(a)) * (aa1 * np.ceil(n) + b2) + zoff - z1

    return dz1 if abs(dz1) < abs(dz) else dz


def absolute_to_zoff_aa(zoff, R0, w0, a, R1, w1, ph1_1, ph1_2, p_ap):
    """
    Convert from absolute Z offset to Zaa' Z offset.

    Parameters
    ----------
    zoff : float
        Absolute Z offset
    R0, R1 : float
        Superhelical and helical radii
    w0, w1 : float
        Superhelical and helical frequencies
    a : float
        Pitch angle
    ph1_1, ph1_2 : float
        Helical phases for chains 1 and 2
    p_ap : int
        Parallel (1) or antiparallel (0)

    Returns
    -------
    float
        Zaa' Z offset
    """
    from .math import canonical_phases, angle_diff, get_heptad_pos

    assert p_ap in [0, 1], "p_ap flag expected to be either 0 or 1"

    # find first a-positions on chain 1
    rng = np.arange(7)
    phase_diffs = np.abs(
        angle_diff(ph1_1 + w1 * rng, canonical_phases(1) * np.pi / 180)
    )
    mi = np.argmin(phase_diffs)
    aph1_1 = np.mod(ph1_1 + w1 * rng[mi], 2 * np.pi)
    az1 = w0 * rng[mi] * R0 / np.tan(a) - R1 * np.sin(a) * np.sin(w1 * rng[mi] + ph1_1)

    # find closest 'a' position on chain 2
    if p_ap == 0:
        n = round((zoff - az1) * np.tan(a) / w0 / R0)
    else:
        n = round((az1 - zoff) * np.tan(a) / w0 / R0)

    sgn = np.nan
    zaa = np.inf

    for count in range(100):
        for ni in [n - count, n + count]:
            aph1_2 = ph1_2 + w1 * ni
            if get_heptad_pos(aph1_2, as_int=True) != 1:
                continue

            if p_ap == 0:
                az2 = zoff - (
                    w0 * ni * R0 / np.tan(a) - R1 * np.sin(a) * np.sin(w1 * ni + ph1_2)
                )
            else:
                az2 = (
                    zoff
                    + w0 * ni * R0 / np.tan(a)
                    - R1 * np.sin(a) * np.sin(w1 * ni + ph1_2)
                )

            if abs(zaa) > abs(az2 - az1):
                zaa = az2 - az1

            if np.isnan(sgn) and np.sign(az2 - az1) != 0:
                sgn = np.sign(az2 - az1)

            if sgn * np.sign(az2 - az1) <= 0:
                return zaa

    return zaa
