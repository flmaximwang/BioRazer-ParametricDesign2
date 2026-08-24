"""
Generate coiled-coil backbone from Crick parameters.
"""

import numpy as np
from .geometry import crick_eq, absolute_to_register_zoff, absolute_to_zoff_aa


def generate_crick_bb(
    chains,
    ch_length,
    cr,
    ph1=-9,
    dph0=180,
    zoff=0,
    alpha=-12.01,
    r0=5,
    r1=2.26,
    w0=-3.6,
    w1=102.857,
    z_type=None,
):
    """
    Generate ideal coiled-coil backbone from Crick parameters.

    Parameters
    ----------
    chains : int
        Number of chains
    ch_length : int
        Length of each chain (number of residues)
    r0, r1 : float
        Superhelical and helical radii (Angstroms)
    w0, w1 : float
        Superhelical and helical frequencies (degrees per residue)
    alpha : float
        Pitch angle (degrees)
    ph1 : float or array_like
        Helical phase angle(s) (degrees)
    cr : array_like
        Chain orientations (length chains-1)
        1 = parallel, 0 = antiparallel to first chain
    dph0 : array_like
        Superhelical phase offsets (degrees, length chains-1)
    zoff : array_like
        Z-offsets (length chains-1)
    z_type : dict, optional
        Z-offset interpretation type:
        - {'zoffaa': True}: offset between 'a' positions
        - {'registerzoff': True}: register offset
        - {'apNNzoff': True}: N-N or N-C offset

    Returns
    -------
    ndarray
        (chains × ch_length) × 3 matrix of coordinates
    """
    # Validate inputs
    if chains <= 0:
        raise ValueError("Number of chains must be positive")
    if ch_length <= 0:
        raise ValueError("Chain length must be positive")

    # Convert degrees to radians
    w0_rad = np.deg2rad(w0)
    w1_rad = np.deg2rad(w1)
    alpha_rad = np.deg2rad(alpha)
    ph1_rad = np.deg2rad(ph1)
    dph0_rad = np.deg2rad(dph0)

    # Ensure proper array shapes
    if np.isscalar(cr):
        cr = [cr]
    if np.isscalar(dph0_rad):
        dph0_rad = [dph0_rad]
    if np.isscalar(zoff):
        zoff = [zoff]

    # Pad arrays to proper length
    if len(cr) == chains - 1:
        cr = [1] + list(cr)  # First chain is always parallel to itself
    if len(dph0_rad) == chains - 1:
        dph0_rad = [0] + list(dph0_rad)  # First chain has zero phase offset
    if len(zoff) == chains - 1:
        zoff = [0] + list(zoff)  # First chain has zero z-offset

    if len(cr) > 0 and cr[0] == 0:
        raise ValueError("The first entry of the orientation vector cannot be 0")

    # Handle helical phases
    if np.isscalar(ph1_rad):
        ph1_rad = ph1_rad * np.ones(chains)
    elif len(ph1_rad) != chains:
        raise ValueError(f"{len(ph1_rad)} helical phases specified for {chains} chains")

    # Initialize coordinate array
    xyz = np.zeros((ch_length * chains, 3))
    t = np.arange(ch_length)

    # Set up z-offset options
    opts = z_type if z_type is not None else {}

    for i in range(chains):
        if cr[i] == 0:  # Antiparallel chain
            x, y, z = crick_eq(r0, r1, -w0_rad, -w1_rad, alpha_rad, 0, -ph1_rad[i], t)

            # Handle different z-offset types for antiparallel chains
            if "apNNzoff" in opts:
                zoff[i] = zoff[i] + xyz[(i - 1) * ch_length, 2] - z[-1]
            elif "registerzoff" in opts:
                zo = xyz[0, 2] - z[-1]  # Bring termini together
                dz = absolute_to_register_zoff(
                    zo, r0, w0_rad, alpha_rad, w1_rad, ph1_rad[0], ph1_rad[i], cr[i]
                )
                zo = zo - dz
                zoff[i] = zoff[i] + zo
            elif "zoffaa" in opts:
                zo = xyz[0, 2] - z[-1]  # Bring termini together
                dz = absolute_to_zoff_aa(
                    zo, r0, w0_rad, alpha_rad, r1, w1_rad, ph1_rad[0], ph1_rad[i], cr[i]
                )
                zo = zo - dz
                zoff[i] = zoff[i] + zo

            # Rotation matrix for antiparallel chains
            angle = dph0_rad[i] - zoff[i] * np.tan(alpha_rad) / r0
            T = np.array(
                [
                    [np.cos(angle), np.sin(angle), 0],
                    [-np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1],
                ]
            )
        else:  # Parallel chain
            x, y, z = crick_eq(r0, r1, w0_rad, w1_rad, alpha_rad, 0, ph1_rad[i], t)

            # Handle different z-offset types for parallel chains
            if "registerzoff" in opts:
                zo = 0  # Start with zero offset
                dz = absolute_to_register_zoff(
                    zo, r0, w0_rad, alpha_rad, w1_rad, ph1_rad[0], ph1_rad[i], cr[i]
                )
                zo = zo - dz
                zoff[i] = zoff[i] + zo
            elif "zoffaa" in opts:
                zo = 0  # Start with zero offset
                dz = absolute_to_zoff_aa(
                    zo, r0, w0_rad, alpha_rad, r1, w1_rad, ph1_rad[0], ph1_rad[i], cr[i]
                )
                zo = zo - dz
                zoff[i] = zoff[i] + zo

            # Rotation matrix for parallel chains
            angle = dph0_rad[i] - zoff[i] * np.tan(alpha_rad) / r0
            T = np.array(
                [
                    [np.cos(angle), np.sin(angle), 0],
                    [-np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1],
                ]
            )

        # Apply transformation
        coords = np.array([x, y, z])
        if len(r0) == ch_length if hasattr(r0, "__len__") else False:
            # Variable radius case (not typically used)
            for k in range(ch_length):
                coords[:, k] = T @ coords[:, k]
        else:
            coords = T @ coords

        # Store coordinates
        start_idx = i * ch_length
        end_idx = (i + 1) * ch_length
        xyz[start_idx:end_idx, 0] = coords[0]
        xyz[start_idx:end_idx, 1] = coords[1]
        xyz[start_idx:end_idx, 2] = coords[2] + zoff[i]

    return xyz


def generate_crick_bb_rad(
    chains, ch_length, r0, r1, w0, w1, alpha, ph1, cr, dph0, zoff, opts=None
):
    """
    Generate coiled-coil backbone with parameters in radians.

    This is the internal function that works with parameters already in radians.

    Parameters
    ----------
    chains : int
        Number of chains
    ch_length : int
        Length of each chain
    r0, r1 : float
        Superhelical and helical radii
    w0, w1 : float
        Superhelical and helical frequencies (radians per residue)
    alpha : float
        Pitch angle (radians)
    ph1 : array_like
        Helical phases (radians)
    cr : array_like
        Chain orientations
    dph0 : array_like
        Superhelical phase offsets (radians)
    zoff : array_like
        Z-offsets
    opts : dict, optional
        Options for z-offset handling

    Returns
    -------
    ndarray
        Coordinate array
    """
    opts = opts if opts is not None else {}

    # Ensure proper array shapes
    if len(cr) == chains - 1:
        cr = np.concatenate([[1], cr])
    if len(dph0) == chains - 1:
        dph0 = np.concatenate([[0], dph0])
    if len(zoff) == chains - 1:
        zoff = np.concatenate([[0], zoff])

    if cr[0] == 0:
        raise ValueError("The first entry of the orientation vector cannot be 0")

    if np.isscalar(ph1):
        ph1 = ph1 * np.ones(chains)
    elif len(ph1) != chains:
        raise ValueError(f"{len(ph1)} helical phases specified for {chains} chains")

    xyz = np.zeros((ch_length * chains, 3))
    t = np.arange(ch_length)

    for i in range(chains):
        if cr[i] == 0:  # Antiparallel
            x, y, z = crick_eq(r0, r1, -w0, -w1, alpha, 0, -ph1[i], t)

            if "apNNzoff" in opts:
                zoff[i] = zoff[i] + xyz[(i - 1) * ch_length, 2] - z[-1]
            elif "registerzoff" in opts:
                zo = xyz[0, 2] - z[-1]
                dz = absolute_to_register_zoff(
                    zo, r0, w0, alpha, w1, ph1[0], ph1[i], cr[i]
                )
                zo = zo - dz
                zoff[i] = zoff[i] + zo
            elif "zoffaa" in opts:
                zo = xyz[0, 2] - z[-1]
                dz = absolute_to_zoff_aa(
                    zo, r0, w0, alpha, r1, w1, ph1[0], ph1[i], cr[i]
                )
                zo = zo - dz
                zoff[i] = zoff[i] + zo

            angle = dph0[i] - zoff[i] * np.tan(alpha) / r0
            T = np.array(
                [
                    [np.cos(angle), np.sin(angle), 0],
                    [-np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1],
                ]
            )
        else:  # Parallel
            x, y, z = crick_eq(r0, r1, w0, w1, alpha, 0, ph1[i], t)

            if "registerzoff" in opts:
                zo = 0
                dz = absolute_to_register_zoff(
                    zo, r0, w0, alpha, w1, ph1[0], ph1[i], cr[i]
                )
                zo = zo - dz
                zoff[i] = zoff[i] + zo
            elif "zoffaa" in opts:
                zo = 0
                dz = absolute_to_zoff_aa(
                    zo, r0, w0, alpha, r1, w1, ph1[0], ph1[i], cr[i]
                )
                zo = zo - dz
                zoff[i] = zoff[i] + zo

            angle = dph0[i] - zoff[i] * np.tan(alpha) / r0
            T = np.array(
                [
                    [np.cos(angle), np.sin(angle), 0],
                    [-np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1],
                ]
            )

        # Apply transformation
        coords = T @ np.array([x, y, z])

        start_idx = i * ch_length
        end_idx = (i + 1) * ch_length
        xyz[start_idx:end_idx, 0] = coords[0]
        xyz[start_idx:end_idx, 1] = coords[1]
        xyz[start_idx:end_idx, 2] = coords[2] + zoff[i]

    return xyz
