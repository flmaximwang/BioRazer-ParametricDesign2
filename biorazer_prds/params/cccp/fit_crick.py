"""
Fit Crick parameters to input structure.
"""

import numpy as np
from scipy.optimize import minimize, least_squares
import warnings
from .geometry import crick_eq, superimpose, crossing_angle
from .math import angle_pmp, get_heptad_position, canonical_phases
from .read_ca import read_ca
from .generate_crick_bb import generate_crick_bb_rad


class CrickParameter:
    """Container for Crick parameter with bounds and metadata."""

    def __init__(self, value, name, lb=None, ub=None, priority=0):
        self.val = value
        self.name = name
        self.LB = lb if lb is not None else value
        self.UB = ub if ub is not None else value
        self.pri = priority

    def __repr__(self):
        return f"{self.name}: {self.val} (Lower boundary: {self.LB}, Upper boundary: {self.UB}, Priority: {self.pri})"


def fit_crick(
    file,
    chains,
    par_type="GENERAL",
    coor_type=1,
    out_type=0.5,
    initial_params={
        "R0": 5.0,
        "R1": 2.26,
        "w0": -2 * np.pi / 100,
        "w1": 4 * np.pi / 7,
        "alpha": -12 * np.pi / 180,
        "ph1": 0,
    },
    lower_bounds=None,
    upper_bounds=None,
    mask=None,
):
    """
    Fit Crick parameters to an input structure. Notice that the input structure must contain chains of the same length.

    Parameters
    ----------
    file : str or array_like
        Input structure file or coordinates
    chains : int
        Number of chains in the structure
    par_type : str
        Parameterization type:
        - 'GENERAL': Most general option
        - 'SYMMETRIC': Most symmetric option
        - 'ZOFF-SYMM': Like GENERAL but symmetric Z-offsets
        - 'DPH0-SYMM': Like GENERAL but symmetric superhelical phases
        - 'GENERAL-HLXPH': Like GENERAL but shared helical phases
        - 'SYMMETRIC-HLXPH': Like SYMMETRIC but shared helical phases
        - 'ZOFF-SYMM-HLXPH': Like ZOFF-SYMM but shared helical phases
        - 'DPH0-SYMM-HLXPH': Like DPH0-SYMM but shared helical phases
    coor_type : int
        Coordinate type (0: XYZ file, 1: PDB file, other: coordinates array)
    out_type : float
        Output type
        - 0: no output,
        - 0.5: display params,
        - 1: display + plot,
        - str: save to file
    initial_params : array_like, optional
        Initial parameter values [R0, R1, w0, w1, alpha, ph1]
        - R0: Radius of the coiled coil (A)
        - R1: Radius of the each helix (A)
        - w0: Initial twist per residue (rad/res)
        - w1: Final twist per residue (rad/res)
        - alpha: Pitch angle (rad)
    lower_bounds : array_like, optional
        Lower bounds for parameters
    upper_bounds : array_like, optional
        Upper bounds for parameters
    mask : array_like, optional
        Mask for excluding residues from fit (1=include, 0=exclude)

    Returns
    -------
    err : float
        RMSD between ideal and input structure
    xyz : ndarray
        N×3 matrix of ideal structure coordinates
    params : list
        List of CrickParameter objects with fitted values
        - R0 (A): Radius of the coiled coil
        - R1 (A): Radius of each helix
        - w0 (rad/res): Initial twist per residue
        - w1 (rad/res): Final twist per residue
        - alpha (rad): Pitch angle
        - ph1 (rad): Helical phase for each chain
        - zoff (A): Z-offset for each chain
        - dph0 (rad): Superhelical phase offset for each chain
        - olig (int): Number of chains in the structure
        - type (str): Parameterization type
        - co (list): Chain order
        - cr (list): Chain orientations (1=parallel, 0=antiparallel)
        - pitch (A): Rise per cycle of the superhelix
        - rise per residue (A): Rise per residue in each helix
        - heptad position: Starting heptad position for each chain
        - message: Additional messages about the fit
    """
    global _M, _x0, _p0, _sym, _co, _cr, _par_type, _show, _extr, _mask

    par_type = par_type.upper()
    valid_types = [
        "GENERAL",
        "SYMMETRIC",
        "ZOFF-SYMM",
        "DPH0-SYMM",
        "GENERAL-HLXPH",
        "SYMMETRIC-HLXPH",
        "ZOFF-SYMM-HLXPH",
        "DPH0-SYMM-HLXPH",
    ]

    if par_type not in valid_types:
        raise ValueError(f'Unknown parameterization type "{par_type}"')

    # Check initial parameters
    for key in ["R0", "R1", "w0", "w1", "alpha", "ph1"]:
        if key not in initial_params:
            raise ValueError(f'Missing initial parameter "{key}"')

    # Read coordinates
    _M = read_ca(file, coor_type)

    n = len(_M)
    if n % chains != 0:
        raise ValueError(
            f"Number of coordinates ({n}) not divisible by chains ({chains})"
        )

    _sym = {"olig": chains, "type": "GENERAL"}
    _show = 0
    _mask = mask
    _par_type = par_type

    # Determine chain order and orientation
    _co, _cr = _determine_chain_properties(_M, chains, mask)

    # Set up initial parameters
    _p0 = _setup_initial_parameters(initial_params, par_type, chains, _cr, _M)
    _x0 = [p.val for p in _p0]

    # Optimize parameters
    err, fitted_params = _optimize_parameters(_x0, _p0, lower_bounds, upper_bounds)

    # Update parameter values
    for i, val in enumerate(fitted_params):
        _p0[i].val = val

    # Clean up parameter values
    _p0 = _cleanup_parameters(_p0)

    # Compute final coordinates
    _show = 0.5 if out_type in [0.5, 1] else 0
    err, xyz = _crick_ssd(fitted_params)

    # Add derived parameters
    _p0 = _add_derived_parameters(_p0, chains, _cr, _M, xyz)

    # Display results if requested
    if out_type >= 0.5:
        _display_results(_p0, err)

    if isinstance(out_type, str):
        _save_parameters(_p0, err, out_type)

    return err, xyz, _p0


def _determine_chain_properties(M, chains, mask):
    """Determine chain order and orientation."""
    n = len(M)
    nc = n // chains

    if chains > 2:
        # Determine chain order (clockwise)
        if mask is not None:
            ind = [i for i in range(nc) if mask[i] == 1]
        else:
            ind = list(range(nc))

        if len(ind) == 0:
            ind = [nc // 2]  # fallback

        c = M[ind[len(ind) // 2]]  # middle CA in each layer
        pz = M[ind[-1]] - M[ind[0]]  # first chain defines positive axis

        chain_centers = [c]
        for i in range(1, chains):
            start_idx = i * nc
            if mask is not None:
                chain_ind = [
                    start_idx + j for j in range(nc) if mask[start_idx + j] == 1
                ]
            else:
                chain_ind = list(range(start_idx, start_idx + nc))

            if len(chain_ind) > 0:
                mid_idx = len(chain_ind) // 2
                chain_centers.append(M[chain_ind[mid_idx]])
            else:
                chain_centers.append(c)  # fallback

        # Calculate rotation angles
        cc = np.mean(chain_centers, axis=0)  # bundle center
        angles = [0]  # first chain at 0

        for i in range(1, chains):
            v1 = cc - chain_centers[0]
            v2 = cc - chain_centers[i]
            angle = np.arccos(
                np.clip(
                    np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1
                )
            )

            # Determine sign using cross product
            cross = np.cross(v1, v2)
            if np.dot(cross, pz) < 0:
                angle = 2 * np.pi - angle
            angles.append(angle)

        # Sort by angle to get clockwise order
        co = np.argsort(angles)
    else:
        # For dimers and monomers, order doesn't matter
        co = np.arange(chains)

    # Determine chain orientations
    if chains >= 2:
        nc = n // chains
        # First chain defines positive direction
        if mask is not None:
            ind = [i for i in range(nc) if mask[i] == 1]
        else:
            ind = list(range(nc))

        if len(ind) < 2:
            pz = np.array([0, 0, 1])  # fallback
        else:
            pz = M[ind[-1]] - M[ind[0]]

        cr = [1]  # first chain is parallel to itself

        for i in range(1, chains):
            start_idx = i * nc
            if mask is not None:
                chain_ind = [
                    start_idx + j for j in range(nc) if mask[start_idx + j] == 1
                ]
            else:
                chain_ind = list(range(start_idx, start_idx + nc))

            if len(chain_ind) < 2:
                cr.append(1)  # fallback to parallel
            else:
                chain_vec = M[chain_ind[-1]] - M[chain_ind[0]]
                parallel = np.dot(chain_vec, pz) > 0
                cr.append(1 if parallel else 0)
    else:
        cr = [1]

    return co, cr


def _setup_initial_parameters(initial_params, par_type, chains, cr, M):
    """Set up initial parameter structure."""
    IP = initial_params
    p0 = []

    # Basic parameters
    p0.append(CrickParameter(IP["R0"], "R0 (A)", 0, 30, 1))
    p0.append(CrickParameter(IP["R1"], "R1 (A)", 2, 3, 1))
    p0.append(
        CrickParameter(IP["w0"], "w0 (rad/res)", -10 * np.pi / 100, 10 * np.pi / 100, 1)
    )
    p0.append(
        CrickParameter(IP["w1"], "w1 (rad/res)", 1 * np.pi / 100, 8 * np.pi / 100, 1)
    )
    p0.append(CrickParameter(IP["alpha"], "alpha (rad)", -np.pi / 6, np.pi / 6, 1))

    # Helical phases
    if "HLXPH" not in par_type:
        p0.append(CrickParameter(IP["ph1"], "ph1 (rad)", -np.pi, np.pi, 1))
    else:
        for i in range(chains):
            p0.append(
                CrickParameter(
                    IP["ph1"], f"ph1 for chain {i+1} (rad)", -np.pi, np.pi, 1
                )
            )

    # Z-offsets
    z_range = np.max(M[:, 2]) - np.min(M[:, 2])
    if "SYMMETRIC" in par_type or "ZOFF-SYMM" in par_type:
        # One z-offset for all antiparallel chains
        if any(c == 0 for c in cr):
            p0.append(
                CrickParameter(z_range, "absolute ap zoff (A)", -z_range, z_range, 4)
            )
    else:
        # Individual z-offsets for each chain (except first)
        for i in range(1, chains):
            if cr[i] == 1:  # parallel
                p0.append(
                    CrickParameter(0, f"absolute zoff_{i+1} (A)", -z_range, z_range, 4)
                )
            else:  # antiparallel
                p0.append(
                    CrickParameter(
                        z_range, f"absolute zoff_{i+1} (A)", -z_range, z_range, 4
                    )
                )

    # Superhelical phase offsets
    if "SYMMETRIC" not in par_type and "DPH0-SYMM" not in par_type:
        for i in range(1, chains):
            ideal_offset = -(i) * 2 * np.pi / chains
            p0.append(
                CrickParameter(ideal_offset, f"dph0_{i+1} (rad)", -2 * np.pi, 0, 1)
            )
    elif "SYMMETRIC" in par_type:
        # Check for alternating up/down pattern
        parallel_chains = [i for i, c in enumerate(cr) if c == 1]
        antiparallel_chains = [i for i, c in enumerate(cr) if c == 0]

        if len(parallel_chains) > 0 and len(antiparallel_chains) > 0:
            p0.append(
                CrickParameter(-2 * np.pi / chains, "dph0_p_ap (rad)", -2 * np.pi, 0, 1)
            )

    return p0


def _optimize_parameters(x0, p0, lower_bounds, upper_bounds):
    """Optimize Crick parameters."""
    global _extr

    # Set up bounds
    if lower_bounds is None:
        lower_bounds = [p.LB for p in p0]
    if upper_bounds is None:
        upper_bounds = [p.UB for p in p0]

    bounds = list(zip(lower_bounds, upper_bounds))

    # Multi-stage optimization
    best_err = np.inf
    best_x = x0

    # Stage 1: Optimize individual parameter groups
    for iteration in range(10):
        # Optimize orientation parameters first
        orientation_indices = []
        for i, p in enumerate(p0):
            if "zoff" in p.name or "dph0" in p.name or "ph1" in p.name:
                orientation_indices.append(i)

        if orientation_indices:
            _extr = {"vary": orientation_indices}

            def single_objective(x_sub):
                x_full = best_x.copy()
                for i, idx in enumerate(orientation_indices):
                    x_full[idx] = x_sub[i]
                return _crick_ssd(x_full)[0]

            x_sub = [best_x[i] for i in orientation_indices]
            bounds_sub = [bounds[i] for i in orientation_indices]

            try:
                result = minimize(
                    single_objective, x_sub, bounds=bounds_sub, method="L-BFGS-B"
                )
                if result.success:
                    for i, idx in enumerate(orientation_indices):
                        best_x[idx] = result.x[i]
            except:
                pass

        # Stage 2: Optimize all parameters together
        _extr = None

        try:
            result = minimize(_crick_ssd, best_x, bounds=bounds, method="L-BFGS-B")
            if result.success and result.fun < best_err:
                best_err = result.fun
                best_x = result.x

            # Check convergence
            if best_err < 0.001:  # Good enough
                break

        except:
            break

    return best_err, best_x


def _crick_ssd(params):
    """Compute sum of squared deviations between model and data."""
    global _M, _sym, _co, _cr, _par_type, _show, _extr, _mask

    try:
        # Handle single parameter optimization
        if _extr is not None and "vary" in _extr:
            full_params = _x0.copy()
            for i, idx in enumerate(_extr["vary"]):
                full_params[idx] = params[i]
            params = full_params

        # Parse parameters based on parameterization type
        parsed = _parse_parameters(params, _par_type, _sym["olig"], _cr)
        r0, r1, w0, w1, alpha, ph1, zoff, dph0 = parsed

        # Generate ideal coordinates
        xyz = generate_crick_bb_rad(
            _sym["olig"],
            len(_M) // _sym["olig"],
            r0,
            r1,
            w0,
            w1,
            alpha,
            ph1,
            _cr,
            dph0,
            zoff,
        )

        # Handle invalid coordinates
        if np.any(np.isnan(xyz)) or np.any(np.isinf(xyz)):
            return 1e10, xyz

        # Compute RMSD
        if _mask is None:
            rmsd, rot_matrix, _ = superimpose(_M.T, xyz.T)
        else:
            mask_indices = np.where(_mask == 1)[0]
            if len(mask_indices) == 0:
                return 1e10, xyz
            rmsd, rot_matrix, _ = superimpose(_M[mask_indices].T, xyz[mask_indices].T)

        ssd = rmsd**2 * len(_M)

        # Add penalty for unrealistic parameters
        if _sym["olig"] == 2 and _show == 0:  # During optimization
            # Penalize unrealistic superhelical phase offsets
            if len(dph0) > 0:
                ideal_offset = -2 * np.pi / _sym["olig"]
                offset_penalty = abs(ideal_offset - dph0[-1]) / (np.pi / 3) * 0.02
                ssd += offset_penalty * len(_M)

        # Transform coordinates for display if needed
        if _show > 0:
            if _mask is None:
                xyz_centered = xyz - np.mean(xyz, axis=0)
                xyz = xyz_centered @ rot_matrix.T + np.mean(_M, axis=0)
            else:
                mask_indices = np.where(_mask == 1)[0]
                xyz_centered = xyz - np.mean(xyz[mask_indices], axis=0)
                xyz = xyz_centered @ rot_matrix.T + np.mean(_M[mask_indices], axis=0)

        return ssd, xyz

    except Exception as e:
        return 1e10, np.zeros_like(_M)


def _parse_parameters(params, par_type, chains, cr):
    """Parse parameter vector based on parameterization type."""
    p = list(params)

    # Basic parameters (always first 5)
    r0 = p.pop(0)
    r1 = p.pop(0)
    w0 = p.pop(0)
    w1 = p.pop(0)
    alpha = p.pop(0)

    # Ensure alpha and w0 have same sign
    alpha = abs(alpha) * np.sign(w0)

    # Helical phases
    if "HLXPH" not in par_type:
        ph1 = np.full(chains, p.pop(0))
    else:
        ph1 = np.array([p.pop(0) for _ in range(chains)])

    # Z-offsets
    zoff = np.zeros(chains)
    if "SYMMETRIC" in par_type or "ZOFF-SYMM" in par_type:
        # Symmetric z-offsets
        ap_indices = [i for i, c in enumerate(cr) if c == 0]
        if ap_indices and p:
            zoff[ap_indices] = p.pop(0)
    else:
        # Individual z-offsets
        for i in range(1, chains):
            if p:
                zoff[i] = p.pop(0)

    # Superhelical phase offsets
    dph0 = np.zeros(chains)
    if "SYMMETRIC" not in par_type and "DPH0-SYMM" not in par_type:
        # Individual phase offsets
        for i in range(1, chains):
            if p:
                dph0[i] = p.pop(0)
    elif "SYMMETRIC" in par_type:
        # Symmetric phase offsets
        if p:  # If there's a phase offset parameter
            dph0_pap = p.pop(0)
            parallel_indices = [i for i, c in enumerate(cr) if c == 1]
            antiparallel_indices = [i for i, c in enumerate(cr) if c == 0]

            # Set phase offsets for parallel chains
            for i, idx in enumerate(parallel_indices):
                dph0[idx] = -i * 2 * np.pi / len(parallel_indices)

            # Set phase offsets for antiparallel chains
            for i, idx in enumerate(antiparallel_indices):
                dph0[idx] = dph0_pap - i * 2 * np.pi / len(antiparallel_indices)
        else:
            # Default symmetric arrangement
            for i in range(chains):
                dph0[i] = -i * 2 * np.pi / chains
    else:  # DPH0-SYMM
        # Default symmetric arrangement
        for i in range(chains):
            dph0[i] = -i * 2 * np.pi / chains

    return r0, r1, w0, w1, alpha, ph1, zoff, dph0


def _cleanup_parameters(p0):
    """Clean up parameter values (handle periodicity, etc.)."""
    # Find parameter indices
    w0_idx = next((i for i, p in enumerate(p0) if "w0" in p.name), None)
    w1_idx = next((i for i, p in enumerate(p0) if "w1" in p.name), None)
    alpha_idx = next((i for i, p in enumerate(p0) if "alpha" in p.name), None)
    r1_idx = next((i for i, p in enumerate(p0) if "R1" in p.name), None)

    # Remove 2π periodicity
    if w0_idx is not None:
        p0[w0_idx].val = angle_pmp(p0[w0_idx].val)
    if w1_idx is not None:
        p0[w1_idx].val = angle_pmp(p0[w1_idx].val)
    if alpha_idx is not None:
        p0[alpha_idx].val = angle_pmp(p0[alpha_idx].val)

    # Clean up phase parameters
    for p in p0:
        if "ph1" in p.name or "dph0" in p.name:
            p.val = angle_pmp(p.val)

    # Ensure alpha and w0 have same sign
    if w0_idx is not None and alpha_idx is not None:
        p0[alpha_idx].val = abs(p0[alpha_idx].val) * np.sign(p0[w0_idx].val)

    # Handle negative R1
    if r1_idx is not None and p0[r1_idx].val < 0:
        p0[r1_idx].val = -p0[r1_idx].val
        # Adjust helical phases
        for p in p0:
            if "ph1" in p.name:
                p.val = p.val + np.pi

    return p0


def _add_derived_parameters(p0, chains, cr, M, xyz):
    """Add derived parameters to the parameter list."""
    # Find basic parameters
    r0 = next(p.val for p in p0 if "R0" in p.name)
    r1 = next(p.val for p in p0 if "R1" in p.name)
    w0 = next(p.val for p in p0 if "w0" in p.name)
    w1 = next(p.val for p in p0 if "w1" in p.name)
    alpha = next(p.val for p in p0 if "alpha" in p.name)

    # Add derived geometric parameters
    pitch = abs(2 * np.pi * r0 / np.tan(alpha))
    rise_per_res = r0 * w0 / np.sin(alpha)

    p0.append(CrickParameter(pitch, "pitch (A)", pitch, pitch, 1))
    p0.append(
        CrickParameter(
            rise_per_res, "rise per residue (A)", rise_per_res, rise_per_res, 1
        )
    )

    # Add heptad positions
    ph1_params = [p for p in p0 if "ph1" in p.name and "chain" in p.name]
    if ph1_params:
        for i, p in enumerate(ph1_params):
            heptad_pos = get_heptad_position(p.val)
            p0.append(
                CrickParameter(
                    heptad_pos,
                    f"starting heptad position for chain {i+1}",
                    heptad_pos,
                    heptad_pos,
                    1,
                )
            )
    else:
        ph1_val = next(p.val for p in p0 if "ph1" in p.name)
        heptad_pos = get_heptad_position(ph1_val)
        p0.append(
            CrickParameter(
                heptad_pos, "starting heptad position", heptad_pos, heptad_pos, 1
            )
        )

    # Check if structure is canonical coiled coil
    canonical_w1 = 4 * np.pi / 7
    if abs(w1 - canonical_w1) > np.pi / 7:
        message = (
            "The fit structure does not appear to be a canonical 7-residue repeat coiled coil. "
            "Be cautious when interpreting heptad based parameters!"
        )
        p0.append(CrickParameter(message, "message", "", ""))
    else:
        p0.append(CrickParameter("", "message", "", ""))

    # Add structure summary
    nc = len(M) // chains
    summary = f"{chains} (fit {nc} residues in each)"
    p0.append(CrickParameter(summary, "structure summary: chains", None, None, 0))

    for i in range(chains):
        start_z = xyz[i * nc, 2]
        end_z = xyz[(i + 1) * nc - 1, 2]
        length = abs(end_z - start_z)

        summary = f"{length:.3f} Angstrom in Z-direction"
        if i > 0:
            if cr[i] == 1:
                summary += "; parallel to chain 1"
            else:
                summary += "; anti-parallel to chain 1"

        p0.append(
            CrickParameter(summary, f"structure summary: chain {i+1}", None, None, 0)
        )

    return p0


def _display_results(p0, err):
    """Display fitting results."""
    print(f"\nCrick Parameter Fitting Results")
    print(f"==============================")
    print(f"RMSD: {np.sqrt(err):.6f} Å")
    print()

    for p in p0:
        if isinstance(p.val, str):
            print(f"{p.name} = {p.val}")
        else:
            print(f"{p.name} = {p.val:.6f}")


def _save_parameters(p0, err, filename):
    """Save parameters to file."""
    with open(f"{filename}.par", "w") as f:
        f.write(f"error = {np.sqrt(err):.6f}\n")

        # Sort by priority
        priorities = sorted(
            set(p.pri for p in p0 if hasattr(p, "pri") and p.pri is not None)
        )

        for pri in priorities:
            for p in p0:
                if hasattr(p, "pri") and p.pri == pri:
                    if isinstance(p.val, str):
                        f.write(f"{p.name} = {p.val}\n")
                    else:
                        f.write(f"{p.name} = {p.val:.6f}\n")

        # Write parameters without priority
        for p in p0:
            if not hasattr(p, "pri") or p.pri is None:
                if isinstance(p.val, str):
                    f.write(f"{p.name} = {p.val}\n")
                else:
                    f.write(f"{p.name} = {p.val:.6f}\n")
