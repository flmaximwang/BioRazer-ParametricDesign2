"""
Local Crick parameter analysis and accommodation index calculation.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from ...math import erfnorm, erfnorm_jacobian
from .fit_crick import fit_crick
from .read_ca import read_ca


def fit_crick_scan(
    file, coor_type, n_chains, window_size, ph_ideal, output_dir, xl_out=False
):
    """
    Local Crick parameter analysis of an input structure and accommodation index calculation.

    This function calculates local Crick parameters for an input structure by fitting
    the Crick coiled coil equations at every residue position using a sliding window.

    Parameters
    ----------
    file : str
        Input coiled-coil coordinates (PDB file or XYZ file)
    coor_type : int
        Input type (1 = PDB file, 0 = XYZ file)
    n_chains : int
        Number of chains in the input structure
    window_size : int
        Length of the local window in residues (7 is recommended)
    ph_ideal : float
        Ideal minorhelical phase change in degrees between neighboring CA positions
        (For 7/2 canonical coiled coil: 360/3.5 ≈ 102.86°)
    output_dir : str
        Output directory for results
    xl_out : bool, optional
        Whether to write Excel output file (default: False)

    Returns
    -------
    coil_fits : dict
        Chain, residue numbers and AI profile fit parameters with errors
    parameters : list
        Local-window Crick parameters for each position
    ai_phases : list
        Phase information and AI values for each chain
    """
    # Read CA coordinates
    M = read_ca(file, coor_type)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Check coordinate validity
    if len(M) % n_chains != 0:
        raise ValueError(
            "Total number of coordinates is not divisible by number of chains!"
        )

    n_residues = len(M) // n_chains
    n_windows = n_residues - window_size + 1

    print(f"Analyzing {n_residues} residues per chain in {n_chains} chains")
    print(f"Using window size of {window_size} residues")
    print(f"Will fit {n_windows} windows")

    # Perform local Crick parameter fits
    print("\nPerforming local Crick parameter fits...")
    fit_results = []

    for window_idx in range(n_windows):
        print(f"Fitting window {window_idx + 1}/{n_windows}")

        # Extract coordinates for current window
        window_coords = []
        for chain_idx in range(n_chains):
            start_idx = chain_idx * n_residues + window_idx
            end_idx = start_idx + window_size
            window_coords.extend(M[start_idx:end_idx])

        window_coords = np.array(window_coords)

        # Fit Crick parameters to window
        try:
            err, xyz, params = fit_crick(
                window_coords, n_chains, "GENERAL-HLXPH", coor_type=2, out_type=0
            )  # Silent mode

            # Extract parameter values
            param_dict = {}
            for p in params:
                param_dict[p.name] = p.val

            param_dict["rmsd_error"] = np.sqrt(err)
            param_dict["starting_position"] = window_idx + 1
            fit_results.append(param_dict)

        except Exception as e:
            print(f"Warning: Failed to fit window {window_idx + 1}: {str(e)}")
            # Create dummy result
            param_dict = {
                "starting_position": window_idx + 1,
                "R0 (A)": np.nan,
                "R1 (A)": np.nan,
                "w0 (rad/res)": np.nan,
                "w1 (rad/res)": np.nan,
                "alpha (rad)": np.nan,
                "rmsd_error": np.nan,
            }
            # Add phase parameters for each chain
            for chain_idx in range(n_chains):
                param_dict[f"ph1 for chain {chain_idx + 1} (rad)"] = np.nan
            fit_results.append(param_dict)

    print("\nCalculating accommodation indices...")

    # Calculate accommodation indices
    positions = np.arange(1, n_windows + 1)

    # Extract phase data for each chain
    phase_data = {}
    ai_data = {}

    for chain_idx in range(n_chains):
        phase_key = f"ph1 for chain {chain_idx + 1} (rad)"
        phases_obs = np.array([r.get(phase_key, np.nan) for r in fit_results])

        # Convert to degrees
        phases_obs_deg = phases_obs * 180 / np.pi

        # Calculate expected phases
        phases_exp_deg = (positions - 1) * ph_ideal + phases_obs_deg[0]

        # Calculate cumulative phases to handle wraparound
        phase_diffs = np.zeros_like(phases_obs_deg)
        cumulative_phases = np.zeros_like(phases_obs_deg)
        cumulative_phases[0] = phases_exp_deg[0]

        for i in range(1, len(phases_obs_deg)):
            if not np.isnan(phases_obs_deg[i - 1]) and not np.isnan(phases_obs_deg[i]):
                # Calculate phase difference with proper handling of periodicity
                diff = _angle_difference(phases_obs_deg[i], phases_obs_deg[i - 1])
                cumulative_phases[i] = cumulative_phases[i - 1] + diff
            else:
                cumulative_phases[i] = phases_exp_deg[i]

        # Calculate phase differences and AI
        delta_phase = phases_exp_deg - cumulative_phases
        ai = delta_phase / ph_ideal

        # Store data
        phase_data[chain_idx] = {
            "expected": phases_exp_deg,
            "observed": cumulative_phases,
            "difference": delta_phase,
        }
        ai_data[chain_idx] = ai

    # Fit accommodation index profiles
    print("Fitting accommodation index profiles...")

    coil_fits = {}
    fit_data = {}

    for chain_idx in range(n_chains):
        ai_values = ai_data[chain_idx]
        valid_indices = ~np.isnan(ai_values)

        if np.sum(valid_indices) < 4:  # Need at least 4 points for fitting
            print(f"Warning: Insufficient data for chain {chain_idx + 1}")
            fit_data[chain_idx] = {
                "insertion_index": np.nan,
                "background": np.nan,
                "midpoint": np.nan,
                "sigma": np.nan,
                "errors": [np.nan] * 4,
            }
            continue

        valid_positions = positions[valid_indices]
        valid_ai = ai_values[valid_indices]

        # Initial parameter guess
        p0 = [0.0, 0.0, (n_residues + 1) / 2, 10.0]

        # Parameter bounds
        bounds_lower = [-0.6, -0.1, 1, 0.01]
        bounds_upper = [1.2, 0.1, n_residues, (n_residues + 1) / 2]

        try:
            # Fit error function
            popt, pcov = curve_fit(
                erfnorm,
                valid_positions,
                valid_ai,
                p0=p0,
                bounds=(bounds_lower, bounds_upper),
                maxfev=100000,
            )

            # Calculate parameter errors
            if pcov is not None and not np.any(np.isinf(pcov)):
                param_errors = np.sqrt(np.diag(pcov))
            else:
                param_errors = [np.nan] * 4

            fit_data[chain_idx] = {
                "insertion_index": popt[0],
                "background": popt[1],
                "midpoint": popt[2],
                "sigma": popt[3],
                "errors": param_errors,
            }

        except Exception as e:
            print(
                f"Warning: Failed to fit AI profile for chain {chain_idx + 1}: {str(e)}"
            )
            fit_data[chain_idx] = {
                "insertion_index": np.nan,
                "background": np.nan,
                "midpoint": np.nan,
                "sigma": np.nan,
                "errors": [np.nan] * 4,
            }

    # Generate plots
    print("Generating plots...")
    _generate_plots(
        fit_results, positions, phase_data, ai_data, fit_data, n_chains, output_dir
    )

    # Prepare output data structures
    coil_fits = _prepare_coil_fits_output(fit_data, n_chains, n_residues)
    parameters = _prepare_parameters_output(fit_results, positions)
    ai_phases = _prepare_ai_phases_output(phase_data, ai_data, n_chains, positions)

    # Write Excel file if requested
    if xl_out:
        _write_excel_output(coil_fits, parameters, ai_phases, output_dir)

    print(f"\nResults saved to: {output_dir}")
    print("Finished local Crick parameter analysis")

    return coil_fits, parameters, ai_phases


def _angle_difference(a1, a2):
    """Calculate the difference between two angles in degrees, handling periodicity."""
    diff = a1 - a2
    while diff > 180:
        diff -= 360
    while diff < -180:
        diff += 360
    return diff


def _generate_plots(
    fit_results, positions, phase_data, ai_data, fit_data, n_chains, output_dir
):
    """Generate Crick parameter and AI profile plots."""

    # Plot 1: Crick parameters
    fig1, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig1.suptitle("Local Crick Parameters", fontsize=14, fontweight="bold")

    # Extract parameter arrays
    r0_vals = [r.get("R0 (A)", np.nan) for r in fit_results]
    r1_vals = [r.get("R1 (A)", np.nan) for r in fit_results]
    w0_vals = [
        r.get("w0 (rad/res)", np.nan) * 180 / np.pi for r in fit_results
    ]  # Convert to degrees
    w1_vals = [
        r.get("w1 (rad/res)", np.nan) * 180 / np.pi for r in fit_results
    ]  # Convert to degrees
    alpha_vals = [
        r.get("alpha (rad)", np.nan) * 180 / np.pi for r in fit_results
    ]  # Convert to degrees
    rise_vals = []

    # Calculate rise per residue
    for r in fit_results:
        r0 = r.get("R0 (A)", np.nan)
        w0 = r.get("w0 (rad/res)", np.nan)
        alpha = r.get("alpha (rad)", np.nan)
        if not any(np.isnan([r0, w0, alpha])):
            rise = r0 * w0 / np.sin(alpha)
            rise_vals.append(rise)
        else:
            rise_vals.append(np.nan)

    # Plot each parameter
    axes[0, 0].plot(positions, r0_vals, "b-o", linewidth=2, markersize=4)
    axes[0, 0].set_xlabel("Position", fontweight="bold")
    axes[0, 0].set_ylabel("R₀ (Å)", fontweight="bold")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(positions, r1_vals, "b-o", linewidth=2, markersize=4)
    axes[0, 1].set_xlabel("Position", fontweight="bold")
    axes[0, 1].set_ylabel("R₁ (Å)", fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(positions, w0_vals, "b-o", linewidth=2, markersize=4)
    axes[1, 0].set_xlabel("Position", fontweight="bold")
    axes[1, 0].set_ylabel("ω₀ (degrees)", fontweight="bold")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(positions, w1_vals, "b-o", linewidth=2, markersize=4)
    axes[1, 1].set_xlabel("Position", fontweight="bold")
    axes[1, 1].set_ylabel("ω₁ (degrees)", fontweight="bold")
    axes[1, 1].grid(True, alpha=0.3)

    axes[2, 0].plot(positions, alpha_vals, "b-o", linewidth=2, markersize=4)
    axes[2, 0].set_xlabel("Position", fontweight="bold")
    axes[2, 0].set_ylabel("α (degrees)", fontweight="bold")
    axes[2, 0].grid(True, alpha=0.3)

    axes[2, 1].plot(positions, rise_vals, "b-o", linewidth=2, markersize=4)
    axes[2, 1].set_xlabel("Position", fontweight="bold")
    axes[2, 1].set_ylabel("Rise/Residue (Å)", fontweight="bold")
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "CrickParam_plots.pdf"), dpi=300, bbox_inches="tight"
    )
    plt.savefig(
        os.path.join(output_dir, "CrickParam_plots.jpg"), dpi=300, bbox_inches="tight"
    )
    plt.savefig(
        os.path.join(output_dir, "CrickParam_plots.small.jpg"),
        dpi=72,
        bbox_inches="tight",
    )
    plt.close()

    # Plot 2: AI profiles and phase differences
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig2.suptitle(
        "Accommodation Index and Phase Profiles", fontsize=14, fontweight="bold"
    )

    colors = plt.cm.tab10(np.linspace(0, 1, n_chains))

    # Plot AI profiles with fits
    for chain_idx in range(n_chains):
        ai_values = ai_data[chain_idx]
        valid_indices = ~np.isnan(ai_values)

        if np.sum(valid_indices) > 0:
            # Plot data points
            ax1.plot(
                positions[valid_indices],
                ai_values[valid_indices],
                "o",
                color=colors[chain_idx],
                label=f"Chain {chain_idx + 1}",
                markersize=6,
                alpha=0.7,
            )

            # Plot fit if available
            if not np.isnan(fit_data[chain_idx]["insertion_index"]):
                fit_params = [
                    fit_data[chain_idx]["insertion_index"],
                    fit_data[chain_idx]["background"],
                    fit_data[chain_idx]["midpoint"],
                    fit_data[chain_idx]["sigma"],
                ]
                fit_curve = erfnorm(fit_params, positions)
                ax1.plot(
                    positions,
                    fit_curve,
                    "-",
                    color=colors[chain_idx],
                    linewidth=2,
                    alpha=0.8,
                )

    ax1.set_xlabel("Position", fontweight="bold")
    ax1.set_ylabel("AI", fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot phase differences
    for chain_idx in range(n_chains):
        phase_diff = phase_data[chain_idx]["difference"]
        valid_indices = ~np.isnan(phase_diff)

        if np.sum(valid_indices) > 0:
            ax2.plot(
                positions[valid_indices],
                phase_diff[valid_indices],
                "o-",
                color=colors[chain_idx],
                label=f"Chain {chain_idx + 1}",
                linewidth=2,
                markersize=4,
            )

    ax2.set_xlabel("Position", fontweight="bold")
    ax2.set_ylabel("Phase Difference (degrees)", fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "AI_profiles.pdf"), dpi=300, bbox_inches="tight"
    )
    plt.savefig(
        os.path.join(output_dir, "AI_profiles.jpg"), dpi=300, bbox_inches="tight"
    )
    plt.savefig(
        os.path.join(output_dir, "AI_profiles.small.jpg"), dpi=72, bbox_inches="tight"
    )
    plt.close()


def _prepare_coil_fits_output(fit_data, n_chains, n_residues):
    """Prepare coil fits output data structure."""
    header = [
        "Chain",
        "First Position",
        "Second Position",
        "Insertion Index",
        "Background",
        "Midpoint Position",
        "Gaussian Accommodation, sigma (Residues)",
        "Insertion Index Error",
        "Background Error",
        "Midpoint Position Error",
        "Gaussian Accommodation, sigma, Error",
    ]

    coil_fits = {"header": header, "data": []}

    for chain_idx in range(n_chains):
        data = fit_data[chain_idx]
        row = [
            chain_idx + 1,  # Chain
            1,  # First Position
            n_residues,  # Second Position
            data["insertion_index"],
            data["background"],
            data["midpoint"],
            data["sigma"],
            data["errors"][0],  # Insertion Index Error
            data["errors"][1],  # Background Error
            data["errors"][2],  # Midpoint Position Error
            data["errors"][3],  # Sigma Error
        ]
        coil_fits["data"].append(row)

    return coil_fits


def _prepare_parameters_output(fit_results, positions):
    """Prepare parameters output data structure."""
    if not fit_results:
        return []

    # Create header from first result
    header = ["Starting Position"]
    first_result = fit_results[0]
    param_names = [key for key in first_result.keys() if key != "starting_position"]
    header.extend(param_names)

    # Prepare data
    output = [header]
    for result in fit_results:
        row = [result["starting_position"]]
        for name in param_names:
            row.append(result.get(name, np.nan))
        output.append(row)

    return output


def _prepare_ai_phases_output(phase_data, ai_data, n_chains, positions):
    """Prepare AI phases output data structure."""
    # Create header
    header = []
    for chain_idx in range(n_chains):
        chain_num = chain_idx + 1
        header.extend(
            [
                f"Phase Expected {chain_num}",
                f"Phase Observed {chain_num}",
                f"Phase Difference {chain_num}",
                f"AI {chain_num}",
            ]
        )

    # Prepare data
    data = []
    for i, pos in enumerate(positions):
        row = []
        for chain_idx in range(n_chains):
            if chain_idx in phase_data and i < len(phase_data[chain_idx]["expected"]):
                row.extend(
                    [
                        phase_data[chain_idx]["expected"][i],
                        phase_data[chain_idx]["observed"][i],
                        phase_data[chain_idx]["difference"][i],
                        ai_data[chain_idx][i],
                    ]
                )
            else:
                row.extend([np.nan, np.nan, np.nan, np.nan])
        data.append(row)

    return [header] + data


def _write_excel_output(coil_fits, parameters, ai_phases, output_dir):
    """Write results to Excel file."""
    try:
        import pandas as pd

        excel_file = os.path.join(output_dir, "CrickParam.xlsx")

        with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
            # Coil fits sheet
            coil_fits_df = pd.DataFrame(coil_fits["data"], columns=coil_fits["header"])
            coil_fits_df.to_excel(writer, sheet_name="CoilFits", index=False)

            # Parameters sheet
            if len(parameters) > 1:
                params_df = pd.DataFrame(parameters[1:], columns=parameters[0])
                params_df.to_excel(writer, sheet_name="Parameters", index=False)

            # AI profiles sheet
            if len(ai_phases) > 1:
                ai_df = pd.DataFrame(ai_phases[1:], columns=ai_phases[0])
                ai_df.to_excel(writer, sheet_name="AIprofiles", index=False)

        print(f"Excel file saved: {excel_file}")

    except ImportError:
        print("Warning: pandas not available, skipping Excel output")
    except Exception as e:
        print(f"Warning: Failed to write Excel file: {str(e)}")
