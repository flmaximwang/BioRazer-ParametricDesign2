import numpy as np
import biotite.structure as bio_struct
from .cccp.fit_crick import fit_crick


def calculate_helix_direction_by_hbonds(helix: bio_struct.AtomArray):
    """
    Calculate the direction of a helix given the AtomArray.
    Calculation is based on mean of hydrogen bonds
    """
    mask_N = helix.get_annotation("atom_name") == "N"
    mask_O = helix.get_annotation("atom_name") == "O"
    array_N = bio_struct.coord(helix[mask_N])
    array_O = bio_struct.coord(helix[mask_O])
    assert len(array_N) == len(array_O), "Number of N and O atoms must match"
    assert len(array_N) > 4, "Not enough atoms to calculate direction"
    hbond_vecs = []
    for i in range(len(array_N) - 4):
        hbond_vec = array_N[i + 4] - array_O[i]
        if np.linalg.norm(hbond_vec) < 3.5:
            hbond_vecs.append(hbond_vec)
    if len(hbond_vecs) == 0:
        raise ValueError(f"No hydrogen bonds found for the given helix")
    hbond_vec = np.sum(hbond_vecs, axis=0)
    hbond_vec /= np.linalg.norm(hbond_vec)
    return hbond_vec


def calculate_helix_direction_by_ca_svd(helix: bio_struct.AtomArray):
    """
    Calculate the direction of a helix given the AtomArray.
    Calculation is based on SVD of CA atoms
    """
    ca_mask = helix.get_annotation("atom_name") == "CA"
    ca_atoms = helix[ca_mask]
    if len(ca_atoms) < 5:
        raise ValueError("Not enough CA atoms to calculate direction")
    xyz = bio_struct.coord(ca_atoms)
    if len(xyz) < 5:
        raise ValueError("Not enough CA atoms to calculate direction")
    u, s, vh = np.linalg.svd(xyz - np.mean(xyz, axis=0))
    N_to_C_vec = ca_atoms[-1].coord - ca_atoms[0].coord
    main_comp = vh[0]
    if main_comp @ N_to_C_vec < 0:
        direction = -main_comp / np.linalg.norm(main_comp)
    else:
        direction = main_comp / np.linalg.norm(main_comp)
    return direction


def calculate_local_direction(
    helix: bio_struct.AtomArray, res_id: int, window: int = 7
):
    """
    Calculate the local direction of a helix based on its CA atoms based on CCCP
    """
    ca_mask = helix.get_annotation("atom_name") == "CA"
    ca_atoms = helix[ca_mask]
    if len(ca_atoms) < window:
        raise ValueError("Not enough CA atoms to calculate local direction")
    if res_id < window // 2 or res_id >= len(ca_atoms) - window // 2:
        raise ValueError(
            f"res_id {res_id} is out of bounds for the helix with {len(ca_atoms)} CA atoms"
        )
    start = res_id - window // 2
    end = res_id + window // 2 + 1
    local_ca_atoms = ca_atoms[start:end]
    xyz = bio_struct.coord(local_ca_atoms)
    if len(xyz) < 2:
        raise ValueError("Not enough CA atoms to calculate local direction")
