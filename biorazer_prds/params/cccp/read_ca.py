"""
Read CA coordinates from PDB files or coordinate files.
"""

import numpy as np
import re
import os


def read_ca(file, coor_type):
    """
    Read CA coordinates from file.
    
    Parameters
    ----------
    file : str or array_like
        Input file path or coordinate array
    coor_type : int
        Input type:
        - 1: PDB file
        - 0: flat text file with XYZ coordinates
        - other: treat file as coordinate array
        
    Returns
    -------
    ndarray
        N×3 matrix of CA coordinates
    """
    if coor_type == 1:
        # PDB file - extract CA atoms
        return _read_pdb_ca(file)
    elif coor_type == 0:
        # Flat text file with coordinates
        return _read_xyz_file(file)
    else:
        # Treat as coordinate array
        return np.array(file)


def _read_pdb_ca(pdb_file):
    """
    Extract CA coordinates from PDB file.
    
    Parameters
    ----------
    pdb_file : str
        Path to PDB file
        
    Returns
    -------
    ndarray
        N×3 matrix of CA coordinates
    """
    coords = []
    
    try:
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith('ATOM') and ' CA ' in line:
                    # Extract coordinates from PDB ATOM line
                    # Columns 31-38 (X), 39-46 (Y), 47-54 (Z) in PDB format
                    try:
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        coords.append([x, y, z])
                    except (ValueError, IndexError):
                        continue
                        
    except FileNotFoundError:
        raise FileNotFoundError(f"PDB file not found: {pdb_file}")
    except Exception as e:
        raise RuntimeError(f"Error reading PDB file {pdb_file}: {str(e)}")
    
    if len(coords) == 0:
        raise ValueError(f"No CA atoms found in PDB file: {pdb_file}")
        
    return np.array(coords)


def _read_xyz_file(xyz_file):
    """
    Read coordinates from flat text file.
    
    Parameters
    ----------
    xyz_file : str
        Path to XYZ coordinate file
        
    Returns
    -------
    ndarray
        N×3 matrix of coordinates
    """
    try:
        coords = np.loadtxt(xyz_file)
        
        # Handle different file formats
        if coords.ndim == 1:
            # Single row, reshape to N×3
            if len(coords) % 3 != 0:
                raise ValueError("Number of coordinates must be divisible by 3")
            coords = coords.reshape(-1, 3)
        elif coords.ndim == 2:
            if coords.shape[1] == 1:
                # Single column, reshape to N×3
                coords = coords.flatten()
                if len(coords) % 3 != 0:
                    raise ValueError("Number of coordinates must be divisible by 3")
                coords = coords.reshape(-1, 3)
            elif coords.shape[1] != 3:
                raise ValueError("Coordinate file must have 3 columns (X, Y, Z)")
        else:
            raise ValueError("Invalid coordinate file format")
            
    except FileNotFoundError:
        raise FileNotFoundError(f"Coordinate file not found: {xyz_file}")
    except Exception as e:
        raise RuntimeError(f"Error reading coordinate file {xyz_file}: {str(e)}")
    
    if coords.size == 0:
        raise ValueError(f"No coordinates found in file: {xyz_file}")
        
    return coords


def write_pdb_ca(coords, output_file, chain_ids=None):
    """
    Write CA coordinates to PDB file.
    
    Parameters
    ----------
    coords : array_like
        N×3 matrix of CA coordinates
    output_file : str
        Output PDB file path
    chain_ids : list, optional
        Chain identifiers for each residue
    """
    coords = np.array(coords)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("Coordinates must be N×3 matrix")
    
    if chain_ids is None:
        chain_ids = ['A'] * len(coords)
    elif len(chain_ids) != len(coords):
        raise ValueError("Number of chain IDs must match number of coordinates")
    
    with open(output_file, 'w') as f:
        f.write("HEADER    COILED COIL STRUCTURE\n")
        
        for i, (coord, chain_id) in enumerate(zip(coords, chain_ids)):
            x, y, z = coord
            line = (f"ATOM  {i+1:5d}  CA  ALA {chain_id}{i+1:4d}    "
                   f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C\n")
            f.write(line)
            
        f.write("END\n")


def write_xyz_file(coords, output_file):
    """
    Write coordinates to flat text file.
    
    Parameters
    ----------
    coords : array_like
        N×3 matrix of coordinates
    output_file : str
        Output file path
    """
    coords = np.array(coords)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("Coordinates must be N×3 matrix")
    
    np.savetxt(output_file, coords, fmt='%8.3f')
