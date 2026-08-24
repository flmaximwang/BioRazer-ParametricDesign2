"""
Mathematical utility functions for CCCP package.
"""

import numpy as np
from scipy.optimize import curve_fit


def angle_pmp(a):
    """
    Puts an angle between +pi and -pi.
    
    Parameters
    ----------
    a : float or array_like
        Angle(s) in radians
        
    Returns
    -------
    float or array_like
        Angle(s) between -pi and pi
    """
    a = np.asarray(a)
    is_scalar = a.ndim == 0
    
    # Match MATLAB behavior: mod(a, 2*pi), then subtract 2*pi if > pi
    a = np.mod(a, 2*np.pi)
    mask = a > np.pi
    a = np.where(mask, a - 2*np.pi, a)
    
    return a.item() if is_scalar else a


def angle_diff(a, b):
    """
    Calculate angular difference between two angles.
    
    Parameters
    ----------
    a, b : float or array_like
        Angles in radians
        
    Returns
    -------
    float or array_like
        Angular difference
    """
    d = np.mod((np.mod(a, 2*np.pi) - np.mod(b, 2*np.pi)), 2*np.pi)
    idx = np.where(d > np.pi)
    d[idx] = d[idx] - 2*np.pi
    return d


def erfnorm(params, xdata):
    """
    Error function for fitting AI profile.
    
    The insertion index; Iindex = 0, & 0.5, 1, -0.5 for canonical coils, and
    ones containing 4, 1, and 3 residue insertions, respectively.
    
    Parameters
    ----------
    params : array_like
        [Iindex, Back, mu, sig] parameters
    xdata : array_like
        Position data
        
    Returns
    -------
    array_like
        Fitted function values
    """
    from scipy.special import erf
    
    Iindex, Back, mu, sig = params
    F = (Iindex/2) * (1 + erf((xdata - mu)/(sig*np.sqrt(2)))) + Back
    return F


def erfnorm_jacobian(params, xdata):
    """
    Jacobian for erfnorm function.
    
    Parameters
    ----------
    params : array_like
        [Iindex, Back, mu, sig] parameters
    xdata : array_like
        Position data
        
    Returns
    -------
    ndarray
        Jacobian matrix
    """
    from scipy.special import erf
    
    Iindex, Back, mu, sig = params
    if xdata.ndim > 1:
        xdata = xdata.flatten()
        
    J = np.zeros((len(xdata), 4))
    J[:, 0] = (1/2) * (1 + erf((xdata - mu)/(sig*np.sqrt(2))))
    J[:, 1] = 1
    J[:, 2] = -(Iindex/np.sqrt(2*np.pi)) * np.exp(-(xdata - mu)**2/(2*sig**2)) * (1/sig)
    J[:, 3] = J[:, 2] * (xdata - mu) / sig
    
    return J


def canonical_phases(ind):
    """
    Get canonical phases for heptad positions.
    
    Parameters
    ----------
    ind : int or array_like
        Index(es) for heptad positions (1-7 for a-g)
        
    Returns
    -------
    float or array_like
        Canonical phase(s) in degrees
    """
    # canonical phases are:
    # [41.0, 95.0, 146.0, 197.0, 249.0, 300.0, 351.0]
    # corresponding to {'c', 'g', 'd', 'a', 'e', 'b', 'f'}, respectively
    
    # in the order a-g:
    median_phases = np.array([197.0, 300.0, 41.0, 146.0, 249.0, 351.0, 95.0])
    return median_phases[ind-1] if np.isscalar(ind) else median_phases[np.array(ind)-1]


def get_heptad_position(ph1, as_int=False):
    """
    Returns the heptad position corresponding to phase ph1.
    
    Parameters
    ----------
    ph1 : float
        Helical phase in radians
    as_int : bool
        If True, return integer 1-7 instead of character
        
    Returns
    -------
    str or int
        Heptad position
    """
    ph1 = ph1 + np.pi/7
    
    # Ensure phase is in [0, 2π] range
    while ph1 < 0:
        ph1 += 2*np.pi
    while ph1 > 2*np.pi:
        ph1 -= 2*np.pi
        
    fhpi = int(np.floor(7*ph1/(2*np.pi))) + 1
    if fhpi <= 0 or fhpi > 7:
        fhpi = 1  # fallback
        
    hps = 'fcgdaeb'
    
    if as_int:
        return ord(hps[fhpi-1]) - ord('a') + 1
    else:
        return hps[fhpi-1]


def get_heptad_pos(ph1, as_int=False):
    """
    Returns the heptad position corresponding to phase ph1 (alternative implementation).
    
    Parameters
    ----------
    ph1 : float
        Helical phase in radians
    as_int : bool
        If True, return integer 1-7 instead of character
        
    Returns
    -------
    str or int
        Heptad position
    """
    meds = canonical_phases(np.arange(1, 8)) * np.pi / 180
    hps = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    
    # sort phases in the order they appear on the helical wheel
    sorted_indices = np.argsort(meds)
    meds = meds[sorted_indices]
    hps = [hps[i] for i in sorted_indices]
    
    ph1 = np.mod(ph1, 2*np.pi)
    
    for i in range(len(meds)):
        pin = i - 1 if i > 0 else len(meds) - 1
        nin = i + 1 if i < len(meds) - 1 else 0
        
        lb = np.mod(angle_diff(meds[pin], meds[i])/2 + meds[i], 2*np.pi)
        ub = np.mod(angle_diff(meds[nin], meds[i])/2 + meds[i], 2*np.pi)
        
        if angle_diff(ph1, lb) > 0 and angle_diff(ub, ph1) > 0:
            if as_int:
                return ord(hps[i]) - ord('a') + 1
            else:
                return hps[i]
    
    # fallback
    return 1 if as_int else 'a'
