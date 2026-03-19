from __future__ import annotations

from typing import Optional, Tuple
import numpy as np


def spectral_trim_cube(cube: np.ndarray,
                       lam: np.ndarray,
                       lam_min: Optional[float] = None,
                       lam_max: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Trim a cube (nlambda, ny, nx) to a wavelength range.

    Parameters
    ----------
    cube : ndarray
        Shape (nlambda, ny, nx)
    lam : ndarray
        Shape (nlambda,)
    lam_min, lam_max : float or None
        Inclusive bounds. If either is None, that bound is left unchanged.

    Returns
    -------
    cube_t, lam_t
    """
    if lam is None:
        raise ValueError("spectral_trim_cube: lam axis is None")

    if lam_min is None and lam_max is None:
        return cube, lam

    if lam_min is None:
        lam_min = float(np.nanmin(lam))
    if lam_max is None:
        lam_max = float(np.nanmax(lam))

    mask = (lam >= lam_min) & (lam <= lam_max)
    if not np.any(mask):
        raise ValueError(f"spectral_trim_cube: no wavelengths in range [{lam_min}, {lam_max}]")

    return cube[mask, :, :], lam[mask]
