from __future__ import annotations

from typing import Tuple
import numpy as np

from .config import TargetBackgroundApertures
from .apertures import mask_from_shape


def extract_1d_counts(cube: np.ndarray, lam: np.ndarray, aps: TargetBackgroundApertures) -> np.ndarray:
    """Extract 1D counts spectrum = sum(target) - median(background)*area(target)."""
    nz, ny, nx = cube.shape
    m_tgt = mask_from_shape(ny, nx, aps.target)
    m_bkg = mask_from_shape(ny, nx, aps.background)
    tgt_area = float(np.sum(m_tgt))

    out = np.zeros(nz, dtype=float)
    for k in range(nz):
        img = cube[k, :, :]
        tgt_sum = np.nansum(img[m_tgt])
        bkg = np.nanmedian(img[m_bkg]) if np.any(m_bkg) else 0.0
        out[k] = tgt_sum - bkg * tgt_area
    return out
