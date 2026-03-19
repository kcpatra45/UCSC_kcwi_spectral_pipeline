from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


def load_cube(path: Path) -> Tuple[np.ndarray, fits.Header]:
    """Load a KCWI cube (nz, ny, nx) and a representative header."""
    with fits.open(path, memmap=False) as hdul:
        if hdul[0].data is not None and getattr(hdul[0].data, "ndim", 0) == 3:
            return np.array(hdul[0].data, dtype=np.float32), hdul[0].header.copy()
        for hdu in hdul[1:]:
            if getattr(hdu, "data", None) is not None and getattr(hdu.data, "ndim", 0) == 3:
                hdr = hdu.header.copy()
                prim = hdul[0].header
                for k in ("TARGNAME", "CAMERA"):
                    if k in prim and k not in hdr:
                        hdr[k] = prim[k]
                return np.array(hdu.data, dtype=np.float32), hdr
    raise ValueError(f"No 3D cube found in {path}")


def get_lambda_axis(hdr: fits.Header, shape3: Tuple[int, int, int]) -> np.ndarray:
    """Return wavelength axis in Angstrom."""
    nz, ny, nx = shape3

    # Try WCS
    try:
        w = WCS(hdr)
        xs = np.full(nz, (nx - 1) / 2.0)
        ys = np.full(nz, (ny - 1) / 2.0)
        zs = np.arange(nz, dtype=float)
        world = w.all_pix2world(xs, ys, zs, 0)
        arrs = [np.array(world[i]) for i in range(len(world))]
        ranges = [np.nanmax(a) - np.nanmin(a) for a in arrs]
        idx = int(np.argmax(ranges))
        lam = arrs[idx].astype(float)
        if np.nanmedian(lam) < 1e-3:  # meters -> Angstrom
            lam = lam * 1e10
        return lam
    except Exception:
        pass

    # Linear fallback
    crval = hdr.get("CRVAL3")
    cdelt = hdr.get("CDELT3", hdr.get("CD3_3"))
    crpix = hdr.get("CRPIX3", 1.0)
    if crval is None or cdelt is None:
        raise ValueError("Could not determine wavelength axis (no WCS and no CRVAL3/CDELT3).")
    pix = np.arange(nz, dtype=float) + 1.0
    lam = crval + (pix - crpix) * cdelt
    if np.nanmedian(lam) < 1e-3:
        lam = lam * 1e10
    return lam.astype(float)


def white_light(cube: np.ndarray, lam: np.ndarray,
                lam_min: Optional[float] = None, lam_max: Optional[float] = None) -> np.ndarray:
    """Median-collapsed white-light image in a wavelength range."""
    if lam_min is None:
        lam_min = float(np.nanpercentile(lam, 10))
    if lam_max is None:
        lam_max = float(np.nanpercentile(lam, 90))
    m = (lam >= lam_min) & (lam <= lam_max)
    if not np.any(m):
        m[:] = True
    return np.nanmedian(cube[m, :, :], axis=0)


def get_airmass_from_header(hdr: fits.Header) -> Optional[float]:
    """KCWI convention: airmass in header keyword AIRMASS."""
    if "AIRMASS" not in hdr:
        return None
    try:
        return float(hdr["AIRMASS"])
    except Exception:
        return None
