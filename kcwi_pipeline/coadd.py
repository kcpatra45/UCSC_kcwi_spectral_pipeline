from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip


@dataclass
class CubeProduct:
    path: Path
    science: np.ndarray
    header: fits.Header
    uncert: Optional[np.ndarray]
    mask: Optional[np.ndarray]
    flags: Optional[np.ndarray]
    noskysub: Optional[np.ndarray]


def clean_object_name(obj) -> str:
    if obj is None:
        return "UNKNOWN"
    s = str(obj).strip()
    if s == "" or s.upper() in ("UNKNOWN", "UNDEF", "NONE", "N/A"):
        return "UNKNOWN"
    return s


def clean_camera(cam) -> str:
    if cam is None:
        return "UNKNOWN"
    s = str(cam).strip().upper()
    if s in ("B", "BLUE", "KCWI-BLUE"):
        return "BLUE"
    if s in ("R", "RED", "KCWI-RED"):
        return "RED"
    return s


def safe_filename(s: str) -> str:
    return "".join(c if (c.isalnum() or c in ("-", "_", ".")) else "_" for c in str(s))


def _first_3d_hdu(hdul: fits.HDUList) -> Tuple[int, fits.ImageHDU | fits.PrimaryHDU]:
    if hdul[0].data is not None and getattr(hdul[0].data, "ndim", 0) == 3:
        return 0, hdul[0]
    for i, hdu in enumerate(hdul[1:], start=1):
        if getattr(hdu, "data", None) is not None and getattr(hdu.data, "ndim", 0) == 3:
            return i, hdu
    raise ValueError("No 3D science cube found")


def _get_ext(hdul: fits.HDUList, extname: str, shape: Tuple[int, int, int]) -> Optional[np.ndarray]:
    extname_u = extname.upper()
    for hdu in hdul[1:]:
        if str(hdu.header.get("EXTNAME", "")).upper() != extname_u:
            continue
        data = getattr(hdu, "data", None)
        if data is None:
            return None
        if getattr(data, "shape", None) != shape:
            raise ValueError(f"{extname} extension has shape {data.shape}, expected {shape}")
        return np.array(data)
    return None


def read_cube_product(path: Path) -> CubeProduct:
    with fits.open(path, memmap=False) as hdul:
        _, sci_hdu = _first_3d_hdu(hdul)
        science = np.array(sci_hdu.data, dtype=np.float32)
        header = sci_hdu.header.copy()
        prim = hdul[0].header
        for k in ("OBJECT", "TARGNAME", "CAMERA"):
            if k in prim and k not in header:
                header[k] = prim[k]

        shape = science.shape
        uncert = _get_ext(hdul, "UNCERT", shape)
        mask = _get_ext(hdul, "MASK", shape)
        flags = _get_ext(hdul, "FLAGS", shape)
        noskysub = _get_ext(hdul, "NOSKYSUB", shape)

    if uncert is not None:
        uncert = uncert.astype(np.float32)
    if mask is not None:
        mask = mask.astype(np.uint8)
    if noskysub is not None:
        noskysub = noskysub.astype(np.float32)

    return CubeProduct(
        path=path,
        science=science,
        header=header,
        uncert=uncert,
        mask=mask,
        flags=flags,
        noskysub=noskysub,
    )


def discover_fits_files(data_dir: Path, list_file: Optional[Path] = None) -> List[Path]:
    data_dir = data_dir.expanduser().resolve()
    if list_file is None:
        files = sorted(data_dir.glob("*_icubed.fits"))
        if not files:
            files = sorted(data_dir.glob("*.fits"))
        return files

    list_path = list_file.expanduser().resolve()
    if not list_path.exists():
        raise FileNotFoundError(f"List file not found: {list_path}")

    files: List[Path] = []
    for line in list_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = Path(line)
        if not p.is_absolute():
            p = data_dir / p
        if not p.exists():
            raise FileNotFoundError(f"Listed file not found: {p}")
        files.append(p)
    return files


def group_files_by_object_camera(files: List[Path]) -> Dict[Tuple[str, str], List[Path]]:
    groups: Dict[Tuple[str, str], List[Path]] = defaultdict(list)
    for f in files:
        with fits.open(f, memmap=False) as hdul:
            hdr0 = hdul[0].header
            obj = clean_object_name(hdr0.get("TARGNAME", hdr0.get("OBJECT")))
            cam = clean_camera(hdr0.get("CAMERA"))
        groups[(obj, cam)].append(f)
    return groups


def _validate_same_wavelength_grid(products: List[CubeProduct]) -> None:
    keys = ("NAXIS3", "CRVAL3", "CDELT3", "CD3_3", "CRPIX3", "CTYPE3", "CUNIT3")
    ref = products[0].header
    for prod in products[1:]:
        for key in keys:
            if key not in ref and key not in prod.header:
                continue
            if ref.get(key) != prod.header.get(key):
                raise ValueError(
                    f"Wavelength grid mismatch for {prod.path.name}: "
                    f"{key}={prod.header.get(key)!r}, expected {ref.get(key)!r}"
                )


def _finite_good_mask(
    stack: np.ndarray,
    var_stack: Optional[np.ndarray],
    mask_stack: Optional[np.ndarray],
    flags_stack: Optional[np.ndarray],
) -> np.ndarray:
    good = np.isfinite(stack)
    if var_stack is not None:
        good &= np.isfinite(var_stack) & (var_stack > 0)
    if mask_stack is not None:
        good &= mask_stack == 0
    if flags_stack is not None:
        good &= flags_stack == 0
    return good


def _sigma_clip_good(stack: np.ndarray, good: np.ndarray, sigma: float, iters: int) -> np.ndarray:
    masked = np.ma.array(stack, mask=~good)
    clipped = sigma_clip(masked, sigma=sigma, maxiters=iters, axis=0, masked=True)
    return ~np.ma.getmaskarray(clipped)


def _weighted_mean_and_uncert(
    stack: np.ndarray,
    uncert_stack: np.ndarray,
    good: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    var_stack = uncert_stack.astype(np.float64) ** 2
    weights = np.zeros_like(var_stack, dtype=np.float64)
    weights[good] = 1.0 / var_stack[good]

    sumw = np.sum(weights, axis=0)
    numerator = np.sum(np.where(good, stack, 0.0) * weights, axis=0)
    coadd = np.full(stack.shape[1:], np.nan, dtype=np.float64)
    out_var = np.full(stack.shape[1:], np.nan, dtype=np.float64)

    valid = sumw > 0
    coadd[valid] = numerator[valid] / sumw[valid]
    out_var[valid] = 1.0 / sumw[valid]
    return coadd.astype(np.float32), np.sqrt(out_var).astype(np.float32)


def _unweighted_mean(stack: np.ndarray, good: np.ndarray) -> np.ndarray:
    n = np.sum(good, axis=0)
    total = np.sum(np.where(good, stack, 0.0), axis=0)
    out = np.full(stack.shape[1:], np.nan, dtype=np.float64)
    valid = n > 0
    out[valid] = total[valid] / n[valid]
    return out.astype(np.float32)


def _median_and_uncert(
    stack: np.ndarray,
    uncert_stack: Optional[np.ndarray],
    good: np.ndarray,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    arr = np.where(good, stack, np.nan)
    coadd = np.nanmedian(arr, axis=0).astype(np.float32)
    if uncert_stack is None:
        return coadd, None

    var_stack = uncert_stack.astype(np.float64) ** 2
    n = np.sum(good, axis=0)
    var_sum = np.sum(np.where(good, var_stack, 0.0), axis=0)
    out = np.full(stack.shape[1:], np.nan, dtype=np.float64)
    valid = n > 0
    out[valid] = (np.pi / 2.0) * var_sum[valid] / (n[valid] ** 2)
    return coadd, np.sqrt(out).astype(np.float32)


def coadd_products(
    products: List[CubeProduct],
    *,
    method: str = "ivar_sigma_clip_mean",
    sigma: float = 4.0,
    iters: int = 3,
    require_uncert: bool = True,
) -> fits.HDUList:
    if not products:
        raise ValueError("No products to coadd")

    shapes = [p.science.shape for p in products]
    if len(set(shapes)) != 1:
        msg = "Cubes in this group do not share the same shape; regridding needed.\n"
        for p in products:
            msg += f"  {p.path.name}: {p.science.shape}\n"
        raise ValueError(msg)

    _validate_same_wavelength_grid(products)

    has_uncert = [p.uncert is not None for p in products]
    if any(has_uncert) and not all(has_uncert):
        raise ValueError("Only some inputs have UNCERT extensions; refusing mixed uncertainty coadd")
    if require_uncert and not all(has_uncert):
        raise ValueError("Input cubes do not all have UNCERT extensions")

    has_mask = all(p.mask is not None for p in products)
    has_flags = all(p.flags is not None for p in products)
    has_noskysub = all(p.noskysub is not None for p in products)

    stack = np.stack([p.science for p in products], axis=0).astype(np.float32)
    uncert_stack = (
        np.stack([p.uncert for p in products], axis=0).astype(np.float32)
        if all(has_uncert)
        else None
    )
    mask_stack = (
        np.stack([p.mask for p in products], axis=0).astype(np.uint8)
        if has_mask
        else None
    )
    flags_stack = (
        np.stack([p.flags for p in products], axis=0)
        if has_flags
        else None
    )

    var_stack = uncert_stack.astype(np.float64) ** 2 if uncert_stack is not None else None
    good = _finite_good_mask(stack, var_stack, mask_stack, flags_stack)
    if "sigma_clip" in method:
        good = _sigma_clip_good(stack, good, sigma=sigma, iters=iters)

    if method == "ivar_sigma_clip_mean":
        if uncert_stack is None:
            raise ValueError("ivar_sigma_clip_mean requires UNCERT extensions")
        coadd, out_uncert = _weighted_mean_and_uncert(stack, uncert_stack, good)
    elif method == "sigma_clip_mean":
        coadd = _unweighted_mean(stack, good)
        out_uncert = None
    elif method == "median":
        coadd, out_uncert = _median_and_uncert(stack, uncert_stack, good)
    else:
        raise ValueError(f"Unknown coadd method: {method}")

    n_good = np.sum(good, axis=0).astype(np.int16)

    out_hdr = products[0].header.copy()
    out_hdr["NCOMBINE"] = (len(products), "Number of exposures combined")
    out_hdr["COMBMETH"] = (method, "Cube combine method")
    out_hdr["CLIPSIG"] = (float(sigma), "Sigma clipping threshold")
    out_hdr["CLIPITER"] = (int(iters), "Sigma clipping max iterations")
    out_hdr["HISTORY"] = f"Coadded {len(products)} cubes with {method}"
    if out_uncert is not None:
        out_hdr["HISTORY"] = "UNCERT propagated as StdDevUncertainty"

    hdus: List[fits.ImageHDU | fits.PrimaryHDU] = [fits.PrimaryHDU(data=coadd, header=out_hdr)]
    if out_uncert is not None:
        uhdr = fits.Header()
        uhdr["EXTNAME"] = ("UNCERT", "extension name")
        uhdr["UTYPE"] = ("StdDevUncertainty", "")
        hdus.append(fits.ImageHDU(data=out_uncert, header=uhdr, name="UNCERT"))

    mask_out = (n_good == 0).astype(np.uint8)
    hdus.append(fits.ImageHDU(data=mask_out, name="MASK"))

    if has_flags:
        flags_out = np.bitwise_or.reduce(flags_stack, axis=0)
        flags_out[n_good == 0] |= np.array(1, dtype=flags_out.dtype)
        hdus.append(fits.ImageHDU(data=flags_out, name="FLAGS"))

    if has_noskysub:
        ns_stack = np.stack([p.noskysub for p in products], axis=0).astype(np.float32)
        if uncert_stack is not None and method == "ivar_sigma_clip_mean":
            noskysub, _ = _weighted_mean_and_uncert(ns_stack, uncert_stack, good)
        elif method == "median":
            noskysub, _ = _median_and_uncert(ns_stack, None, good)
        else:
            noskysub = _unweighted_mean(ns_stack, good)
        hdus.append(fits.ImageHDU(data=noskysub, name="NOSKYSUB"))

    hdus.append(fits.ImageHDU(data=n_good, name="NEXP"))
    return fits.HDUList(hdus)


def coadd_file_group(
    files: List[Path],
    outdir: Path,
    *,
    obj: str,
    cam: str,
    method: str = "ivar_sigma_clip_mean",
    sigma: float = 4.0,
    iters: int = 3,
    require_uncert: bool = True,
) -> Path:
    products = [read_cube_product(f) for f in files]
    hdul = coadd_products(
        products,
        method=method,
        sigma=sigma,
        iters=iters,
        require_uncert=require_uncert,
    )
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{safe_filename(obj)}_{cam.lower()}_coadd_icube.fits"
    hdul.writeto(outpath, overwrite=True)
    hdul.close()
    return outpath


def coadd_directory(
    data_dir: Path,
    outdir: Path,
    *,
    list_file: Optional[Path] = None,
    method: str = "ivar_sigma_clip_mean",
    sigma: float = 4.0,
    iters: int = 3,
    require_uncert: bool = True,
) -> Dict[str, str]:
    files = discover_fits_files(data_dir, list_file=list_file)
    if not files:
        raise FileNotFoundError(f"No FITS files found in {data_dir}")

    groups = group_files_by_object_camera(files)
    outputs: Dict[str, str] = {}
    print(f"Found {len(files)} FITS files")
    print(f"Identified {len(groups)} groups by (OBJECT, CAMERA)")

    for (obj, cam), flist in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        print(f"\nGroup OBJECT={obj!r}, CAMERA={cam!r}, N={len(flist)}")
        for f in flist:
            print(f"  {f.name}")
        outpath = coadd_file_group(
            flist,
            outdir,
            obj=obj,
            cam=cam,
            method=method,
            sigma=sigma,
            iters=iters,
            require_uncert=require_uncert,
        )
        outputs[f"{obj}:{cam}"] = str(outpath)
        print(f"  -> wrote {outpath}")

    return outputs
