from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from astropy.io import fits
from astropy.stats import sigma_clip
from scipy.interpolate import UnivariateSpline

from .apertures import interactive_define_apertures, mask_from_shape, plot_apertures
from .calibration import (
    apply_o2_telluric_correction,
    apply_sensitivity,
    build_o2_transmission_template,
    plot_calibration_diagnostics,
    plot_o2_template_diagnostic,
)
from .config import TargetBackgroundApertures
from .io import get_airmass_from_header, get_lambda_axis, white_light
from .join import concat_join, plot_join_diagnostic
from .project import find_project_root
from .standard_flux import STANDARD_NAMES, list_standard_stars, reference_flux
from .utils import prompt, safe_filename


DEFAULT_SIDE_RANGES = {
    "BLUE": (3550.0, 5550.0),
    "RED": (5650.0, 8800.0),
}


@dataclass
class ExposureSpectrum:
    path: str
    side: str
    lam_path: str
    spectrum_path: str
    aperture_path: str
    airmass: Optional[float]


def _side_limits(side: str) -> Tuple[float, float]:
    return DEFAULT_SIDE_RANGES[side.upper()]


def _trim_side_arrays(side: str, lam: np.ndarray, *arrays: Optional[np.ndarray]):
    lo, hi = _side_limits(side)
    mask = np.isfinite(lam) & (lam >= lo) & (lam <= hi)
    if not np.any(mask):
        raise ValueError(f"No wavelengths for {side} in default range {lo:.0f}-{hi:.0f} A")
    out = [np.asarray(lam)[mask]]
    for arr in arrays:
        out.append(None if arr is None else np.asarray(arr)[mask])
    return tuple(out)


def _load_cube_product(path: Path) -> Tuple[np.ndarray, fits.Header, Optional[np.ndarray], Optional[np.ndarray]]:
    with fits.open(path, memmap=False) as hdul:
        science = np.array(hdul[0].data, dtype=np.float32)
        header = hdul[0].header.copy()
        uncert = None
        flags = None
        for hdu in hdul[1:]:
            name = str(hdu.header.get("EXTNAME", "")).upper()
            if name == "UNCERT" and hdu.data is not None:
                uncert = np.array(hdu.data, dtype=np.float32)
            elif name in {"MASK", "FLAGS"} and hdu.data is not None:
                arr = np.array(hdu.data)
                flags = arr if flags is None else np.bitwise_or(flags, arr)
    return science, header, uncert, flags


def _aperture_to_json(path: Path, aps: TargetBackgroundApertures) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(aps.to_dict(), f, indent=2)


def _aperture_from_json(path: Path) -> TargetBackgroundApertures:
    with open(path, "r", encoding="utf-8") as f:
        return TargetBackgroundApertures.from_dict(json.load(f))


def _extract_counts_with_uncert(
    cube: np.ndarray,
    uncert: Optional[np.ndarray],
    flags: Optional[np.ndarray],
    aps: TargetBackgroundApertures,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    nz, ny, nx = cube.shape
    m_tgt = mask_from_shape(ny, nx, aps.target)
    m_bkg = mask_from_shape(ny, nx, aps.background)
    tgt_area = float(np.sum(m_tgt))
    n_bkg = int(np.sum(m_bkg))

    counts = np.zeros(nz, dtype=float)
    sigma = np.full(nz, np.nan, dtype=float) if uncert is not None else None

    for k in range(nz):
        img = cube[k, :, :]
        bad = np.zeros((ny, nx), dtype=bool)
        if flags is not None:
            bad |= flags[k, :, :] != 0

        tgt_vals = img[m_tgt & ~bad]
        bkg_vals = img[m_bkg & ~bad]
        bkg = np.nanmedian(bkg_vals) if bkg_vals.size else 0.0
        counts[k] = np.nansum(tgt_vals) - bkg * tgt_area

        if uncert is not None:
            var = uncert[k, :, :].astype(float) ** 2
            var_tgt = np.nansum(var[m_tgt & ~bad])
            if n_bkg > 0:
                bkg_var_vals = var[m_bkg & ~bad]
                if bkg_var_vals.size:
                    var_bkg_median = (np.pi / 2.0) * np.nanmedian(bkg_var_vals) / bkg_var_vals.size
                else:
                    var_bkg_median = 0.0
            else:
                var_bkg_median = 0.0
            sigma[k] = np.sqrt(var_tgt + (tgt_area ** 2) * var_bkg_median)

    return counts, sigma


def _save_spectrum(path: Path, lam: np.ndarray, y: np.ndarray, sigma: Optional[np.ndarray], header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if sigma is None:
        arr = np.c_[lam, y]
        hdr = f"lambda_A  {header}"
    else:
        arr = np.c_[lam, y, sigma]
        hdr = f"lambda_A  {header}  sigma_{header}"
    np.savetxt(path, arr, header=hdr)


def _coadd_1d_spectra(
    spectra: List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]],
    *,
    sigma_clip_value: float = 4.0,
    maxiters: int = 3,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    if not spectra:
        raise ValueError("No spectra to coadd")

    lam_ref = spectra[0][0]
    values = []
    sigmas = []
    have_sigma = all(s[2] is not None for s in spectra)

    for lam, y, sig in spectra:
        if lam.shape == lam_ref.shape and np.allclose(lam, lam_ref, rtol=0.0, atol=1e-7):
            y_i = y
            sig_i = sig
        else:
            y_i = np.interp(lam_ref, lam, y, left=np.nan, right=np.nan)
            sig_i = np.interp(lam_ref, lam, sig, left=np.nan, right=np.nan) if sig is not None else None
        values.append(y_i)
        if have_sigma:
            sigmas.append(sig_i)

    stack = np.asarray(values, dtype=float)
    good = np.isfinite(stack)
    if have_sigma:
        sigma_stack = np.asarray(sigmas, dtype=float)
        good &= np.isfinite(sigma_stack) & (sigma_stack > 0)
    else:
        sigma_stack = None

    clipped = sigma_clip(np.ma.array(stack, mask=~good), sigma=sigma_clip_value, maxiters=maxiters, axis=0)
    good = ~np.ma.getmaskarray(clipped)
    n_good = np.sum(good, axis=0).astype(np.int16)

    out = np.full(lam_ref.shape, np.nan, dtype=float)
    out_sigma = np.full(lam_ref.shape, np.nan, dtype=float) if have_sigma else None

    if have_sigma and sigma_stack is not None:
        var = sigma_stack ** 2
        weights = np.zeros_like(var)
        weights[good] = 1.0 / var[good]
        sumw = np.sum(weights, axis=0)
        valid = sumw > 0
        out[valid] = np.sum(np.where(good, stack, 0.0) * weights, axis=0)[valid] / sumw[valid]
        out_sigma[valid] = np.sqrt(1.0 / sumw[valid])
    else:
        valid = n_good > 0
        out[valid] = np.sum(np.where(good, stack, 0.0), axis=0)[valid] / n_good[valid]

    return lam_ref, out, out_sigma, n_good


def _side_files(object_dir: Path, side: str) -> List[Path]:
    return sorted((object_dir / side).glob("*_icubes.fits"))


def _project_calib_dir(object_dir: Path, calib_dir: Optional[Path]) -> Path:
    if calib_dir is not None:
        return calib_dir.expanduser().resolve()
    root = find_project_root(object_dir)
    if root is not None:
        return root / "calibrations"
    return object_dir / "calibrations"


def _load_registry(calib_dir: Path) -> Dict[str, object]:
    path = calib_dir / "calibration_registry.json"
    if not path.exists():
        return {"standards": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_registry(calib_dir: Path, registry: Dict[str, object]) -> None:
    calib_dir.mkdir(parents=True, exist_ok=True)
    with open(calib_dir / "calibration_registry.json", "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def _choose_calibration(calib_dir: Path, side: str) -> Optional[Dict[str, object]]:
    registry = _load_registry(calib_dir)
    matches = [item for item in registry.get("standards", []) if item.get("side") == side]
    if not matches:
        return None
    print(f"\nAvailable {side} calibrations:")
    for i, item in enumerate(matches):
        print(f"  {i}: {item.get('standard_name')}  {item.get('sensitivity_file')}")
    if len(matches) == 1:
        ans = prompt("Use this calibration? (y/n)", "y").lower()
        return matches[0] if ans.startswith("y") else None
    idx = int(prompt("Calibration index", "0"))
    return matches[idx]


def _choose_standard_star(default_name: str) -> Tuple[int, str]:
    standards = list_standard_stars()
    print("\nAvailable AB standard stars:")
    for idx, name in standards:
        print(f"  {idx:2d}: {name}")
    default_id = None
    clean_default = default_name.replace("_", " ").strip().lower()
    for idx, name in standards:
        if clean_default and clean_default == name.lower():
            default_id = idx
            break
    default_text = str(default_id) if default_id is not None else ""
    raw = prompt("Standard star number", default_text if default_text else None)
    star_id = int(raw)
    if star_id not in STANDARD_NAMES:
        raise ValueError(f"Unknown standard star number: {star_id}")
    return star_id, STANDARD_NAMES[star_id]


def _continuum_from_points(lam: np.ndarray, points: List[Tuple[float, float]]) -> np.ndarray:
    points = sorted(points, key=lambda item: item[0])
    xp = np.asarray([p[0] for p in points], dtype=float)
    yp = np.asarray([p[1] for p in points], dtype=float)
    if len(points) >= 4:
        spline = UnivariateSpline(xp, yp, s=0, k=min(3, len(points) - 1))
        return spline(lam)
    return np.interp(lam, xp, yp, left=np.nan, right=np.nan)


def _nearest_point(points: List[Tuple[float, float]], x: float, y: float) -> Optional[int]:
    if not points:
        return None
    arr = np.asarray(points, dtype=float)
    dx = (arr[:, 0] - x) / max(np.nanmax(arr[:, 0]) - np.nanmin(arr[:, 0]), 1.0)
    dy = (arr[:, 1] - y) / max(np.nanmax(arr[:, 1]) - np.nanmin(arr[:, 1]), 1.0)
    return int(np.argmin(dx * dx + dy * dy))


def _nearest_point_pixels(ax, points: List[Tuple[float, float]], event) -> Tuple[Optional[int], float]:
    if not points:
        return None, np.inf
    pts = ax.transData.transform(np.asarray(points, dtype=float))
    mouse = np.asarray([event.x, event.y], dtype=float)
    dist = np.hypot(pts[:, 0] - mouse[0], pts[:, 1] - mouse[1])
    idx = int(np.argmin(dist))
    return idx, float(dist[idx])


def interactive_continuum_spline(
    lam: np.ndarray,
    counts: np.ndarray,
    ref_flux: np.ndarray,
    *,
    title: str,
    show: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pick continuum points on a standard spectrum and return continuum + sensitivity."""
    good = np.isfinite(lam) & np.isfinite(counts) & np.isfinite(ref_flux) & (ref_flux > 0)
    if np.count_nonzero(good) < 8:
        raise ValueError("Not enough finite standard/reference samples to build sensitivity")

    lam_g = lam[good]
    counts_g = counts[good]
    ref_g = ref_flux[good]

    n_init = min(12, max(6, lam_g.size // 250))
    qs = np.linspace(5, 95, n_init)
    x_init = np.nanpercentile(lam_g, qs)
    y_init = np.interp(x_init, lam_g, counts_g)
    points: List[Tuple[float, float]] = list(zip(x_init, y_init))
    accepted = {"done": False}

    fig, (ax_obs, ax_sens) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    ax_ref = ax_obs.twinx()
    original_view = {"xlim": None, "obs_ylim": None, "sens_ylim": None}
    zoom = {"active": False, "start": None, "axis": None, "patch": None}

    def redraw() -> None:
        current_view = {
            "xlim": ax_obs.get_xlim() if original_view["xlim"] is not None else None,
            "obs_ylim": ax_obs.get_ylim() if original_view["obs_ylim"] is not None else None,
        }
        ax_obs.clear()
        ax_ref.clear()
        ax_sens.clear()
        cont = _continuum_from_points(lam_g, points) if len(points) >= 2 else np.full_like(lam_g, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            sens = ref_g / cont

        ax_obs.plot(lam_g, counts_g, lw=0.8, color="0.35", label="Extracted standard")
        if len(points) >= 2:
            ax_obs.plot(lam_g, cont, lw=1.5, color="tab:red", label="Continuum spline")
        if points:
            xp = [p[0] for p in points]
            yp = [p[1] for p in points]
            ax_obs.scatter(xp, yp, s=35, color="tab:red", zorder=5)
        ax_obs.set_ylabel("Observed counts")
        ax_obs.set_title(
            title
            + "\nleft-click add point, drag marker to move, right-click delete, z=zoom box, o=original zoom, a=accept, r=reset, q=quit"
        )
        ax_obs.legend(loc="best")
        ax_obs.grid(alpha=0.2)

        ax_ref.plot(lam_g, ref_g, lw=0.9, color="tab:blue", alpha=0.55, label="AB reference flux")
        ax_ref.set_ylabel("Reference flux (1e-16 cgs/A)")
        ax_ref.yaxis.set_label_position("right")
        ax_ref.yaxis.tick_right()
        ax_ref.tick_params(axis="y", colors="tab:blue")
        ax_ref.yaxis.label.set_color("tab:blue")
        ax_obs.set_zorder(ax_ref.get_zorder() + 1)
        ax_obs.patch.set_visible(False)

        ax_sens.plot(lam_g, sens, lw=1.0, color="tab:green")
        ax_sens.set_ylabel("Sensitivity")
        ax_sens.set_xlabel("Wavelength (A)")
        ax_sens.grid(alpha=0.2)
        sens_good = sens[np.isfinite(sens) & (sens > 0)]
        if sens_good.size >= 5:
            ylo, yhi = np.nanpercentile(sens_good, [2, 98])
            if np.isfinite(ylo) and np.isfinite(yhi) and yhi > ylo:
                pad = 0.08 * (yhi - ylo)
                ax_sens.set_ylim(ylo - pad, yhi + pad)
        if original_view["xlim"] is None:
            original_view["xlim"] = ax_obs.get_xlim()
            original_view["obs_ylim"] = ax_obs.get_ylim()
            original_view["sens_ylim"] = ax_sens.get_ylim()
        elif current_view["xlim"] is not None:
            ax_obs.set_xlim(current_view["xlim"])
            ax_obs.set_ylim(current_view["obs_ylim"])
            ax_ref.relim()
            ax_ref.autoscale_view(scalex=False, scaley=True)
        fig.canvas.draw_idle()

    drag = {"idx": None}

    def event_obs_xy(event) -> Optional[Tuple[float, float]]:
        if event.inaxes not in (ax_obs, ax_ref) or event.x is None or event.y is None:
            return None
        x, y = ax_obs.transData.inverted().transform((event.x, event.y))
        xlim = ax_obs.get_xlim()
        ylim = ax_obs.get_ylim()
        if not (min(xlim) <= x <= max(xlim) and min(ylim) <= y <= max(ylim)):
            return None
        return float(x), float(y)

    def event_data_xy(event) -> Optional[Tuple[float, float]]:
        if event.inaxes not in (ax_obs, ax_sens) or event.x is None or event.y is None:
            return None
        x, y = event.inaxes.transData.inverted().transform((event.x, event.y))
        return float(x), float(y)

    def clear_zoom_patch() -> None:
        if zoom["patch"] is not None:
            try:
                zoom["patch"].remove()
            except Exception:
                pass
            zoom["patch"] = None

    def draw_zoom_patch(event) -> None:
        if zoom["start"] is None or zoom["axis"] is None:
            return
        xy = event_data_xy(event)
        if xy is None:
            return
        x0, y0 = zoom["start"]
        x1, y1 = xy
        clear_zoom_patch()
        rect = Rectangle(
            (min(x0, x1), min(y0, y1)),
            abs(x1 - x0),
            abs(y1 - y0),
            fill=False,
            edgecolor="tab:purple",
            linewidth=1.5,
            linestyle="--",
        )
        zoom["axis"].add_patch(rect)
        zoom["patch"] = rect
        fig.canvas.draw_idle()

    def on_press(event):
        if zoom["active"]:
            xy = event_data_xy(event)
            if xy is not None:
                zoom["start"] = xy
                zoom["axis"] = event.inaxes
            return

        xy = event_obs_xy(event)
        if xy is None:
            return
        x, y = xy
        if event.button == 1:
            idx, dist_px = _nearest_point_pixels(ax_obs, points, event)
            if idx is not None and dist_px <= 8.0:
                drag["idx"] = idx
                return
            points.append((x, y))
            redraw()
        elif event.button == 3:
            idx = _nearest_point(points, x, y)
            if idx is not None and len(points) > 2:
                points.pop(idx)
                redraw()

    def on_motion(event):
        if zoom["active"]:
            draw_zoom_patch(event)
            return
        xy = event_obs_xy(event)
        if drag["idx"] is None or xy is None:
            return
        points[drag["idx"]] = xy
        redraw()

    def on_release(event):
        if zoom["active"]:
            xy = event_data_xy(event)
            if zoom["start"] is not None and xy is not None and zoom["axis"] is not None:
                x0, y0 = zoom["start"]
                x1, y1 = xy
                if abs(x1 - x0) > 0 and abs(y1 - y0) > 0:
                    ax_obs.set_xlim(min(x0, x1), max(x0, x1))
                    if zoom["axis"] is ax_obs:
                        ax_obs.set_ylim(min(y0, y1), max(y0, y1))
                    elif zoom["axis"] is ax_sens:
                        ax_sens.set_ylim(min(y0, y1), max(y0, y1))
                    fig.canvas.draw_idle()
            clear_zoom_patch()
            zoom["active"] = False
            zoom["start"] = None
            zoom["axis"] = None
            return
        drag["idx"] = None

    def on_key(event):
        if event.key in ("a", "enter", "return"):
            accepted["done"] = True
            plt.close(fig)
        elif event.key == "q":
            plt.close(fig)
        elif event.key == "z":
            zoom["active"] = True
            zoom["start"] = None
            zoom["axis"] = None
            clear_zoom_patch()
            ax_obs.set_title("Zoom mode: drag a box on either panel. Press o for original zoom.")
            fig.canvas.draw_idle()
        elif event.key == "o":
            clear_zoom_patch()
            zoom["active"] = False
            if original_view["xlim"] is not None:
                ax_obs.set_xlim(original_view["xlim"])
                ax_obs.set_ylim(original_view["obs_ylim"])
                ax_sens.set_ylim(original_view["sens_ylim"])
                fig.canvas.draw_idle()
        elif event.key == "r":
            points.clear()
            points.extend(zip(x_init, y_init))
            redraw()

    cids = [
        fig.canvas.mpl_connect("button_press_event", on_press),
        fig.canvas.mpl_connect("motion_notify_event", on_motion),
        fig.canvas.mpl_connect("button_release_event", on_release),
        fig.canvas.mpl_connect("key_press_event", on_key),
    ]
    redraw()
    plt.show()
    for cid in cids:
        fig.canvas.mpl_disconnect(cid)

    if not accepted["done"]:
        raise RuntimeError("Continuum spline was not accepted")
    continuum = _continuum_from_points(lam, points)
    with np.errstate(divide="ignore", invalid="ignore"):
        sensitivity = ref_flux / continuum
    return continuum, sensitivity


def _extract_side(object_dir: Path, side: str, *, show_plots: bool, redo_apertures: bool) -> Optional[Path]:
    files = _side_files(object_dir, side)
    if not files:
        return None

    spectra_dir = object_dir / "extracted" / side
    ap_dir = object_dir / "apertures" / side
    diag_dir = object_dir / "diagnostics" / side
    coadd_dir = object_dir / "coadded_spectra"

    extracted: List[ExposureSpectrum] = []
    spectra_for_coadd: List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = []
    current_aps: Optional[TargetBackgroundApertures] = None

    for i, path in enumerate(files):
        exposure_label = f"{object_dir.name} {side} exposure {i + 1}/{len(files)}"
        cube, hdr, uncert, flags = _load_cube_product(path)
        lam = get_lambda_axis(hdr, cube.shape)
        lo, hi = _side_limits(side)
        img = white_light(cube, lam, lam_min=lo, lam_max=hi)

        ap_path = ap_dir / f"{path.stem}_aperture.json"
        if ap_path.exists() and not redo_apertures:
            aps = _aperture_from_json(ap_path)
        elif current_aps is not None:
            plot_apertures(img, current_aps, diag_dir / f"{path.stem}_aperture_reuse_preview.png",
                           title=f"{exposure_label}: aperture reuse preview", show=show_plots)
            ans = prompt(f"[{exposure_label}] Reuse previous aperture? (Y/n)", "y").lower()
            aps = current_aps if ans.startswith("y") else interactive_define_apertures(img, exposure_label, show=show_plots)
        else:
            aps = interactive_define_apertures(img, exposure_label, show=show_plots)

        current_aps = aps
        _aperture_to_json(ap_path, aps)
        plot_apertures(img, aps, diag_dir / f"{path.stem}_aperture.png", title=f"{exposure_label}: aperture", show=False)

        counts, sigma = _extract_counts_with_uncert(cube, uncert, flags, aps)
        lam, counts, sigma = _trim_side_arrays(side, lam, counts, sigma)
        spec_path = spectra_dir / f"{path.stem}_counts.txt"
        _save_spectrum(spec_path, lam, counts, sigma, "counts")
        spectra_for_coadd.append((lam, counts, sigma))
        extracted.append(
            ExposureSpectrum(
                path=str(path),
                side=side,
                lam_path=str(spec_path),
                spectrum_path=str(spec_path),
                aperture_path=str(ap_path),
                airmass=get_airmass_from_header(hdr),
            )
        )
        print(f"Extracted {exposure_label} -> {spec_path}")

    lam_c, counts_c, sigma_c, n_good = _coadd_1d_spectra(spectra_for_coadd)
    out_path = coadd_dir / f"{object_dir.name}_{side}_counts_coadd.txt"
    _save_spectrum(out_path, lam_c, counts_c, sigma_c, "counts_coadd")
    np.savetxt(coadd_dir / f"{object_dir.name}_{side}_nexp.txt", np.c_[lam_c, n_good], header="lambda_A  n_exposures_used")

    state_path = object_dir / "extraction_state.json"
    state = {}
    if state_path.exists():
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
    state.setdefault("sides", {})[side] = {
        "coadd_counts": str(out_path),
        "exposures": [asdict(item) for item in extracted],
    }
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    print(f"Coadded {side} 1D spectra -> {out_path}")
    return out_path


def _load_txt_spectrum(path: Path) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    arr = np.loadtxt(path, comments="#")
    if arr.ndim == 1:
        arr = arr[None, :]
    lam = arr[:, 0]
    y = arr[:, 1]
    sigma = arr[:, 2] if arr.shape[1] >= 3 else None
    return lam, y, sigma


def _build_standard_calibrations(
    object_dir: Path,
    coadd_paths: Dict[str, Path],
    calib_dir: Path,
    *,
    show_plots: bool,
) -> None:
    standard_name = object_dir.name
    registry = _load_registry(calib_dir)
    registry.setdefault("standards", [])

    for side, counts_path in coadd_paths.items():
        lam_std, counts, _ = _load_txt_spectrum(counts_path)
        lam_std, counts = _trim_side_arrays(side, lam_std, counts)
        star_id, star_name = _choose_standard_star(standard_name)
        flux_ref = reference_flux(star_id, lam_std, scaled_1e16=True)
        continuum, sens = interactive_continuum_spline(
            lam_std,
            counts,
            flux_ref,
            title=f"{standard_name} {side}: continuum fit for {star_name}",
            show=show_plots,
        )
        lam_cal_std, flux_cal_std = apply_sensitivity(lam_std, sens, lam_std, counts)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = flux_ref / continuum

        outdir = calib_dir / standard_name / side
        outdir.mkdir(parents=True, exist_ok=True)
        sens_path = outdir / f"sensitivity_{side}.txt"
        continuum_path = outdir / f"observed_continuum_{side}.txt"
        ref_path = outdir / f"ab_reference_flux_{side}.txt"
        np.savetxt(sens_path, np.c_[lam_std, sens], header="lambda_A  S_lambda")
        np.savetxt(continuum_path, np.c_[lam_std, continuum], header="lambda_A  observed_continuum_counts")
        np.savetxt(ref_path, np.c_[lam_std, flux_ref], header="lambda_A  reference_flux_1e-16_erg_s_cm2_A")

        item: Dict[str, object] = {
            "standard_name": standard_name,
            "ab_standard_id": star_id,
            "ab_standard_name": star_name,
            "side": side,
            "wavelength_range_A": list(_side_limits(side)),
            "counts_file": str(counts_path),
            "reference_flux_file": str(ref_path),
            "observed_continuum_file": str(continuum_path),
            "sensitivity_file": str(sens_path),
            "reference_units": "1e-16 erg/s/cm^2/A",
        }

        tell_path = None
        mask_path = None
        if side == "RED":
            t_o2, o2_mask = build_o2_transmission_template(
                lam_ref=lam_std,
                F_ref=flux_ref,
                lam_std=lam_std,
                C_std=counts,
                S_sens=sens,
                o2_windows=[(6860, 6925), (7590, 7670)],
            )
            tell_path = outdir / "telluric_O2_template_RED.txt"
            np.savetxt(tell_path, np.c_[lam_std, t_o2, o2_mask.astype(int)],
                       header="lambda_A  T_O2_std  in_O2mask")
            mask_path = str(tell_path)
            plot_o2_template_diagnostic(lam_std, t_o2, [(6860, 6925), (7590, 7670)],
                                        outdir / "O2_template_RED.png", show=show_plots)
            item["telluric_file"] = str(tell_path)
            item["x_std"] = None

        plot_calibration_diagnostics(
            side=side,
            std_name=standard_name,
            lam_std=lam_std,
            C_std=counts,
            lam_ref=lam_std,
            F_ref=flux_ref,
            ratio=ratio,
            S=sens,
            F_std_cal=flux_cal_std,
            outdir=outdir / "diagnostics",
            show=show_plots,
        )

        registry["standards"] = [
            old for old in registry["standards"]
            if not (old.get("standard_name") == standard_name and old.get("side") == side)
        ]
        registry["standards"].append(item)
        print(f"Saved {side} calibration -> {sens_path}")
        if mask_path:
            print(f"Saved RED telluric template -> {tell_path}")

    _save_registry(calib_dir, registry)
    print(f"Updated calibration registry -> {calib_dir / 'calibration_registry.json'}")


def _apply_science_calibrations(
    object_dir: Path,
    coadd_paths: Dict[str, Path],
    calib_dir: Path,
    *,
    show_plots: bool,
) -> None:
    flux_paths: Dict[str, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = {}
    flux_dir = object_dir / "fluxcal"
    final_dir = object_dir / "final"
    flux_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    for side, counts_path in coadd_paths.items():
        cal = _choose_calibration(calib_dir, side)
        if cal is None:
            print(f"No {side} calibration selected; leaving counts-only product.")
            continue

        lam_counts, counts, sigma_counts = _load_txt_spectrum(counts_path)
        lam_counts, counts, sigma_counts = _trim_side_arrays(side, lam_counts, counts, sigma_counts)
        sens_arr = np.loadtxt(cal["sensitivity_file"], comments="#")
        lam_sens = sens_arr[:, 0]
        sens = sens_arr[:, 1]
        lam_sens, sens = _trim_side_arrays(side, lam_sens, sens)
        lam_flux, flux = apply_sensitivity(lam_sens, sens, lam_counts, counts)
        sigma_flux = None
        if sigma_counts is not None:
            sigma_interp = np.interp(lam_sens, lam_counts, sigma_counts, left=np.nan, right=np.nan)
            sigma_flux = np.abs(sens) * sigma_interp

        if side == "RED" and cal.get("telluric_file"):
            tell = np.loadtxt(cal["telluric_file"], comments="#")
            t_std = tell[:, 1]
            o2_mask = tell[:, 2].astype(bool)
            flux = apply_o2_telluric_correction(flux, t_std, None, None, o2_mask)
            if sigma_flux is not None:
                sigma_flux = apply_o2_telluric_correction(sigma_flux, t_std, None, None, o2_mask)

        out_path = flux_dir / f"{object_dir.name}_{side}_fluxcal.txt"
        _save_spectrum(out_path, lam_flux, flux, sigma_flux, "flux")
        flux_paths[side] = (lam_flux, flux, sigma_flux)
        print(f"Saved flux-calibrated {side} spectrum -> {out_path}")

    if "BLUE" in flux_paths and "RED" in flux_paths:
        lam_b, flux_b, sig_b = flux_paths["BLUE"]
        lam_r, flux_r, sig_r = flux_paths["RED"]
        lam_j, flux_j = concat_join(lam_b, flux_b, lam_r, flux_r)
        if sig_b is not None and sig_r is not None:
            lam_s, sig_j = concat_join(lam_b, sig_b, lam_r, sig_r)
            order = np.argsort(lam_s)
            sig_j = sig_j[order] if not np.allclose(lam_s, lam_j) else sig_j
        else:
            sig_j = None
        out_path = final_dir / f"{object_dir.name}_BLUE+RED_spectrum.txt"
        _save_spectrum(out_path, lam_j, flux_j, sig_j, "flux")
        plot_join_diagnostic(object_dir.name, lam_b, flux_b, lam_r, flux_r,
                             outpng=final_dir / f"{object_dir.name}_joined.png", show=show_plots)
        print(f"Saved joined spectrum -> {out_path}")


def extract_object(
    object_dir: Path,
    *,
    calib_dir: Optional[Path] = None,
    standard: Optional[bool] = None,
    side: str = "both",
    show_plots: bool = False,
    redo_apertures: bool = False,
) -> None:
    object_dir = object_dir.expanduser().resolve()
    if not object_dir.exists():
        raise FileNotFoundError(object_dir)

    if standard is None:
        standard = prompt("Is this a standard star? (y/n)", "n").lower().startswith("y")

    side = side.lower().strip()
    if side == "both":
        sides = ("BLUE", "RED")
    elif side == "blue":
        sides = ("BLUE",)
    elif side == "red":
        sides = ("RED",)
    else:
        raise ValueError("side must be one of: blue, red, both")

    coadd_paths: Dict[str, Path] = {}
    for side_name in sides:
        path = _extract_side(object_dir, side_name, show_plots=show_plots, redo_apertures=redo_apertures)
        if path is not None:
            coadd_paths[side_name] = path

    if not coadd_paths:
        raise FileNotFoundError(f"No requested-side *_icubes.fits files found under {object_dir}")

    resolved_calib_dir = _project_calib_dir(object_dir, calib_dir)
    if standard:
        _build_standard_calibrations(object_dir, coadd_paths, resolved_calib_dir, show_plots=show_plots)
    else:
        _apply_science_calibrations(object_dir, coadd_paths, resolved_calib_dir, show_plots=show_plots)
