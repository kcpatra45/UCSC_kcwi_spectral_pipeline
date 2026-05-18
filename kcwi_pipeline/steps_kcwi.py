# UPDATED steps_kcwi.py (arm-specific spectral trim + spline approval loop)

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

from .config import PipelineConfig, save_config
from .pipeline import Step, PipelineContext
from .utils import SIDE_PAT, die, prompt, choose_from_list, safe_filename, parse_ranges, parse_trim, apply_trim_and_exclude
from .io import load_cube, get_lambda_axis, white_light, get_airmass_from_header
from .apertures import interactive_define_apertures, plot_apertures
from .extract import extract_1d_counts
from .calibration import (
    load_reference_flux,
    build_sensitivity,
    apply_sensitivity,
    build_o2_transmission_template,
    apply_o2_telluric_correction,
    plot_calibration_diagnostics,
    plot_o2_template_diagnostic,
    plot_o2_before_after,
)
from .join import concat_join, plot_join_diagnostic, interactive_rescale_and_approve_flux
from .trim import spectral_trim_cube
from .coadd import coadd_directory


def _ensure_dirs(ctx: PipelineContext) -> None:
    for d in (ctx.coadd_outdir, ctx.apdir, ctx.caldir, ctx.countsdir, ctx.fluxdir, ctx.finaldir, ctx.diagdir):
        d.mkdir(parents=True, exist_ok=True)



def _maybe_spectral_trim(ctx: PipelineContext, side: str, cube: np.ndarray, lam: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Apply arm-specific spectral trim if enabled in config."""
    side_u = side.upper()
    arm_cfg = ctx.cfg.spectral_trim.blue if side_u == "BLUE" else ctx.cfg.spectral_trim.red
    if not getattr(arm_cfg, "enabled", False):
        return cube, lam
    lam_min = arm_cfg.lam_min
    lam_max = arm_cfg.lam_max
    cube_t, lam_t = spectral_trim_cube(cube, lam, lam_min=lam_min, lam_max=lam_max)
    return cube_t, lam_t


def step00_setup(ctx: PipelineContext) -> None:
    """Create output directories and write config.json."""
    _ensure_dirs(ctx)
    save_config(ctx.cfg, ctx.config_path)


def step01_coadd_level2_cubes(ctx: PipelineContext) -> None:
    """Optionally coadd individual KCWI DRP Level 2 cubes before discovery."""
    if not ctx.cfg.coadd.enabled:
        print("Coadd step disabled; using existing coadd_dir.")
        return

    input_dir = Path(ctx.cfg.coadd.input_dir or ctx.cfg.coadd_dir).expanduser().resolve()
    outdir = Path(ctx.cfg.coadd.outdir).expanduser().resolve() if ctx.cfg.coadd.outdir else ctx.coadd_outdir
    list_file = Path(ctx.cfg.coadd.list_file).expanduser().resolve() if ctx.cfg.coadd.list_file else None

    outputs = coadd_directory(
        input_dir,
        outdir,
        list_file=list_file,
        method=ctx.cfg.coadd.method,
        sigma=ctx.cfg.coadd.sigma,
        iters=ctx.cfg.coadd.iters,
        require_uncert=ctx.cfg.coadd.require_uncert,
    )

    ctx.cfg.coadd_dir = str(outdir)
    save_config(ctx.cfg, ctx.config_path)
    ctx.state.artifacts["coadd_outputs"] = outputs


def step01_discover_coadds(ctx: PipelineContext) -> None:
    """Discover <obj>_(blue|red)_coadd_icube.fits in coadd_dir."""
    coadd_dir = Path(ctx.cfg.coadd_dir).expanduser().resolve()
    files = list(coadd_dir.glob("*_blue_coadd_icube.fits")) + list(coadd_dir.glob("*_red_coadd_icube.fits"))
    if not files:
        die(f"No coadds found in {coadd_dir} matching *_blue/red_coadd_icube.fits")

    cube_map: Dict[str, Dict[str, str]] = {}
    for f in files:
        m = SIDE_PAT.match(f.name)
        if not m:
            continue
        obj = m.group("obj")
        side = m.group("side").upper()
        cube_map.setdefault(obj, {})[side] = str(f.resolve())

    objects = sorted(cube_map.keys())
    print(f"Discovered {len(objects)} objects in {coadd_dir}")
    if len(objects) <= 30:
        print("Objects:", ", ".join(objects))
    else:
        print("Objects (first 30):", ", ".join(objects[:30]), "...")

    ctx.state.artifacts["cube_map"] = cube_map
    ctx.state.artifacts["objects"] = objects


def step02_choose_standards_and_refs(ctx: PipelineContext) -> None:
    """Choose flux standards + reference flux files (BLUE/RED)."""
    objects: List[str] = ctx.state.artifacts.get("objects", [])
    if not objects:
        die("No objects in state; run step01_discover_coadds first.")

    if ctx.cfg.fluxstd_blue is None:
        ctx.cfg.fluxstd_blue = choose_from_list("Choose FLUX standard for BLUE:", objects)
    if ctx.cfg.fluxstd_red is None:
        ctx.cfg.fluxstd_red = choose_from_list("Choose FLUX standard for RED:", objects)

    if ctx.cfg.ref_blue is None:
        ctx.cfg.ref_blue = str(Path(prompt("Path to BLUE reference flux file:")).expanduser().resolve())
    if ctx.cfg.ref_red is None:
        ctx.cfg.ref_red = str(Path(prompt("Path to RED reference flux file:")).expanduser().resolve())

    if not Path(ctx.cfg.ref_blue).exists():
        die(f"Not found: {ctx.cfg.ref_blue}")
    if not Path(ctx.cfg.ref_red).exists():
        die(f"Not found: {ctx.cfg.ref_red}")

    save_config(ctx.cfg, ctx.config_path)


def step03_define_default_apertures(ctx: PipelineContext) -> None:
    """Define default target/background apertures independently for BLUE and RED."""
    cube_map: Dict[str, Dict[str, str]] = ctx.state.artifacts.get("cube_map", {})
    if not cube_map:
        die("No cube_map in state; run step01_discover_coadds first.")

    # Use the chosen standard cube as a convenient image for defining apertures.
    std_b = ctx.cfg.fluxstd_blue
    std_r = ctx.cfg.fluxstd_red
    if std_b is None or std_r is None:
        die("Standards not set; run step02_choose_standards_and_refs first.")

    if ctx.cfg.apertures_blue is None:
        if "BLUE" not in cube_map.get(std_b, {}):
            die(f"Chosen BLUE standard '{std_b}' has no BLUE cube.")
        cube, hdr = load_cube(Path(cube_map[std_b]["BLUE"]))
        lam = get_lambda_axis(hdr, cube.shape)
        cube, lam = _maybe_spectral_trim(ctx, "BLUE", cube, lam)
        img = white_light(cube, lam)
        ctx.cfg.apertures_blue = interactive_define_apertures(img, side_label="BLUE", show=ctx.cfg.show_plots)
        plot_apertures(img, ctx.cfg.apertures_blue, ctx.apdir / f"DEFAULT_BLUE_apertures.png",
                       title=f"DEFAULT BLUE apertures (defined on {std_b})", show=False)

    if ctx.cfg.apertures_red is None:
        if "RED" not in cube_map.get(std_r, {}):
            die(f"Chosen RED standard '{std_r}' has no RED cube.")
        cube, hdr = load_cube(Path(cube_map[std_r]["RED"]))
        lam = get_lambda_axis(hdr, cube.shape)
        cube, lam = _maybe_spectral_trim(ctx, "RED", cube, lam)
        img = white_light(cube, lam)
        ctx.cfg.apertures_red = interactive_define_apertures(img, side_label="RED", show=ctx.cfg.show_plots)
        plot_apertures(img, ctx.cfg.apertures_red, ctx.apdir / f"DEFAULT_RED_apertures.png",
                       title=f"DEFAULT RED apertures (defined on {std_r})", show=False)

    save_config(ctx.cfg, ctx.config_path)


def _extract_object_side(ctx: PipelineContext, objname: str, side: str) -> Tuple[np.ndarray, np.ndarray]:
    cube_map: Dict[str, Dict[str, str]] = ctx.state.artifacts["cube_map"]
    cube_path = Path(cube_map[objname][side])
    cube, hdr = load_cube(cube_path)
    lam = get_lambda_axis(hdr, cube.shape)
    cube, lam = _maybe_spectral_trim(ctx, side, cube, lam)
    img = white_light(cube, lam)

    default_aps = ctx.cfg.apertures_blue if side == "BLUE" else ctx.cfg.apertures_red
    if default_aps is None:
        raise RuntimeError(f"Default apertures for {side} are not defined.")

    aps = default_aps

    if ctx.cfg.interactive:
        ans = prompt(f"[{objname} {side}] Use default apertures? (y=default, n=define override)", "y").lower()
        if ans != "y":
            aps = interactive_define_apertures(img, side_label=f"{objname} {side}", show=ctx.cfg.show_plots)

    # Save overlay used for this extraction
    plot_apertures(img, aps, ctx.apdir / f"{safe_filename(objname)}_{side}_aperture.png",
                   title=f"{objname} {side} apertures", show=False)

    counts = extract_1d_counts(cube, lam, aps)
    return lam, counts


def step04_extract_standards_build_calibration(ctx: PipelineContext) -> None:
    """Extract standards, build sensitivity curves, build O2 template, write calibration products.

    This step is interactive by default: it will show sensitivity diagnostics and allow the user
    to adjust spline_s until they approve the sensitivity curve for each arm.
    """
    cube_map: Dict[str, Dict[str, str]] = ctx.state.artifacts.get("cube_map", {})
    if not cube_map:
        die("No cube_map in state; run step01_discover_coadds first.")

    if ctx.cfg.fluxstd_blue is None or ctx.cfg.fluxstd_red is None:
        die("Standards not set; run step02_choose_standards_and_refs first.")
    if ctx.cfg.apertures_blue is None or ctx.cfg.apertures_red is None:
        die("Apertures not defined; run step03_define_default_apertures first.")

    def _arm_lims(side: str) -> Tuple[Optional[float], Optional[float]]:
        st = ctx.cfg.spectral_trim
        if side.upper() == "BLUE":
            return st.blue.lam_min, st.blue.lam_max
        if side.upper() == "RED":
            return st.red.lam_min, st.red.lam_max
        return None, None

    # ----------------
    # BLUE calibration
    # ----------------
    stdB = ctx.cfg.fluxstd_blue
    if "BLUE" not in cube_map.get(stdB, {}):
        die(f"Chosen BLUE flux standard '{stdB}' has no BLUE cube.")

    lam_b_std, C_b_std = _extract_object_side(ctx, stdB, "BLUE")
    np.savetxt(
        ctx.caldir / f"std_counts_{safe_filename(stdB)}_BLUE.flm",
        np.c_[lam_b_std, C_b_std],
        header="lambda_A  counts_tgt_bkgsub",
    )

    lam_b_ref_full, F_b_ref_full = load_reference_flux(Path(ctx.cfg.ref_blue))
    lam_b_min, lam_b_max = _arm_lims("BLUE")

    spline_b = ctx.cfg.calibration.spline_s_init
    while True:
        lam_b_ref, F_b_ref, ratio_b, S_b = build_sensitivity(
            lam_ref=lam_b_ref_full,
            F_ref=F_b_ref_full,
            lam_std=lam_b_std,
            C_std=C_b_std,
            spline_s=spline_b,
            exclude_windows=None,
            lam_min=lam_b_min,
            lam_max=lam_b_max,
        )

        lam_b_cal_std, F_b_cal_std = apply_sensitivity(lam_b_ref, S_b, lam_b_std, C_b_std)

        np.savetxt(
            ctx.caldir / "sensitivity_BLUE.txt",
            np.c_[lam_b_ref, S_b],
            header=f"lambda_A  S_lambda   (spline_s={spline_b})",
        )

        plot_calibration_diagnostics(
            side="BLUE",
            std_name=stdB,
            lam_std=lam_b_std,
            C_std=C_b_std,
            lam_ref=lam_b_ref,
            F_ref=F_b_ref,
            ratio=ratio_b,
            S=S_b,
            F_std_cal=F_b_cal_std,
            outdir=ctx.diagdir,
            show=ctx.cfg.show_plots,
            lam_min=lam_b_min,
            lam_max=lam_b_max,
        )

        if not ctx.cfg.interactive:
            break

        ans = prompt(
            f"[BLUE] Approve sensitivity curve? (a=accept, c=change spline_s, q=quit)",
            default="a",
        ).strip().lower()

        if ans in ("a", "accept", "y", "yes"):
            break
        if ans in ("q", "quit", "n", "no"):
            die("User aborted during BLUE spline_s selection.")
        if ans.startswith("c"):
            val = prompt(f"Enter new spline_s for BLUE (current={spline_b})", default=str(spline_b))
            try:
                spline_b = float(val)
            except Exception:
                die(f"Invalid spline_s: {val}")
            continue

    # --------------
    # RED calibration
    # --------------
    stdR = ctx.cfg.fluxstd_red
    if "RED" not in cube_map.get(stdR, {}):
        die(f"Chosen RED flux standard '{stdR}' has no RED cube.")

    lam_r_std, C_r_std = _extract_object_side(ctx, stdR, "RED")
    np.savetxt(
        ctx.caldir / f"std_counts_{safe_filename(stdR)}_RED.flm",
        np.c_[lam_r_std, C_r_std],
        header="lambda_A  counts_tgt_bkgsub",
    )

    lam_r_ref_full, F_r_ref_full = load_reference_flux(Path(ctx.cfg.ref_red))
    lam_r_min, lam_r_max = _arm_lims("RED")

    spline_r = ctx.cfg.calibration.spline_s_init
    while True:
        lam_r_ref, F_r_ref, ratio_r, S_r = build_sensitivity(
            lam_ref=lam_r_ref_full,
            F_ref=F_r_ref_full,
            lam_std=lam_r_std,
            C_std=C_r_std,
            spline_s=spline_r,
            exclude_windows=ctx.cfg.calibration.telluric_windows,
            lam_min=lam_r_min,
            lam_max=lam_r_max,
        )
        lam_r_cal_std, F_r_cal_std = apply_sensitivity(lam_r_ref, S_r, lam_r_std, C_r_std)

        np.savetxt(
            ctx.caldir / "sensitivity_RED.txt",
            np.c_[lam_r_ref, S_r],
            header=f"lambda_A  S_lambda   (spline_s={spline_r})",
        )

        # Airmass + O2 template (built on the truncated RED reference grid)
        _, hdr_std_red = load_cube(Path(cube_map[stdR]["RED"]))
        X_std_red = get_airmass_from_header(hdr_std_red)

        T_o2_std, o2_mask = build_o2_transmission_template(
            lam_ref=lam_r_ref,
            F_ref=F_r_ref,
            lam_std=lam_r_std,
            C_std=C_r_std,
            S_sens=S_r,
            o2_windows=ctx.cfg.calibration.telluric_windows,
            min_T=ctx.cfg.calibration.telluric_min_T,
            smooth_s=ctx.cfg.calibration.telluric_template_smooth_s,
        )

        np.savetxt(
            ctx.caldir / "telluric_O2_template_RED.txt",
            np.c_[lam_r_ref, T_o2_std, o2_mask.astype(int)],
            header="lambda_A  T_O2_std  in_O2mask(1/0)",
        )
        plot_o2_template_diagnostic(
            lam_ref=lam_r_ref,
            T_std=T_o2_std,
            o2_windows=ctx.cfg.calibration.telluric_windows,
            outpng=ctx.diagdir / "O2_template_RED.png",
            show=ctx.cfg.show_plots,
        )

        # Standard telluric corrected diagnostic (use X_std for both so this is "shape only")
        F_r_cal_std_tc = apply_o2_telluric_correction(
            F_fluxcal=F_r_cal_std.copy(),
            T_std=T_o2_std,
            X_std=X_std_red,
            X_sci=X_std_red,
            o2_mask=o2_mask,
            min_T=ctx.cfg.calibration.telluric_min_T,
        )

        plot_calibration_diagnostics(
            side="RED",
            std_name=stdR,
            lam_std=lam_r_std,
            C_std=C_r_std,
            lam_ref=lam_r_ref,
            F_ref=F_r_ref,
            ratio=ratio_r,
            S=S_r,
            F_std_cal=F_r_cal_std_tc,
            outdir=ctx.diagdir,
            show=ctx.cfg.show_plots,
            telluric_windows=ctx.cfg.calibration.telluric_windows,
            red_tell_before=F_r_cal_std,
            red_tell_after=F_r_cal_std_tc,
            lam_min=lam_r_min,
            lam_max=lam_r_max,
        )

        if not ctx.cfg.interactive:
            break

        ans = prompt(
            f"[RED] Approve sensitivity curve? (a=accept, c=change spline_s, q=quit)",
            default="a",
        ).strip().lower()

        if ans in ("a", "accept", "y", "yes"):
            break
        if ans in ("q", "quit", "n", "no"):
            die("User aborted during RED spline_s selection.")
        if ans.startswith("c"):
            val = prompt(f"Enter new spline_s for RED (current={spline_r})", default=str(spline_r))
            try:
                spline_r = float(val)
            except Exception:
                die(f"Invalid spline_s: {val}")
            continue

    # Persist calibration artifacts to state (note: reference grids are truncated per arm)
    ctx.state.artifacts["calibration"] = {
        "lam_b_ref": lam_b_ref.tolist(),
        "S_b": S_b.tolist(),
        "lam_r_ref": lam_r_ref.tolist(),
        "S_r": S_r.tolist(),
        "T_o2_std": T_o2_std.tolist(),
        "o2_mask": o2_mask.astype(int).tolist(),
        "X_std_red": X_std_red,
        "spline_s_blue": spline_b,
        "spline_s_red": spline_r,
        "lam_min_blue": lam_b_min,
        "lam_max_blue": lam_b_max,
        "lam_min_red": lam_r_min,
        "lam_max_red": lam_r_max,
    }



def step05_process_all_objects(ctx: PipelineContext) -> None:
    """Extract all objects, apply calibration, apply O2 correction in RED, join, save products."""
    cube_map: Dict[str, Dict[str, str]] = ctx.state.artifacts.get("cube_map", {})
    objects: List[str] = ctx.state.artifacts.get("objects", [])
    cal = ctx.state.artifacts.get("calibration")
    if not cube_map or not objects:
        die("No objects/cube_map in state; run step01_discover_coadds first.")
    if cal is None:
        die("No calibration in state; run step04_extract_standards_build_calibration first.")


    # Object-level resume/redo controls (set from CLI on ctx)
    only = getattr(ctx, "only_objects", None) or []
    skip = getattr(ctx, "skip_objects", None) or []
    redo = set(getattr(ctx, "redo_objects", None) or [])

    if only:
        objects = [o for o in objects if o in set(only)]
    if skip:
        objects = [o for o in objects if o not in set(skip)]

    obj_state_map = ctx.state.artifacts.setdefault("object_state", {})
    # If user asked to redo specific objects, clear their completion flags here
    for o in redo:
        if o in obj_state_map:
            obj_state_map[o].pop("process_complete", None)
            obj_state_map[o].pop("join_scale", None)
            obj_state_map[o].pop("outputs", None)
    lam_b_ref = np.asarray(cal["lam_b_ref"], dtype=float)
    S_b = np.asarray(cal["S_b"], dtype=float)
    lam_r_ref = np.asarray(cal["lam_r_ref"], dtype=float)
    S_r = np.asarray(cal["S_r"], dtype=float)
    T_o2_std = np.asarray(cal["T_o2_std"], dtype=float)
    o2_mask = np.asarray(cal["o2_mask"], dtype=int).astype(bool)
    X_std_red = cal.get("X_std_red")

    tell_windows = ctx.cfg.calibration.telluric_windows

    blue_trim_min, blue_trim_max = parse_trim(ctx.cfg.join.blue_trim)
    red_trim_min, red_trim_max = parse_trim(ctx.cfg.join.red_trim)

    global_excl_blue = list(ctx.cfg.join.global_excl_blue)
    global_excl_red = list(ctx.cfg.join.global_excl_red)

    if ctx.cfg.interactive:
        ans = prompt("\nSet GLOBAL wavelength exclusions for joined output/plots? (y/n)", "n").lower()
        if ans == "y":
            s = prompt("Global BLUE exclusions (e.g. 3600-3650,5450-5500) or blank", "")
            global_excl_blue = parse_ranges(s)
            s = prompt("Global RED exclusions (e.g. 7400-7500) or blank", "")
            global_excl_red = parse_ranges(s)

    for objname in objects:
        # Skip objects already processed unless explicitly redoing them
        st = obj_state_map.get(objname, {})
        if st.get("process_complete", False) and objname not in redo:
            print(f"\n=== {objname} ===  [SKIP: already processed]")
            continue

        print(f"\n=== {objname} ===")

        blue_full: Optional[Tuple[np.ndarray, np.ndarray]] = None
        red_full: Optional[Tuple[np.ndarray, np.ndarray]] = None

        # BLUE
        if "BLUE" in cube_map[objname]:
            lam_b, C_b = _extract_object_side(ctx, objname, "BLUE")
            np.savetxt(ctx.countsdir / f"{safe_filename(objname)}_BLUE_counts.flm",
                       np.c_[lam_b, C_b], header="lambda_A  counts_tgt_bkgsub")
            lam_b_cal, F_b_cal = apply_sensitivity(lam_b_ref, S_b, lam_b, C_b)
            np.savetxt(ctx.fluxdir / f"{safe_filename(objname)}_BLUE_fluxcal_full.flm",
                       np.c_[lam_b_cal, F_b_cal], header="lambda_A  fluxcal_full")
            blue_full = (lam_b_cal, F_b_cal)

        # RED
        if "RED" in cube_map[objname]:
            lam_r, C_r = _extract_object_side(ctx, objname, "RED")
            np.savetxt(ctx.countsdir / f"{safe_filename(objname)}_RED_counts.flm",
                       np.c_[lam_r, C_r], header="lambda_A  counts_tgt_bkgsub")
            lam_r_cal, F_r_cal = apply_sensitivity(lam_r_ref, S_r, lam_r, C_r)

            # Airmass from object header
            _, hdr_obj_red = load_cube(Path(cube_map[objname]["RED"]))
            X_sci = get_airmass_from_header(hdr_obj_red)
            F_r_tc = apply_o2_telluric_correction(
                F_fluxcal=F_r_cal,
                T_std=T_o2_std,
                X_std=X_std_red,
                X_sci=X_sci,
                o2_mask=o2_mask,
                min_T=ctx.cfg.calibration.telluric_min_T,
            )
            plot_o2_before_after(objname, lam_r_ref, F_r_cal, F_r_tc, tell_windows,
                                 ctx.diagdir / f"{safe_filename(objname)}_RED_O2corr.png",
                                 show=ctx.cfg.show_plots)

            np.savetxt(ctx.fluxdir / f"{safe_filename(objname)}_RED_fluxcal_full_tellcorr.flm",
                       np.c_[lam_r_cal, F_r_tc], header="lambda_A  fluxcal_full_tellcorr")
            red_full = (lam_r_cal, F_r_tc)

        if blue_full is None and red_full is None:
            print("No BLUE or RED cube; skipping.")
            continue

        # Per-object exclusions (only affect joined)
        excl_b = list(global_excl_blue)
        excl_r = list(global_excl_red)
        if ctx.cfg.interactive:
            ans = prompt("Add OBJECT-SPECIFIC wavelength exclusions for JOINED output/plot? (y/n)", "n").lower()
            if ans == "y":
                if blue_full is not None:
                    s = prompt("  BLUE exclusions lo-hi,lo-hi (blank for none)", "")
                    excl_b += parse_ranges(s)
                if red_full is not None:
                    s = prompt("  RED exclusions lo-hi,lo-hi (blank for none)", "")
                    excl_r += parse_ranges(s)

        if blue_full is not None:
            lam_b2, F_b2 = apply_trim_and_exclude(blue_full[0], blue_full[1], blue_trim_min, blue_trim_max, excl_b)
        else:
            lam_b2, F_b2 = np.array([]), np.array([])

        if red_full is not None:
            lam_r2, F_r2 = apply_trim_and_exclude(red_full[0], red_full[1], red_trim_min, red_trim_max, excl_r)
        else:
            lam_r2, F_r2 = np.array([]), np.array([])

        if blue_full is not None:
            np.savetxt(ctx.finaldir / f"{safe_filename(objname)}_BLUE_fluxcal_trimmed_for_join.flm",
                       np.c_[lam_b2, F_b2], header="lambda_A  flux_blue_trimmed_for_join")
        if red_full is not None:
            np.savetxt(ctx.finaldir / f"{safe_filename(objname)}_RED_fluxcal_trimmed_for_join.flm",
                       np.c_[lam_r2, F_r2], header="lambda_A  flux_red_trimmed_for_join")

        # --- Join (KCWI-style concatenation) with optional interactive rescaling ---
        blue_scale = 1.0
        red_scale = 1.0

        if lam_b2.size and lam_r2.size and ctx.cfg.interactive:
            blue_scale, red_scale = interactive_rescale_and_approve_flux(
                objname=objname,
                lam_blue=lam_b2, flux_blue=F_b2,
                lam_red=lam_r2,  flux_red=F_r2,
                outdir=ctx.finaldir,
                show=ctx.cfg.show_plots,
                interactive=True,
            )

        # Apply approved scales BEFORE saving joined spectrum.
        F_b2s = F_b2 * blue_scale
        F_r2s = F_r2 * red_scale

        if lam_b2.size and lam_r2.size:
            lamJ, FJ = concat_join(lam_b2, F_b2s, lam_r2, F_r2s)
        elif lam_b2.size:
            lamJ, FJ = lam_b2, F_b2s
        else:
            lamJ, FJ = lam_r2, F_r2s

        np.savetxt(
            ctx.finaldir / f"{safe_filename(objname)}_BLUE+RED_joined_concat.flm",
            np.c_[lamJ, FJ],
            header=f"lambda_A  flux_joined_concat  (blue_scale={blue_scale}, red_scale={red_scale})",
        )

        # Diagnostic plot. While open: zoom/pan with toolbar and press 'z' to save a zoomed view.
        plot_join_diagnostic(
            objname, lam_b2, F_b2s, lam_r2, F_r2s,
            outpng=ctx.finaldir / f"{safe_filename(objname)}_joined_concat.png",
            show=ctx.cfg.show_plots,
        )

        # Mark this object complete in object_state and persist after each object
        obj_state_map[objname] = {
            "process_complete": True,
            "join_scale": {"blue": float(blue_scale), "red": float(red_scale)},
            "outputs": {
                "blue_counts": str(ctx.countsdir / f"{safe_filename(objname)}_BLUE_counts.flm"),
                "red_counts": str(ctx.countsdir / f"{safe_filename(objname)}_RED_counts.flm"),
                "blue_fluxcal": str(ctx.fluxdir / f"{safe_filename(objname)}_BLUE_fluxcal_full.flm") if ("BLUE" in cube_map[objname]) else None,
                "red_fluxcal": str(ctx.fluxdir / f"{safe_filename(objname)}_RED_fluxcal_full.flm") if ("RED" in cube_map[objname]) else None,
                "joined": str(ctx.finaldir / f"{safe_filename(objname)}_BLUE+RED_joined_concat.flm"),
            },
        }
        ctx.save_state()



def make_steps() -> List[Step]:
    return [
        Step("setup", "Create output dirs + write config.json", step00_setup),
        Step("coadd", "Coadd KCWI Level 2 cubes if enabled", step01_coadd_level2_cubes),
        Step("discover", "Discover coadd cubes", step01_discover_coadds),
        Step("choose", "Choose flux standards + reference flux files", step02_choose_standards_and_refs),
        Step("apertures", "Define default target/background apertures (independent)", step03_define_default_apertures),
        Step("calibrate", "Extract standards + build sensitivity + build O2 template", step04_extract_standards_build_calibration),
        Step("process", "Process all objects (extract, fluxcal, O2 correct, join)", step05_process_all_objects),
    ]
