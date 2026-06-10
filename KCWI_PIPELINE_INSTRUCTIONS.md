# KCWI IFU Reduction Pipeline Instructions

This guide describes the current object-local KCWI reduction workflow for KOA/KCWI DRP Level 2 `*_icubes.fits` files.

The expected workflow is:

1. Put all downloaded KOA `*_icubes.fits` files in one input directory.
2. Organize them into a project directory grouped by object and side.
3. Extract one or more standard stars to build master calibrations.
4. Extract science targets object by object.
5. Re-run individual sides or apertures as needed without affecting other objects.

## Dependencies

Beyond the Python standard library, the pipeline uses:

```text
numpy
scipy
astropy
photutils
matplotlib
```

The user should run the pipeline from a Python environment where these packages are installed.

## 1. Organize KOA Downloads

Start with a directory containing all downloaded Level 2 `*_icubes.fits` files.

```bash
python run_kcwi_reduction.py organize /path/to/koa_download --project /path/to/kcwi_project
```

By default, files are symlinked into the project. This is usually preferred because it does not duplicate large FITS files.

The project structure will look like:

```text
kcwi_project/
  objects/
    OBJECT_A/
      BLUE/
        *_icubes.fits
      RED/
        *_icubes.fits
    OBJECT_B/
      BLUE/
      RED/
  calibrations/
    calibration_registry.json
  project_manifest.json
```

Alternative organization modes:

```bash
python run_kcwi_reduction.py organize /path/to/koa_download --project /path/to/kcwi_project --mode symlink
python run_kcwi_reduction.py organize /path/to/koa_download --project /path/to/kcwi_project --mode copy
python run_kcwi_reduction.py organize /path/to/koa_download --project /path/to/kcwi_project --mode move
```

Use `move` only if you really want the original KOA files moved.

## 2. Extract a Standard Star

From the pipeline repository directory, pass the standard star object directory to `extract`:

Extract only the red side:

```bash
python run_kcwi_reduction.py extract /path/to/kcwi_project/objects/STD_OBJECT_NAME --standard --side red
```

Extract only the blue side:

```bash
python run_kcwi_reduction.py extract /path/to/kcwi_project/objects/STD_OBJECT_NAME --standard --side blue
```

Extract both sides:

```bash
python run_kcwi_reduction.py extract /path/to/kcwi_project/objects/STD_OBJECT_NAME --standard --side both
```

If `--side` is omitted, the default is `both`.

During standard extraction, the pipeline will:

- extract every `*_icubes.fits` exposure separately;
- ask for target/background apertures;
- optionally reuse and review apertures for later exposures;
- coadd the extracted 1D spectra with sigma clipping;
- ask for the AB standard-star identity from `kcwi_pipeline/abcalc.py`;
- show the extracted standard spectrum and reference flux;
- let you add/delete/move spline points for the continuum fit;
- load previously saved continuum spline points when the same standard/side is redone;
- for RED standards, exclude O2 telluric windows from the continuum fit;
- build sensitivity functions;
- for RED standards, build an O2 telluric template;
- save calibration products in the master project calibration directory;
- also save object-local standard diagnostics and processed spectra.

Master calibration outputs are saved under:

```text
/path/to/kcwi_project/calibrations/STD_OBJECT_NAME/SIDE/
```

Object-local standard outputs are saved under:

```text
objects/STD_OBJECT_NAME/extracted/SIDE/
objects/STD_OBJECT_NAME/coadded_spectra/
objects/STD_OBJECT_NAME/diagnostics/SIDE/
objects/STD_OBJECT_NAME/diagnostics/SIDE/standard_calibration/
objects/STD_OBJECT_NAME/final/
```

Standard processed spectra are ASCII `.flm` files:

```text
objects/STD_OBJECT_NAME/final/STD_OBJECT_NAME_BLUE_standard_processed.flm
objects/STD_OBJECT_NAME/final/STD_OBJECT_NAME_RED_standard_processed.flm
```

The corresponding PNGs include uncertainty shading if uncertainty is available:

```text
objects/STD_OBJECT_NAME/final/STD_OBJECT_NAME_BLUE_standard_processed.png
objects/STD_OBJECT_NAME/final/STD_OBJECT_NAME_RED_standard_processed.png
```

Accepted standard-star continuum spline points are saved in both the calibration directory and the object-local diagnostics directory:

```text
calibrations/STD_OBJECT_NAME/SIDE/continuum_spline_points_SIDE.txt
objects/STD_OBJECT_NAME/diagnostics/SIDE/standard_calibration/continuum_spline_points_SIDE.txt
```

When the standard side is rerun, these points are loaded as the initial spline points. The user can accept them, move them, add/delete points, or reset to automatically generated defaults.

## 3. Extract a Science Object

From the pipeline repository directory, pass the science object directory to `extract`:

Extract both sides:

```bash
python run_kcwi_reduction.py extract /path/to/kcwi_project/objects/SCIENCE_OBJECT_NAME --science --side both
```

Extract only one side:

```bash
python run_kcwi_reduction.py extract /path/to/kcwi_project/objects/SCIENCE_OBJECT_NAME --science --side blue
python run_kcwi_reduction.py extract /path/to/kcwi_project/objects/SCIENCE_OBJECT_NAME --science --side red
```

During science extraction, the pipeline will:

- extract each exposure separately;
- coadd the 1D spectra with sigma clipping;
- save and display a coadd diagnostic plot;
- ask which standard calibration to use for each side;
- apply sensitivity functions;
- for RED, apply the O2 telluric correction using the standard/science airmass ratio;
- save side-level flux-calibrated spectra;
- if both sides are available, run the join/scale/approve step;
- save the final spectrum as `.flm` and `.png`.

Science outputs are saved under:

```text
objects/SCIENCE_OBJECT_NAME/apertures/
objects/SCIENCE_OBJECT_NAME/extracted/
objects/SCIENCE_OBJECT_NAME/coadded_spectra/
objects/SCIENCE_OBJECT_NAME/fluxcal/
objects/SCIENCE_OBJECT_NAME/final/
objects/SCIENCE_OBJECT_NAME/diagnostics/
objects/SCIENCE_OBJECT_NAME/extraction_state.json
```

Final science spectra:

```text
objects/SCIENCE_OBJECT_NAME/final/SCIENCE_OBJECT_NAME_BLUE+RED_spectrum.flm
objects/SCIENCE_OBJECT_NAME/final/SCIENCE_OBJECT_NAME_BLUE+RED_spectrum.png
```

If only one side was extracted:

```text
objects/SCIENCE_OBJECT_NAME/final/SCIENCE_OBJECT_NAME_BLUE_spectrum.flm
objects/SCIENCE_OBJECT_NAME/final/SCIENCE_OBJECT_NAME_RED_spectrum.flm
```

## 4. Aperture Editing Controls

White-light images are shown in two panels:

- left: original aspect ratio;
- right: y-compressed by a factor of 3 for easier visual comparison to sky charts.

Both panels use the same image coordinates. Apertures can be drawn, moved, and resized from either panel.

The white-light display includes real-time contrast controls:

- `Low %` slider: lower percentile cut;
- `High %` slider: upper percentile cut;
- `Reset`: return to 5-99 percent scaling.

Aperture drawing/editing keys:

```text
m      move mode
e      resize mode
a      accept aperture
Enter  accept aperture
r      redraw from scratch, where available
q      cancel/quit
```

General aperture review prompt:

```text
a   approve apertures
rt  redraw target
st  change target shape
rb  redraw background
sb  change background shape
t   move/resize target
b   move/resize background
p   edit numeric parameters
q   quit
```

When setting or editing the background aperture, the target aperture is shown for reference.

## 5. Continuum Spline Controls for Standards

The standard-star spline plot is used to define the observed continuum for sensitivity creation.

Controls:

```text
left-click       add point
drag marker      move point
right-click      delete nearest point
z                zoom-box mode
o                original zoom
a                accept spline
Enter            accept spline
r                reset spline points
q                quit
```

For RED standards, the O2 telluric windows are shaded and excluded from the continuum fit:

```text
6860-6935 A
7590-7690 A
```

This preserves the telluric absorption troughs so the RED telluric template is built correctly.

## 6. Join and Approve Controls

When both BLUE and RED sides are available for a science object, the pipeline opens a join approval plot.

Controls:

```text
Blue scale slider  multiply BLUE spectrum by 0.1-10
Red scale slider   multiply RED spectrum by 0.1-10
Reset              reset both scales to 1
Approve            approve current scaling
a      approve current scaling
Enter  approve current scaling
z      save current zoomed view
q      abort
```

If you close the plot without approving, the terminal still offers fallback scaling choices:

```text
a   approve current scaling
1   multiply RED by factor
2   multiply BLUE by factor
3   set RED scale absolute
4   set BLUE scale absolute
q   abort pipeline
```

The final joined spectrum uses the approved BLUE and RED scale factors.

## 7. Common Commands

### Organize New Data

```bash
python run_kcwi_reduction.py organize /path/to/koa_download --project /path/to/kcwi_project
```

### Organize by Copy Instead of Symlink

```bash
python run_kcwi_reduction.py organize /path/to/koa_download --project /path/to/kcwi_project --mode copy
```

### Extract a RED Standard

```bash
python run_kcwi_reduction.py extract /path/to/kcwi_project/objects/STD_OBJECT_NAME --standard --side red
```

### Extract a BLUE Standard

```bash
python run_kcwi_reduction.py extract /path/to/kcwi_project/objects/STD_OBJECT_NAME --standard --side blue
```

### Extract Both Sides of a Science Object

```bash
python run_kcwi_reduction.py extract /path/to/kcwi_project/objects/SCIENCE_OBJECT_NAME --science --side both
```

### Extract Only RED Science

```bash
python run_kcwi_reduction.py extract /path/to/kcwi_project/objects/SCIENCE_OBJECT_NAME --science --side red
```

### Extract Only BLUE Science

```bash
python run_kcwi_reduction.py extract /path/to/kcwi_project/objects/SCIENCE_OBJECT_NAME --science --side blue
```

### Redo Apertures

Use this when you want to ignore saved apertures and redefine them.

```bash
python run_kcwi_reduction.py extract /path/to/kcwi_project/objects/SCIENCE_OBJECT_NAME --science --side red --redo-apertures
```

For a standard:

```bash
python run_kcwi_reduction.py extract /path/to/kcwi_project/objects/STD_OBJECT_NAME --standard --side red --redo-apertures
```

### Force Diagnostic Plots

```bash
python run_kcwi_reduction.py extract /path/to/kcwi_project/objects/SCIENCE_OBJECT_NAME --science --side both --show-plots
```

### Use a Calibration Directory Explicitly

Normally the pipeline finds the project calibration directory automatically. To specify one:

```bash
python run_kcwi_reduction.py extract /path/to/kcwi_project/objects/SCIENCE_OBJECT_NAME --science --side both --calib-dir /path/to/kcwi_project/calibrations
```

### Redo One Side and Rebuild the Join

If both sides were processed before, you can rerun one side. The pipeline will reuse the existing other side from `fluxcal/` and rerun the join approval.

Redo RED only:

```bash
python run_kcwi_reduction.py extract /path/to/kcwi_project/objects/SCIENCE_OBJECT_NAME --science --side red
```

Redo BLUE only:

```bash
python run_kcwi_reduction.py extract /path/to/kcwi_project/objects/SCIENCE_OBJECT_NAME --science --side blue
```

The reused side is loaded from:

```text
fluxcal/SCIENCE_OBJECT_NAME_BLUE_fluxcal.flm
fluxcal/SCIENCE_OBJECT_NAME_RED_fluxcal.flm
```

For backward compatibility, older `.txt` side flux files can still be read if present.

### Redo Only the BLUE+RED Scaling and Join

Use this when both side spectra already exist and you only want to adjust the relative BLUE/RED scaling again.

```bash
python run_kcwi_reduction.py extract /path/to/kcwi_project/objects/SCIENCE_OBJECT_NAME --science --join-only
```

This skips aperture extraction, 1D coaddition, flux calibration, and telluric correction. It loads:

```text
fluxcal/SCIENCE_OBJECT_NAME_BLUE_fluxcal.flm
fluxcal/SCIENCE_OBJECT_NAME_RED_fluxcal.flm
```

Then it opens the join scaling window and rewrites:

```text
final/SCIENCE_OBJECT_NAME_BLUE+RED_spectrum.flm
final/SCIENCE_OBJECT_NAME_BLUE+RED_spectrum.png
final/SCIENCE_OBJECT_NAME_joined.png
final/SCIENCE_OBJECT_NAME_join_scale.txt
```

## 8. Output File Conventions

Spectrum products are ASCII files with `.flm` extension.

Flux-calibrated spectra are saved in units of:

```text
1e-15 erg/s/cm^2/A
```

KCWI DRP `*_icubes.fits` fluxes are commonly in `1e-16 erg/s/cm^2/A`, but this pipeline rescales calibrated saved spectra by a factor of 10 so the `.flm` flux and `sigma_flux` columns are in `1e-15 erg/s/cm^2/A`. For example, a saved flux value of `2.4` means:

```text
2.4 x 10^-15 erg/s/cm^2/A
```

Examples:

```text
extracted/BLUE/*_counts.flm
extracted/RED/*_counts.flm
coadded_spectra/OBJECT_BLUE_counts_coadd.flm
coadded_spectra/OBJECT_RED_counts_coadd.flm
fluxcal/OBJECT_BLUE_fluxcal.flm
fluxcal/OBJECT_RED_fluxcal.flm
final/OBJECT_BLUE+RED_spectrum.flm
final/OBJECT_BLUE_spectrum.flm
final/OBJECT_RED_spectrum.flm
```

If uncertainty exists, spectra have three columns:

```text
lambda_A  flux_or_counts  sigma_flux_or_counts
```

For flux-calibrated products, `flux` and `sigma_flux` are in `1e-15 erg/s/cm^2/A`. For counts products, the values remain in the native extracted cube/count scale.

If uncertainty is unavailable, spectra have two columns:

```text
lambda_A  flux_or_counts
```

Non-spectrum metadata and calibration tables remain `.txt`, for example:

```text
coadded_spectra/OBJECT_SIDE_nexp.txt
calibrations/STANDARD/SIDE/sensitivity_SIDE.txt
calibrations/STANDARD/SIDE/observed_continuum_SIDE.txt
calibrations/STANDARD/SIDE/ab_reference_flux_SIDE.txt
calibrations/STANDARD/RED/telluric_O2_template_RED.txt
fluxcal/OBJECT_RED_telluric_correction_arrays.txt
final/OBJECT_join_scale.txt
```

## 9. Coaddition Logic

The pipeline no longer coadds cubes before extraction in the main object-local workflow.

Instead:

1. Extract each individual `*_icubes.fits` exposure.
2. Interpolate spectra onto the first exposure wavelength grid if needed.
3. Sigma-clip the stack at each wavelength.
4. If uncertainty exists for all exposures, inverse-variance weight the surviving samples.
5. If uncertainty is unavailable, mean-combine the surviving samples.
6. Save the number of accepted spectra per wavelength in `*_nexp.txt`.

The current 1D coadd clipping is symmetric:

```text
sigma = 3.0
maxiters = 5
```

The background aperture is also sigma-clipped at each wavelength before estimating the weighted mean background:

```text
sigma = 2.5
maxiters = 5
```

Science coadd diagnostics are saved and displayed:

```text
diagnostics/SIDE/OBJECT_SIDE_coadd_diagnostic.png
```

The plot shows:

- every extracted exposure with a vertical offset;
- the sigma-clipped coadd;
- coadd uncertainty if available;
- `N used` versus wavelength.

## 10. Flux Calibration and Telluric Correction

Standard-star flux calibration uses AB magnitudes from:

```text
kcwi_pipeline/abcalc.py
```

The AB reference flux is interpolated onto the extracted standard wavelength grid. The sensitivity function is:

```text
sensitivity = reference_flux / observed_standard_continuum
```

The AB reference flux and sensitivity function are scaled so calibrated outputs are in `1e-15 erg/s/cm^2/A`.

If an older calibration registry entry was created when the pipeline used `1e-16 erg/s/cm^2/A`, the science extraction step converts that sensitivity to the new `1e-15` scale before applying it. For consistency, rebuilding standards after this change is still recommended.

RED telluric correction uses an O2 transmission template from the RED standard:

```text
T_std = flux_calibrated_standard / reference_flux
```

For science RED spectra, the template is scaled by the airmass ratio:

```text
T_scaled = T_std ** (X_sci / X_std)
flux_corrected = flux_uncorrected / T_scaled
```

The correction is applied only in the O2 windows:

```text
6860-6935 A
7590-7690 A
```

Science RED telluric diagnostics:

```text
fluxcal/OBJECT_RED_telluric_correction.png
fluxcal/OBJECT_RED_telluric_detail.png
fluxcal/OBJECT_RED_telluric_correction_arrays.txt
```

## 11. Default Wavelength Ranges

The pipeline trims all major extraction, calibration, plotting, and final products to:

```text
BLUE: 3550-5550 A
RED:  5650-8800 A
```

These defaults are currently defined in:

```text
kcwi_pipeline/object_workflow.py
```

Look for:

```python
DEFAULT_SIDE_RANGES = {
    "BLUE": (3550.0, 5550.0),
    "RED": (5650.0, 8800.0),
}
```

## 12. Practical Run Order for a New Dataset

1. Organize the KOA directory:

```bash
python run_kcwi_reduction.py organize /path/to/koa_download --project /path/to/kcwi_project
```

2. Inspect the object directories:

```bash
ls /path/to/kcwi_project/objects
```

3. Extract the relevant standard stars first:

```bash
python run_kcwi_reduction.py extract /path/to/kcwi_project/objects/STD_RED --standard --side red
python run_kcwi_reduction.py extract /path/to/kcwi_project/objects/STD_BLUE --standard --side blue
```

4. Extract science objects:

```bash
python run_kcwi_reduction.py extract /path/to/kcwi_project/objects/SCIENCE_OBJECT --science --side both
```

5. If one side needs improvement, rerun only that side:

```bash
python run_kcwi_reduction.py extract /path/to/kcwi_project/objects/SCIENCE_OBJECT --science --side red --redo-apertures
```

The pipeline will reuse the existing other side, rerun join approval, and refresh the final combined spectrum.

## 13. Notes and Caveats

- The workflow expects KOA/KCWI Level 2 `*_icubes.fits`, not `*_icubed.fits`.
- The final science spectrum is based on per-exposure extraction followed by 1D spectral coaddition.
- Aperture definitions are saved in JSON files under `apertures/SIDE/`.
- Existing apertures are displayed for approval before reuse.
- RED telluric correction requires valid standard and science airmass values. If either is missing, the pipeline prints a warning and skips the correction.
- Final PNGs show uncertainty shading only when an uncertainty column exists.
- Master calibration products are shared across science objects through `calibrations/calibration_registry.json`.
