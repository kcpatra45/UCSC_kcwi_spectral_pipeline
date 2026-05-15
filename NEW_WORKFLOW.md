# KCWI Object-Local Reduction Workflow

This workflow is centered on KOA/KCWI DRP Level 2 `*_icubes.fits` files.

## 1. Organize KOA Downloads

Start with a directory containing all downloaded `*_icubes.fits` files:

```bash
/Users/kcpatra/miniforge3/envs/astronomy/bin/python run_kcwi_reduction.py organize /path/to/koa_download --project /path/to/kcwi_project
```

By default this creates symlinks rather than moving the original downloads:

```text
kcwi_project/
  objects/
    OBJECT_A/
      BLUE/
      RED/
    OBJECT_B/
      BLUE/
      RED/
  calibrations/
  project_manifest.json
```

Use `--mode copy` or `--mode move` only when you really want copied or moved FITS files.

## 2. Extract A Standard Star

Change into the standard star object directory:

```bash
cd /path/to/kcwi_project/objects/Feige34
/Users/kcpatra/miniforge3/envs/astronomy/bin/python /path/to/repo/run_kcwi_reduction.py extract . --standard
```

The extraction step extracts every exposure separately, reusing the previous aperture by default for later exposures. It then coadds the 1D spectra and saves standard-star calibration products under:

```text
kcwi_project/calibrations/
```

## 3. Extract A Science Object

```bash
cd /path/to/kcwi_project/objects/2026dix
/Users/kcpatra/miniforge3/envs/astronomy/bin/python /path/to/repo/run_kcwi_reduction.py extract . --science
```

The science extraction writes object-local products:

```text
apertures/
extracted/
coadded_spectra/
fluxcal/
final/
diagnostics/
extraction_state.json
```

If compatible standard-star calibrations exist in the project calibration registry, the command offers them for flux calibration.

## Notes

- The core reduction strategy is now: per-exposure cube extraction, then 1D spectral coaddition.
- The default aperture behavior is to define apertures on the first exposure and reuse them for subsequent exposures unless the user says no.
- The old batch pipeline entry point still exists, but new work should target `run_kcwi_reduction.py`.
