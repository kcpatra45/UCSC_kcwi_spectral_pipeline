# KCWI modular reduction pipeline

This is a refactor of the original monolithic script into a small package + a short runner script.

## Run

```bash
python3 run_kcwi_pipeline.py /path/to/coadds --outdir kcwi_fluxcal_out
```

List steps:

```bash
python3 run_kcwi_pipeline.py /path/to/coadds --outdir kcwi_fluxcal_out --list-steps
```

Resume from a step:

```bash
python3 run_kcwi_pipeline.py /path/to/coadds --outdir kcwi_fluxcal_out --start-at calibrate
```

Redo a step (and everything after it):

```bash
python3 run_kcwi_pipeline.py /path/to/coadds --outdir kcwi_fluxcal_out --redo-from apertures
```

## Key changes vs original

- PSF fitting removed.
- Target aperture and background region are defined independently (shape + location).
  Background can be an annulus (ellipse_annulus/circle_annulus) or any other supported shape.
- Step-state written to `<outdir>/state.json` so you can restart without repeating earlier steps.
- Steps live in `kcwi_pipeline/steps_kcwi.py` and can be extended.

## Adding a new step later (e.g. cosmic rays)

1. Implement a function in `kcwi_pipeline/steps_kcwi.py` with signature:

```python
def stepXX_my_new_thing(ctx: PipelineContext) -> None:
    ...
```

2. Insert it into the `make_steps()` list, with a unique `step_id`.
3. Write outputs to a new subdirectory under `ctx.outdir` and stash any small metadata in `ctx.state.artifacts`.

