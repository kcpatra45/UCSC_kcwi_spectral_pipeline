#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from kcwi_pipeline.config import CoaddConfig, PipelineConfig, load_config, save_config
from kcwi_pipeline.pipeline import Pipeline, PipelineContext
from kcwi_pipeline.steps_kcwi import make_steps


def main() -> None:
    ap = argparse.ArgumentParser(
        description="KCWI IFU cube reduction pipeline (coadd, extract, flux-calibrate, join)"
    )
    ap.add_argument(
        "input_dir",
        type=str,
        help="Directory with existing coadds, or Level 2 cubes when --make-coadds is set",
    )
    ap.add_argument("--outdir", type=str, default="kcwi_fluxcal_out", help="Output directory")

    ap.add_argument("--config", type=str, default=None,
                    help="Path to config.json (default: <outdir>/config.json)")

    ap.add_argument("--list-steps", action="store_true", help="List pipeline steps and exit")
    ap.add_argument("--start-at", type=str, default=None, help="Start execution at step id")
    ap.add_argument("--redo-from", type=str, default=None, help="Mark step id and later as incomplete, then run")

    ap.add_argument("--only", type=str, default=None,
                    help="Comma-separated list of object names to process (others skipped)")
    ap.add_argument("--skip", type=str, default=None,
                    help="Comma-separated list of object names to skip")
    ap.add_argument("--redo-objects", type=str, default=None,
                    help="Comma-separated list of object names to redo in the object-processing step")

    ap.add_argument(
        "--make-coadds",
        action="store_true",
        help="Coadd individual KCWI DRP Level 2 cubes before extraction",
    )
    ap.add_argument(
        "--coadd-list",
        type=str,
        default=None,
        help="Optional text file listing Level 2 FITS cubes to coadd",
    )
    ap.add_argument(
        "--coadd-outdir",
        type=str,
        default=None,
        help="Output directory for generated coadds (default: <outdir>/coadds)",
    )
    ap.add_argument(
        "--coadd-method",
        choices=("ivar_sigma_clip_mean", "sigma_clip_mean", "median"),
        default="ivar_sigma_clip_mean",
        help="Coadd method used when --make-coadds is set",
    )
    ap.add_argument("--coadd-sigma", type=float, default=4.0, help="Sigma threshold for coadd clipping")
    ap.add_argument("--coadd-iters", type=int, default=3, help="Max coadd clipping iterations")
    ap.add_argument(
        "--allow-missing-uncert",
        action="store_true",
        help="Allow coadd methods that do not require UNCERT extensions",
    )

    ap.add_argument("--interactive", action="store_true", help="Enable interactive prompts")
    ap.add_argument("--no-interactive", dest="interactive", action="store_false")
    ap.set_defaults(interactive=True)

    ap.add_argument("--show-plots", action="store_true", help="Display plots interactively (still saves PNGs)")
    ap.add_argument("--no-show-plots", dest="show_plots", action="store_false")
    ap.set_defaults(show_plots=True)

    args = ap.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    config_path = Path(args.config).expanduser().resolve() if args.config else (outdir / "config.json")

    if config_path.exists():
        cfg = load_config(config_path)
        # input_dir is positional for convenience; keep config in sync
        if args.make_coadds:
            cfg.coadd.input_dir = args.input_dir
        else:
            cfg.coadd_dir = args.input_dir
        cfg.outdir = str(outdir)
    else:
        coadd_outdir = (
            str(Path(args.coadd_outdir).expanduser().resolve())
            if args.coadd_outdir
            else str(outdir / "coadds")
        )
        cfg = PipelineConfig(
            coadd_dir=coadd_outdir if args.make_coadds else args.input_dir,
            outdir=str(outdir),
            interactive=bool(args.interactive),
            show_plots=bool(args.show_plots),
            coadd=CoaddConfig(
                enabled=bool(args.make_coadds),
                input_dir=args.input_dir if args.make_coadds else None,
                list_file=args.coadd_list,
                outdir=coadd_outdir if args.make_coadds else None,
                method=args.coadd_method,
                sigma=float(args.coadd_sigma),
                iters=int(args.coadd_iters),
                require_uncert=not bool(args.allow_missing_uncert),
            ),
        )
        save_config(cfg, config_path)

    if args.make_coadds:
        cfg.coadd.enabled = True
        cfg.coadd.input_dir = args.input_dir
        cfg.coadd.list_file = args.coadd_list
        cfg.coadd.outdir = (
            str(Path(args.coadd_outdir).expanduser().resolve())
            if args.coadd_outdir
            else cfg.coadd.outdir or str(outdir / "coadds")
        )
        cfg.coadd.method = args.coadd_method
        cfg.coadd.sigma = float(args.coadd_sigma)
        cfg.coadd.iters = int(args.coadd_iters)
        cfg.coadd.require_uncert = not bool(args.allow_missing_uncert)
        cfg.coadd_dir = cfg.coadd.outdir
    else:
        cfg.coadd.enabled = False

    # update UI flags from CLI (let CLI override config)
    cfg.interactive = bool(args.interactive)
    cfg.show_plots = bool(args.show_plots)
    save_config(cfg, config_path)

    def _parse_list(s: str | None):
        if not s:
            return None
        parts = [p.strip() for p in s.split(",") if p.strip()]
        return parts or None

    only_objects = _parse_list(args.only)
    skip_objects = _parse_list(args.skip) or []
    redo_objects = _parse_list(args.redo_objects) or []

    ctx = PipelineContext(cfg=cfg, outdir=outdir, only_objects=only_objects, skip_objects=skip_objects, redo_objects=redo_objects)
    steps = make_steps()
    pipe = Pipeline(steps)

    if args.list_steps:
        for line in pipe.list_steps():
            print(line)
        return

    pipe.run(ctx, start_at=args.start_at, redo_from=args.redo_from)


if __name__ == "__main__":
    main()
