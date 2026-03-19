#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from kcwi_pipeline.config import PipelineConfig, load_config, save_config
from kcwi_pipeline.pipeline import Pipeline, PipelineContext
from kcwi_pipeline.steps_kcwi import make_steps


def main() -> None:
    ap = argparse.ArgumentParser(
        description="KCWI coadd-cube flux-calibration pipeline (modular, resumable)"
    )
    ap.add_argument(
        "coadd_dir",
        type=str,
        help="Directory with <obj>_blue/red_coadd_icube.fits",
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
        # coadd_dir is positional for convenience; keep config in sync
        cfg.coadd_dir = args.coadd_dir
        cfg.outdir = str(outdir)
    else:
        cfg = PipelineConfig(
            coadd_dir=args.coadd_dir,
            outdir=str(outdir),
            interactive=bool(args.interactive),
            show_plots=bool(args.show_plots),
        )
        save_config(cfg, config_path)

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
