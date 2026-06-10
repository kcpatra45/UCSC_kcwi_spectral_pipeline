#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="KCWI object-local reduction workflow for KOA Level 2 *_icubes.fits products"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_org = sub.add_parser("organize", help="Group KOA *_icubes.fits files by object and side")
    p_org.add_argument("fits_dir", type=str, help="Directory containing KOA *_icubes.fits files")
    p_org.add_argument("--project", type=str, required=True, help="Project directory to create/update")
    p_org.add_argument(
        "--mode",
        choices=("symlink", "copy", "move"),
        default="symlink",
        help="How to place FITS files into object directories",
    )

    p_ext = sub.add_parser("extract", help="Extract all exposures in one object directory")
    p_ext.add_argument("object_dir", type=str, help="Object directory created by organize")
    role = p_ext.add_mutually_exclusive_group()
    role.add_argument("--standard", action="store_true", help="Treat this object as a standard star")
    role.add_argument("--science", action="store_true", help="Treat this object as a science target")
    p_ext.add_argument(
        "--side",
        choices=("blue", "red", "both"),
        default="both",
        help="Which side to extract",
    )
    p_ext.add_argument("--calib-dir", type=str, default=None, help="Master calibration directory")
    p_ext.add_argument("--show-plots", action="store_true", help="Show interactive/diagnostic plots")
    p_ext.add_argument("--redo-apertures", action="store_true", help="Ignore saved apertures and redefine them")
    p_ext.add_argument(
        "--join-only",
        action="store_true",
        help="Skip extraction/calibration and redo only the BLUE+RED scaling/join from existing fluxcal spectra",
    )

    args = parser.parse_args()

    if args.command == "organize":
        from kcwi_pipeline.project import organize_project

        manifest = organize_project(Path(args.fits_dir), Path(args.project), mode=args.mode)
        objects = manifest.get("objects", {})
        print(f"Organized {len(manifest.get('files', []))} files into {len(objects)} object directories")
        print(f"Project: {Path(args.project).expanduser().resolve()}")
        return

    if args.command == "extract":
        from kcwi_pipeline.object_workflow import extract_object

        standard = True if args.standard else False if args.science else None
        extract_object(
            Path(args.object_dir),
            calib_dir=Path(args.calib_dir) if args.calib_dir else None,
            standard=standard,
            side=args.side,
            show_plots=bool(args.show_plots),
            redo_apertures=bool(args.redo_apertures),
            join_only=bool(args.join_only),
        )
        return


if __name__ == "__main__":
    main()
