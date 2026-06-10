#!/usr/bin/env python3
"""
Coadd wavelength+flux calibrated KCWI DRP cube products.

Inputs are individual Level 2 cube FITS files, typically *_icubed.fits.
Outputs are <object>_<blue/red>_coadd_icube.fits multi-extension products.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from kcwi_pipeline.coadd import coadd_directory


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", type=str, help="Directory containing the FITS cubes")
    ap.add_argument(
        "list_file",
        type=str,
        nargs="?",
        default=None,
        help="Optional text file listing FITS filenames, one per line",
    )
    ap.add_argument("--outdir", type=str, default="coadds", help="Output directory")
    ap.add_argument(
        "--method",
        choices=("ivar_sigma_clip_mean", "sigma_clip_mean", "median"),
        default="ivar_sigma_clip_mean",
        help="Coadd method",
    )
    ap.add_argument("--sigma", type=float, default=4.0, help="Sigma threshold for clipping")
    ap.add_argument("--iters", type=int, default=3, help="Max iterations for clipping")
    ap.add_argument(
        "--allow-missing-uncert",
        action="store_true",
        help="Allow methods that do not require UNCERT when inputs lack uncertainty extensions",
    )
    args = ap.parse_args()

    coadd_directory(
        Path(args.data_dir),
        Path(args.outdir),
        list_file=Path(args.list_file) if args.list_file else None,
        method=args.method,
        sigma=args.sigma,
        iters=args.iters,
        require_uncert=not args.allow_missing_uncert,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
