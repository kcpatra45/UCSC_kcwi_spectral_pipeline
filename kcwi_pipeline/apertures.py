from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle, Circle

from .config import ApertureShape, TargetBackgroundApertures
from .utils import prompt, savefig_show


def _rotated_coords(ny: int, nx: int, x0: float, y0: float, theta: float):
    """Return (xr, yr) = coordinates rotated by theta around (x0, y0)."""
    y, x = np.mgrid[0:ny, 0:nx]
    ct, st = np.cos(theta), np.sin(theta)
    xr = (x - x0) * ct + (y - y0) * st
    yr = -(x - x0) * st + (y - y0) * ct
    return xr, yr


def mask_from_shape(ny: int, nx: int, shape: ApertureShape) -> np.ndarray:
    """Boolean mask for a given ApertureShape."""
    sh = shape.shape
    p = shape.params

    if sh == "ellipse":
        x0, y0, a, b, theta = p
        xr, yr = _rotated_coords(ny, nx, x0, y0, theta)
        return (xr / a) ** 2 + (yr / b) ** 2 <= 1.0

    if sh == "circle":
        x0, y0, r = p
        y, x = np.mgrid[0:ny, 0:nx]
        return (x - x0) ** 2 + (y - y0) ** 2 <= r ** 2

    if sh == "rect":
        x0, y0, w, h, theta = p
        xr, yr = _rotated_coords(ny, nx, x0, y0, theta)
        return (np.abs(xr) <= (w / 2.0)) & (np.abs(yr) <= (h / 2.0))

    if sh == "ellipse_annulus":
        x0, y0, a_in, b_in, a_out, b_out, theta = p
        xr, yr = _rotated_coords(ny, nx, x0, y0, theta)
        inner = (xr / a_in) ** 2 + (yr / b_in) ** 2 <= 1.0
        outer = (xr / a_out) ** 2 + (yr / b_out) ** 2 <= 1.0
        return outer & (~inner)

    if sh == "circle_annulus":
        x0, y0, r_in, r_out = p
        y, x = np.mgrid[0:ny, 0:nx]
        rr2 = (x - x0) ** 2 + (y - y0) ** 2
        return (rr2 <= r_out ** 2) & (rr2 > r_in ** 2)

    raise ValueError(f"Unknown shape: {sh}")


def _patch_for_shape(shape: ApertureShape, edgecolor: str, linestyle: str = "-", lw: float = 2.0):
    """Matplotlib patch for quick overlays."""
    sh = shape.shape
    p = shape.params

    if sh == "ellipse":
        x0, y0, a, b, theta = p
        return Ellipse((x0, y0), 2 * a, 2 * b, angle=np.degrees(theta), fill=False,
                       linewidth=lw, edgecolor=edgecolor, linestyle=linestyle)

    if sh == "circle":
        x0, y0, r = p
        return Circle((x0, y0), r, fill=False, linewidth=lw, edgecolor=edgecolor, linestyle=linestyle)

    if sh == "rect":
        x0, y0, w, h, theta = p
        # Rectangle expects bottom-left; we give center + angle
        return Rectangle((x0 - w / 2.0, y0 - h / 2.0), w, h, angle=np.degrees(theta),
                         fill=False, linewidth=lw, edgecolor=edgecolor, linestyle=linestyle)

    if sh == "ellipse_annulus":
        x0, y0, a_in, b_in, a_out, b_out, theta = p
        # Return both boundaries
        return (
            Ellipse((x0, y0), 2 * a_in, 2 * b_in, angle=np.degrees(theta), fill=False,
                    linewidth=1.2, edgecolor=edgecolor, linestyle="--"),
            Ellipse((x0, y0), 2 * a_out, 2 * b_out, angle=np.degrees(theta), fill=False,
                    linewidth=1.2, edgecolor=edgecolor, linestyle="--"),
        )

    if sh == "circle_annulus":
        x0, y0, r_in, r_out = p
        return (
            Circle((x0, y0), r_in, fill=False, linewidth=1.2, edgecolor=edgecolor, linestyle="--"),
            Circle((x0, y0), r_out, fill=False, linewidth=1.2, edgecolor=edgecolor, linestyle="--"),
        )

    raise ValueError(f"Unknown shape: {sh}")


def plot_apertures(img: np.ndarray,
                   apertures: TargetBackgroundApertures,
                   outpng: Path,
                   title: str,
                   show: bool = False) -> None:
    v1, v2 = np.nanpercentile(img, [5, 99])
    plt.figure(figsize=(7, 6))
    plt.imshow(img, origin="lower", vmin=v1, vmax=v2, cmap="viridis")
    plt.colorbar(label="White-light")
    ax = plt.gca()

    tgt = _patch_for_shape(apertures.target, edgecolor="red", lw=2.0)
    bkg = _patch_for_shape(apertures.background, edgecolor="orange", lw=1.6)

    # annuli return tuples
    if isinstance(tgt, tuple):
        for t in tgt:
            ax.add_patch(t)
    else:
        ax.add_patch(tgt)

    if isinstance(bkg, tuple):
        for b in bkg:
            ax.add_patch(b)
    else:
        ax.add_patch(bkg)

    plt.title(title)
    savefig_show(outpng, show)


def _default_center(img: np.ndarray) -> Tuple[float, float]:
    ny, nx = img.shape
    return (nx - 1) / 2.0, (ny - 1) / 2.0


def prompt_aperture(shape_kind: str,
                    default_center: Tuple[float, float],
                    default_theta: float = 0.0) -> ApertureShape:
    """CLI prompts to define an ApertureShape."""
    shape_kind = shape_kind.lower().strip()

    if shape_kind == "ellipse":
        x0 = float(prompt("x0", f"{default_center[0]:.2f}"))
        y0 = float(prompt("y0", f"{default_center[1]:.2f}"))
        a = float(prompt("a (semi-major, pix)", "4.0"))
        b = float(prompt("b (semi-minor, pix)", "4.0"))
        theta = float(prompt("theta (rad)", f"{default_theta:.6f}"))
        return ApertureShape("ellipse", (x0, y0, a, b, theta))

    if shape_kind == "circle":
        x0 = float(prompt("x0", f"{default_center[0]:.2f}"))
        y0 = float(prompt("y0", f"{default_center[1]:.2f}"))
        r = float(prompt("r (pix)", "4.0"))
        return ApertureShape("circle", (x0, y0, r))

    if shape_kind == "rect":
        x0 = float(prompt("x0", f"{default_center[0]:.2f}"))
        y0 = float(prompt("y0", f"{default_center[1]:.2f}"))
        w = float(prompt("width (pix)", "6.0"))
        h = float(prompt("height (pix)", "6.0"))
        theta = float(prompt("theta (rad)", f"{default_theta:.6f}"))
        return ApertureShape("rect", (x0, y0, w, h, theta))

    if shape_kind == "ellipse_annulus":
        x0 = float(prompt("x0", f"{default_center[0]:.2f}"))
        y0 = float(prompt("y0", f"{default_center[1]:.2f}"))
        a_in = float(prompt("a_in (pix)", "6.0"))
        b_in = float(prompt("b_in (pix)", "6.0"))
        a_out = float(prompt("a_out (pix)", "10.0"))
        b_out = float(prompt("b_out (pix)", "10.0"))
        theta = float(prompt("theta (rad)", f"{default_theta:.6f}"))
        return ApertureShape("ellipse_annulus", (x0, y0, a_in, b_in, a_out, b_out, theta))

    if shape_kind == "circle_annulus":
        x0 = float(prompt("x0", f"{default_center[0]:.2f}"))
        y0 = float(prompt("y0", f"{default_center[1]:.2f}"))
        r_in = float(prompt("r_in (pix)", "6.0"))
        r_out = float(prompt("r_out (pix)", "10.0"))
        return ApertureShape("circle_annulus", (x0, y0, r_in, r_out))

    raise ValueError(f"Unknown shape kind: {shape_kind}")


def _click_center(img: np.ndarray, title: str) -> Tuple[float, float]:
    """Pop up a figure and return a single clicked (x, y) center."""
    v1, v2 = np.nanpercentile(img, [5, 99])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(img, origin="lower", vmin=v1, vmax=v2, cmap="viridis")
    ax.set_title(title + "\n(Click once to set center; close window if needed)")
    plt.tight_layout()
    # Block until one click is received.
    pts = plt.ginput(1, timeout=-1)
    plt.close(fig)
    if not pts:
        # If user closes window, keep current center by raising a controlled error
        raise RuntimeError("No click received (window closed).")
    x, y = pts[0]
    return float(x), float(y)


def _update_shape_center(shape: ApertureShape, x0: float, y0: float) -> ApertureShape:
    p = list(shape.params)
    if len(p) < 2:
        raise ValueError("Shape params must include x0,y0 in first two entries.")
    p[0], p[1] = float(x0), float(y0)
    return ApertureShape(shape.shape, tuple(p))


def _preview_apertures_blocking(img: np.ndarray,
                               aps: TargetBackgroundApertures,
                               title: str) -> None:
    """Show apertures overlay in a blocking window (no file write)."""
    v1, v2 = np.nanpercentile(img, [5, 99])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(img, origin="lower", vmin=v1, vmax=v2, cmap="viridis")
    ax.set_title(title)
    tgt = _patch_for_shape(aps.target, edgecolor="red", lw=2.0)
    bkg = _patch_for_shape(aps.background, edgecolor="orange", lw=1.6)

    if isinstance(tgt, tuple):
        for t in tgt:
            ax.add_patch(t)
    else:
        ax.add_patch(tgt)

    if isinstance(bkg, tuple):
        for b in bkg:
            ax.add_patch(b)
    else:
        ax.add_patch(bkg)

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def interactive_define_apertures(img: np.ndarray,
                                 side_label: str,
                                 show: bool = False) -> TargetBackgroundApertures:
    """Interactively define target + background apertures (independent) with iterative recentering.

    Workflow
    --------
    1) Choose shape + size parameters for TARGET and BACKGROUND.
    2) Optionally click to set their centers.
    3) Preview overlay.
    4) Iterate:
        - recenter target by clicking
        - recenter background by clicking
        - edit numeric parameters
      until user approves.

    The pipeline will only proceed once the user approves the apertures.
    """
    xcen, ycen = _default_center(img)

    print(f"\n[{side_label}] Define TARGET aperture")
    tgt_kind = prompt("Target shape (ellipse/circle/rect)", "ellipse").lower().strip()
    tgt = prompt_aperture(tgt_kind, (xcen, ycen), default_theta=0.0)

    if prompt("Click to set TARGET center? (y/n)", "y").lower().startswith("y"):
        try:
            x0, y0 = _click_center(img, title=f"{side_label} TARGET center")
            tgt = _update_shape_center(tgt, x0, y0)
        except RuntimeError:
            print("No click received; keeping numeric center for TARGET.")

    print(f"\n[{side_label}] Define BACKGROUND region")
    bkg_kind = prompt("Background shape (ellipse_annulus/circle_annulus/ellipse/circle/rect)", "ellipse_annulus").lower().strip()
    bkg = prompt_aperture(bkg_kind, (xcen, ycen), default_theta=0.0)

    if prompt("Click to set BACKGROUND center? (y/n)", "y").lower().startswith("y"):
        try:
            x0, y0 = _click_center(img, title=f"{side_label} BACKGROUND center")
            bkg = _update_shape_center(bkg, x0, y0)
        except RuntimeError:
            print("No click received; keeping numeric center for BACKGROUND.")

    aps = TargetBackgroundApertures(target=tgt, background=bkg)

    # Approval loop
    while True:
        if show:
            _preview_apertures_blocking(img, aps, title=f"{side_label} apertures preview")
        else:
            # Even if show_plots is False, we still must show for approval in interactive mode.
            _preview_apertures_blocking(img, aps, title=f"{side_label} apertures preview")

        choice = prompt(
            "Approve apertures? [a=approve, t=recenter target, b=recenter background, p=edit params, q=quit]",
            "a",
        ).lower().strip()

        if choice in ("a", "y", "yes", ""):
            return aps

        if choice == "q":
            raise SystemExit("User quit during aperture definition.")

        if choice == "t":
            try:
                x0, y0 = _click_center(img, title=f"{side_label} recenter TARGET")
                aps = TargetBackgroundApertures(target=_update_shape_center(aps.target, x0, y0),
                                                background=aps.background)
            except RuntimeError:
                print("No click received; TARGET not changed.")
            continue

        if choice == "b":
            try:
                x0, y0 = _click_center(img, title=f"{side_label} recenter BACKGROUND")
                aps = TargetBackgroundApertures(target=aps.target,
                                                background=_update_shape_center(aps.background, x0, y0))
            except RuntimeError:
                print("No click received; BACKGROUND not changed.")
            continue

        if choice == "p":
            print(f"\n[{side_label}] Edit TARGET parameters")
            tgt_kind = aps.target.shape
            aps_t = prompt_aperture(tgt_kind, default_center=(aps.target.params[0], aps.target.params[1]), default_theta=0.0)
            print(f"\n[{side_label}] Edit BACKGROUND parameters")
            bkg_kind = aps.background.shape
            aps_b = prompt_aperture(bkg_kind, default_center=(aps.background.params[0], aps.background.params[1]), default_theta=0.0)
            aps = TargetBackgroundApertures(target=aps_t, background=aps_b)
            continue

        print("Unrecognized option; please choose a/t/b/p/q.")
