from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle, Circle

from .config import ApertureShape, TargetBackgroundApertures
from .utils import prompt


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


def _white_light_limits(img: np.ndarray) -> Tuple[float, float]:
    return tuple(float(x) for x in np.nanpercentile(img, [5, 99]))


def _add_shape_patch(ax, shape: ApertureShape, edgecolor: str, linestyle: str = "-", lw: float = 2.0) -> None:
    patch = _patch_for_shape(shape, edgecolor=edgecolor, linestyle=linestyle, lw=lw)
    if isinstance(patch, tuple):
        for item in patch:
            ax.add_patch(item)
    else:
        ax.add_patch(patch)


def _white_light_two_panel(
    img: np.ndarray,
    title: str,
    *,
    apertures: Optional[TargetBackgroundApertures] = None,
    shapes: Optional[Tuple[ApertureShape, ...]] = None,
):
    """Create a two-panel white-light view.

    Left panel keeps the original aspect ratio and is the aperture-editing view.
    Right panel uses the same data coordinates but compresses the y display scale
    by a factor of 3 to make elongated sources easier to compare to sky charts.
    """
    v1, v2 = _white_light_limits(img)
    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(12, 5.8),
        gridspec_kw={"width_ratios": [1.2, 1.0]},
        constrained_layout=True,
    )

    im = ax_left.imshow(img, origin="lower", vmin=v1, vmax=v2, cmap="viridis", aspect="equal")
    ax_left.set_title("White light")
    ax_left.set_xlabel("x pixel")
    ax_left.set_ylabel("y pixel")

    ax_right.imshow(img, origin="lower", vmin=v1, vmax=v2, cmap="viridis", aspect=(1.0 / 3.0))
    ax_right.set_title("White light, y compressed x3")
    ax_right.set_xlabel("x pixel")
    ax_right.set_ylabel("y pixel")

    if apertures is not None:
        _add_shape_patch(ax_left, apertures.target, edgecolor="red", lw=2.0)
        _add_shape_patch(ax_left, apertures.background, edgecolor="orange", lw=1.6)
        _add_shape_patch(ax_right, apertures.target, edgecolor="red", lw=2.0)
        _add_shape_patch(ax_right, apertures.background, edgecolor="orange", lw=1.6)
    if shapes is not None:
        for shape in shapes:
            _add_shape_patch(ax_left, shape, edgecolor="cyan", lw=1.8)
            _add_shape_patch(ax_right, shape, edgecolor="cyan", lw=1.8)

    fig.suptitle(title)
    fig.colorbar(im, ax=[ax_left, ax_right], label="White-light", fraction=0.035, pad=0.03)
    return fig, ax_left, ax_right


def plot_apertures(img: np.ndarray,
                   apertures: TargetBackgroundApertures,
                   outpng: Path,
                   title: str,
                   show: bool = False) -> None:
    fig, _, _ = _white_light_two_panel(img, title, apertures=apertures)
    outpng.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpng, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


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


def prompt_aperture_current(shape: ApertureShape) -> ApertureShape:
    """Prompt for an aperture's parameters, using the current values as defaults."""
    sh = shape.shape
    p = shape.params

    if sh == "ellipse":
        x0 = float(prompt("x0", f"{p[0]:.2f}"))
        y0 = float(prompt("y0", f"{p[1]:.2f}"))
        a = float(prompt("a (semi-major, pix)", f"{p[2]:.2f}"))
        b = float(prompt("b (semi-minor, pix)", f"{p[3]:.2f}"))
        theta = float(prompt("theta (rad)", f"{p[4]:.6f}"))
        return ApertureShape("ellipse", (x0, y0, a, b, theta))

    if sh == "circle":
        x0 = float(prompt("x0", f"{p[0]:.2f}"))
        y0 = float(prompt("y0", f"{p[1]:.2f}"))
        r = float(prompt("r (pix)", f"{p[2]:.2f}"))
        return ApertureShape("circle", (x0, y0, r))

    if sh == "rect":
        x0 = float(prompt("x0", f"{p[0]:.2f}"))
        y0 = float(prompt("y0", f"{p[1]:.2f}"))
        w = float(prompt("width (pix)", f"{p[2]:.2f}"))
        h = float(prompt("height (pix)", f"{p[3]:.2f}"))
        theta = float(prompt("theta (rad)", f"{p[4]:.6f}"))
        return ApertureShape("rect", (x0, y0, w, h, theta))

    if sh == "ellipse_annulus":
        x0 = float(prompt("x0", f"{p[0]:.2f}"))
        y0 = float(prompt("y0", f"{p[1]:.2f}"))
        a_in = float(prompt("a_in (pix)", f"{p[2]:.2f}"))
        b_in = float(prompt("b_in (pix)", f"{p[3]:.2f}"))
        a_out = float(prompt("a_out (pix)", f"{p[4]:.2f}"))
        b_out = float(prompt("b_out (pix)", f"{p[5]:.2f}"))
        theta = float(prompt("theta (rad)", f"{p[6]:.6f}"))
        return ApertureShape("ellipse_annulus", (x0, y0, a_in, b_in, a_out, b_out, theta))

    if sh == "circle_annulus":
        x0 = float(prompt("x0", f"{p[0]:.2f}"))
        y0 = float(prompt("y0", f"{p[1]:.2f}"))
        r_in = float(prompt("r_in (pix)", f"{p[2]:.2f}"))
        r_out = float(prompt("r_out (pix)", f"{p[3]:.2f}"))
        return ApertureShape("circle_annulus", (x0, y0, r_in, r_out))

    raise ValueError(f"Unknown shape kind: {sh}")


def _click_center(img: np.ndarray, title: str) -> Tuple[float, float]:
    """Pop up a figure and return a single clicked (x, y) center."""
    fig, _, _ = _white_light_two_panel(
        img,
        title + "\nClick once in either panel to set center; close window if needed",
    )
    # Block until one click is received.
    pts = plt.ginput(1, timeout=-1)
    plt.close(fig)
    if not pts:
        # If user closes window, keep current center by raising a controlled error
        raise RuntimeError("No click received (window closed).")
    x, y = pts[0]
    return float(x), float(y)


def _shape_center(shape: ApertureShape) -> Tuple[float, float]:
    return float(shape.params[0]), float(shape.params[1])


def _shape_from_drag(shape_kind: str, p0: Tuple[float, float], p1: Tuple[float, float]) -> ApertureShape:
    shape_kind = shape_kind.lower().strip()
    x0, y0 = p0
    x1, y1 = p1
    xc = 0.5 * (x0 + x1)
    yc = 0.5 * (y0 + y1)
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    eps = 0.5

    if shape_kind == "circle":
        r = max(eps, float(np.hypot(x1 - x0, y1 - y0)))
        return ApertureShape("circle", (float(x0), float(y0), r))

    if shape_kind == "ellipse":
        return ApertureShape("ellipse", (float(xc), float(yc), max(dx / 2.0, eps), max(dy / 2.0, eps), 0.0))

    if shape_kind in ("rect", "rectangle", "square"):
        if shape_kind == "square":
            side = max(dx, dy, eps)
            return ApertureShape("rect", (float(xc), float(yc), side, side, 0.0))
        return ApertureShape("rect", (float(xc), float(yc), max(dx, eps), max(dy, eps), 0.0))

    if shape_kind == "circle_annulus":
        r_in = max(eps, min(dx, dy) / 2.0)
        r_out = max(r_in + eps, max(dx, dy) / 2.0)
        return ApertureShape("circle_annulus", (float(xc), float(yc), r_in, r_out))

    if shape_kind == "ellipse_annulus":
        a_out = max(dx / 2.0, eps)
        b_out = max(dy / 2.0, eps)
        return ApertureShape(
            "ellipse_annulus",
            (float(xc), float(yc), max(a_out * 0.6, eps), max(b_out * 0.6, eps), a_out, b_out, 0.0),
        )

    raise ValueError(f"Unknown shape kind: {shape_kind}")


def _draw_shape_by_drag(img: np.ndarray, shape_kind: str, title: str) -> ApertureShape:
    """Create an aperture shape from one mouse drag on either white-light panel."""
    print(f"{title}: click-drag-release on either panel.")
    fig, ax_left, ax_right = _white_light_two_panel(
        img,
        title + "\nClick-drag on either panel. Circle uses click=center, release=edge.",
    )
    state = {"start": None, "shape": None, "patches": []}

    def clear_preview() -> None:
        for patch in state["patches"]:
            try:
                patch.remove()
            except Exception:
                pass
        state["patches"] = []

    def draw_preview(shape: ApertureShape) -> None:
        clear_preview()
        for ax in (ax_left, ax_right):
            patch = _patch_for_shape(shape, edgecolor="cyan", linestyle="-", lw=1.5)
            patches = patch if isinstance(patch, tuple) else (patch,)
            for item in patches:
                ax.add_patch(item)
                state["patches"].append(item)
        fig.canvas.draw_idle()

    def on_press(event):
        if event.inaxes not in (ax_left, ax_right) or event.xdata is None or event.ydata is None:
            return
        state["start"] = (float(event.xdata), float(event.ydata))

    def on_motion(event):
        if state["start"] is None or event.inaxes not in (ax_left, ax_right) or event.xdata is None or event.ydata is None:
            return
        try:
            shape = _shape_from_drag(shape_kind, state["start"], (float(event.xdata), float(event.ydata)))
        except ValueError:
            return
        draw_preview(shape)

    def on_release(event):
        if state["start"] is None or event.inaxes not in (ax_left, ax_right) or event.xdata is None or event.ydata is None:
            return
        state["shape"] = _shape_from_drag(shape_kind, state["start"], (float(event.xdata), float(event.ydata)))
        plt.close(fig)

    def on_key(event):
        if event.key == "q":
            state["shape"] = None
            plt.close(fig)

    cids = [
        fig.canvas.mpl_connect("button_press_event", on_press),
        fig.canvas.mpl_connect("motion_notify_event", on_motion),
        fig.canvas.mpl_connect("button_release_event", on_release),
        fig.canvas.mpl_connect("key_press_event", on_key),
    ]
    plt.show()
    for cid in cids:
        fig.canvas.mpl_disconnect(cid)
    clear_preview()
    if state["shape"] is None:
        raise RuntimeError("No aperture was drawn.")
    return state["shape"]


def _drag_move_shape(img: np.ndarray, shape: ApertureShape, title: str) -> ApertureShape:
    """Move an existing aperture by dragging its center on either panel."""
    print(f"{title}: click-drag the aperture center on either panel.")
    fig, ax_left, ax_right = _white_light_two_panel(
        img,
        title + "\nDrag aperture center on either panel to move. Press q to cancel.",
        shapes=(shape,),
    )
    # Redraw cyan overlays on both panels during dragging.
    for ax in (ax_left, ax_right):
        for patch in list(ax.patches):
            patch.remove()

    state = {"shape": shape, "start_xy": None, "start_center": None, "done": False, "cancel": False, "patches": []}

    def clear_preview() -> None:
        for patch in state["patches"]:
            try:
                patch.remove()
            except Exception:
                pass
        state["patches"] = []

    def draw_preview(current: ApertureShape) -> None:
        clear_preview()
        for ax in (ax_left, ax_right):
            patch = _patch_for_shape(current, edgecolor="cyan", lw=1.8)
            patches = patch if isinstance(patch, tuple) else (patch,)
            for item in patches:
                ax.add_patch(item)
                state["patches"].append(item)
        fig.canvas.draw_idle()

    draw_preview(shape)

    def on_press(event):
        if event.inaxes not in (ax_left, ax_right) or event.xdata is None or event.ydata is None:
            return
        state["start_xy"] = (float(event.xdata), float(event.ydata))
        state["start_center"] = _shape_center(state["shape"])

    def on_motion(event):
        if state["start_xy"] is None or event.inaxes not in (ax_left, ax_right) or event.xdata is None or event.ydata is None:
            return
        sx, sy = state["start_xy"]
        cx, cy = state["start_center"]
        moved = _update_shape_center(state["shape"], cx + float(event.xdata) - sx, cy + float(event.ydata) - sy)
        draw_preview(moved)

    def on_release(event):
        if state["start_xy"] is None or event.inaxes not in (ax_left, ax_right) or event.xdata is None or event.ydata is None:
            return
        sx, sy = state["start_xy"]
        cx, cy = state["start_center"]
        state["shape"] = _update_shape_center(state["shape"], cx + float(event.xdata) - sx, cy + float(event.ydata) - sy)
        state["done"] = True
        plt.close(fig)

    def on_key(event):
        if event.key == "q":
            state["cancel"] = True
            plt.close(fig)

    cids = [
        fig.canvas.mpl_connect("button_press_event", on_press),
        fig.canvas.mpl_connect("motion_notify_event", on_motion),
        fig.canvas.mpl_connect("button_release_event", on_release),
        fig.canvas.mpl_connect("key_press_event", on_key),
    ]
    plt.show()
    for cid in cids:
        fig.canvas.mpl_disconnect(cid)
    clear_preview()
    if state["cancel"] or not state["done"]:
        raise RuntimeError("Aperture move cancelled.")
    return state["shape"]


def _auto_background_from_target(target: ApertureShape, kind: str) -> ApertureShape:
    kind = kind.lower().strip()
    p = target.params
    if kind == "circle_annulus":
        if target.shape == "circle":
            x0, y0, r = p
            return ApertureShape("circle_annulus", (x0, y0, r * 1.8, r * 3.0))
        x0, y0 = p[0], p[1]
        scale = max(p[2:4]) if len(p) >= 4 else 4.0
        return ApertureShape("circle_annulus", (x0, y0, scale * 1.8, scale * 3.0))

    if kind == "ellipse_annulus":
        x0, y0 = p[0], p[1]
        if target.shape == "ellipse":
            _, _, a, b, theta = p
        elif target.shape == "circle":
            _, _, r = p
            a, b, theta = r, r, 0.0
        elif target.shape == "rect":
            _, _, w, h, theta = p
            a, b = w / 2.0, h / 2.0
        else:
            a, b, theta = 4.0, 4.0, 0.0
        return ApertureShape("ellipse_annulus", (x0, y0, a * 1.8, b * 1.8, a * 3.0, b * 3.0, theta))

    raise ValueError(f"Cannot auto-create background kind: {kind}")


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
    fig, _, _ = _white_light_two_panel(img, title, apertures=aps)
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
    print(f"\n[{side_label}] Define TARGET aperture")
    tgt_kind = prompt("Target shape (circle/ellipse/rect/square)", "ellipse").lower().strip()
    while True:
        try:
            tgt = _draw_shape_by_drag(img, tgt_kind, title=f"{side_label} TARGET aperture")
        except RuntimeError as exc:
            print(exc)
            if not prompt("Retry target aperture? (y/n)", "y").lower().startswith("y"):
                raise SystemExit("User quit during target aperture definition.")
            continue
        break

    print(f"\n[{side_label}] Define BACKGROUND region")
    bkg_kind = prompt("Background shape (auto_ellipse_annulus/auto_circle_annulus/ellipse_annulus/circle_annulus/ellipse/circle/rect)", "auto_ellipse_annulus").lower().strip()
    if bkg_kind == "auto_ellipse_annulus":
        bkg = _auto_background_from_target(tgt, "ellipse_annulus")
    elif bkg_kind == "auto_circle_annulus":
        bkg = _auto_background_from_target(tgt, "circle_annulus")
    else:
        draw_kind = bkg_kind
        if draw_kind in ("ellipse", "circle", "rect", "rectangle", "square"):
            print("Background will be drawn as a filled region, not an annulus.")
        while True:
            try:
                bkg = _draw_shape_by_drag(img, draw_kind, title=f"{side_label} BACKGROUND region")
            except RuntimeError as exc:
                print(exc)
                if not prompt("Retry background region? (y/n)", "y").lower().startswith("y"):
                    raise SystemExit("User quit during background definition.")
                continue
            break

    if prompt("Recenter BACKGROUND by clicking? (y/n)", "n").lower().startswith("y"):
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
            "Approve apertures? [a=approve, rt=redraw target, rb=redraw background, t=recenter target, b=recenter background, p=edit params, q=quit]",
            "a",
        ).lower().strip()

        if choice in ("a", "y", "yes", ""):
            return aps

        if choice == "q":
            raise SystemExit("User quit during aperture definition.")

        if choice == "rt":
            try:
                aps = TargetBackgroundApertures(
                    target=_draw_shape_by_drag(img, aps.target.shape, title=f"{side_label} redraw TARGET aperture"),
                    background=aps.background,
                )
            except RuntimeError:
                print("No target aperture drawn; TARGET not changed.")
            continue

        if choice == "rb":
            try:
                aps = TargetBackgroundApertures(
                    target=aps.target,
                    background=_draw_shape_by_drag(img, aps.background.shape, title=f"{side_label} redraw BACKGROUND region"),
                )
            except RuntimeError:
                print("No background region drawn; BACKGROUND not changed.")
            continue

        if choice == "t":
            try:
                aps = TargetBackgroundApertures(
                    target=_drag_move_shape(img, aps.target, title=f"{side_label} move TARGET"),
                    background=aps.background,
                )
            except RuntimeError:
                print("TARGET not changed.")
            continue

        if choice == "b":
            try:
                aps = TargetBackgroundApertures(
                    target=aps.target,
                    background=_drag_move_shape(img, aps.background, title=f"{side_label} move BACKGROUND"),
                )
            except RuntimeError:
                print("BACKGROUND not changed.")
            continue

        if choice == "p":
            print(f"\n[{side_label}] Edit TARGET parameters")
            aps_t = prompt_aperture_current(aps.target)
            print(f"\n[{side_label}] Edit BACKGROUND parameters")
            aps_b = prompt_aperture_current(aps.background)
            aps = TargetBackgroundApertures(target=aps_t, background=aps_b)
            continue

        print("Unrecognized option; please choose a/t/b/p/q.")
