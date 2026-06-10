from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle, Circle
from matplotlib.widgets import Slider, Button
from photutils.aperture import (
    CircularAnnulus,
    CircularAperture,
    EllipticalAnnulus,
    EllipticalAperture,
    RectangularAperture,
)

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


def aperture_weight_mask(ny: int, nx: int, shape: ApertureShape) -> np.ndarray:
    """Fractional aperture mask using exact pixel/aperture intersection.

    Returns weights in [0, 1] with shape (ny, nx). A value of 0.25 means
    one quarter of that spaxel is covered by the aperture.
    """
    sh = shape.shape
    p = shape.params

    if sh == "circle":
        x0, y0, r = p
        aper = CircularAperture((x0, y0), r=r)
    elif sh == "circle_annulus":
        x0, y0, r_in, r_out = p
        aper = CircularAnnulus((x0, y0), r_in=r_in, r_out=r_out)
    elif sh == "ellipse":
        x0, y0, a, b, theta = p
        aper = EllipticalAperture((x0, y0), a=a, b=b, theta=theta)
    elif sh == "ellipse_annulus":
        x0, y0, a_in, b_in, a_out, b_out, theta = p
        aper = EllipticalAnnulus((x0, y0), a_in=a_in, a_out=a_out, b_in=b_in, b_out=b_out, theta=theta)
    elif sh == "rect":
        x0, y0, w, h, theta = p
        aper = RectangularAperture((x0, y0), w=w, h=h, theta=theta)
    else:
        raise ValueError(f"Unknown shape: {sh}")

    mask = aper.to_mask(method="exact")
    image = mask.to_image((ny, nx))
    if image is None:
        return np.zeros((ny, nx), dtype=float)
    return np.clip(np.asarray(image, dtype=float), 0.0, 1.0)


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
    )
    fig.subplots_adjust(left=0.07, right=0.90, bottom=0.18, top=0.88, wspace=0.25)

    im = ax_left.imshow(img, origin="lower", vmin=v1, vmax=v2, cmap="viridis", aspect="equal")
    ax_left.set_title("White light")
    ax_left.set_xlabel("x pixel")
    ax_left.set_ylabel("y pixel")

    im_right = ax_right.imshow(img, origin="lower", vmin=v1, vmax=v2, cmap="viridis", aspect=(1.0 / 3.0))
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

    finite = np.asarray(img)[np.isfinite(img)]
    if finite.size:
        p_min, p_max = 0.0, 100.0
    else:
        finite = np.asarray([0.0, 1.0])
        p_min, p_max = 0.0, 100.0

    ax_low = fig.add_axes([0.12, 0.075, 0.58, 0.025])
    ax_high = fig.add_axes([0.12, 0.035, 0.58, 0.025])
    ax_reset = fig.add_axes([0.73, 0.04, 0.10, 0.055])
    low_slider = Slider(ax_low, "Low %", p_min, p_max, valinit=5.0, valstep=0.1)
    high_slider = Slider(ax_high, "High %", p_min, p_max, valinit=99.0, valstep=0.1)
    reset_button = Button(ax_reset, "Reset")

    def update_scale(_val=None) -> None:
        lo = float(low_slider.val)
        hi = float(high_slider.val)
        if hi <= lo:
            return
        new_v1, new_v2 = np.nanpercentile(finite, [lo, hi])
        if not np.isfinite(new_v1) or not np.isfinite(new_v2) or new_v2 <= new_v1:
            return
        im.set_clim(new_v1, new_v2)
        im_right.set_clim(new_v1, new_v2)
        fig.canvas.draw_idle()

    def reset_scale(_event=None) -> None:
        low_slider.set_val(5.0)
        high_slider.set_val(99.0)

    low_slider.on_changed(update_scale)
    high_slider.on_changed(update_scale)
    reset_button.on_clicked(reset_scale)
    fig._kcwi_scale_widgets = (low_slider, high_slider, reset_button)

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


def _draw_shape_by_drag(
    img: np.ndarray,
    shape_kind: str,
    title: str,
    *,
    reference_shapes: Tuple[ApertureShape, ...] = (),
) -> ApertureShape:
    """Create and adjust an aperture shape on either white-light panel."""
    print(f"{title}: click-drag-release to draw, then move/resize. Press a/Enter when done.")
    fig, ax_left, ax_right = _white_light_two_panel(
        img,
        title + "\nClick-drag to draw. m=move, e=resize, drag to adjust. a/Enter=accept, r=redraw, q=cancel.",
    )
    state = {
        "start": None,
        "shape": None,
        "patches": [],
        "move_start": None,
        "move_center": None,
        "mode": "move",
        "accepted": False,
        "cancel": False,
    }

    for ax in (ax_left, ax_right):
        for ref in reference_shapes:
            _add_shape_patch(ax, ref, edgecolor="red", linestyle="-", lw=1.8)

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
        xy = (float(event.xdata), float(event.ydata))
        if state["shape"] is None:
            state["start"] = xy
            return
        state["move_start"] = xy
        state["move_center"] = _shape_center(state["shape"])

    def on_motion(event):
        if event.inaxes not in (ax_left, ax_right) or event.xdata is None or event.ydata is None:
            return
        if state["start"] is not None:
            try:
                shape = _shape_from_drag(shape_kind, state["start"], (float(event.xdata), float(event.ydata)))
            except ValueError:
                return
            draw_preview(shape)
            return
        if state["shape"] is not None and state["move_start"] is not None:
            if state["mode"] == "resize":
                draw_preview(_resize_shape_to_point(state["shape"], float(event.xdata), float(event.ydata)))
            else:
                sx, sy = state["move_start"]
                cx, cy = state["move_center"]
                moved = _update_shape_center(state["shape"], cx + float(event.xdata) - sx, cy + float(event.ydata) - sy)
                draw_preview(moved)

    def on_release(event):
        if state["start"] is None or event.inaxes not in (ax_left, ax_right) or event.xdata is None or event.ydata is None:
            if state["move_start"] is not None and event.inaxes in (ax_left, ax_right) and event.xdata is not None and event.ydata is not None:
                if state["mode"] == "resize":
                    state["shape"] = _resize_shape_to_point(state["shape"], float(event.xdata), float(event.ydata))
                else:
                    sx, sy = state["move_start"]
                    cx, cy = state["move_center"]
                    state["shape"] = _update_shape_center(state["shape"], cx + float(event.xdata) - sx, cy + float(event.ydata) - sy)
                draw_preview(state["shape"])
            state["move_start"] = None
            state["move_center"] = None
            return
        state["shape"] = _shape_from_drag(shape_kind, state["start"], (float(event.xdata), float(event.ydata)))
        state["start"] = None
        draw_preview(state["shape"])

    def on_key(event):
        if event.key in ("a", "enter", "return"):
            state["accepted"] = True
            plt.close(fig)
        elif event.key == "m":
            state["mode"] = "move"
            fig.suptitle(title + "\nMOVE mode: drag aperture to move. e=resize, a/Enter=accept, r=redraw, q=cancel.")
            fig.canvas.draw_idle()
        elif event.key == "e":
            state["mode"] = "resize"
            fig.suptitle(title + "\nRESIZE mode: drag to set edge/corner from current center. m=move, a/Enter=accept, r=redraw, q=cancel.")
            fig.canvas.draw_idle()
        elif event.key == "r":
            state["shape"] = None
            state["start"] = None
            state["move_start"] = None
            state["move_center"] = None
            clear_preview()
            fig.canvas.draw_idle()
        elif event.key == "q":
            state["cancel"] = True
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
    if state["cancel"] or not state["accepted"] or state["shape"] is None:
        raise RuntimeError("No aperture was drawn.")
    return state["shape"]


def _drag_move_shape(
    img: np.ndarray,
    shape: ApertureShape,
    title: str,
    *,
    reference_shapes: Tuple[ApertureShape, ...] = (),
) -> ApertureShape:
    """Move an existing aperture by dragging on either panel until accepted."""
    print(f"{title}: click-drag the aperture on either panel to move/resize. Press a/Enter when done.")
    fig, ax_left, ax_right = _white_light_two_panel(
        img,
        title + "\nm=move, e=resize, drag to adjust. a/Enter=accept, q=cancel.",
    )

    for ax in (ax_left, ax_right):
        for ref in reference_shapes:
            _add_shape_patch(ax, ref, edgecolor="red", linestyle="-", lw=1.8)

    state = {
        "shape": shape,
        "start_xy": None,
        "start_center": None,
        "mode": "move",
        "accepted": False,
        "cancel": False,
        "patches": [],
    }

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
        if state["mode"] == "resize":
            draw_preview(_resize_shape_to_point(state["shape"], float(event.xdata), float(event.ydata)))
        else:
            sx, sy = state["start_xy"]
            cx, cy = state["start_center"]
            moved = _update_shape_center(state["shape"], cx + float(event.xdata) - sx, cy + float(event.ydata) - sy)
            draw_preview(moved)

    def on_release(event):
        if state["start_xy"] is None or event.inaxes not in (ax_left, ax_right) or event.xdata is None or event.ydata is None:
            state["start_xy"] = None
            state["start_center"] = None
            return
        if state["mode"] == "resize":
            state["shape"] = _resize_shape_to_point(state["shape"], float(event.xdata), float(event.ydata))
        else:
            sx, sy = state["start_xy"]
            cx, cy = state["start_center"]
            state["shape"] = _update_shape_center(state["shape"], cx + float(event.xdata) - sx, cy + float(event.ydata) - sy)
        state["start_xy"] = None
        state["start_center"] = None
        draw_preview(state["shape"])

    def on_key(event):
        if event.key in ("a", "enter", "return"):
            state["accepted"] = True
            plt.close(fig)
        elif event.key == "m":
            state["mode"] = "move"
            fig.suptitle(title + "\nMOVE mode: drag aperture to move. e=resize, a/Enter=accept, q=cancel.")
            fig.canvas.draw_idle()
        elif event.key == "e":
            state["mode"] = "resize"
            fig.suptitle(title + "\nRESIZE mode: drag to set edge/corner from current center. m=move, a/Enter=accept, q=cancel.")
            fig.canvas.draw_idle()
        elif event.key == "q":
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
    if state["cancel"] or not state["accepted"]:
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


def _resize_shape_to_point(shape: ApertureShape, x: float, y: float) -> ApertureShape:
    """Resize a shape around its current center using the pointer as an edge/corner."""
    sh = shape.shape
    p = list(shape.params)
    x0, y0 = float(p[0]), float(p[1])
    dx = abs(float(x) - x0)
    dy = abs(float(y) - y0)
    eps = 0.5

    if sh == "circle":
        return ApertureShape(sh, (x0, y0, max(float(np.hypot(float(x) - x0, float(y) - y0)), eps)))

    if sh == "ellipse":
        return ApertureShape(sh, (x0, y0, max(dx, eps), max(dy, eps), p[4]))

    if sh == "rect":
        return ApertureShape(sh, (x0, y0, max(2.0 * dx, eps), max(2.0 * dy, eps), p[4]))

    if sh == "circle_annulus":
        r_in_old, r_out_old = max(float(p[2]), eps), max(float(p[3]), eps)
        ratio = min(0.95, r_in_old / r_out_old)
        r_out = max(float(np.hypot(float(x) - x0, float(y) - y0)), r_in_old + eps, eps)
        return ApertureShape(sh, (x0, y0, max(r_out * ratio, eps), r_out))

    if sh == "ellipse_annulus":
        a_in_old, b_in_old = max(float(p[2]), eps), max(float(p[3]), eps)
        a_out_old, b_out_old = max(float(p[4]), eps), max(float(p[5]), eps)
        a_ratio = min(0.95, a_in_old / a_out_old)
        b_ratio = min(0.95, b_in_old / b_out_old)
        a_out = max(dx, eps)
        b_out = max(dy, eps)
        return ApertureShape(sh, (x0, y0, max(a_out * a_ratio, eps), max(b_out * b_ratio, eps), a_out, b_out, p[6]))

    raise ValueError(f"Unknown shape: {sh}")


def _preview_apertures_blocking(img: np.ndarray,
                               aps: TargetBackgroundApertures,
                               title: str) -> None:
    """Show apertures overlay in a blocking window (no file write)."""
    fig, _, _ = _white_light_two_panel(img, title, apertures=aps)
    plt.show()
    plt.close(fig)


def review_apertures(img: np.ndarray,
                     apertures: TargetBackgroundApertures,
                     side_label: str,
                     show: bool = False) -> TargetBackgroundApertures:
    """Show proposed apertures and allow approval or modification."""
    aps = apertures
    while True:
        _preview_apertures_blocking(img, aps, title=f"{side_label} apertures preview")

        choice = prompt(
            "Approve apertures? [a=approve, rt=redraw target, st=change target shape, rb=redraw background, sb=change background shape, t=move target, b=move background, p=edit params, q=quit]",
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

        if choice == "st":
            new_kind = prompt("New target shape (circle/ellipse/rect/square)", aps.target.shape).lower().strip()
            try:
                aps = TargetBackgroundApertures(
                    target=_draw_shape_by_drag(img, new_kind, title=f"{side_label} change TARGET shape"),
                    background=aps.background,
                )
            except RuntimeError:
                print("No target aperture drawn; TARGET not changed.")
            continue

        if choice == "rb":
            try:
                aps = TargetBackgroundApertures(
                    target=aps.target,
                    background=_draw_shape_by_drag(
                        img,
                        aps.background.shape,
                        title=f"{side_label} redraw BACKGROUND region",
                        reference_shapes=(aps.target,),
                    ),
                )
            except RuntimeError:
                print("No background region drawn; BACKGROUND not changed.")
            continue

        if choice == "sb":
            new_kind = prompt(
                "New background shape (auto_ellipse_annulus/auto_circle_annulus/ellipse_annulus/circle_annulus/ellipse/circle/rect)",
                aps.background.shape,
            ).lower().strip()
            try:
                if new_kind == "auto_ellipse_annulus":
                    bkg = _auto_background_from_target(aps.target, "ellipse_annulus")
                    bkg = _drag_move_shape(
                        img,
                        bkg,
                        title=f"{side_label} change BACKGROUND shape",
                        reference_shapes=(aps.target,),
                    )
                elif new_kind == "auto_circle_annulus":
                    bkg = _auto_background_from_target(aps.target, "circle_annulus")
                    bkg = _drag_move_shape(
                        img,
                        bkg,
                        title=f"{side_label} change BACKGROUND shape",
                        reference_shapes=(aps.target,),
                    )
                else:
                    bkg = _draw_shape_by_drag(
                        img,
                        new_kind,
                        title=f"{side_label} change BACKGROUND shape",
                        reference_shapes=(aps.target,),
                    )
                aps = TargetBackgroundApertures(target=aps.target, background=bkg)
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
                    background=_drag_move_shape(
                        img,
                        aps.background,
                        title=f"{side_label} move BACKGROUND",
                        reference_shapes=(aps.target,),
                    ),
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

        print("Unrecognized option; please choose a/rt/st/rb/sb/t/b/p/q.")


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
        try:
            bkg = _drag_move_shape(
                img,
                bkg,
                title=f"{side_label} BACKGROUND region",
                reference_shapes=(tgt,),
            )
        except RuntimeError:
            print("BACKGROUND not changed.")
    elif bkg_kind == "auto_circle_annulus":
        bkg = _auto_background_from_target(tgt, "circle_annulus")
        try:
            bkg = _drag_move_shape(
                img,
                bkg,
                title=f"{side_label} BACKGROUND region",
                reference_shapes=(tgt,),
            )
        except RuntimeError:
            print("BACKGROUND not changed.")
    else:
        draw_kind = bkg_kind
        if draw_kind in ("ellipse", "circle", "rect", "rectangle", "square"):
            print("Background will be drawn as a filled region, not an annulus.")
        while True:
            try:
                bkg = _draw_shape_by_drag(
                    img,
                    draw_kind,
                    title=f"{side_label} BACKGROUND region",
                    reference_shapes=(tgt,),
                )
            except RuntimeError as exc:
                print(exc)
                if not prompt("Retry background region? (y/n)", "y").lower().startswith("y"):
                    raise SystemExit("User quit during background definition.")
                continue
            break

    aps = TargetBackgroundApertures(target=tgt, background=bkg)
    return review_apertures(img, aps, side_label=side_label, show=show)
