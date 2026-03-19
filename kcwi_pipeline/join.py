from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt


def concat_join(
    lam_blue: np.ndarray,
    flux_blue: np.ndarray,
    lam_red: np.ndarray,
    flux_red: np.ndarray,
    *,
    sort_by_wavelength: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """KCWI join: simple concatenation (NO overlap stitching)."""
    lam = np.concatenate([np.asarray(lam_blue).ravel(), np.asarray(lam_red).ravel()])
    flux = np.concatenate([np.asarray(flux_blue).ravel(), np.asarray(flux_red).ravel()])
    if sort_by_wavelength:
        s = np.argsort(lam)
        lam, flux = lam[s], flux[s]
    return lam, flux


def plot_join_diagnostic(
    objname: str,
    lam_blue: np.ndarray,
    flux_blue: np.ndarray,
    lam_red: np.ndarray,
    flux_red: np.ndarray,
    *,
    outpng: Path,
    show: bool,
) -> None:
    """Plot BLUE and RED together; supports saving a zoomed-in version.

    While the plot window is open:
      - Use the matplotlib toolbar to zoom/pan.
      - Press 'z' to save a zoomed PNG using the current axis limits.
    """
    outpng = Path(outpng)
    outpng.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    if np.asarray(lam_blue).size:
        ax.plot(lam_blue, flux_blue, lw=1.0, label="Blue")
    if np.asarray(lam_red).size:
        ax.plot(lam_red, flux_red, lw=1.0, label="Red")
    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel("Flux")
    ax.set_title(f"{objname}: BLUE+RED diagnostic (press 'z' to save zoom)")
    ax.legend()
    ax.grid(alpha=0.2)

    # Always save the full-range diagnostic
    fig.savefig(outpng, dpi=200, bbox_inches="tight")

    def save_zoomed() -> Path:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        fn = outpng.with_name(
            outpng.stem + f"_zoom_x{x0:.0f}-{x1:.0f}_y{y0:.3g}-{y1:.3g}" + outpng.suffix
        )
        fig.savefig(fn, dpi=200, bbox_inches="tight")
        return fn

    def on_key(event):
        if event.key == "z":
            try:
                fn = save_zoomed()
                ax.set_title(f"{objname}: BLUE+RED diagnostic [zoom saved]")
                fig.canvas.draw_idle()
                print(f"Saved zoomed plot: {fn}")
            except Exception as e:
                print(f"Could not save zoomed plot: {e}")

    cid = fig.canvas.mpl_connect("key_press_event", on_key)

    if show:
        plt.show()
    else:
        plt.close(fig)

    fig.canvas.mpl_disconnect(cid)


def interactive_rescale_and_approve_flux(
    *,
    objname: str,
    lam_blue: np.ndarray,
    flux_blue: np.ndarray,
    lam_red: np.ndarray,
    flux_red: np.ndarray,
    outdir: Path,
    show: bool = True,
    interactive: bool = True,
) -> Tuple[float, float]:
    """Iteratively rescale BLUE/RED until user approves.

    Returns (blue_scale, red_scale). These multiplicative factors are applied
    to the provided flux arrays *before* saving the final joined spectrum.

    Controls (interactive=True):
      - Use toolbar to zoom/pan.
      - Press 'z' to save zoomed view.
      - Press 'a' (or Enter) in the figure to approve current scaling and continue.
      - Press 'q' in the figure to abort.
      - If figure key events are missed, terminal prompts also provide approve/abort.

    After closing without approval, you'll be prompted in the terminal to update
    scales and the plot will reopen.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    blue_scale = 1.0
    red_scale = 1.0

    if not interactive:
        with open(outdir / f"{objname}_join_scale.txt", "w") as f:
            f.write(f"blue_scale {blue_scale}\nred_scale {red_scale}\n")
        return blue_scale, red_scale

    while True:
        fb = np.asarray(flux_blue) * blue_scale
        fr = np.asarray(flux_red)  * red_scale

        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(lam_blue, fb, lw=1.0, label=f"Blue x{blue_scale:.4g}")
        ax.plot(lam_red,  fr, lw=1.0, label=f"Red  x{red_scale:.4g}")
        ax.set_xlabel("Wavelength (Å)")
        ax.set_ylabel("Flux")
        ax.set_title(f"{objname}: adjust join scaling (a=approve, z=save zoom, q=abort)")
        ax.legend()
        ax.grid(alpha=0.2)

        # Save full-range current scaling diagnostic
        fig.savefig(outdir / f"{objname}_join_scaling.png", dpi=200, bbox_inches="tight")

        def save_zoomed() -> Path:
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            fn = outdir / f"{objname}_join_scaling_zoom_x{x0:.0f}-{x1:.0f}_y{y0:.3g}-{y1:.3g}.png"
            fig.savefig(fn, dpi=200, bbox_inches="tight")
            return fn

        decision = {"approved": False, "abort": False}

        def on_key(event):
            if event.key in ("a", "enter", "return"):
                decision["approved"] = True
                plt.close(fig)
            elif event.key == "q":
                decision["abort"] = True
                plt.close(fig)
            elif event.key == "z":
                try:
                    fn = save_zoomed()
                    ax.set_title(f"{objname}: adjust join scaling [zoom saved]")
                    fig.canvas.draw_idle()
                    print(f"Saved zoomed plot: {fn}")
                except Exception as e:
                    print(f"Could not save zoomed plot: {e}")

        cid = fig.canvas.mpl_connect("key_press_event", on_key)

        if show:
            plt.show()
        else:
            plt.close(fig)

        fig.canvas.mpl_disconnect(cid)

        if decision["abort"]:
            raise RuntimeError("User aborted join rescale approval (pressed 'q').")

        if decision["approved"]:
            with open(outdir / f"{objname}_join_scale.txt", "w") as f:
                f.write(f"blue_scale {blue_scale}\n")
                f.write(f"red_scale {red_scale}\n")
            return blue_scale, red_scale

        print("\nRescale options:")
        print("  [a] Approve current scaling and continue")
        print("  [1] Multiply RED by factor")
        print("  [2] Multiply BLUE by factor")
        print("  [3] Set RED scale (absolute)")
        print("  [4] Set BLUE scale (absolute)")
        print("  [q] Abort pipeline")
        print("  [Enter] Re-open plot with same scales")
        choice = input("Choice: ").strip()

        if choice.lower() == "a":
            with open(outdir / f"{objname}_join_scale.txt", "w") as f:
                f.write(f"blue_scale {blue_scale}\n")
                f.write(f"red_scale {red_scale}\n")
            return blue_scale, red_scale
        elif choice.lower() == "q":
            raise RuntimeError("User aborted join rescale approval (terminal 'q').")
        elif choice == "1":
            fac = float(input(f"Multiply RED by factor (current {red_scale}): ").strip())
            red_scale *= fac
        elif choice == "2":
            fac = float(input(f"Multiply BLUE by factor (current {blue_scale}): ").strip())
            blue_scale *= fac
        elif choice == "3":
            red_scale = float(input(f"Set RED scale (current {red_scale}): ").strip())
        elif choice == "4":
            blue_scale = float(input(f"Set BLUE scale (current {blue_scale}): ").strip())
        else:
            pass
