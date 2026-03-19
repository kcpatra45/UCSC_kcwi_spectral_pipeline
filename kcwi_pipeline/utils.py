from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional
import re
import numpy as np
import matplotlib.pyplot as plt

SIDE_PAT = re.compile(r"^(?P<obj>.+)_(?P<side>blue|red)_coadd_icube\.fits$", re.IGNORECASE)


def safe_filename(s: str) -> str:
    return "".join(c if (c.isalnum() or c in ("-", "_", ".")) else "_" for c in str(s))


def die(msg: str) -> None:
    raise SystemExit(msg)


def prompt(msg: str, default: Optional[str] = None) -> str:
    if default is None:
        return input(msg).strip()
    out = input(f"{msg} [{default}] ").strip()
    return out if out else default


def choose_from_list(title: str, items: List[str]) -> str:
    print("\n" + title)
    for i, it in enumerate(items):
        print(f"  {i:2d}: {it}")
    while True:
        s = prompt("Enter index: ")
        try:
            k = int(s)
            if 0 <= k < len(items):
                return items[k]
        except Exception:
            pass
        print("Invalid index, try again.")


def savefig_show(path: Path, show: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    if show:
        plt.show()
    plt.close()


def parse_trim(s: str) -> Tuple[Optional[float], Optional[float]]:
    """Parse 'min:max' where either can be blank."""
    s = (s or "").strip()
    if s == "":
        return (None, None)
    if ":" not in s:
        raise ValueError("Trim must be 'min:max' with ':' present.")
    a, b = s.split(":", 1)
    amin = float(a) if a.strip() else None
    amax = float(b) if b.strip() else None
    return amin, amax


def parse_ranges(s: str) -> List[Tuple[float, float]]:
    """Parse comma-separated ranges like '6860-6890,7600-7630'."""
    s = (s or "").strip()
    if not s:
        return []
    out: List[Tuple[float, float]] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" not in part:
            raise ValueError(f"Range '{part}' must look like lo-hi")
        lo, hi = part.split("-", 1)
        lo_f = float(lo.strip())
        hi_f = float(hi.strip())
        if hi_f < lo_f:
            lo_f, hi_f = hi_f, lo_f
        out.append((lo_f, hi_f))
    return out


def apply_trim_and_exclude(
    lam: np.ndarray,
    flux: np.ndarray,
    trim_min: Optional[float],
    trim_max: Optional[float],
    exclude: List[Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    m = np.isfinite(lam) & np.isfinite(flux)
    if trim_min is not None:
        m &= lam >= trim_min
    if trim_max is not None:
        m &= lam <= trim_max
    for lo, hi in exclude:
        m &= ~((lam >= lo) & (lam <= hi))
    return lam[m], flux[m]
