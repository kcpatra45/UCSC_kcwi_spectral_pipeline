from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


STANDARD_NAMES: Dict[int, str] = {
    1: "HD 19445",
    2: "HD 84937",
    3: "BD+26 2606",
    4: "BD+17 4708",
    5: "HD 140283",
    6: "GD-248",
    7: "G158-100",
    8: "LTT 1788",
    9: "LTT 377",
    10: "HZ 4",
    11: "HZ 7",
    12: "ROSS 627",
    13: "LDS 235B",
    14: "G99-37",
    15: "LB 227",
    16: "L745-46A",
    17: "FEIGE 24",
    18: "G60-54",
    19: "G24-9",
    20: "BD+28 4211",
    21: "G191B2B",
    22: "G157-34",
    23: "G138-31",
    24: "HZ 44",
    25: "LTT 9491",
    26: "FEIGE 110",
    27: "FEIGE 34",
    28: "LTT 1020",
    29: "LTT 9239",
    30: "HILTNER 600",
    31: "BD+25 3941",
    32: "BD+33 2642",
    33: "FEIGE 56",
    34: "G193-74",
    35: "EG145",
    36: "FEIGE 25",
    37: "PG0823+546",
    38: "HD 217086",
    39: "HZ 14",
    40: "FEIGE 66",
    41: "FEIGE 67",
    42: "LTT 377",
    43: "LTT 2415",
    44: "LTT 4364",
    45: "FEIGE 15",
    46: "HILTNER 102",
    47: "LTT 3864",
    48: "LTT 3218",
    49: "CYG OB2",
    50: "VMa 2",
    51: "GD 71",
    52: "HZ 43",
    53: "LTT 7379",
    54: "LTT 7987",
    55: "GD 153",
    56: "CD32D9927",
}


def _ab_column_index(target: ast.AST) -> int | None:
    if not isinstance(target, ast.Subscript):
        return None
    if not isinstance(target.value, ast.Name) or target.value.id != "ab":
        return None
    sl = target.slice
    if isinstance(sl, ast.Tuple) and len(sl.elts) == 2:
        col = sl.elts[1]
        if isinstance(col, ast.Constant) and isinstance(col.value, int):
            return int(col.value)
    return None


@lru_cache(maxsize=1)
def load_ab_table() -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    """Load the coarse AB magnitude table embedded in abcalc.py."""
    path = Path(__file__).with_name("abcalc.py")
    tree = ast.parse(path.read_text())
    wave = None
    columns: Dict[int, np.ndarray] = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "waveab":
                wave = np.asarray(ast.literal_eval(node.value), dtype=float)
                continue
            col = _ab_column_index(target)
            if col is not None:
                columns[col] = np.asarray(ast.literal_eval(node.value), dtype=float)

    if wave is None:
        raise RuntimeError("Could not find waveab table in abcalc.py")
    missing = [idx for idx in STANDARD_NAMES if idx not in columns]
    if missing:
        raise RuntimeError(f"Missing AB magnitude columns in abcalc.py: {missing}")
    for idx, values in columns.items():
        if values.shape != wave.shape:
            raise RuntimeError(f"AB table length mismatch for standard {idx}")
    return wave, columns


def list_standard_stars() -> List[Tuple[int, str]]:
    return [(idx, STANDARD_NAMES[idx]) for idx in sorted(STANDARD_NAMES)]


def ab_magnitudes(star_id: int) -> Tuple[np.ndarray, np.ndarray]:
    wave, columns = load_ab_table()
    if star_id not in columns:
        raise KeyError(f"Unknown standard star id: {star_id}")
    return wave.copy(), columns[star_id].copy()


def abmag_to_flambda(wave_a: np.ndarray, abmag: np.ndarray, *, scaled_1e16: bool = True) -> np.ndarray:
    """Convert AB magnitude to f_lambda.

    Returns erg/s/cm^2/A by default scaled into KCWI cube units of
    1e-16 erg/s/cm^2/A when scaled_1e16=True.
    """
    wave_a = np.asarray(wave_a, dtype=float)
    abmag = np.asarray(abmag, dtype=float)
    c_a_per_s = 2.99792458e18
    fnu = 10.0 ** (-0.4 * (abmag + 48.60))
    flam = fnu * c_a_per_s / (wave_a ** 2)
    return flam / 1e-16 if scaled_1e16 else flam


def reference_flux(star_id: int, wave_a: np.ndarray, *, scaled_1e16: bool = True) -> np.ndarray:
    wave_ab, ab = ab_magnitudes(star_id)
    ref_coarse = abmag_to_flambda(wave_ab, ab, scaled_1e16=scaled_1e16)
    return np.interp(np.asarray(wave_a, dtype=float), wave_ab, ref_coarse, left=np.nan, right=np.nan)
