"""
finopt.geometry — airfoil coordinate generation and loading.

D0 needs only `naca4_coords` (to feed the coordinate path and as a D1 seed).
D1 will add `.dat` loaders (Eppler/CH/UIUC) that return the SAME (N,2)
array format, so the solver never knows which family it came from — that
uniform representation is the whole point of the family-agnostic front end.

Convention: coordinates returned as an (N,2) array of (x, y), ordered
XFOIL-style: trailing edge -> upper surface -> leading edge -> lower
surface -> trailing edge, with chord from x=0 to x=1.
"""

from __future__ import annotations

import numpy as np


def naca4_coords(code: str, n: int = 160, closed_te: bool = False) -> np.ndarray:
    """Generate NACA 4-digit airfoil coordinates.

    Parameters
    ----------
    code : 4-char string, e.g. "0012", "2412", "3211".
           digits = (m*100, p*10, tt) : m=max camber %, p=camber position
           /10, tt=thickness %.
    n    : number of points per surface (cosine-spaced).
    closed_te : if True use the closed-trailing-edge thickness constant
                (-0.1036) instead of the classic open-TE (-0.1015). XFOIL
                re-panels anyway, but Eppler/CH .dat files are often sharp,
                so matching TE closure matters for apples-to-apples.

    Returns
    -------
    (2n-1, 2) array of (x, y), TE -> upper -> LE -> lower -> TE.
    """
    if len(code) != 4 or not code.isdigit():
        raise ValueError(f"NACA 4-digit code must be 4 digits, got {code!r}")
    m = int(code[0]) / 100.0
    p = int(code[1]) / 10.0
    t = int(code[2:]) / 100.0

    # cosine spacing clusters points at LE and TE where curvature is high
    beta = np.linspace(0.0, np.pi, n)
    x = (1 - np.cos(beta)) / 2.0

    a4 = -0.1036 if closed_te else -0.1015
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2
                  + 0.2843 * x**3 + a4 * x**4)

    if m > 0 and p > 0:
        yc = np.where(x < p,
                      m / p**2 * (2 * p * x - x**2),
                      m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x - x**2))
        dyc = np.where(x < p,
                       2 * m / p**2 * (p - x),
                       2 * m / (1 - p)**2 * (p - x))
        theta = np.arctan(dyc)
    else:
        yc = np.zeros_like(x)
        theta = np.zeros_like(x)

    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    # assemble TE -> upper -> LE -> lower -> TE (drop duplicated LE point)
    x_coords = np.concatenate([xu[::-1], xl[1:]])
    y_coords = np.concatenate([yu[::-1], yl[1:]])
    return np.column_stack([x_coords, y_coords])


def load_dat(path: str) -> np.ndarray:
    """Load a .dat airfoil file -> (N,2) coords in XFOIL order.

    Handles both coordinate conventions robustly and returns a single loop
    ordered TE -> upper -> LE -> lower -> TE (Selig order), which is what
    XFOIL's LOAD expects:

      * Selig  : one block, x runs 1 -> 0 -> 1 (TE over the top to LE, then
                 back under to TE). Optional name header line.
      * Lednicer: a "count" line like "61. 61." after the name, then TWO
                 blocks each running x: 0 -> 1 (upper LE->TE, then lower
                 LE->TE). We detect the count line and reassemble to Selig
                 order.

    Raises ValueError if it can't find a sane coordinate set.
    """
    with open(path) as f:
        raw = [ln.strip() for ln in f.readlines()]

    # collect (x,y) numeric pairs and remember where a Lednicer count line sits
    pts = []
    count_line_idx = None
    for i, ln in enumerate(raw):
        p = ln.split()
        if len(p) != 2:
            continue
        try:
            a, b = float(p[0]), float(p[1])
        except ValueError:
            continue
        # Lednicer count line: both values > 1 and (near-)integral, appears early
        if (count_line_idx is None and len(pts) == 0
                and a > 1.5 and b > 1.5
                and abs(a - round(a)) < 1e-6 and abs(b - round(b)) < 1e-6):
            count_line_idx = (int(round(a)), int(round(b)))
            continue
        pts.append((a, b))

    coords = np.array(pts, float)
    if coords.ndim != 2 or coords.shape[0] < 10:
        raise ValueError(f"Could not parse enough coordinates from {path!r}")

    if count_line_idx is not None:
        # Lednicer: first n_upper points are upper LE->TE, next n_lower are
        # lower LE->TE. Reassemble: upper reversed (TE->LE) + lower (LE->TE).
        n_up, n_lo = count_line_idx
        if n_up + n_lo <= coords.shape[0]:
            upper = coords[:n_up]
            lower = coords[n_up:n_up + n_lo]
            coords = np.vstack([upper[::-1], lower[1:]])  # drop dup LE
    # else: already Selig order, leave as-is

    return _normalize_chord(coords)


def _normalize_chord(coords: np.ndarray) -> np.ndarray:
    """Shift/scale so the chord runs x in [0,1] (leading edge at x=0)."""
    x = coords[:, 0]
    xmin, xmax = x.min(), x.max()
    c = xmax - xmin
    if c <= 0:
        return coords
    out = coords.copy()
    out[:, 0] = (coords[:, 0] - xmin) / c
    out[:, 1] = coords[:, 1] / c
    return out


def write_dat(coords: np.ndarray, path: str, name: str = "airfoil"):
    """Write coords to a Selig .dat file (for inspection / feeding XFOIL)."""
    with open(path, "w") as f:
        f.write(f"{name}\n")
        for x, y in coords:
            f.write(f"{x:.6f} {y:.6f}\n")
