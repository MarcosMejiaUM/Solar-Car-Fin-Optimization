"""
finopt.cst — Class-Shape Transformation (CST) airfoil parametrization (D4).

CST (Kulfan) represents an airfoil as a class function times a sum of Bernstein
polynomials. With ~8-16 coefficients it spans NACA, Eppler, CH, supercritical
sections IN ONE CONTINUOUS, SMOOTH space — so the optimizer can morph between
and beyond the seed families instead of being locked to NACA-4's 3 knobs.

  surface:   y(x) = C(x) * S(x) + x * dz
  class:     C(x) = x^N1 * (1-x)^N2         (N1=0.5 round nose, N2=1 sharp tail)
  shape:     S(x) = sum_i A_i * binom(n,i) * x^i * (1-x)^(n-i)   (Bernstein)
  dz        = trailing-edge half-thickness (y at x=1); 0 for a sharp TE.

An airfoil = upper coeffs Au[0..n] + lower coeffs Al[0..n]  ->  2(n+1) design
variables. order n ~ 4-6 is a good balance (too few: can't fit a sharp nose;
too many: wiggly, overfit-prone space with spurious optima).

Provides:
  cst_to_coords(Au, Al, ...)  — generate (N,2) Selig coords from coefficients
  coords_to_cst(coords, order) — fit coefficients to an existing airfoil (LSQ)
  round-trip fit is the D4 known-answer check.
"""

from __future__ import annotations
import numpy as np
from math import comb


# --------------------------------------------------------------------------
def _bernstein_basis(x, n):
    """Matrix B[k, i] = binom(n,i) x_k^i (1-x_k)^(n-i), shape (len(x), n+1)."""
    x = np.asarray(x, float)
    B = np.empty((len(x), n + 1))
    for i in range(n + 1):
        B[:, i] = comb(n, i) * x ** i * (1 - x) ** (n - i)
    return B


def cst_surface(x, A, N1=0.5, N2=1.0, dz=0.0):
    """Evaluate one CST surface y(x) for coefficients A (length n+1)."""
    x = np.asarray(x, float)
    A = np.asarray(A, float)
    n = len(A) - 1
    C = x ** N1 * (1 - x) ** N2
    S = _bernstein_basis(x, n) @ A
    return C * S + x * dz


def cst_to_coords(Au, Al, n_points=140, N1=0.5, N2=1.0, dz_u=0.0, dz_l=0.0):
    """Generate (N,2) airfoil coordinates in Selig order (TE->upper->LE->lower->TE).

    Cosine x-spacing clusters points at LE/TE. Upper uses Au, lower uses Al."""
    beta = np.linspace(0.0, np.pi, n_points)
    x = (1 - np.cos(beta)) / 2.0                      # 0..1, cosine spaced
    yu = cst_surface(x, Au, N1, N2, dz_u)
    yl = cst_surface(x, Al, N1, N2, dz_l)
    # Selig: TE -> upper -> LE -> lower -> TE (drop duplicate LE)
    xc = np.concatenate([x[::-1], x[1:]])
    yc = np.concatenate([yu[::-1], yl[1:]])
    return np.column_stack([xc, yc])


def _split_surfaces(coords):
    """Split Selig-ordered coords into upper and lower (each LE->TE, x ascending)."""
    coords = np.asarray(coords, float)
    x = coords[:, 0]
    le = int(np.argmin(x))
    upper = coords[:le + 1][::-1]      # was TE->LE, now LE->TE
    lower = coords[le:]                # LE->TE
    return upper, lower


def _fit_surface(xy, order, N1=0.5, N2=1.0):
    """Least-squares fit CST coefficients to one surface's (x,y) points.
    Returns (A, dz)."""
    x = xy[:, 0].astype(float)
    y = xy[:, 1].astype(float)
    # normalize x to [0,1] defensively
    x = np.clip(x, 0.0, 1.0)
    dz = float(y[np.argmax(x)])        # TE y-value (at x≈1)
    C = x ** N1 * (1 - x) ** N2
    B = _bernstein_basis(x, order)
    M = B * C[:, None]                 # y - x*dz = M @ A
    rhs = y - x * dz
    # drop degenerate rows where C≈0 (LE/TE) to avoid zero-weighting noise
    keep = C > 1e-6
    A, *_ = np.linalg.lstsq(M[keep], rhs[keep], rcond=None)
    return A, dz


def coords_to_cst(coords, order=5, N1=0.5, N2=1.0):
    """Fit CST coefficients (Au, Al, dz_u, dz_l) to an existing airfoil.

    Round-trip fidelity is the D4 known-answer check: coords_to_cst then
    cst_to_coords should reproduce the airfoil to a small tolerance."""
    upper, lower = _split_surfaces(coords)
    Au, dz_u = _fit_surface(upper, order, N1, N2)
    Al, dz_l = _fit_surface(lower, order, N1, N2)
    return {"Au": Au, "Al": Al, "dz_u": dz_u, "dz_l": dz_l,
            "order": order, "N1": N1, "N2": N2}


def cst_from_params(cst, n_points=140):
    """Convenience: coords from a cst dict (as returned by coords_to_cst)."""
    return cst_to_coords(cst["Au"], cst["Al"], n_points=n_points,
                         N1=cst["N1"], N2=cst["N2"],
                         dz_u=cst["dz_u"], dz_l=cst["dz_l"])


# --------------------------------------------------------------------------
# geometry helpers used by the optimizer (thickness / validity)
# --------------------------------------------------------------------------
def thickness_distribution(Au, Al, N1=0.5, N2=1.0, dz_u=0.0, dz_l=0.0,
                           n=100):
    """t(x) = y_upper(x) - y_lower(x) on a uniform interior grid."""
    x = np.linspace(0.02, 0.98, n)
    return x, cst_surface(x, Au, N1, N2, dz_u) - cst_surface(x, Al, N1, N2, dz_l)


def max_thickness(Au, Al, **kw):
    _, t = thickness_distribution(Au, Al, **kw)
    return float(np.max(t))


def is_valid(Au, Al, min_thick=0.06, **kw):
    """Valid if upper is above lower everywhere and max thickness >= min_thick
    (structural). Returns (ok, reason)."""
    _, t = thickness_distribution(Au, Al, **kw)
    if np.any(t <= 0):
        return False, "surfaces cross (negative thickness)"
    if np.max(t) < min_thick:
        return False, f"too thin (max t={np.max(t):.3f} < {min_thick})"
    return True, "ok"
