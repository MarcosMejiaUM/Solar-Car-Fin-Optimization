"""
finopt.optimize — minimize the thrust objective over CST airfoil space (D4).

The "morphing optimizer": design variables are CST coefficients (continuous,
family-agnostic), the objective is cd_eff_A (D2), with geometric validity
enforced by a graded penalty (no crossing surfaces, minimum thickness for
structure). Multi-start from seed airfoils (NACA/CH10/E327) explores the space;
each start is a local search that morphs the seed toward lower effective drag.

Gradient-free (Nelder-Mead / Powell) for now — the analytical-gradient (adjoint)
upgrade is D6. A coarse solver (sweep_step=1.0, tight alpha range) keeps each
evaluation fast during the search.
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

from finopt.cst import (coords_to_cst, cst_to_coords, thickness_distribution)
from finopt.objective import cd_eff_A


# ---- design-vector packing -------------------------------------------------
def pack(Au, Al):
    return np.concatenate([np.asarray(Au, float), np.asarray(Al, float)])


def unpack(x, order):
    k = order + 1
    return np.asarray(x[:k], float), np.asarray(x[k:], float)


def design_to_coords(x, order, dz_u, dz_l, N1=0.5, N2=1.0, n_points=140):
    Au, Al = unpack(x, order)
    return cst_to_coords(Au, Al, n_points=n_points, N1=N1, N2=N2,
                         dz_u=dz_u, dz_l=dz_l)


# ---- graded geometric penalty ---------------------------------------------
def geom_penalty(Au, Al, N1, N2, dz_u, dz_l, min_thick=0.06):
    """0 if valid; else a POSITIVE value that grows with the violation, so the
    optimizer has a slope back toward valid airfoils (not a flat wall)."""
    _, t = thickness_distribution(Au, Al, N1=N1, N2=N2, dz_u=dz_u, dz_l=dz_l)
    pen = 0.0
    cross = np.clip(-t, 0, None)            # negative thickness = crossing
    if np.any(cross > 0):
        pen += 1.0 + 20.0 * float(np.sum(cross))
    tmax = float(np.max(t))
    if tmax < min_thick:
        pen += 1.0 + 10.0 * (min_thick - tmax)
    return pen


# ---- objective over the design vector -------------------------------------
def make_objective(solver, reynolds, yaw, order, dz_u, dz_l, N1, N2,
                   alphas, A_ref, c0, min_thick, history=None):
    def f(x):
        Au, Al = unpack(x, order)
        pen = geom_penalty(Au, Al, N1, N2, dz_u, dz_l, min_thick)
        if pen > 0:
            if history is not None:
                history.append((np.asarray(x).copy(), pen, "invalid"))
            return pen                        # dominates any real cd_eff_A (~1e-2)
        coords = cst_to_coords(Au, Al, n_points=140, N1=N1, N2=N2,
                               dz_u=dz_u, dz_l=dz_l)
        polar = solver.run_coords(coords, reynolds, alphas, name="opt")
        if polar.n_converged < 4:
            if history is not None:
                history.append((np.asarray(x).copy(), 0.5, "poor convergence"))
            return 0.5
        val = cd_eff_A(polar, yaw, A=A_ref, c0=c0).cd_eff_A
        if not np.isfinite(val):
            val = 0.5
        if history is not None:
            history.append((np.asarray(x).copy(), float(val), "ok"))
        return float(val)
    return f


def run_min(f, x0, method="Nelder-Mead", maxfev=200, bound_delta=0.25):
    """Thin wrapper so the same optimizer core is testable on a synthetic f."""
    x0 = np.asarray(x0, float)
    if method in ("L-BFGS-B", "Powell"):
        bounds = [(xi - bound_delta, xi + bound_delta) for xi in x0]
        return minimize(f, x0, method=method, bounds=bounds,
                        options={"maxiter": maxfev, "maxfev": maxfev})
    return minimize(f, x0, method="Nelder-Mead",
                    options={"maxfev": maxfev, "xatol": 1e-4, "fatol": 1e-7})


def optimize_airfoil(seed_coords, solver, reynolds, yaw, order=5,
                     min_thick=0.06, maxfev=200, method="Nelder-Mead",
                     alphas=None, A_ref=0.264, c0=0.0):
    """Morph a seed airfoil to minimize cd_eff_A. Returns a result dict."""
    if alphas is None:
        alphas = np.arange(-12.0, 16.1, 1.0)
    cst = coords_to_cst(seed_coords, order=order)
    Au0, Al0 = cst["Au"], cst["Al"]
    dz_u, dz_l, N1, N2 = cst["dz_u"], cst["dz_l"], cst["N1"], cst["N2"]
    x0 = pack(Au0, Al0)
    history = []
    f = make_objective(solver, reynolds, yaw, order, dz_u, dz_l, N1, N2,
                       alphas, A_ref, c0, min_thick, history=history)
    f0 = f(x0)
    res = run_min(f, x0, method=method, maxfev=maxfev)
    x_opt = np.asarray(res.x, float)
    return {"seed_cst": cst, "x0": x0, "x_opt": x_opt,
            "f0": float(f0), "f_opt": float(res.fun),
            "seed_coords": np.asarray(seed_coords, float),
            "opt_coords": design_to_coords(x_opt, order, dz_u, dz_l, N1, N2),
            "history": history, "order": order, "dz_u": dz_u, "dz_l": dz_l,
            "n_evals": len(history), "scipy": res}


def multi_start(seeds, solver, reynolds, yaw, **kw):
    """seeds: {name: coords}. Returns (results_by_name, best_name)."""
    results = {}
    for nm, coords in seeds.items():
        results[nm] = optimize_airfoil(coords, solver, reynolds, yaw, **kw)
    best = min(results.items(), key=lambda kv: kv[1]["f_opt"])[0]
    return results, best
