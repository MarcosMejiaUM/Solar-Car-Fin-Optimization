"""
D4 known-answer checks — CST parametrization + optimizer.

Run on laptop (real XFOIL):  python tests/test_d4_optimizer.py
Here (no XFOIL):             CHECK 1 & 2 are solver-free and run fully;
                             CHECK 3 (real improvement) is wiring-only on mock.

  CHECK 1 (CST round-trip): fit a NACA to CST, regenerate, and the shape
     matches to a small tolerance (the parametrization is faithful).
  CHECK 2 (optimizer core): on a SYNTHETIC objective over the CST design
     vector (drive max-thickness to a target), the optimizer reduces it to
     near zero. Tests the optimize plumbing with no solver.
  CHECK 3 (real improvement): optimizing from a seed under a +biased W yields
     a VALID airfoil with LOWER cd_eff_A than the seed. (Real XFOIL only.)
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finopt.geometry import naca4_coords                              # noqa
from finopt.cst import (coords_to_cst, cst_from_params, _split_surfaces,  # noqa
                        max_thickness, is_valid)
from finopt.optimize import (pack, unpack, run_min, optimize_airfoil,  # noqa
                             design_to_coords)
from finopt.objective import GaussianYaw                              # noqa


def check1():
    print("-" * 60)
    print("CHECK 1: CST round-trip fidelity (fit -> regenerate)")
    ref = naca4_coords("2412", n=160)
    cst = coords_to_cst(ref, order=5)
    gen = cst_from_params(cst, n_points=160)
    xq = np.linspace(0.05, 0.95, 40)
    uref, _ = _split_surfaces(ref); ugen, _ = _split_surfaces(gen)
    rms = np.sqrt(np.mean((np.interp(xq, uref[:, 0], uref[:, 1])
                           - np.interp(xq, ugen[:, 0], ugen[:, 1])) ** 2))
    good = rms < 1e-3
    print(f"   upper-surface RMS deviation = {rms*1e3:.3f}e-3 chord  "
          f"-> [{'PASS' if good else 'FAIL'}] (want < 1e-3)")
    return good


def check2():
    print("-" * 60)
    print("CHECK 2: optimizer core on a synthetic objective (no solver)")
    # drive max thickness of a 0012 CST toward a target 0.18
    cst = coords_to_cst(naca4_coords("0012"), order=5)
    dz_u, dz_l, N1, N2 = cst["dz_u"], cst["dz_l"], cst["N1"], cst["N2"]
    order = 5
    target = 0.18

    def f(x):
        Au, Al = unpack(x, order)
        tmax = max_thickness(Au, Al, N1=N1, N2=N2, dz_u=dz_u, dz_l=dz_l)
        return (tmax - target) ** 2

    x0 = pack(cst["Au"], cst["Al"])
    f0 = f(x0)
    res = run_min(f, x0, method="Nelder-Mead", maxfev=2000)
    good = res.fun < 1e-4 and res.fun < f0
    tmax_final = max_thickness(*unpack(res.x, order), N1=N1, N2=N2,
                               dz_u=dz_u, dz_l=dz_l)
    print(f"   f: {f0:.5f} -> {res.fun:.2e}   max-thick -> {tmax_final:.3f} "
          f"(target {target})  -> [{'PASS' if good else 'FAIL'}]")
    return good


def check3(real):
    print("-" * 60)
    print("CHECK 3: optimizing a seed LOWERS cd_eff_A (valid airfoil)")
    if real:
        from finopt.solver import XfoilLibSolver
        solver = XfoilLibSolver(n_crit=9.0, sweep_step=1.0)
    else:
        from finopt.solver import MockSolver
        solver = MockSolver()
    yaw = GaussianYaw(6.0, 6.0, 18)      # +biased: camber should help
    r = optimize_airfoil(naca4_coords("0012"), solver, 6.0e5, yaw,
                         order=5, maxfev=(120 if real else 40))
    Au, Al = unpack(r["x_opt"], r["order"])
    ok_valid, reason = is_valid(Au, Al, dz_u=r["dz_u"], dz_l=r["dz_l"])
    improved = r["f_opt"] < r["f0"] - 1e-6
    good = improved and ok_valid
    print(f"   cd_eff_A: {r['f0']:+.5f} -> {r['f_opt']:+.5f}  "
          f"({r['n_evals']} evals)  valid={ok_valid}")
    print(f"   -> [{'PASS' if good else 'FAIL' if real else 'wiring-ok'}]")
    return good or not real


def main():
    from finopt.solver import _HAS_XFOIL_LIB
    real = _HAS_XFOIL_LIB
    print("=" * 60)
    print("D4 KNOWN-ANSWER CHECKS — CST parametrization + optimizer")
    print("=" * 60)
    if not real:
        print("No XFOIL — CHECK 3 is wiring-only (mock).\n")
    c1 = check1()
    c2 = check2()
    c3 = check3(real)
    ok = c1 and c2 and c3
    print("=" * 60)
    tag = "" if real else "  (CHECK 3 mock-wiring; rerun on laptop for real)"
    print("VERDICT:", ("D4 PASS" if ok else "D4 FAIL") + tag)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
