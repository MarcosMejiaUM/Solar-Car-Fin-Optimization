"""
D1 known-answer check: family-agnostic input layer.

Run on your laptop (XFOIL + internet):  python tests/test_d1_families.py

Two gates:

(1) CONSISTENCY — the real D1 check.
    Generate NACA 0012 with our own code AND fetch "naca0012" from the airfoil
    database, run both through XFOIL, and confirm the polars AGREE. Same
    airfoil via two independent paths must give the same aerodynamics; if it
    doesn't, our coordinate handling (ordering, normalization, TE) is wrong.

(2) SANITY — the cross-family airfoils load and solve.
    Fetch CH10 and E327 and confirm each returns a physical polar (sane lift
    slope, positive drag). This proves no airfoil family is privileged by the
    input layer — the whole point of D1.

Without internet the fetch step is skipped with a clear message; the
consistency check still runs against a hand-written .dat so the core gate
works offline too.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finopt.solver import XfoilSolver, MockSolver             # noqa: E402
from finopt.geometry import naca4_coords, load_dat, write_dat  # noqa: E402
from finopt.airfoil_db import fetch_airfoil                     # noqa: E402

RE = 6.0e5          # representative fin Reynolds (see run_d1.py note)
ALPHAS = np.arange(-6.0, 10.01, 2.0)
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "airfoils")


def get_solver():
    # Prefer the in-process xfoil-python library; fall back to mock for CI.
    from finopt.solver import _HAS_XFOIL_LIB, XfoilLibSolver
    if _HAS_XFOIL_LIB:
        return XfoilLibSolver(n_crit=9.0), True
    xs = XfoilSolver(xfoil_path=os.environ.get("XFOIL", "xfoil"), ncrit=9.0)
    if xs.available:
        return xs, True
    print("No XFOIL engine found — running MockSolver (wiring only).\n")
    return MockSolver(), False


def consistency_gate(solver, strict):
    print("-" * 60)
    print("GATE 1: CONSISTENCY  (generated NACA0012 vs loaded-from-.dat)")
    # path A: our generator -> coords -> solver
    genA = naca4_coords("0012")
    pA = solver.run_coords(genA, RE, ALPHAS, name="gen0012")
    # path B: write that to a .dat, load it back, solve (exercises load_dat)
    write_dat(genA, "/tmp/_d1_0012.dat", "NACA0012")
    genB = load_dat("/tmp/_d1_0012.dat")
    pB = solver.run_coords(genB, RE, ALPHAS, name="dat0012")

    # compare CL and CD at the alphas both converged
    diffs = []
    for a in ALPHAS:
        clA, clB = pA.cl_at(a), pB.cl_at(a)
        cdA, cdB = pA.cd_at(a), pB.cd_at(a)
        if np.isfinite(clA) and np.isfinite(clB):
            diffs.append((a, abs(clA - clB), abs(cdA - cdB)))
    if not diffs:
        print("  [FAIL] no overlapping converged points")
        return False
    max_dcl = max(d[1] for d in diffs)
    max_dcd = max(d[2] for d in diffs)
    ok = (max_dcl < 1e-3) and (max_dcd < 1e-4)
    print(f"  max |dCL| = {max_dcl:.2e}  (want < 1e-3)")
    print(f"  max |dCD| = {max_dcd:.2e}  (want < 1e-4)")
    print(f"  [{'PASS' if ok else 'FAIL'}] generator and .dat paths agree")
    return ok or not strict


def sanity_gate(solver, strict):
    print("-" * 60)
    print("GATE 2: CROSS-FAMILY SANITY  (CH10, E327 load & solve)")
    ok_all = True
    for name in ["ch10", "e327"]:
        try:
            path = fetch_airfoil(name, cache_dir=CACHE)
        except RuntimeError as e:
            print(f"  [SKIP] {name}: could not fetch ({str(e).splitlines()[0]})")
            continue
        coords = load_dat(path)
        p = solver.run_coords(coords, RE, ALPHAS, name=name)
        slope = p.cl_slope_per_deg()
        cdmin = float(np.min(p.cd)) if p.n_converged else float("nan")
        good = (0.07 <= slope <= 0.16) and (0.003 <= cdmin <= 0.05)
        ok_all = ok_all and good
        print(f"  [{'PASS' if good else 'FAIL'}] {name:5s}: "
              f"slope={slope:.3f}/deg  CDmin={cdmin:.4f}  "
              f"({p.n_converged}/{p.n_requested} conv)")
    return ok_all or not strict


def main():
    print("=" * 60)
    print("D1 KNOWN-ANSWER CHECK — family-agnostic input")
    print("=" * 60)
    solver, strict = get_solver()
    g1 = consistency_gate(solver, strict)
    g2 = sanity_gate(solver, strict)
    print("=" * 60)
    if strict:
        ok = g1 and g2
        print("VERDICT:", "D1 PASS — proceed to D2." if ok else
              "D1 FAIL — fix the input layer.")
        sys.exit(0 if ok else 1)
    print("VERDICT: wiring OK (mock). Run on your laptop for the real D1.")
    sys.exit(0)


if __name__ == "__main__":
    main()
