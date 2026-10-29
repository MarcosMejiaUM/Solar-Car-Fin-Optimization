"""
D3 credibility checks — does the tool behave the way Anderson's result implies?

Run on laptop (real XFOIL):  python tests/test_d3_baseline.py
Here (no XFOIL):             wiring only on MockSolver.

KEY PHYSICS: camber tracks yaw ASYMMETRY (bias), not yaw magnitude. A
symmetric W is best served by a SYMMETRIC airfoil; a +biased W rewards
+camber. (This is the corrected framing -- an earlier version wrongly varied
yaw magnitude with symmetric Gaussians, for which m=0 is correctly optimal.)

  CHECK A (physics sanity): under a SYMMETRIC W (mean yaw 0), optimal camber
      is ~0 -- a symmetric airfoil wins when there's no left/right bias.
  CHECK B (direction): optimal camber INCREASES as the yaw bias goes positive
      (mean yaw 0 -> +8 deg).
  CHECK C (Anderson consistency): under a +biased W, best camber > 0 -- the
      mechanism behind his cambered 3211.

Strict only on real XFOIL; on the mock they just exercise wiring (the mock's
crude camber model can't represent camber-helps-sailing).
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finopt.solver import MockSolver                              # noqa
from finopt.geometry import naca4_coords                          # noqa
from finopt.objective import cd_eff_A, GaussianYaw                # noqa


def get():
    from finopt.solver import _HAS_XFOIL_LIB, XfoilLibSolver
    if _HAS_XFOIL_LIB:
        return XfoilLibSolver(n_crit=9.0), True
    print("No XFOIL — MockSolver (wiring only; direction needs real aero).\n")
    return MockSolver(), False


def best_camber(polars, cambers, W):
    scored = [(m, cd_eff_A(polars[m], W, A=0.264).cd_eff_A) for m in cambers
              if polars[m].n_converged]
    return min(scored, key=lambda x: x[1])[0]


def main():
    print("=" * 62)
    print("D3 CREDIBILITY CHECKS — camber vs yaw distribution")
    print("=" * 62)
    solver, real = get()
    RE = 6.0e5
    alphas = np.arange(-16.0, 18.1, 1.0)
    cambers = [0, 1, 2, 3, 4, 5]

    polars = {}
    for m in cambers:
        code = f"{m}211"
        polars[m] = solver.run_coords(naca4_coords(code), RE, alphas, name=code)

    W_sym = GaussianYaw(0.0, 6.0, 18)     # symmetric (no bias)
    W_bias = GaussianYaw(8.0, 6.0, 18)    # strongly +biased
    ok = True

    m_sym = best_camber(polars, cambers, W_sym)
    m_bias = best_camber(polars, cambers, W_bias)

    print("-" * 62)
    print(f"CHECK A: symmetric W -> ~symmetric airfoil (best m small)")
    good = m_sym <= 1
    ok = ok and (good or not real)
    print(f"   best m (symmetric W) = {m_sym}   -> [{'PASS' if good else 'FAIL'}]"
          f"  (want <= 1)")

    print("-" * 62)
    print(f"CHECK B: optimal camber increases with +yaw bias")
    good = m_bias > m_sym
    ok = ok and (good or not real)
    print(f"   best m: symmetric={m_sym} -> +biased={m_bias}   "
          f"-> [{'PASS' if good else 'FAIL'}]")

    print("-" * 62)
    print(f"CHECK C: camber helps under +biased W (best m > 0)")
    good = m_bias > 0
    ok = ok and (good or not real)
    print(f"   best m (+biased W) = {m_bias}   -> [{'PASS' if good else 'FAIL'}]")

    print("=" * 62)
    tag = "" if real else "  (MOCK wiring — rerun on laptop for the real check)"
    print("VERDICT:", ("D3 PASS" if ok else "D3 FAIL") + tag)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
