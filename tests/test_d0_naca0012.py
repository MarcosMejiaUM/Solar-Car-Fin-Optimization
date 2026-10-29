"""
D0 known-answer check: NACA 0012 at Re = 1e6.

Run this on your laptop (XFOIL on PATH):   python tests/test_d0_naca0012.py
Here in dev (no XFOIL):                     exercises MockSolver wiring.

Reference bands for NACA 0012, Re=1e6, Ncrit=9 (typical XFOIL; exact values
shift with Ncrit and panel count, so bands are intentionally loose — the
point is "is the solver wired up and physical", not a 4-digit match):

  * lift-curve slope      ~0.105 - 0.120  /deg   (thin-airfoil ideal 0.1097)
  * CL at alpha = 0        |CL| < 0.03           (symmetric airfoil)
  * CL at alpha = 5 deg    0.50 - 0.62
  * min CD                 0.0050 - 0.0075
  * symmetry               CL(+a) ~= -CL(-a), CD(+a) ~= CD(-a)

Cross-check your own numbers against Airfoil Tools / a trusted XFOIL run;
if they land in these bands, D0 is trustworthy and we move to D1.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finopt.solver import XfoilSolver, MockSolver          # noqa: E402

RE = 1.0e6
ALPHAS = np.arange(-8.0, 12.01, 1.0)

BANDS = dict(slope=(0.105, 0.120), cl0_abs=0.03, cl5=(0.50, 0.62),
             cdmin=(0.0050, 0.0075))


def check(polar, strict):
    """Return (passed, messages). strict=True only for a real XFOIL run."""
    msgs, ok = [], True

    def band(name, val, lo, hi):
        nonlocal ok
        good = (val == val) and (lo <= val <= hi)   # NaN-safe
        ok = ok and (good or not strict)
        msgs.append(f"  [{'PASS' if good else 'FAIL'}] {name}: "
                    f"{val:.4f}  (band {lo}-{hi})")

    slope = polar.cl_slope_per_deg()
    band("lift slope /deg", slope, *BANDS["slope"])

    cl0 = polar.cl_at(0.0)
    good0 = abs(cl0) <= BANDS["cl0_abs"]
    ok = ok and (good0 or not strict)
    msgs.append(f"  [{'PASS' if good0 else 'FAIL'}] CL at 0 deg: {cl0:.4f}  "
                f"(|CL| <= {BANDS['cl0_abs']})")

    cl5 = polar.cl_at(5.0)
    band("CL at 5 deg", cl5, *BANDS["cl5"])

    if polar.n_converged:
        cdmin = float(np.min(polar.cd))
        band("min CD", cdmin, *BANDS["cdmin"])

    # symmetry
    cl_p5, cl_m5 = polar.cl_at(5.0), polar.cl_at(-5.0)
    sym = abs(cl_p5 + cl_m5)
    good_s = sym < 0.05
    ok = ok and (good_s or not strict)
    msgs.append(f"  [{'PASS' if good_s else 'FAIL'}] CL symmetry "
                f"|CL(5)+CL(-5)|: {sym:.4f}  (< 0.05)")

    msgs.append(f"  convergence: {polar.n_converged}/{polar.n_requested} "
                f"({100*polar.convergence_rate:.0f}%)")
    return ok, msgs


def main():
    print("=" * 66)
    print("D0 KNOWN-ANSWER CHECK — NACA 0012 @ Re = 1e6")
    print("=" * 66)

    xs = XfoilSolver(xfoil_path=os.environ.get("XFOIL", "xfoil"))
    if xs.available:
        print(f"XFOIL found at: {xs._resolved}\nRunning real solver...\n")
        polar = xs.run_naca("0012", RE, ALPHAS)
        strict = True
    else:
        print("XFOIL NOT found on PATH.")
        print("Running MockSolver — this validates PIPELINE WIRING only,")
        print("NOT aerodynamics. Run on your laptop for the real check.\n")
        polar = MockSolver().run_naca("0012", RE, ALPHAS)
        strict = False

    print(f"Result: {polar}\n")
    passed, msgs = check(polar, strict)
    print("\n".join(msgs))
    print("-" * 66)
    if strict:
        print("VERDICT:", "D0 PASS — solver trustworthy, proceed to D1."
              if passed else "D0 FAIL — fix solver before building on it.")
        sys.exit(0 if passed else 1)
    else:
        print("VERDICT: wiring OK (mock). Re-run on your laptop with XFOIL "
              "for the real D0 verdict.")
        sys.exit(0)


if __name__ == "__main__":
    main()
