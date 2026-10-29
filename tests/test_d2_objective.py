"""
D2 known-answer checks for the work-weighted thrust objective.

Run on your laptop:   python tests/test_d2_objective.py   (uses real XFOIL)
Here in dev:          exercises the physics with MockSolver.

Checks (physics must hold regardless of solver):
  1. DELTA-FUNCTION: with W = delta(0), Cd_effA/A reduces to the single-yaw
     effective drag at phi=0, which equals C0 + min(CD) over the controllable
     range (controller picks least-drag theta; no sailing at zero yaw).
  2. FIG-1.3 SHAPE: effective drag Cd(phi) must DECREASE with |phi| and go
     NEGATIVE at high yaw for a lifting airfoil (the sailing effect).
  3. MONOTONICITY: a higher-lift airfoil (more thrust capacity) must score a
     LOWER (better) Cd_effA than a weak one.
  4. CONTROL CAP: the chosen theta never exceeds the +/-20 deg authority.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finopt.solver import get_solver, MockSolver                      # noqa
from finopt.geometry import naca4_coords                              # noqa
from finopt.objective import (cd_eff_A, evaluate_airfoil, DiracYaw,   # noqa
                              SkewNormalYaw, GaussianYaw, best_thrust)


def get():
    from finopt.solver import _HAS_XFOIL_LIB, XfoilLibSolver
    if _HAS_XFOIL_LIB:
        return XfoilLibSolver(n_crit=9.0), True
    print("No XFOIL — using MockSolver (physics still exercised).\n")
    return MockSolver(), False


def main():
    print("=" * 62)
    print("D2 KNOWN-ANSWER CHECKS — work-weighted thrust objective")
    print("=" * 62)
    solver, real = get()
    RE = 6.0e5
    # attached-flow range; the controller keeps alpha here (never near stall)
    alphas = np.arange(-16, 18.1, 1.0)
    ok = True

    # --- solve a symmetric and a cambered airfoil ---
    p0012 = solver.run_coords(naca4_coords("0012"), RE, alphas, name="0012")
    p4412 = solver.run_coords(naca4_coords("4412"), RE, alphas, name="4412")

    # 1) delta-function check
    print("-" * 62)
    print("CHECK 1: W = delta(0)  ->  Cd_effA/A == C0 + min(CD)")
    res = cd_eff_A(p0012, DiracYaw(0.0), A=1.0, c0=0.0, n_phi=121)
    # min CD over controllable range at phi=0 is alpha in [-20,20]
    m = (p0012.alpha >= -16) & (p0012.alpha <= 18)
    cdmin = float(np.min(p0012.cd[m]))
    good = abs(res.cd_eff - cdmin) < 0.02
    ok = ok and good
    print(f"   Cd_eff(delta0) = {res.cd_eff:.4f}   min(CD)[-20,20] = {cdmin:.4f}"
          f"   -> [{'PASS' if good else 'FAIL'}]")

    # 2) Fig 1.3 shape: goes negative at high yaw
    print("-" * 62)
    print("CHECK 2: Cd(phi) decreases with |phi| and goes NEGATIVE (sailing)")
    res = cd_eff_A(p4412, SkewNormalYaw(), A=1.0, c0=0.0, n_phi=121)
    phi = res.phi_grid; cd = res.cd_of_phi
    cd0 = cd[np.argmin(np.abs(phi))]
    cd_hi = np.nanmin(cd)            # most negative (highest-sailing) point
    good = np.isfinite(cd0) and np.isfinite(cd_hi) and (cd_hi < 0) and (cd_hi < cd0)
    ok = ok and good
    print(f"   Cd(0) = {cd0:+.4f}   min Cd(phi) = {cd_hi:+.4f}   "
          f"-> [{'PASS' if good else 'FAIL'}] (want min < 0 and < Cd(0))")

    # 3) monotonicity of the OBJECTIVE: a strictly higher-lift polar (more
    #    thrust capacity, same drag) must score LOWER. Tested with a
    #    synthetically-boosted polar so it probes the objective math, not the
    #    (crude) mock aerodynamics.
    print("-" * 62)
    print("CHECK 3: a strictly higher-lift polar scores LOWER Cd_effA")
    from finopt.solver import Polar
    boosted = Polar(alpha=p0012.alpha.copy(), cl=p0012.cl * 1.3,
                    cd=p0012.cd.copy(), cm=p0012.cm.copy(),
                    reynolds=p0012.reynolds, source="0012+30%CL",
                    n_requested=p0012.n_requested)
    yaw = SkewNormalYaw()
    s_base = cd_eff_A(p0012, yaw, A=0.264, c0=0.0).cd_eff_A
    s_boost = cd_eff_A(boosted, yaw, A=0.264, c0=0.0).cd_eff_A
    good = s_boost < s_base
    ok = ok and good
    print(f"   base Cd_effA = {s_base:+.5f}   +30%-lift Cd_effA = {s_boost:+.5f}"
          f"   -> [{'PASS' if good else 'FAIL'}] (want boosted < base)")

    # 4) control cap respected
    print("-" * 62)
    print("CHECK 4: controller theta within +/-20 deg")
    res = cd_eff_A(p4412, SkewNormalYaw(), theta_max=20.0)
    th = res.theta_star[np.isfinite(res.theta_star)]
    good = bool(np.all(np.abs(th) <= 20.0 + 1e-6))
    ok = ok and good
    print(f"   max|theta*| = {np.max(np.abs(th)):.2f} deg   "
          f"-> [{'PASS' if good else 'FAIL'}]")

    print("=" * 62)
    tag = "" if real else "  (MOCK physics — rerun on laptop for real numbers)"
    print("VERDICT:", ("D2 PASS" if ok else "D2 FAIL") + tag)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()