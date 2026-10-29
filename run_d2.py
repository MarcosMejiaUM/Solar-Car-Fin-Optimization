#!/usr/bin/env python3
"""
run_d2.py — rank airfoils by the WORK-WEIGHTED THRUST objective (D2).

This is the first time "which airfoil is best for the fin" is asked correctly:
by projected thrust over the yaw distribution, not raw L/D. Lower Cd_effA = better.

    python run_d2.py                 # NACA0012 vs CH10 vs E327, skew-normal W
    python run_d2.py --re 500000

Also shows how the RANKING shifts with the (stubbed) yaw distribution, to make
the point that the answer depends on W(phi) — which is exactly why the real
SOLAR-SET distribution matters and why raw L/D is the wrong metric.
"""
import sys, os, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from finopt.solver import get_solver                          # noqa
from finopt.geometry import naca4_coords, load_dat            # noqa
from finopt.airfoil_db import fetch_airfoil                   # noqa
from finopt.objective import (cd_eff_A, SkewNormalYaw,        # noqa
                              GaussianYaw, UniformYaw)

CACHE = os.path.join(HERE, "airfoils")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--re", type=float, default=6.0e5)
    ap.add_argument("--ncrit", type=float, default=9.0)
    args = ap.parse_args()

    solver = get_solver(ncrit=args.ncrit)
    RE = args.re
    # alpha must cover theta+phi (|theta|<=20, |phi|<=18) plus margin
    alphas = np.arange(-16.0, 18.1, 1.0)

    # --- solve the three airfoils once ---
    polars = {}
    print(f"Re = {RE:.0f}, Ncrit = {args.ncrit}, solver = {type(solver).__name__}\n")
    print("solving NACA0012 ...")
    polars["NACA0012"] = solver.run_coords(naca4_coords("0012"), RE, alphas,
                                           name="NACA0012")
    for nm in ["ch10", "e327"]:
        try:
            path = fetch_airfoil(nm, cache_dir=CACHE)
        except RuntimeError as e:
            print(f"  skip {nm}: {str(e).splitlines()[0]}"); continue
        print(f"solving {nm} ...")
        polars[nm] = solver.run_coords(load_dat(path), RE, alphas, name=nm)

    # --- primary ranking under the Fig-1.5-like skew-normal W ---
    yaw = SkewNormalYaw(xi=3.0, omega=6.0, alpha=-4.0, phimax=18.0)
    print("\n" + "=" * 68)
    print("RANKING by work-weighted effective drag area  (lower = better)")
    print("  W(phi): skew-normal, mode ~+3 deg, left-skew (approx Fig 1.5)")
    print("=" * 68)
    print(f"{'airfoil':12s} {'Cd_effA':>12s} {'mean Cd':>10s} "
          f"{'feasible':>9s} {'(L/D)max':>9s}")
    print("-" * 68)
    rows = []
    for nm, p in polars.items():
        if p.n_converged == 0:
            print(f"{nm:12s}  (no converged points)"); continue
        res = cd_eff_A(p, yaw, A=0.264, c0=0.0)
        ld = np.nanmax(p.cl / np.where(p.cd > 0, p.cd, np.nan))
        rows.append((nm, res, ld))
    for nm, res, ld in sorted(rows, key=lambda r: r[1].cd_eff_A):
        print(f"{nm:12s} {res.cd_eff_A:+12.5f} {res.cd_eff:+10.4f} "
              f"{100*res.feasible_frac:7.0f}% {ld:9.1f}")
    print("-" * 68)
    if rows:
        best = min(rows, key=lambda r: r[1].cd_eff_A)[0]
        print(f"BEST by thrust objective: {best}")
        # contrast with L/D winner
        ld_best = max(rows, key=lambda r: r[2])[0]
        print(f"BEST by raw L/D (the WRONG metric): {ld_best}")
        if best != ld_best:
            print("  -> note: the thrust objective and raw L/D DISAGREE. "
                  "That's the whole point.")

    # --- sensitivity: how the ranking shifts with the yaw distribution ---
    print("\n" + "=" * 68)
    print("RANKING SENSITIVITY to the (stubbed) yaw distribution")
    print("=" * 68)
    stubs = {
        "skew-normal (~Fig1.5)": yaw,
        "gaussian mode 0":       GaussianYaw(mu=0.0, sigma=6.0, phimax=18.0),
        "gaussian mode +8":      GaussianYaw(mu=8.0, sigma=5.0, phimax=18.0),
        "uniform +/-18":         UniformYaw(phimax=18.0),
    }
    for label, w in stubs.items():
        scores = [(nm, cd_eff_A(p, w, A=0.264).cd_eff_A)
                  for nm, p in polars.items() if p.n_converged]
        order = " > ".join(nm for nm, _ in sorted(scores, key=lambda s: s[1]))
        print(f"  {label:24s}: {order}")
    print("  (ranking changing with W is WHY the real SOLAR-SET W(phi) matters)")

    _plot(polars, yaw, args)


def _plot(polars, yaw, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed — skipping plot)"); return
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for nm, p in polars.items():
        if p.n_converged == 0:
            continue
        res = cd_eff_A(p, yaw, A=0.264)
        ax[0].plot(res.phi_grid, res.cd_of_phi, "-", label=nm)
        ax[1].plot(res.phi_grid, res.theta_star, "-", label=nm)
    ax[0].axhline(0, color="k", lw=0.8, ls="--")
    ax[0].set(xlabel="yaw phi [deg]", ylabel="effective drag Cd(phi)",
              title="Effective drag vs yaw  (negative = sailing, cf. Fig 1.3)")
    ax[1].set(xlabel="yaw phi [deg]", ylabel="controller theta* [deg]",
              title="Optimal control angle vs yaw")
    # overlay the yaw weight on ax[0] (right axis)
    axw = ax[0].twinx()
    axw.fill_between(res.phi_grid, yaw.normalized_pdf(res.phi_grid),
                     alpha=0.12, color="gray")
    axw.set_ylabel("W(phi) weight", color="gray")
    for a in ax:
        a.grid(alpha=.3); a.legend(fontsize=8)
    fig.suptitle(f"D2 work-weighted thrust objective  —  Re={args.re:.0e}")
    fig.tight_layout()
    out = os.path.join(HERE, "outputs", "d2_objective.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"\nPlot saved: {out}")


if __name__ == "__main__":
    main()