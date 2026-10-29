#!/usr/bin/env python3
"""
run_d1.py — first cross-family polar comparison (D1 milestone).

Fetches CH10, E327, and a NACA baseline, runs them all through the SAME
verified XFOIL wrapper at the SAME Reynolds number and Ncrit, and overlays:
  * lift curve (CL vs alpha)
  * drag polar (CL vs CD)
  * lift-to-drag (CL/CD vs alpha)

    python run_d1.py                 # default airfoils & Re
    python run_d1.py --re 500000     # set Reynolds
    python run_d1.py --ncrit 5       # rougher-flow transition (see note)

NOTE ON REYNOLDS: the fin chord is ~0.366 m. At ~100 km/h apparent wind and
sea-level air (nu ~ 1.5e-5 m^2/s), Re ~ V*c/nu ~ 6.8e5. Apparent wind at yaw
can push this higher; low apparent winds lower it. 6e5 is a reasonable
representative default, but THIS IS A PLACEHOLDER — the honest comparison
sweeps a range, which D2/D3 will do with the work-weighted distribution.
At these low Re, airfoil ranking can shift with Ncrit, so it's exposed here.
This is a LOOK, not the verdict — the verdict needs the race-time objective.
"""
import sys
import os
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from finopt.solver import get_solver                   # noqa: E402
from finopt.geometry import naca4_coords, load_dat     # noqa: E402
from finopt.airfoil_db import fetch_airfoil            # noqa: E402

CACHE = os.path.join(HERE, "airfoils")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--re", type=float, default=6.0e5)
    ap.add_argument("--ncrit", type=float, default=9.0)
    ap.add_argument("--naca", default="0012", help="NACA 4-digit baseline")
    ap.add_argument("--airfoils", nargs="*", default=["ch10", "e327"])
    args = ap.parse_args()

    alphas = np.arange(-6.0, 12.01, 1.0)
    solver = get_solver(xfoil_path=os.environ.get("XFOIL", "xfoil"),
                        ncrit=args.ncrit)

    print(f"Re = {args.re:.0f}   Ncrit = {args.ncrit}   "
          f"solver = {type(solver).__name__}\n")

    polars = {}

    # NACA baseline from our generator
    nm = f"NACA{args.naca}"
    print(f"solving {nm} ...")
    polars[nm] = solver.run_coords(naca4_coords(args.naca), args.re, alphas,
                                   name=nm)

    # fetched families
    for name in args.airfoils:
        try:
            path = fetch_airfoil(name, cache_dir=CACHE)
        except RuntimeError as e:
            print(f"  skip {name}: {str(e).splitlines()[0]}")
            continue
        print(f"solving {name} ...")
        polars[name] = solver.run_coords(load_dat(path), args.re, alphas,
                                         name=name)

    # --- report table ---
    print("\n" + "=" * 58)
    print(f"{'airfoil':12s} {'slope/deg':>10s} {'CLmax*':>8s} "
          f"{'CDmin':>8s} {'(L/D)max':>9s}")
    print("-" * 58)
    for nm, p in polars.items():
        if p.n_converged == 0:
            print(f"{nm:12s}  (no converged points)")
            continue
        ld = p.cl / np.where(p.cd > 0, p.cd, np.nan)
        print(f"{nm:12s} {p.cl_slope_per_deg():10.4f} "
              f"{np.nanmax(p.cl):8.3f} {np.nanmin(p.cd):8.4f} "
              f"{np.nanmax(ld):9.1f}")
    print("=" * 58)
    print("*CLmax here = max over the converged range, NOT true stall CLmax.")

    _plot(polars, args)


def _plot(polars, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed — skipping plot)")
        return
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for nm, p in polars.items():
        if p.n_converged == 0:
            continue
        ld = p.cl / np.where(p.cd > 0, p.cd, np.nan)
        ax[0].plot(p.alpha, p.cl, "o-", ms=3, label=nm)
        ax[1].plot(p.cd, p.cl, "o-", ms=3, label=nm)
        ax[2].plot(p.alpha, ld, "o-", ms=3, label=nm)
    ax[0].set(xlabel="alpha [deg]", ylabel="CL", title="Lift curve")
    ax[1].set(xlabel="CD", ylabel="CL", title="Drag polar")
    ax[2].set(xlabel="alpha [deg]", ylabel="CL/CD", title="Lift-to-drag")
    for a in ax:
        a.grid(alpha=.3); a.legend(fontsize=8)
    fig.suptitle(f"D1 cross-family comparison  —  Re={args.re:.0e}, "
                 f"Ncrit={args.ncrit}  (a LOOK, not the verdict)")
    fig.tight_layout()
    out = os.path.join(HERE, "outputs", "d1_cross_family.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"\nPlot saved: {out}")


if __name__ == "__main__":
    main()
