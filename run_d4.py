#!/usr/bin/env python3
"""
run_d4.py — the morphing airfoil optimizer (D4).

Minimizes the work-weighted thrust objective over CST shape space, starting
from seed airfoil(s) and morphing them toward lower effective drag. This is the
"true optimizer" (not a grid/catalog search): the design variables are
continuous CST coefficients, so the result can be a shape that's none of the
seeds.

    python run_d4.py                         # morph NACA0012, +biased demo W
    python run_d4.py --seeds 0012,e327,ch10  # multi-start incl. current fin
    python run_d4.py --wbias 3 --maxfev 150

NOTE: the yaw distribution here is a PLACEHOLDER +biased Gaussian (mean yaw
= --wbias deg), chosen so camber is beneficial and the morphing is visible.
The real optimum needs the real SOLAR-SET W(phi); swap it in and rerun. This
run proves the MACHINERY: continuous shape optimization against the right
objective, with geometric constraints, ready for the real inputs.
"""
import sys, os, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from finopt.solver import get_solver, XfoilLibSolver, _HAS_XFOIL_LIB  # noqa
from finopt.geometry import naca4_coords, load_dat                    # noqa
from finopt.airfoil_db import fetch_airfoil                           # noqa
from finopt.objective import GaussianYaw, cd_eff_A                     # noqa
from finopt.optimize import optimize_airfoil, unpack                  # noqa
from finopt.cst import is_valid, max_thickness                        # noqa

CACHE = os.path.join(HERE, "airfoils")


def load_seed(token):
    token = token.strip()
    if len(token) == 4 and token.isdigit():
        return f"NACA{token}", naca4_coords(token)
    path = fetch_airfoil(token, cache_dir=CACHE)
    return token, load_dat(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0012",
                    help="comma list: NACA codes (e.g. 0012) or names (e327,ch10)")
    ap.add_argument("--re", type=float, default=6.0e5)
    ap.add_argument("--wbias", type=float, default=5.0,
                    help="mean yaw [deg] of the placeholder +biased W")
    ap.add_argument("--order", type=int, default=5)
    ap.add_argument("--maxfev", type=int, default=120)
    args = ap.parse_args()

    if _HAS_XFOIL_LIB:
        solver = XfoilLibSolver(n_crit=9.0, sweep_step=1.0)   # fast mode
    else:
        solver = get_solver()
        print("WARNING: no XFOIL library — using fallback (results not real).")

    yaw = GaussianYaw(args.wbias, 6.0, 18)
    print(f"Re={args.re:.0f}  placeholder W: +biased Gaussian mean~{args.wbias} "
          f"deg  CST order={args.order}  maxfev={args.maxfev}\n")

    seeds = {}
    for tok in args.seeds.split(","):
        nm, coords = load_seed(tok)
        seeds[nm] = coords

    results = {}
    for nm, coords in seeds.items():
        print(f"optimizing from seed {nm} ...")
        r = optimize_airfoil(coords, solver, args.re, yaw, order=args.order,
                             maxfev=args.maxfev)
        results[nm] = r
        pct = 100 * (r["f0"] - r["f_opt"]) / abs(r["f0"]) if r["f0"] else 0
        print(f"   cd_eff_A {r['f0']:+.5f} -> {r['f_opt']:+.5f}  "
              f"({pct:+.1f}%, {r['n_evals']} evals)")

    # --- report ---
    print("\n" + "=" * 66)
    print("D4 OPTIMIZATION RESULTS  (lower cd_eff_A = better)")
    print("=" * 66)
    print(f"{'seed':10s} {'seed Cd_effA':>13s} {'opt Cd_effA':>13s} "
          f"{'improve':>9s} {'opt tmax':>9s}")
    print("-" * 66)
    for nm, r in results.items():
        Au, Al = unpack(r["x_opt"], r["order"])
        tmax = max_thickness(Au, Al, dz_u=r["dz_u"], dz_l=r["dz_l"])
        imp = 100 * (r["f0"] - r["f_opt"]) / abs(r["f0"]) if r["f0"] else 0
        print(f"{nm:10s} {r['f0']:+13.5f} {r['f_opt']:+13.5f} "
              f"{imp:+8.1f}% {tmax*100:8.1f}%")
    best = min(results.items(), key=lambda kv: kv[1]["f_opt"])
    print("-" * 66)
    print(f"BEST optimized airfoil: morph of {best[0]}  "
          f"(Cd_effA = {best[1]['f_opt']:+.5f})")
    print("  NOTE: placeholder W — rerun with real SOLAR-SET W(phi) for the"
          " real optimum.")

    _plot(results, args)


def _plot(results, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed — skipping plot)"); return
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 7), squeeze=False)
    for j, (nm, r) in enumerate(results.items()):
        ax = axes[0][j]
        ax.plot(r["seed_coords"][:, 0], r["seed_coords"][:, 1], "--",
                color="gray", label=f"seed {nm}")
        ax.plot(r["opt_coords"][:, 0], r["opt_coords"][:, 1], "-",
                color="C0", label="optimized")
        ax.set_aspect("equal"); ax.grid(alpha=.3); ax.legend(fontsize=8)
        ax.set_title(f"{nm}: shape morph")
        # convergence (best-so-far of valid evals)
        axc = axes[1][j]
        vals = [v for _, v, tag in r["history"] if tag == "ok"]
        if vals:
            best = np.minimum.accumulate(vals)
            axc.plot(best, "-")
        axc.set(xlabel="valid eval #", ylabel="best cd_eff_A",
                title=f"{nm}: convergence")
        axc.grid(alpha=.3)
    fig.suptitle(f"D4 morphing optimizer (placeholder +biased W, Re={args.re:.0e})")
    fig.tight_layout()
    out = os.path.join(HERE, "outputs", "d4_optimization.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"\nPlot saved: {out}")


if __name__ == "__main__":
    main()
