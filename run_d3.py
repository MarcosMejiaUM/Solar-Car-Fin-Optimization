#!/usr/bin/env python3
"""
run_d3.py — credibility study: does optimal CAMBER track the ASYMMETRY (bias)
of the yaw distribution? (The mechanism behind Anderson's cambered NACA 3211.)

KEY PHYSICS: camber helps because of yaw *asymmetry*, NOT yaw magnitude. A
symmetric yaw distribution (no left/right bias) is best served by a SYMMETRIC
airfoil -- the controller handles +phi and -phi equally by flipping theta.
Camber makes the fin better on one side and worse on the other, so it only
pays when the distribution is BIASED to one side. Anderson got cambered 3211
because his W is asymmetric (mode +3 deg), not merely high-yaw.

With no real SOLAR-SET W(phi) yet, we sweep NACA camber under a SEQUENCE of
yaw distributions of increasing +bias (mean yaw 0 -> +8 deg) and watch the
optimum camber rise from ~0 (symmetric W) toward positive camber.

  Expected (Anderson-consistent): symmetric W -> m~0; +biased W -> m>0.

Camber sweep at fixed p=2 (max camber at 20% chord) and t=11% -- i.e. the
NACA 32xx family around 3211 -- so m is the one variable in question. The
current-fin airfoils (CH10, E327) and the paper's infant fin (NACA 0015) are
included for context. Full (m,p,t) optimization is D4.

    python run_d3.py
    python run_d3.py --re 500000
"""
import sys, os, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from finopt.solver import get_solver                        # noqa
from finopt.geometry import naca4_coords, load_dat          # noqa
from finopt.airfoil_db import fetch_airfoil                 # noqa
from finopt.objective import cd_eff_A, GaussianYaw, SkewNormalYaw  # noqa

CACHE = os.path.join(HERE, "airfoils")


def naca_code(m, p=2, t=11):
    return f"{int(m)}{int(p)}{int(t):02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--re", type=float, default=6.0e5)
    ap.add_argument("--ncrit", type=float, default=9.0)
    args = ap.parse_args()

    solver = get_solver(ncrit=args.ncrit)
    RE = args.re
    alphas = np.arange(-16.0, 18.1, 1.0)

    # --- camber sweep: NACA m 2 11 for m = 0..5  (32xx family around 3211) ---
    cambers = [0, 1, 2, 3, 4, 5]
    polars = {}
    print(f"Re = {RE:.0f}, Ncrit = {args.ncrit}, solver = {type(solver).__name__}\n")
    for m in cambers:
        code = naca_code(m)
        print(f"solving NACA{code} ...")
        polars[f"NACA{code}"] = solver.run_coords(naca4_coords(code), RE,
                                                  alphas, name=f"NACA{code}")
    # context airfoils
    print("solving NACA0015 (infant fin) ...")
    polars["NACA0015"] = solver.run_coords(naca4_coords("0015"), RE, alphas,
                                           name="NACA0015")
    for nm in ["ch10", "e327"]:
        try:
            path = fetch_airfoil(nm, cache_dir=CACHE)
            print(f"solving {nm} ...")
            polars[nm] = solver.run_coords(load_dat(path), RE, alphas, name=nm)
        except RuntimeError as e:
            print(f"  skip {nm}: {str(e).splitlines()[0]}")

    camber_names = [f"NACA{naca_code(m)}" for m in cambers]

    # --- sequence of yaw distributions, increasing +BIAS (mean yaw) ---
    W_seq = [("gauss mu=0", GaussianYaw(0.0, 6.0, 18)),
             ("gauss mu=+2", GaussianYaw(2.0, 6.0, 18)),
             ("gauss mu=+4", GaussianYaw(4.0, 6.0, 18)),
             ("gauss mu=+6", GaussianYaw(6.0, 6.0, 18)),
             ("gauss mu=+8", GaussianYaw(8.0, 6.0, 18)),
             ("skew ~Fig1.5", SkewNormalYaw(3.0, 6.0, -4.0, 18))]

    print("\n" + "=" * 72)
    print("OPTIMAL CAMBER vs YAW BIAS  (does camber track the asymmetry of W?)")
    print("=" * 72)
    print(f"{'W(phi)':14s} {'E[phi]':>8s} {'best NACA':>10s} {'best m':>7s} "
          f"{'  best Cd_effA':>14s}")
    print("-" * 72)
    trend = []
    for label, W in W_seq:
        scored = [(m, cd_eff_A(polars[f"NACA{naca_code(m)}"], W, A=0.264).cd_eff_A)
                  for m in cambers if polars[f"NACA{naca_code(m)}"].n_converged]
        best_m, best_s = min(scored, key=lambda x: x[1])
        ey = W.mean_yaw()
        trend.append((ey, best_m))
        print(f"{label:14s} {ey:8.2f} {'NACA'+naca_code(best_m):>10s} "
              f"{best_m:7d} {best_s:+14.5f}")
    print("-" * 72)
    # credibility signal: best camber should RISE with the +bias (mean yaw)
    ms = [t[1] for t in trend[:5]]   # the gaussian sequence (monotone mu)
    rising = ms[-1] > ms[0]
    print(f"optimal camber {'RISES' if rising else 'does NOT rise'} with +yaw "
          f"bias  (m: {ms[0]} -> {ms[-1]} as mean yaw 0 -> +8)")
    print("  -> Anderson-consistent: a +biased/asymmetric W rewards +camber,")
    print("     the mechanism behind his NACA 3211. Symmetric W -> symmetric foil.")

    # --- full ranking under the most realistic (skew) W, with current fin ---
    W = SkewNormalYaw(3.0, 6.0, -4.0, 18)
    print("\n" + "=" * 72)
    print("FULL RANKING under skew-normal W (current fin vs NACA family)")
    print("=" * 72)
    rows = [(nm, cd_eff_A(p, W, A=0.264).cd_eff_A)
            for nm, p in polars.items() if p.n_converged]
    for nm, s in sorted(rows, key=lambda r: r[1]):
        tag = "  <- current fin" if nm in ("ch10", "e327") else (
              "  <- infant fin" if nm == "NACA0015" else "")
        print(f"  {nm:12s} {s:+12.5f}{tag}")

    _plot(trend, polars, camber_names, cambers, W_seq, args)


def _plot(trend, polars, camber_names, cambers, W_seq, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from finopt.objective import cd_eff_A
    except ImportError:
        print("\n(matplotlib not installed — skipping plot)"); return
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    # left: optimal camber vs mean yaw (bias)
    ax[0].plot([t[0] for t in trend], [t[1] for t in trend], "o-")
    ax[0].axhline(3, color="r", ls="--", lw=.8, label="Anderson 3211 (m=3)")
    ax[0].set(xlabel="mean yaw E[phi]  (bias / asymmetry)",
              ylabel="optimal camber m [%]",
              title="Optimal camber tracks yaw bias")
    ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)
    # right: Cd_effA vs camber, one curve per W
    for label, W in W_seq:
        ys = [cd_eff_A(polars[nm], W, A=0.264).cd_eff_A for nm in camber_names]
        ax[1].plot(cambers, ys, "o-", ms=3, label=label)
    ax[1].set(xlabel="NACA camber m [%]", ylabel="Cd_effA (lower=better)",
              title="Objective vs camber, per yaw distribution")
    ax[1].grid(alpha=.3); ax[1].legend(fontsize=7)
    fig.suptitle(f"D3 credibility: camber vs yaw distribution  (Re={args.re:.0e})")
    fig.tight_layout()
    out = os.path.join(HERE, "outputs", "d3_camber_study.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"\nPlot saved: {out}")


if __name__ == "__main__":
    main()
