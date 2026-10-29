#!/usr/bin/env python3
"""
run_d0.py — one-command D0 check for your laptop.

    python run_d0.py                 # uses 'xfoil' on PATH
    XFOIL=/path/to/xfoil python run_d0.py
    python run_d0.py --plot          # also save a polar plot (needs matplotlib)

If XFOIL is found it runs the real NACA 0012 check and prints PASS/FAIL.
If not, it runs the mock wiring check. Either way it tells you what to do.
"""
import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    plot = "--plot" in sys.argv
    # run the known-answer check as a subprocess so its exit code is the verdict
    rc = subprocess.call([sys.executable,
                          os.path.join(HERE, "tests", "test_d0_naca0012.py")])
    if plot:
        _plot()
    sys.exit(rc)


def _plot():
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sys.path.insert(0, HERE)
        from finopt.solver import XfoilSolver, MockSolver
        xs = XfoilSolver(xfoil_path=os.environ.get("XFOIL", "xfoil"))
        solver = xs if xs.available else MockSolver()
        alphas = np.arange(-8, 14.01, 1.0)
        p = solver.run_naca("0012", 1e6, alphas)
        fig, ax = plt.subplots(1, 2, figsize=(9, 4))
        ax[0].plot(p.alpha, p.cl, "o-"); ax[0].set_xlabel("alpha [deg]")
        ax[0].set_ylabel("CL"); ax[0].grid(alpha=.3); ax[0].set_title("Lift curve")
        ax[1].plot(p.cd, p.cl, "o-"); ax[1].set_xlabel("CD")
        ax[1].set_ylabel("CL"); ax[1].grid(alpha=.3); ax[1].set_title("Drag polar")
        fig.suptitle(f"D0: {p.source}  Re={p.reynolds:.0e}")
        fig.tight_layout()
        out = os.path.join(HERE, "outputs", "d0_naca0012.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, dpi=130)
        print(f"\nPlot saved: {out}")
    except ImportError:
        print("\n(matplotlib not installed — skipping plot)")


if __name__ == "__main__":
    main()
