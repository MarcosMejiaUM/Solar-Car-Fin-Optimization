#!/usr/bin/env python3
"""Confirm the xfoil-python API on this install, and do a minimal NACA solve,
so the solver wrapper matches the real interface. Also a first paneling test
on E327/CH10 (Kfid says they SHOULD converge — likely just paneling)."""
import sys, os, glob
import numpy as np

sys.path.insert(0, os.path.expanduser("~/SolarCar"))
from finopt.geometry import naca4_coords, load_dat

from xfoil import XFoil
from xfoil.model import Airfoil

print("=== XFoil attributes/methods ===")
print([a for a in dir(XFoil()) if not a.startswith("_")])
print("\n=== Airfoil attributes/methods ===")
print([a for a in dir(Airfoil) if not a.startswith("_")])

print("\n=== Minimal NACA0012 solve (Re=1e6) ===")
xf = XFoil()
try:
    xf.print = False
except Exception as e:
    print("  (xf.print not settable:", e, ")")
c = naca4_coords("0012")
xf.airfoil = Airfoil(c[:, 0].copy(), c[:, 1].copy())
# try to repanel if the method exists
if hasattr(xf, "repanel"):
    try:
        xf.repanel()
        print("  repanel() ok (defaults)")
    except Exception as e:
        print("  repanel() error:", e)
xf.Re = 1e6
xf.M = 0.0
if hasattr(xf, "n_crit"):
    xf.n_crit = 9
xf.max_iter = 100
a, cl, cd, cm, cp = xf.aseq(-4, 10, 2)
for ai, cli, cdi in zip(a, cl, cd):
    print(f"   a={ai:+.1f}  CL={cli:.4f}  CD={cdi:.5f}")

print("\n=== E327 / CH10 first try (Re=6e5), with/without repanel ===")
for nm in ["e327", "ch10"]:
    dat = glob.glob(os.path.expanduser(f"~/SolarCar/airfoils/{nm}*.dat"))
    if not dat:
        print(f"  {nm}: no .dat"); continue
    coords = load_dat(dat[0])
    for rp in [False, True]:
        xf = XFoil()
        try: xf.print = False
        except Exception: pass
        xf.airfoil = Airfoil(coords[:, 0].copy(), coords[:, 1].copy())
        if rp and hasattr(xf, "repanel"):
            try: xf.repanel(n_nodes=240) if "n_nodes" in xf.repanel.__doc__ else xf.repanel()
            except Exception:
                try: xf.repanel()
                except Exception: pass
        xf.Re = 6e5; xf.M = 0.0
        if hasattr(xf, "n_crit"): xf.n_crit = 9
        xf.max_iter = 200
        a, cl, cd, cm, cp = xf.aseq(-4, 10, 1)
        nconv = int(np.isfinite(cl).sum())
        rng = (f"{np.nanmin(cl):.2f}..{np.nanmax(cl):.2f}" if nconv else "--")
        print(f"  {nm:5s} repanel={str(rp):5s}: {nconv}/{len(a)} converged, CL {rng}")
