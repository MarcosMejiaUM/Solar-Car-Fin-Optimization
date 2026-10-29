#!/usr/bin/env python3
"""Dump RAW output to settle two things on this XFOIL build:
   A) does ASEQ (no PACC) actually run multiple alphas, and in what format?
   B) is the VPAR submenu eating the ITER setting (starving hard airfoils)?
"""
import subprocess, glob, os, tempfile, re


def run_file(cmds, timeout=120):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".in", delete=False)
    f.write("\n".join(cmds) + "\n")
    f.close()
    p = subprocess.run(f"xfoil < {f.name}", shell=True, text=True,
                       capture_output=True, timeout=timeout)
    os.remove(f.name)
    return p.stdout


def summarize(out):
    ni = len(re.findall(r"rms:", out))
    conv = "Convergence failed" not in out
    cl = re.findall(r"a\s*=\s*-?\d+\.\d+\s+CL\s*=\s*(-?\d+\.\d+)", out)
    return f"iters_run={ni}  converged={conv}  last_CL={cl[-1] if cl else None}"


e = glob.glob(os.path.expanduser("~/SolarCar/airfoils/e327*.dat"))[0]

print("=" * 62)
print("A) NACA0012  ASEQ 0 8 2  (NO PACC) — RAW stdout, last 1800 chars")
print("=" * 62)
out = run_file(["PLOP", "G F", "", "NACA 0012", "PANE", "OPER",
                "VISC 1000000", "ITER 200", "ASEQ 0 8 2", "", "QUIT"])
print(out[-1800:])

print("\n" + "=" * 62)
print("B) E327  ALFA=4  NO VPAR   ITER 400")
print("=" * 62)
out = run_file(["PLOP", "G F", "", f"LOAD {e}", "PANE", "OPER",
                "VISC 600000", "ITER 400", "ALFA 4", "", "QUIT"])
print("  ", summarize(out))

print("=" * 62)
print("C) E327  ALFA=4  WITH VPAR N 9  ITER 400  (compare iters_run to B)")
print("=" * 62)
out = run_file(["PLOP", "G F", "", f"LOAD {e}", "PANE", "OPER",
                "VISC 600000", "VPAR", "N 9", "", "ITER 400", "ALFA 4",
                "", "QUIT"])
print("  ", summarize(out))
