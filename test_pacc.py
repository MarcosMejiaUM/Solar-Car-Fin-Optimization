#!/usr/bin/env python3
"""
Run the EXACT MATLAB recipe: file-redirected stdin + PACC polar file + ASEQ,
then DUMP the raw polar file and raw stdout so we see what XFOIL actually
produces (no parsing assumptions).
"""
import subprocess, glob, os, tempfile


def run_via_file(cmd_lines, timeout=120):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".in", delete=False)
    f.write("\n".join(cmd_lines) + "\n")
    f.close()
    p = subprocess.run(f"xfoil < {f.name}", shell=True, text=True,
                       capture_output=True, timeout=timeout)
    os.remove(f.name)
    return p.stdout, p.stderr


def recipe(load_cmds, pol, reynolds, a0, a1, step, iters=200):
    if os.path.exists(pol):
        os.remove(pol)
    cmds = ["PLOP", "G F", ""] + load_cmds + ["OPER", f"VISC {reynolds}",
            f"ITER {iters}", "PACC", pol, "", f"ASEQ {a0} {a1} {step}",
            "PACC", "", "QUIT"]
    out, err = run_via_file(cmds)
    return out, err


print("#" * 66)
print("# NACA0012  (must work — proves the recipe)")
print("#" * 66)
out, err = recipe(["NACA 0012", "PANE"], "/tmp/pol_naca.txt", 1000000, 0, 8, 2)
print(">>> polar file exists:", os.path.exists("/tmp/pol_naca.txt"))
if os.path.exists("/tmp/pol_naca.txt"):
    print(">>> POLAR FILE CONTENTS:")
    print(open("/tmp/pol_naca.txt").read())
print(">>> STDOUT (last 25 lines):")
print("\n".join(out.splitlines()[-25:]))

print("\n" + "#" * 66)
print("# E327  (the real target)")
print("#" * 66)
dat = glob.glob(os.path.expanduser("~/SolarCar/airfoils/e327*.dat"))
if dat:
    out, err = recipe([f"LOAD {dat[0]}", "PANE"], "/tmp/pol_e327.txt",
                      600000, -4, 10, 1)
    print(">>> polar file exists:", os.path.exists("/tmp/pol_e327.txt"))
    if os.path.exists("/tmp/pol_e327.txt"):
        print(">>> POLAR FILE CONTENTS:")
        print(open("/tmp/pol_e327.txt").read())
else:
    print("no e327 .dat found")
