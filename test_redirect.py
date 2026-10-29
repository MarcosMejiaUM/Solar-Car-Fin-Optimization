#!/usr/bin/env python3
"""
Test the KEY hypothesis from the MATLAB FanBladeDesigner.m:
does FILE-REDIRECTED stdin (xfoil < file) succeed at multi-point ASEQ
where PIPED stdin (subprocess input=) fails on this build?

If FILE works and PIPE doesn't, we rebuild the solver around file-redirect
+ ASEQ (internal continuation) — which also matches the proven MATLAB approach.
"""
import subprocess, re, glob, os, tempfile


def run_via_file(cmd_lines, timeout=120):
    """MATLAB-style: write commands to a file, run `xfoil < file`."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".in", delete=False)
    f.write("\n".join(cmd_lines) + "\n")
    f.close()
    try:
        p = subprocess.run(f"xfoil < {f.name}", shell=True, text=True,
                           capture_output=True, timeout=timeout)
        return p.stdout
    finally:
        os.remove(f.name)


def run_via_pipe(cmd_lines, timeout=120):
    """What we've been doing: pipe via subprocess input=."""
    p = subprocess.run(["xfoil"], input="\n".join(cmd_lines) + "\n",
                       text=True, capture_output=True, timeout=timeout)
    return p.stdout


def parse(stdout):
    conv = {}
    for m in re.finditer(r"a\s*=\s*(-?\d+\.\d+)\s+CL\s*=\s*(-?\d+\.\d+)", stdout):
        conv[round(float(m.group(1)), 2)] = float(m.group(2))
    return conv


def fmt(d):
    return {a: round(v, 3) for a, v in sorted(d.items())}


naca = ["PLOP", "G F", "", "NACA 0012", "PANE", "OPER", "VISC 1000000",
        "ITER 200", "ASEQ 0 8 2", "", "QUIT"]

print("=" * 64)
print("PIPE  vs  FILE-REDIRECT   (NACA0012, ASEQ 0->8 step 2)")
print("=" * 64)
print("PIPE :", fmt(parse(run_via_pipe(naca))))
print("FILE :", fmt(parse(run_via_file(naca))))

print("\n" + "=" * 64)
print("Hard airfoils via FILE-REDIRECT + ASEQ (the real test)")
print("=" * 64)
for nm in ["e327", "ch10"]:
    dat = glob.glob(os.path.expanduser(f"~/SolarCar/airfoils/{nm}*.dat"))
    if not dat:
        print(f"{nm}: no .dat found"); continue
    cmds = ["PLOP", "G F", "", f"LOAD {dat[0]}", "PANE", "OPER",
            "VISC 600000", "ITER 300", "ASEQ -4 10 1", "", "QUIT"]
    r = parse(run_via_file(cmds))
    print(f"{nm:5s} ASEQ -4..10 Re6e5 : {len(r)} pts  {fmt(r)}")
