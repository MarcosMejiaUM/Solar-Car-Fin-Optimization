#!/usr/bin/env python3
"""Root-cause test: is X11 graphics (broken under WSL) what breaks ASEQ/PACC?
Run the MATLAB recipe with DISPLAY set vs UNSET and count real polar rows."""
import subprocess, os, tempfile, glob


def run(cmds, unset_display, timeout=120):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".in", delete=False)
    f.write("\n".join(cmds) + "\n")
    f.close()
    env = dict(os.environ)
    if unset_display:
        env.pop("DISPLAY", None)
    p = subprocess.run(f"xfoil < {f.name}", shell=True, text=True,
                       capture_output=True, timeout=timeout, env=env)
    os.remove(f.name)
    return p.stdout


def polar_rows(pol):
    n = 0
    if os.path.exists(pol):
        for ln in open(pol):
            q = ln.split()
            if len(q) >= 7:
                try:
                    float(q[0]); float(q[1]); n += 1
                except ValueError:
                    pass
    return n


print("current DISPLAY =", repr(os.environ.get("DISPLAY")))
pol = "/tmp/t_naca.txt"
for ud in [False, True]:
    if os.path.exists(pol):
        os.remove(pol)
    cmds = ["PLOP", "G F", "", "NACA 0012", "PANE", "OPER", "VISC 1000000",
            "ITER 200", "PACC", pol, "", "ASEQ 0 8 2", "PACC", "", "QUIT"]
    run(cmds, ud)
    print(f"NACA0012  DISPLAY {'UNSET' if ud else 'set  '}: "
          f"polar_data_rows = {polar_rows(pol)}  (want 5)")

# if unsetting DISPLAY works, confirm it on E327 too
e = glob.glob(os.path.expanduser("~/SolarCar/airfoils/e327*.dat"))
if e:
    pol = "/tmp/t_e327.txt"
    if os.path.exists(pol):
        os.remove(pol)
    cmds = ["PLOP", "G F", "", f"LOAD {e[0]}", "PANE", "OPER", "VISC 600000",
            "ITER 300", "PACC", pol, "", "ASEQ -4 10 1", "PACC", "", "QUIT"]
    run(cmds, True)
    print(f"E327      DISPLAY UNSET: polar_data_rows = {polar_rows(pol)}  (of 15)")
    if polar_rows(pol):
        print("--- E327 polar file ---")
        print(open(pol).read())
