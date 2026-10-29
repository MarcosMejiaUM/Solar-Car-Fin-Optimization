#!/usr/bin/env python3
"""Test whether running XFOIL under xvfb (virtual X framebuffer) fixes the
WSL graphics corruption that breaks ASEQ/PACC. Real XFOIL, just a working
fake display. If polar files fill with data, the whole problem is solved."""
import subprocess, os, tempfile, glob


def run_xvfb(cmds, timeout=180):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".in", delete=False)
    f.write("\n".join(cmds) + "\n")
    f.close()
    p = subprocess.run(f"xvfb-run -a xfoil < {f.name}", shell=True, text=True,
                       capture_output=True, timeout=timeout)
    os.remove(f.name)
    return p.stdout, p.stderr


def polar_rows(pol):
    """Count real data rows (7+ all-numeric columns); skips text headers."""
    n = 0
    if not os.path.exists(pol):
        return 0
    for ln in open(pol):
        q = ln.split()
        if len(q) >= 7:
            try:
                [float(x) for x in q[:7]]
                n += 1
            except ValueError:
                pass
    return n


# --- NACA0012: must produce 5 rows if xvfb fixes it ---
pol = "/tmp/xv_naca.txt"
if os.path.exists(pol):
    os.remove(pol)
cmds = ["PLOP", "G F", "", "NACA 0012", "PANE", "OPER", "VISC 1000000",
        "ITER 200", "PACC", pol, "", "ASEQ 0 8 2", "PACC", "", "QUIT"]
out, err = run_xvfb(cmds)
r = polar_rows(pol)
print(f"NACA0012 under xvfb : polar_data_rows = {r}  (want 5)")
if r:
    print(open(pol).read())

# --- E327: the real continuation test ---
e = glob.glob(os.path.expanduser("~/SolarCar/airfoils/e327*.dat"))
if e:
    pol = "/tmp/xv_e327.txt"
    if os.path.exists(pol):
        os.remove(pol)
    cmds = ["PLOP", "G F", "", f"LOAD {e[0]}", "PANE", "OPER", "VISC 600000",
            "ITER 300", "PACC", pol, "", "ASEQ -4 10 1", "PACC", "", "QUIT"]
    out, err = run_xvfb(cmds)
    r = polar_rows(pol)
    print(f"E327 under xvfb     : polar_data_rows = {r}  (of 15)")
    if r:
        print(open(pol).read())

# --- CH10 too ---
ch = glob.glob(os.path.expanduser("~/SolarCar/airfoils/ch10*.dat"))
if ch:
    pol = "/tmp/xv_ch10.txt"
    if os.path.exists(pol):
        os.remove(pol)
    cmds = ["PLOP", "G F", "", f"LOAD {ch[0]}", "PANE", "OPER", "VISC 600000",
            "ITER 300", "PACC", pol, "", "ASEQ -4 10 1", "PACC", "", "QUIT"]
    out, err = run_xvfb(cmds)
    r = polar_rows(pol)
    print(f"CH10 under xvfb     : polar_data_rows = {r}  (of 15)")
    if r:
        print(open(pol).read())
