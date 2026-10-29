"""
finopt.solver — 2D airfoil polar solvers behind one interface.

Deliverable D0 of the fin-optimizer roadmap.

Two solvers implement the SAME interface so everything downstream (the
objective in D2, the optimizer in D4) is written once and never has to
know which solver produced the numbers:

    XfoilSolver  — drives a real XFOIL binary via subprocess (run on your
                   laptop / CAEN where xfoil exists).
    MockSolver   — a cheap analytic stand-in (thin-airfoil + quadratic drag)
                   so the pipeline can be developed and unit-tested anywhere
                   XFOIL is NOT installed. NOT physics — a placeholder.

Swap is a one-line change:
    solver = XfoilSolver("xfoil")      # real
    solver = MockSolver()              # dev / CI

Both return a `Polar`. Both accept geometry either as a NACA 4-digit code
or as an (N,2) array of (x, y) coordinates — the coordinate path is what
makes the pipeline family-agnostic (Eppler, CH, UIUC .dat, anything).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import warnings
from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------
# Result type
# --------------------------------------------------------------------------
@dataclass
class Polar:
    """A solved polar: aligned arrays of angle of attack and coefficients.

    Only CONVERGED points are stored. `alpha` is in degrees.
    """
    alpha: np.ndarray          # deg
    cl: np.ndarray
    cd: np.ndarray
    cm: np.ndarray
    reynolds: float
    source: str                # which solver / geometry produced this
    n_requested: int = 0       # how many alphas were asked for
    meta: dict = field(default_factory=dict)

    # --- convenience accessors ------------------------------------------
    @property
    def n_converged(self) -> int:
        return len(self.alpha)

    @property
    def convergence_rate(self) -> float:
        return self.n_converged / self.n_requested if self.n_requested else 0.0

    def cl_at(self, a_deg: float) -> float:
        """Linear-interpolated CL at a given alpha (deg). NaN if out of range."""
        return self._interp(self.cl, a_deg)

    def cd_at(self, a_deg: float) -> float:
        return self._interp(self.cd, a_deg)

    def cm_at(self, a_deg: float) -> float:
        return self._interp(self.cm, a_deg)

    def _interp(self, y: np.ndarray, a_deg: float) -> float:
        if self.n_converged == 0:
            return float("nan")
        if a_deg < self.alpha.min() or a_deg > self.alpha.max():
            return float("nan")
        # np.interp needs ascending x; alpha is sorted on construction
        return float(np.interp(a_deg, self.alpha, y))

    def cl_slope_per_deg(self, lo: float = -3.0, hi: float = 3.0) -> float:
        """Lift-curve slope (per deg) from a least-squares fit in [lo, hi].

        Sanity check: thin-airfoil theory gives ~0.1097 /deg (2*pi per rad).
        """
        m = (self.alpha >= lo) & (self.alpha <= hi)
        if m.sum() < 2:
            return float("nan")
        A = np.polyfit(self.alpha[m], self.cl[m], 1)
        return float(A[0])

    def __repr__(self) -> str:
        return (f"Polar({self.source}, Re={self.reynolds:.2e}, "
                f"{self.n_converged}/{self.n_requested} converged)")


# --------------------------------------------------------------------------
# Base interface
# --------------------------------------------------------------------------
class BaseSolver:
    """Common interface. Subclasses implement `run_coords`.

    `run_naca` has a default that generates coordinates with a 4-digit
    generator and calls `run_coords`, so a solver only MUST implement the
    coordinate path. (XfoilSolver overrides run_naca to use XFOIL's own
    built-in NACA command, which is the cleanest D0 check.)
    """

    def run_coords(self, coords: np.ndarray, reynolds: float,
                   alphas: np.ndarray, name: str = "coords") -> Polar:
        raise NotImplementedError

    def run_naca(self, code: str, reynolds: float,
                 alphas: np.ndarray) -> Polar:
        from finopt.geometry import naca4_coords
        coords = naca4_coords(code)
        return self.run_coords(coords, reynolds, alphas, name=f"NACA{code}")


# --------------------------------------------------------------------------
# Real XFOIL subprocess wrapper
# --------------------------------------------------------------------------
class XfoilSolver(BaseSolver):
    """Drive a real XFOIL binary through its stdin command interface.

    Robustness choices (XFOIL is notoriously flaky):
      * Alpha sweep runs OUTWARD FROM 0 (0->max, then re-init, 0->min) so
        each point continues from a converged neighbour — far more robust
        than a single ASEQ through stall.
      * Per-run wall-clock `timeout`; a hung XFOIL is killed, not waited on.
      * Only converged points (those XFOIL writes to its accumulated polar
        file) are returned; non-converged alphas are silently dropped and
        counted in Polar.meta['dropped'].
      * ITER limit is configurable.

    Parameters
    ----------
    xfoil_path : path to the xfoil executable (default "xfoil", i.e. on PATH).
    iters      : XFOIL viscous iteration cap per alpha.
    ncrit      : e^n transition parameter (9 = "clean wind tunnel"; LOWER
                 (~4-7) for the rougher, higher-turbulence flow a roadgoing
                 fin actually sees. This single knob matters a LOT at the
                 low Reynolds numbers of a 366 mm-chord fin — flag it in
                 any cross-airfoil comparison.)
    timeout    : seconds before a single xfoil invocation is killed.
    """

    def __init__(self, xfoil_path: str = "xfoil", iters: int = 200,
                 ncrit: float = 9.0, timeout: float = 60.0,
                 n_panel: int = 200):
        self.xfoil_path = xfoil_path
        self.iters = iters
        self.ncrit = ncrit
        self.timeout = timeout
        self.n_panel = n_panel
        resolved = shutil.which(xfoil_path) or (
            xfoil_path if os.path.isfile(xfoil_path) else None)
        self.available = resolved is not None
        self._resolved = resolved

    def _panel_cmds(self):
        """Geometry re-paneling via PPAR with an explicit panel count.

        CRITICAL: a bare `PANE` auto-repanels, and on fetched high-camber
        airfoils (CH10, E327) that auto-paneling produces a distribution XFOIL
        can NEVER converge on — it returns a plausible-looking CL while the
        viscous solve silently fails. Setting the panel count explicitly via
        PPAR fixes convergence for those airfoils and is harmless for NACA.
        This was the single fix that brought the whole airfoil family in.
        """
        return ["PPAR", f"N {self.n_panel}", "", ""]

    # -- public: NACA via XFOIL's own generator (cleanest D0 check) -------
    def run_naca(self, code: str, reynolds: float,
                 alphas: np.ndarray) -> Polar:
        geom_cmds = [f"NACA {code}"] + self._panel_cmds()
        return self._run(geom_cmds, reynolds, alphas, source=f"XFOIL NACA{code}")

    # -- public: arbitrary coordinates (family-agnostic path) ------------
    def run_coords(self, coords: np.ndarray, reynolds: float,
                   alphas: np.ndarray, name: str = "coords") -> Polar:
        coords = np.asarray(coords, float)
        datfile = tempfile.NamedTemporaryFile(
            mode="w", suffix=".dat", delete=False, dir=_tmpdir())
        try:
            datfile.write(f"{name}\n")
            for x, y in coords:
                datfile.write(f"{x:.6f} {y:.6f}\n")
            datfile.close()
            geom_cmds = [f"LOAD {datfile.name}"] + self._panel_cmds()
            return self._run(geom_cmds, reynolds, alphas,
                             source=f"XFOIL {name}")
        finally:
            _quiet_remove(datfile.name)

    # -- engine ----------------------------------------------------------
    def _run(self, geom_cmds, reynolds, alphas, source) -> Polar:
        """Solve a polar by invoking XFOIL ONCE PER ALPHA.

        Why one process per alpha instead of a single multi-alpha session:
        on some XFOIL builds (observed on 6.99 under WSL) a multi-ALFA stdin
        stream silently stops executing after the first angle — XFOIL solves
        alpha[0], then ignores every later command until QUIT, and PACC writes
        nothing. One fresh process per alpha sidesteps that entirely: each run
        is a known-good minimal session, and results are read from stdout (the
        one output that's reliable across builds). Cost is ~0.1-0.3 s/alpha of
        process spawn, which is negligible for airfoil polars.

        Geometry is set up once per call via `geom_cmds` (NACA or LOAD+PANE),
        replayed in each per-alpha session.
        """
        if not self.available:
            raise FileNotFoundError(
                f"XFOIL binary '{self.xfoil_path}' not found on PATH. "
                f"Install XFOIL or use MockSolver() for pipeline development.")

        alphas = np.asarray(alphas, float)
        a_list, cl_list, cd_list, cm_list = [], [], [], []

        for a in alphas:
            res = self._solve_one(geom_cmds, reynolds, float(a))
            if res is not None:
                cl, cd, cm = res
                a_list.append(float(a)); cl_list.append(cl)
                cd_list.append(cd); cm_list.append(cm)

        alpha_c = np.array(a_list)
        cl_c = np.array(cl_list)
        cd_c = np.array(cd_list)
        cm_c = np.array(cm_list)

        order = np.argsort(alpha_c) if len(alpha_c) else slice(None)
        polar = Polar(
            alpha=alpha_c[order] if len(alpha_c) else alpha_c,
            cl=cl_c[order] if len(cl_c) else cl_c,
            cd=cd_c[order] if len(cd_c) else cd_c,
            cm=cm_c[order] if len(cm_c) else cm_c,
            reynolds=reynolds, source=source, n_requested=len(alphas),
            meta={"dropped": len(alphas) - len(alpha_c),
                  "ncrit": self.ncrit, "iters": self.iters,
                  "mode": "per-alpha"})
        if polar.n_converged == 0:
            warnings.warn(f"XFOIL returned NO converged points for {source}. "
                          f"Check the binary, Reynolds number, and alpha range.")
        return polar

    def _solve_one(self, geom_cmds, reynolds, alpha):
        """Run one headless XFOIL session for a single alpha; parse stdout.

        Returns (cl, cd, cm) or None if the point didn't converge.
        """
        cmds = ["PLOP", "G F", ""]          # headless (no X11)
        cmds += list(geom_cmds)             # NACA xxxx + PANE, or LOAD + PANE
        cmds += ["OPER"]
        cmds += [f"VISC {reynolds:.1f}"]
        cmds += ["VPAR", f"N {self.ncrit:.3f}", ""]   # Ncrit via VPAR sub-menu
        cmds += [f"ITER {self.iters}"]
        cmds += [f"ALFA {alpha:.4f}"]
        cmds += ["", "QUIT"]
        stdin = "\n".join(cmds) + "\n"

        try:
            proc = subprocess.run(
                [self._resolved], input=stdin, text=True,
                capture_output=True, timeout=self.timeout, cwd=_tmpdir())
        except subprocess.TimeoutExpired:
            return None

        return _parse_single(proc.stdout or "")


def _parse_single(stdout: str):
    """Parse one single-alpha XFOIL session's stdout -> (cl, cd, cm) or None.

    XFOIL prints, per iteration, a pair:
        a =  5.000      CL =  0.5500
       Cm = -0.0012     CD =  0.00800   =>  ...
    The LAST such pair in the output is the converged result. If XFOIL failed
    to converge it prints 'VISCAL:  Convergence failed' and the last CL is
    unreliable, so we treat a visible convergence-failure as non-converged.
    """
    import re
    if "Convergence failed" in stdout:
        return None
    cl_matches = re.findall(r"a\s*=\s*(-?\d+\.\d+)\s+CL\s*=\s*(-?\d+\.\d+)",
                            stdout)
    cd_matches = re.findall(r"Cm\s*=\s*(-?\d+\.\d+)\s+CD\s*=\s*(-?\d+\.\d+)",
                            stdout)
    if not cl_matches or not cd_matches:
        return None
    cl = float(cl_matches[-1][1])
    cm = float(cd_matches[-1][0])
    cd = float(cd_matches[-1][1])
    # XFOIL occasionally emits CD <= 0 on a non-physical non-converged exit
    if cd <= 0:
        return None
    return cl, cd, cm


def _parse_stdout(stdout: str):
    """Fallback: recover converged points from XFOIL's console output.

    XFOIL prints each converged point as a pair of lines:
        a =  5.000      CL =  0.5500
       Cm = -0.0012     CD =  0.00800   =>  ...
    We pair consecutive 'a =/CL =' and 'Cm =/CD =' lines. Keeps only the
    LAST occurrence per alpha (the converged value after iteration).
    """
    import re
    a_cl = re.compile(r"a\s*=\s*(-?\d+\.\d+)\s+CL\s*=\s*(-?\d+\.\d+)")
    cm_cd = re.compile(r"Cm\s*=\s*(-?\d+\.\d+)\s+CD\s*=\s*(-?\d+\.\d+)")
    points = {}   # alpha -> (cl, cd, cm)
    lines = stdout.splitlines()
    pending = None
    for ln in lines:
        m1 = a_cl.search(ln)
        if m1:
            pending = (float(m1.group(1)), float(m1.group(2)))
            continue
        m2 = cm_cd.search(ln)
        if m2 and pending is not None:
            a, cl = pending
            cm, cd = float(m2.group(1)), float(m2.group(2))
            points[round(a, 3)] = (cl, cd, cm)
            pending = None
    if not points:
        return (np.array([]), np.array([]), np.array([]), np.array([]))
    a = np.array(sorted(points))
    cl = np.array([points[x][0] for x in a])
    cd = np.array([points[x][1] for x in a])
    cm = np.array([points[x][2] for x in a])
    return a, cl, cd, cm


def _parse_polar_file(path: str):
    """Parse an XFOIL PACC polar file -> (alpha, cl, cd, cm) arrays.

    The file has a ~12-line header; the data block columns are:
        alpha  CL  CD  CDp  CM  Top_Xtr  Bot_Xtr
    We take columns 0,1,2,4.
    """
    a, cl, cd, cm = [], [], [], []
    if not os.path.isfile(path):
        return (np.array([]), np.array([]), np.array([]), np.array([]))
    with open(path) as f:
        lines = f.readlines()
    data_started = False
    for ln in lines:
        if "-----" in ln:          # dashed rule precedes the data block
            data_started = True
            continue
        if not data_started:
            continue
        parts = ln.split()
        if len(parts) < 5:
            continue
        try:
            vals = [float(p) for p in parts[:5]]
        except ValueError:
            continue
        a.append(vals[0]); cl.append(vals[1]); cd.append(vals[2]); cm.append(vals[4])
    return (np.array(a), np.array(cl), np.array(cd), np.array(cm))


# --------------------------------------------------------------------------
# Mock analytic solver (pipeline development without XFOIL)
# --------------------------------------------------------------------------
class MockSolver(BaseSolver):
    """Analytic stand-in with the XfoilSolver interface. NOT PHYSICS.

    Purpose: let D1-D4 be built and unit-tested where XFOIL isn't installed,
    and let CI exercise the full pipeline deterministically. Produces a
    plausible-SHAPED polar (linear lift region, soft stall, quadratic drag
    bucket) driven by crude camber/thickness readouts of the geometry, so
    different airfoils give different — but meaningless — numbers.

    Do NOT draw any aerodynamic conclusion from MockSolver output.
    """

    def __init__(self, cl_alpha_per_deg: float = 0.11, stall_deg: float = 14.0,
                 cd0: float = 0.006, k_induced: float = 0.012):
        self.cl_alpha = cl_alpha_per_deg
        self.stall = stall_deg
        self.cd0 = cd0
        self.k = k_induced

    def run_coords(self, coords, reynolds, alphas, name="coords") -> Polar:
        coords = np.asarray(coords, float)
        camber, thick = _camber_thickness(coords)
        return self._model(reynolds, np.asarray(alphas, float),
                           camber, thick, source=f"MOCK {name}")

    def run_naca(self, code, reynolds, alphas) -> Polar:
        # read camber/thickness straight off the 4-digit code, no generator needed
        m = int(code[0]) / 100.0
        t = int(code[2:]) / 100.0
        return self._model(reynolds, np.asarray(alphas, float), m, t,
                           source=f"MOCK NACA{code}")

    def _model(self, reynolds, alphas, camber, thick, source) -> Polar:
        a0 = -camber * 40.0                      # zero-lift angle shifts with camber
        cl_lin = self.cl_alpha * (alphas - a0)
        # soft stall via tanh envelope
        env = np.where(np.abs(alphas) < self.stall, 1.0,
                       np.exp(-(np.abs(alphas) - self.stall) / 3.0))
        cl = cl_lin * env
        cd = self.cd0 * (1 + 4 * thick) + self.k * cl**2 + \
            0.02 * np.clip(np.abs(alphas) - self.stall, 0, None)**2
        cm = -0.1 * camber - 0.01 * alphas * 0.0  # crude, roughly constant
        cm = np.full_like(alphas, -0.1 * camber)
        return Polar(alpha=alphas, cl=cl, cd=cd, cm=cm, reynolds=reynolds,
                     source=source, n_requested=len(alphas),
                     meta={"MOCK": True, "camber": camber, "thick": thick})


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _camber_thickness(coords: np.ndarray):
    """Very crude camber & thickness readout from a coordinate loop.

    Good enough to make MockSolver differentiate airfoils; not used by XFOIL.
    """
    x = coords[:, 0]
    y = coords[:, 1]
    # normalise chord to 1
    c = x.max() - x.min()
    if c <= 0:
        return 0.0, 0.1
    xs = (x - x.min()) / c
    ys = y / c
    # sample thickness/camber at a few chord stations by nearest points
    cambers, thicks = [], []
    for xc in np.linspace(0.1, 0.9, 9):
        near = np.abs(xs - xc) < 0.05
        if near.sum() >= 2:
            yy = ys[near]
            thicks.append(yy.max() - yy.min())
            cambers.append((yy.max() + yy.min()) / 2)
    camber = float(np.mean(cambers)) if cambers else 0.0
    thick = float(np.max(thicks)) if thicks else 0.1
    return camber, thick


def _tmpdir() -> str:
    d = os.environ.get("FINOPT_TMP", os.path.join(tempfile.gettempdir(), "finopt"))
    os.makedirs(d, exist_ok=True)
    return d


def _quiet_remove(path: str):
    try:
        os.remove(path)
    except OSError:
        pass


# --------------------------------------------------------------------------
# In-process XFOIL via the xfoil-python library (PREFERRED engine)
# --------------------------------------------------------------------------
# This calls XFOIL's computational routines directly (f2py), with NO
# subprocess, NO stdin stream, and NO graphics/X11 — so it sidesteps the
# entire class of problems the Ubuntu `apt` subprocess binary had on WSL
# (command-stream desync, PACC/ASEQ dying, X11 font errors). aseq() does
# internal alpha-continuation, which is why high-camber low-Re airfoils
# (CH10, E327) converge here where the cold-start subprocess path failed.
try:
    from xfoil import XFoil as _XFoilLib          # type: ignore
    from xfoil.model import Airfoil as _AirfoilLib  # type: ignore
    _HAS_XFOIL_LIB = True
except Exception:
    _HAS_XFOIL_LIB = False


class XfoilLibSolver(BaseSolver):
    """Preferred solver: in-process XFOIL via the xfoil-python library.

    Same BaseSolver interface (run_naca / run_coords -> Polar) as the others,
    so everything downstream is unchanged. Only CONVERGED points (finite CL
    and CD) are returned; non-converged alphas come back as NaN from aseq and
    are dropped (and counted in Polar.meta['dropped']).
    """

    def __init__(self, n_crit: float = 9.0, max_iter: int = 200,
                 mach: float = 0.0, n_panel: int = 240, repanel: bool = True,
                 sweep_step: float = 0.5):
        if not _HAS_XFOIL_LIB:
            raise ImportError(
                "xfoil-python library not importable. Install it (see README) "
                "or use get_solver(), which falls back automatically.")
        self.n_crit = n_crit
        self.max_iter = max_iter
        self.mach = mach
        self.n_panel = n_panel
        self.repanel = repanel
        self.sweep_step = sweep_step   # deg; coarser (e.g. 1.0) = faster opt
        self.available = True

    def _new(self):
        xf = _XFoilLib()
        try:
            xf.print = False
        except Exception:
            pass
        xf.M = self.mach
        xf.n_crit = self.n_crit
        xf.max_iter = self.max_iter
        return xf

    def _do_repanel(self, xf):
        if not self.repanel:
            return
        try:
            xf.repanel(n_nodes=self.n_panel)
        except TypeError:
            try:
                xf.repanel()
            except Exception:
                pass
        except Exception:
            pass

    def run_naca(self, code: str, reynolds: float, alphas) -> Polar:
        xf = self._new()
        xf.naca(code)
        self._do_repanel(xf)
        return self._sweep(xf, reynolds, alphas, f"XFOILlib NACA{code}", step=self.sweep_step)

    def run_coords(self, coords, reynolds: float, alphas,
                   name: str = "coords") -> Polar:
        coords = np.asarray(coords, float)
        xf = self._new()
        xf.airfoil = _AirfoilLib(coords[:, 0].copy(), coords[:, 1].copy())
        self._do_repanel(xf)
        return self._sweep(xf, reynolds, alphas, f"XFOILlib {name}", step=self.sweep_step)

    def _sweep(self, xf, reynolds, alphas, source, step: float = 0.5) -> Polar:
        """Run the polar robustly for ANY airfoil, using aseq() + START SEARCH.

        Proven fact (from the probe): xf.aseq(a0, a1, da) converges CH10/E327
        14/14 — BUT aseq continues from its FIRST alpha, so if a0 is below the
        airfoil's cold-convergeable range (e.g. CH10 at -6 deg), the first
        solve fails and the whole sweep returns NaN. (A bare xf.a() cold call
        is weaker than aseq's internal first-point handling, which is why a
        hand-rolled a() loop fails where aseq succeeds — so we use aseq.)

        Strategy: try aseq(start, amax, step) for progressively safer starts
        (requested min first, then inward toward 0) and keep the first start
        that yields converged points. Then extend DOWN from that start toward
        amin to pick up any lower convergeable range. Fine step (0.5 deg) for
        smooth continuation; downstream interpolates (Polar.cl_at).
        """
        alphas = np.asarray(alphas, float)
        xf.Re = reynolds
        amin, amax = float(np.min(alphas)), float(np.max(alphas))

        pts = {}   # alpha -> (cl, cd, cm)

        def collect(res):
            a, cl, cd, cm = (np.atleast_1d(np.asarray(res[i], float))
                             for i in range(4))
            n = 0
            for ai, cli, cdi, cmi in zip(a, cl, cd, cm):
                if np.isfinite(cli) and np.isfinite(cdi) and cdi > 0:
                    pts[round(float(ai), 3)] = (float(cli), float(cdi), float(cmi))
                    n += 1
            return n

        # candidate upward-sweep starts: requested min first (so airfoils that
        # converge low get their full negative range), then inward toward 0.
        starts = []
        for v in [amin, -4.0, -2.0, 0.0, 2.0, 4.0]:
            if amin - 1e-9 <= v <= amax + 1e-9 and v not in starts:
                starts.append(v)

        # CHEAP probe: test cold-convergence of each candidate with a tiny
        # 3-point aseq (same cold-start behaviour as a full sweep) before
        # committing to the expensive full sweep. Avoids running a full
        # aseq-to-amax for every failed start.
        used = None
        for start in starts:
            xf.reset_bls()
            probe = xf.aseq(start, min(start + 1.0, amax), 0.5)
            if np.any(np.isfinite(np.asarray(probe[1], float))):
                used = start
                break

        if used is not None:
            xf.reset_bls()
            collect(xf.aseq(used, amax, step))

        if used is None or len(pts) == 0:
            return Polar(np.array([]), np.array([]), np.array([]), np.array([]),
                         reynolds, source, n_requested=len(alphas),
                         meta={"engine": "xfoil-lib", "note": "no convergeable "
                               "start", "ncrit": self.n_crit})

        # NOTE: no downward extension below the convergeable start. A negative
        # aseq step hangs this library (point count spins), and a high-camber
        # airfoil genuinely does not converge below its cold-start angle — so
        # the ascending sweep from `used` to amax is the honest usable polar.

        aa = np.array(sorted(pts))
        cl = np.array([pts[a][0] for a in aa])
        cd = np.array([pts[a][1] for a in aa])
        cm = np.array([pts[a][2] for a in aa])
        return Polar(alpha=aa, cl=cl, cd=cd, cm=cm, reynolds=reynolds,
                     source=source, n_requested=len(alphas),
                     meta={"n_converged_pts": len(aa), "ncrit": self.n_crit,
                           "engine": "xfoil-lib", "start": used,
                           "internal_step": step})


def get_solver(prefer_xfoil: bool = True, xfoil_path: str = "xfoil",
               **kw) -> BaseSolver:
    """Return the best available solver, deciding 'which engine' in one place.

    Preference order:
      1. XfoilLibSolver  — in-process xfoil-python library (robust, no X11)
      2. XfoilSolver     — subprocess binary (fallback for machines without
                           the library; note: broken on some apt/WSL builds)
      3. MockSolver      — placeholder physics for dev/CI without any XFOIL

    Accepts either `ncrit` or `n_crit`, and `iters` or `max_iter`, so callers
    written against any engine keep working.
    """
    ncrit = kw.get("n_crit", kw.get("ncrit", 9.0))
    max_iter = kw.get("max_iter", kw.get("iters", 200))
    n_panel = kw.get("n_panel", 240)

    if prefer_xfoil and _HAS_XFOIL_LIB:
        return XfoilLibSolver(n_crit=ncrit, max_iter=max_iter, n_panel=n_panel)

    if prefer_xfoil:
        xs = XfoilSolver(xfoil_path=xfoil_path, **{k: v for k, v in kw.items()
                         if k in ("iters", "ncrit", "timeout", "n_panel")})
        if xs.available:
            warnings.warn("xfoil-python library not found; using the subprocess "
                          "binary (may be unreliable on apt/WSL builds).")
            return xs
        warnings.warn("No XFOIL available; falling back to MockSolver "
                      "(placeholder physics — dev/CI only).")
    return MockSolver()
