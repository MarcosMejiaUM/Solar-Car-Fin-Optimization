"""
finopt.objective — the work-weighted, thrust-projected fin objective (D2).

This is the heart of the project: it asks "how good is this airfoil for the
FIN" the RIGHT way — projected thrust over the yaw distribution the car
actually sees — instead of by raw L/D.

Physics (Anderson Eq. 1.1-1.3, 2.5-2.7):

  angle of attack:   alpha = theta + phi      (control angle + yaw)
  forward thrust  :  C_thr(theta,phi) = CL(alpha) sin(phi) - CD(alpha) cos(phi)
                     (CL = C_yF lift coeff, CD = C_xF drag coeff;
                      POSITIVE C_thr = force along the direction of travel)
  per-yaw eff drag:  Cd(phi) = C0 - max_theta C_thr(theta,phi)
                     (the onboard controller picks theta, |theta| <= 20 deg,
                      to MAXIMIZE forward thrust; effective drag is the
                      baseline C0 minus that best thrust, so at high yaw the
                      fin "sails" and Cd(phi) goes NEGATIVE — matching Fig 1.3)
  objective       :  Cd_effA = A * integral_{-phimax}^{phimax} Cd(phi) W(phi) dphi
                     (work-weighted average; MINIMIZE it. Lower = faster car.)

SIGN NOTE: the paper writes Cd = C0 + max C_thr (Eq 2.6), but with Eq 1.2 that
makes drag INCREASE with yaw — the opposite of the paper's own Fig 1.3 (drag
goes negative at high yaw). We implement the physically-correct version
(C0 - max C_thr) and VERIFY the Fig-1.3 shape in the D2 known-answer check.
This only flips the sign of the thrust term; minimizing Cd_effA is correct
either way, and the airfoil RANKING is unaffected by C0 or A (both are shared
constants when comparing airfoils at one size).

STUBS (clean seams, per the roadmap):
  * W(phi)       — yaw distribution. Real one is SOLAR-SET; here, parametric
                   curves (skew-normal shaped like Fig 1.5, gaussian, uniform).
  * raytracer    — array-shadow power. raytracer=False omits it (default);
                   the `array_penalty` hook is where it plugs in later.
  * Re(phi)      — apparent wind (hence Re) rises with yaw; here we use ONE
                   representative Re. Re-per-yaw is a documented refinement.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from finopt.solver import Polar

# numpy 2.x renamed trapz -> trapezoid; support both
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


# --------------------------------------------------------------------------
# Yaw distributions W(phi)   (phi in DEGREES; each integrates to 1)
# --------------------------------------------------------------------------
class YawDistribution:
    """Base: a normalized work-weighted yaw density over [-phimax, phimax]."""

    def __init__(self, phimax: float = 18.0):
        self.phimax = phimax

    def pdf(self, phi_deg):
        raise NotImplementedError

    def normalized_pdf(self, phi_grid):
        """pdf evaluated on phi_grid and renormalized to integrate to 1
        (trapezoid) over the grid, so the discrete objective is consistent."""
        w = np.asarray(self.pdf(phi_grid), float)
        w = np.clip(w, 0, None)
        area = _trapz(w, phi_grid)
        return w / area if area > 0 else w

    def mean_abs_yaw(self, n_phi: int = 361):
        """E[|phi|]: 'how much high-yaw weight' (sailing magnitude)."""
        phi = np.linspace(-self.phimax, self.phimax, n_phi)
        w = self.normalized_pdf(phi)
        return float(_trapz(np.abs(phi) * w, phi))

    def mean_yaw(self, n_phi: int = 361):
        """E[phi]: the signed yaw BIAS (asymmetry) of the distribution. This —
        not the magnitude — is what drives optimal CAMBER: a distribution
        biased toward +yaw rewards +camber, a symmetric one rewards a symmetric
        section. (Anderson's cambered 3211 comes from his W's asymmetry.)"""
        phi = np.linspace(-self.phimax, self.phimax, n_phi)
        w = self.normalized_pdf(phi)
        return float(_trapz(phi * w, phi))


class GaussianYaw(YawDistribution):
    def __init__(self, mu: float = 0.0, sigma: float = 6.0, phimax: float = 18.0):
        super().__init__(phimax); self.mu = mu; self.sigma = sigma

    def pdf(self, phi):
        phi = np.asarray(phi, float)
        return np.exp(-0.5 * ((phi - self.mu) / self.sigma) ** 2)


class SkewNormalYaw(YawDistribution):
    """Skew-normal, shaped to resemble Fig 1.5: modal yaw slightly positive,
    left-skewed (high yaws tend negative). alpha<0 gives left skew."""

    def __init__(self, xi: float = 3.0, omega: float = 6.0, alpha: float = -4.0,
                 phimax: float = 18.0):
        super().__init__(phimax)
        self.xi = xi; self.omega = omega; self.alpha = alpha

    def pdf(self, phi):
        from math import sqrt, pi
        phi = np.asarray(phi, float)
        t = (phi - self.xi) / self.omega
        pdf = np.exp(-0.5 * t ** 2) / sqrt(2 * pi)
        # standard-normal CDF via erf
        from math import erf
        cdf = np.array([0.5 * (1 + erf(self.alpha * ti / sqrt(2))) for ti in t])
        return 2.0 / self.omega * pdf * cdf


class UniformYaw(YawDistribution):
    def pdf(self, phi):
        phi = np.asarray(phi, float)
        return np.where(np.abs(phi) <= self.phimax, 1.0, 0.0)


class DiracYaw(YawDistribution):
    """All weight at a single yaw phi0 — for the delta-function known-answer
    check (reduces Cd_effA/A to the single-yaw effective drag)."""

    def __init__(self, phi0: float = 0.0):
        super().__init__(abs(phi0) + 1.0); self.phi0 = phi0

    def normalized_pdf(self, phi_grid):
        w = np.zeros_like(np.asarray(phi_grid, float))
        i = int(np.argmin(np.abs(np.asarray(phi_grid, float) - self.phi0)))
        # unit mass at the nearest grid point (discrete delta)
        dphi = (phi_grid[1] - phi_grid[0]) if len(phi_grid) > 1 else 1.0
        w[i] = 1.0 / dphi
        return w


# --------------------------------------------------------------------------
# Thrust physics
# --------------------------------------------------------------------------
def c_thrust(polar: Polar, theta_deg: float, phi_deg: float):
    """Forward-thrust coefficient at control angle theta, yaw phi (degrees).

    Uses the polar's CL/CD interpolated at alpha = theta + phi. Returns NaN if
    alpha is outside the converged polar range (i.e. the fin would be stalled
    / off-data at that combination — treated as infeasible for the controller).
    """
    alpha = theta_deg + phi_deg
    cl = polar.cl_at(alpha)
    cd = polar.cd_at(alpha)
    if not (np.isfinite(cl) and np.isfinite(cd)):
        return float("nan")
    phi = np.radians(phi_deg)
    return cl * np.sin(phi) - cd * np.cos(phi)


def best_thrust(polar: Polar, phi_deg: float, theta_max: float = 20.0,
                n_theta: int = 81):
    """max over theta in [-theta_max, theta_max] of C_thrust(theta, phi).

    The inner control optimization (Eq 2.6's max_theta). Grid search over
    theta; skips thetas where alpha is off the converged polar (stall). Returns
    (best_thrust, theta_star). If no theta is feasible, returns (nan, nan).
    """
    thetas = np.linspace(-theta_max, theta_max, n_theta)
    vals = np.array([c_thrust(polar, t, phi_deg) for t in thetas])
    if not np.any(np.isfinite(vals)):
        return float("nan"), float("nan")
    i = int(np.nanargmax(vals))
    return float(vals[i]), float(thetas[i])


def cd_eff_of_phi(polar: Polar, phi_deg: float, c0: float = 0.0,
                  theta_max: float = 20.0):
    """Effective drag coefficient at a single yaw: Cd(phi) = C0 - max_theta C_thr.
    Goes NEGATIVE where the fin can sail (matches Fig 1.3). NaN if no feasible
    control at this yaw."""
    bt, _ = best_thrust(polar, phi_deg, theta_max=theta_max)
    if not np.isfinite(bt):
        return float("nan")
    return c0 - bt


# --------------------------------------------------------------------------
# The objective
# --------------------------------------------------------------------------
@dataclass
class ObjectiveResult:
    cd_eff_A: float          # the scalar objective (minimize)
    cd_eff: float            # the work-weighted mean Cd (= cd_eff_A / A)
    phi_grid: np.ndarray
    cd_of_phi: np.ndarray    # Cd(phi) curve (for plotting vs Fig 1.3)
    theta_star: np.ndarray   # controller's chosen theta(phi)
    feasible_frac: float     # fraction of yaw grid with a feasible control
    A: float
    meta: dict


def cd_eff_A(polar: Polar, yaw: YawDistribution, A: float = 0.264,
             c0: float = 0.0, theta_max: float = 20.0,
             n_phi: int = 121, array_penalty=None) -> ObjectiveResult:
    """Compute the work-weighted effective drag area (the D2 objective).

    Parameters
    ----------
    polar : solved Polar for the airfoil (from any solver).
    yaw   : a YawDistribution (the W(phi) stub).
    A     : planform area S*L [m^2]. Default ~0.264 = 0.722*0.366 (paper fin).
            A common scale factor — does not change airfoil RANKING at fixed size.
    c0    : baseline drag constant. A common offset — does not change ranking.
    theta_max : control authority [deg], default 20 (paper).
    n_phi : yaw-grid resolution.
    array_penalty : optional callable(phi)->extra drag (the Raytracer seam).
                    Default None (raytracer=False).

    Returns ObjectiveResult. LOWER cd_eff_A = better (faster car).
    """
    phimax = yaw.phimax
    phi_grid = np.linspace(-phimax, phimax, n_phi)
    w = yaw.normalized_pdf(phi_grid)

    cd = np.empty_like(phi_grid)
    th = np.empty_like(phi_grid)
    for i, phi in enumerate(phi_grid):
        bt, ts = best_thrust(polar, phi, theta_max=theta_max)
        if np.isfinite(bt):
            cd[i] = c0 - bt
            th[i] = ts
        else:
            cd[i] = np.nan
            th[i] = np.nan

    if array_penalty is not None:
        cd = cd + np.array([array_penalty(p) for p in phi_grid])

    feasible = np.isfinite(cd)
    feasible_frac = float(feasible.mean())
    # integrate only over feasible yaws; renormalize the weight there so the
    # mean is well-defined even if a few extreme yaws are infeasible.
    if feasible.any():
        wf = w.copy()
        wf[~feasible] = 0.0
        wn = _trapz(wf, phi_grid)
        mean_cd = _trapz(np.where(feasible, cd, 0.0) * wf, phi_grid) / wn \
            if wn > 0 else float("nan")
    else:
        mean_cd = float("nan")

    return ObjectiveResult(
        cd_eff_A=A * mean_cd, cd_eff=mean_cd, phi_grid=phi_grid,
        cd_of_phi=cd, theta_star=th, feasible_frac=feasible_frac, A=A,
        meta={"c0": c0, "theta_max": theta_max, "source": polar.source,
              "reynolds": polar.reynolds})


def evaluate_airfoil(solver, coords_or_naca, reynolds: float,
                     yaw: YawDistribution, alpha_range=None, **kw) -> ObjectiveResult:
    """Convenience: solve the polar then score it. coords_or_naca is either an
    (N,2) coordinate array or a NACA 4-digit string. Extra kwargs go to cd_eff_A."""
    if alpha_range is None:
        # Attached-flow range only. The controller MAXIMIZES thrust, so it
        # keeps alpha in the good-lift region and never operates near stall;
        # a full +/-(20+phimax) sweep would just burn iterations on stalled
        # angles the controller never picks.
        alpha_range = np.arange(-16.0, 18.1, 1.0)
    if isinstance(coords_or_naca, str):
        polar = solver.run_naca(coords_or_naca, reynolds, alpha_range)
    else:
        polar = solver.run_coords(np.asarray(coords_or_naca, float),
                                  reynolds, alpha_range, name="airfoil")
    return cd_eff_A(polar, yaw, **kw)
