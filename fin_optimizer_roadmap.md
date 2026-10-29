# Fin Airfoil Optimizer — Build Roadmap

A verification-driven build plan. Each deliverable has a **known-answer check** so errors
can't hide, and each is independently useful if the project stops there. We build *down to*
the committed deliverable, not *up forever*.

## Scope decisions (locked for now)

| Decision | Choice | Deferred alternative |
|---|---|---|
| Size variables | **Constant chord**, span fixed per study | Variable span/chord (later) |
| 2D solver | **XFOIL** via Python subprocess, unmodified | MFoil (enables analytical gradients) |
| 3D correction | **Prandtl lifting-line** on 2D polars | Vortex-lattice (AVL/XFLR5), then RANS |
| Wind yaw dist. W(φ) | **Stubbed** — generic parametric curves | Real SOLAR-SET W(φ) from team |
| Array shadowing | **`raytracer=False`** — excluded from objective | Raytracer plugged into same seam |
| Control angle θ | Inner max over θ (perfect control assumed) | Robust/expected-thrust (later) |

## Design principle: clean seams

Everything we can't access yet (Raytracer, real wind data, Simulator race-time map) is
represented by a **stub with the real interface**, so swapping in the true component is a
one-line change, not a refactor. The objective is assembled from pluggable pieces.

---

## The deliverable ladder

### D0 — XFOIL wrapper with a known-answer check
**Build:** A Python function `polar(coords, Re, alpha_range) -> (Cl, Cd, Cm)` that writes
an airfoil coordinate file, drives XFOIL via subprocess, parses the polar, and handles
non-convergence gracefully.
**Known-answer check:** Run NACA 0012 at Re = 1e6. Compare Cl(α) and Cd(α) against
published XFOIL/experimental polars. Must match within a few percent.
**Why first:** Everything downstream is garbage if the solver interface is wrong. This is
the foundation and it must be trustworthy before anything else.

### D1 — Coordinate-table front end (family-agnostic input)
**Build:** Loaders that turn (a) NACA 4-digit params and (b) raw `.dat` tables (Eppler,
CH, UIUC database) into the *same* internal coordinate representation XFOIL consumes.
**Known-answer check:** Feed a known Eppler/CH airfoil through; confirm the polar matches
its published data. Confirm a NACA-4 generated here matches a `.dat` of the same airfoil.
**Why:** This is the core generality unlock — the thing the teammate's sweep has at the
solver level but which must be explicit and validated in our pipeline. After D1, no airfoil
family is privileged by the input layer.

### D2 — The objective function (stubbed physics)
**Build:** `Cd_effA(airfoil; W, raytracer=False)` implementing the work-weighted yaw
integral:  Cd_effA = A · ∫ Cd(F,φ) W(φ) dφ,  with Cd(F,φ) = C0 + max_θ Cthrust(θ,φ).
- Thrust/lateral force from the force-projection equations (paper Eq. 1.1–1.3).
- `max_θ` by a cheap 1-D search over the control angle, seeded near max-L/D.
- `W(φ)` is a **pluggable stub**: generic curves (Gaussian, skew-normal, uniform) with the
  same signature the real SOLAR-SET distribution will have.
- `raytracer=False` leaves array power out of the objective but keeps the seam.
**Known-answer check:** With W(φ) = δ(0) (all weight at zero yaw), Cd_effA must reduce to
the plain sectional drag at α matching the control solution. Sanity-check monotonicity:
a higher-thrust airfoil scores lower Cd_effA.
**Why:** This is where the paper's actual physics lives. Getting the projection and the
work-weighting right is the intellectual core.

### D3 — Reproduce a baseline (the credibility rung)
**Build:** Nothing new — wire D0–D2 together and run the current fin airfoils
(CH10 / E327) and a NACA baseline through the *same* objective.
**Known-answer check (two parts):**
1. Reproduce the **teammate's** existing ranking under *their* criterion (confirm we didn't
   break the aero). → requires knowing what their sweep optimizes.
2. If/when Matthew's settings arrive, reproduce his NACA 3211 result.
**Why:** You cannot claim to improve on a baseline you can't reproduce. This rung turns
"I think" into "I verified." It's also the apples-to-apples that reveals whether the current
airfoil is optimal for the *race-time* objective or just for L/D.

### D4 — The optimizer (search → true optimization)
**Build:** Wrap the objective in an actual optimizer. Start with the paper's structure but
cleaner: Bayesian Optimization (for the global/family phase) and/or L-BFGS-B over a
parameterization. Multi-start from each real airfoil seed.
**Known-answer check:** On a synthetic objective with a known minimum (e.g. a quadratic in
camber), the optimizer must find it. Then confirm it reproduces D3's best-of-sweep when the
search space is restricted to the catalog.
**Why:** This is the headline deliverable — the thing that makes it an optimizer, not a
search. Committed deliverable candidate.

### D5 — 3D finite-span / induced-drag correction
**Build:** Prandtl lifting-line (or a finite-wing correction) layered on the 2D polars to
add induced drag for the fin's low aspect ratio (AR ≈ 2). Constant chord keeps this simple.
**Known-answer check:** Reproduce the analytical induced drag Cdi = Cl²/(π e AR) for an
elliptical loading of known AR before trusting it on the fin.
**Why:** The single biggest fidelity gain for a stubby fin. Tests whether the "flat sectional
optimum" survives once finite-span losses are included.

### D6+ — Future seams (not now, but built toward)
- Plug in real SOLAR-SET W(φ) (swap the stub).
- Plug in Raytracer (`raytracer=True`) for array shadowing.
- Plug in Simulator race-time map (replace Cd_effA proxy with true T).
- CST parameterization for continuous cross-family search.
- Analytical gradients via MFoil + constrained envelope theorem.
- Robust/expected-thrust objective (imperfect control).
- Variable chord; VLM then RANS fidelity tiers.

---

## Committed first target
**D0 → D4**, with D3 as the non-negotiable credibility checkpoint. That yields a *validated,
family-agnostic true optimizer on the race-time proxy objective* — a real, defensible result —
with every deferred component (Raytracer, real wind, 3D, gradients) sitting behind a clean
seam ready to plug in.
