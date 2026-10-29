"""finopt — family-agnostic fin airfoil optimizer (roadmap D0-D1)."""
from finopt.solver import (Polar, BaseSolver, XfoilSolver, XfoilLibSolver,
                           MockSolver, get_solver)
from finopt.geometry import naca4_coords, load_dat, write_dat
from finopt.airfoil_db import fetch_airfoil, ensure_airfoils
from finopt.objective import (cd_eff_A, evaluate_airfoil, ObjectiveResult,
                              YawDistribution, SkewNormalYaw, GaussianYaw,
                              UniformYaw, DiracYaw, c_thrust, best_thrust,
                              cd_eff_of_phi)
from finopt.cst import (cst_to_coords, coords_to_cst, cst_from_params,
                        cst_surface, max_thickness, is_valid,
                        thickness_distribution)
from finopt.optimize import (optimize_airfoil, multi_start, design_to_coords,
                             pack, unpack)

__all__ = ["Polar", "BaseSolver", "XfoilSolver", "XfoilLibSolver",
           "MockSolver", "get_solver",
           "naca4_coords", "load_dat", "write_dat",
           "fetch_airfoil", "ensure_airfoils",
           "cd_eff_A", "evaluate_airfoil", "ObjectiveResult",
           "YawDistribution", "SkewNormalYaw", "GaussianYaw", "UniformYaw",
           "DiracYaw", "c_thrust", "best_thrust", "cd_eff_of_phi",
           "cst_to_coords", "coords_to_cst", "cst_from_params", "cst_surface",
           "max_thickness", "is_valid", "thickness_distribution",
           "optimize_airfoil", "multi_start", "design_to_coords",
           "pack", "unpack"]
