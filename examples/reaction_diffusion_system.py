"""Coupled reaction-diffusion (Barrio-Varea-Aragon-Maini) system with high-order IMEX time stepping.

The two fields are advanced with implicit diffusion and explicit reaction:

    dt(u) - Du lap(u) = alpha u (1 - tau1 v^2) + v (1 - tau2 u)
    dt(v) - Dv lap(v) = beta v (1 + alpha tau1 u v / beta) + u (gamma + tau2 v)

Each field keeps one factorization of (I - gamma dt D lap) for the whole run.
The script compares the accuracy of several schemes at the same step size
against a fine reference solution.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pysurfacefun as psf

PARAMETERS = {
    "Du": 0.516 * 5.0e-3,
    "Dv": 5.0e-3,
    "alpha": 0.899,
    "beta": -0.91,
    "gamma": -0.899,
    "tau1": 0.02,
    "tau2": 0.2,
}


def build_problem(dom):
    bbox = psf.boundingbox(dom)
    u0 = psf.field(psf.randnfun3(0.3, bbox, seed=0), dom)
    v0 = psf.field(psf.randnfun3(0.3, bbox, seed=1), dom)
    problem = psf.SurfaceIVP(dom, variables={"u": u0, "v": v0}, namespace=PARAMETERS)
    problem.add_equation("dt(u) - Du*lap(u) = alpha*u*(1 - tau1*v**2) + v*(1 - tau2*u)")
    problem.add_equation("dt(v) - Dv*lap(v) = beta*v*(1 + (alpha/beta)*tau1*u*v) + u*(gamma + tau2*v)")
    return problem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=8, help="nodes per triangle edge")
    parser.add_argument("--nref", type=int, default=1, help="icosphere refinement level")
    parser.add_argument("--T", type=float, default=5.0, help="final time")
    parser.add_argument("--dt", type=float, default=0.1, help="time step for the comparison")
    parser.add_argument("--vtu", default="", help="write the final u to this VTU file")
    args = parser.parse_args()

    dom = psf.icosphere_tri(args.n, nref=args.nref)
    problem = build_problem(dom)
    steps = round(args.T / args.dt)

    reference = problem.build_solver(args.dt / 16, scheme="rk443").run(16 * steps)
    print(f"patches = {dom.npatches}, degree = {dom.degree}, T = {args.T}, dt = {args.dt}")
    print("scheme   order   max error vs reference   wall time")
    for scheme in ("sbdf1", "sbdf2", "rk222", "sbdf3", "sbdf4", "rk443"):
        solver = problem.build_solver(args.dt, scheme=scheme)
        t0 = time.perf_counter()
        state = solver.run(steps)
        elapsed = time.perf_counter() - t0
        error = max(psf.norm(state[k] - reference[k], "inf") for k in ("u", "v"))
        print(f"{scheme:7s}  {psf.get_scheme(scheme).order:5d}   {error:22.3e}   {elapsed:8.2f} s")

    if args.vtu:
        psf.write_vtu(args.vtu, reference["u"], point_name="u", nvis=2 * args.n)
        print(f"wrote {args.vtu}")


if __name__ == "__main__":
    main()
