"""Variable-coefficient elliptic problem in divergence form on quadrilateral and triangular spheres.

Solves

    -div(k grad u) + u = f,    k(x, y, z) = 1 + x/2,

on the unit sphere with the manufactured solution u = x y z, for which
div(k grad u) = -12 k x y z + y z (1 - 3 x^2) / 2.  The equation is written as a
string; the product rule turns it into k lap(u) + grad(k).grad(u).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pysurfacefun as psf


def exact(x, y, z):
    return x * y * z


def kappa(x, y, z):
    return 1.0 + 0.5 * x


def rhs(x, y, z):
    return -((1.0 + 0.5 * x) * (-12.0 * x * y * z) + 0.5 * y * z * (1.0 - 3.0 * x**2)) + x * y * z


def solve(dom) -> float:
    problem = psf.SurfaceProblem(dom, variables="u", namespace={"k": psf.field(kappa, dom), "f": psf.field(rhs, dom)})
    problem.add_equation("-div(k*grad(u)) + u = f")
    u = problem.solve()
    ue = psf.field(exact, dom)
    return psf.norm(u - ue, "inf") / psf.norm(ue, "inf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-values", default="7 9 11 13", help="points per patch edge")
    args = parser.parse_args()
    ns = [int(v) for v in args.n_values.replace(",", " ").split()]

    print("  n   quad sphere   triangular sphere")
    for n in ns:
        quad = solve(psf.sphere(n, nref=1))
        tri = solve(psf.icosphere_tri(n, nref=1))
        print(f"{n:3d}   {quad:.3e}     {tri:.3e}")


if __name__ == "__main__":
    main()
