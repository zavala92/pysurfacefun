"""Layer 4: triangular SurfaceProblem Helmholtz solve on the sphere."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pysurfacefun as psf


def parse_n_values(text: str) -> list[int]:
    return [int(part) for part in text.replace(",", " ").split()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-values", default="5 7 9 11", help="nodes per triangle edge")
    parser.add_argument("--nref", type=int, default=0, help="icosphere refinement level")
    parser.add_argument("--alpha", type=float, default=20.0, help="Helmholtz shift")
    parser.add_argument("--output", default="tri_surfaceproblem_helmholtz_convergence.txt")
    parser.add_argument("--plot", action="store_true", help="plot the convergence curve")
    parser.add_argument("--vtu", default="", help="optional VTU file for the finest solution")
    args = parser.parse_args()

    rows = []
    finest_solution = None
    for n in parse_n_values(args.n_values):
        dom = psf.icosphere_tri(n=n, nref=args.nref)
        exact = psf.field(lambda x, y, z: x * y * z, dom)
        rhs = (args.alpha - 12.0) * exact

        problem = psf.SurfaceProblem(dom, variables="u", namespace={"alpha": args.alpha, "rhs": rhs})
        problem.add_equation("lap(u) + alpha*u = rhs")
        uh = problem.solve()

        rel_error = psf.norm(uh - exact, "inf") / psf.norm(exact, "inf")
        rows.append((dom.degree, n, dom.npatches, rel_error))
        finest_solution = uh
        print(f"degree={dom.degree:2d}, n={n:2d}, patches={dom.npatches:3d}, rel error={rel_error:.3e}")

    data = np.asarray(rows)
    np.savetxt(
        args.output,
        data,
        header="degree n npatches relative_linf_error",
        fmt=["%d", "%d", "%d", "%.16e"],
    )
    print(f"wrote {args.output}")

    if args.vtu and finest_solution is not None:
        psf.write_tri_vtu(args.vtu, finest_solution, point_name="u_h")
        print(f"wrote {args.vtu}")

    if args.plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6.5, 4.6))
        plt.semilogy(data[:, 0], data[:, 3], "-o", linewidth=2.0, markersize=7)
        plt.xlabel("Polynomial degree")
        plt.ylabel(r"$\|u-u_h\|_\infty/\|u\|_\infty$")
        plt.grid(True, which="both", alpha=0.35)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
