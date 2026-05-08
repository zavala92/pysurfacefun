"""Layer 3: p-refinement check for triangular surface differentiation."""

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
    parser.add_argument("--nref", type=int, default=1, help="icosphere refinement level")
    parser.add_argument("--output", default="tri_prefinement_convergence.txt")
    parser.add_argument("--plot", action="store_true", help="plot the convergence curve")
    args = parser.parse_args()

    rows = []
    exact_area = 4 * np.pi
    for n in parse_n_values(args.n_values):
        dom = psf.icosphere_tri(n=n, nref=args.nref)
        u = psf.tri_surfacefun(lambda x, y, z: x * y * z, dom)
        residual = psf.tri_lap(u) + 12 * u
        area_error = abs(psf.tri_surfacearea(dom) - exact_area)
        lap_error = residual.norm_inf()
        rows.append((dom.degree, n, dom.npatches, area_error, lap_error))
        print(
            f"degree={dom.degree:2d}, n={n:2d}, "
            f"area error={area_error:.3e}, lap identity error={lap_error:.3e}"
        )

    data = np.asarray(rows)
    header = "degree n npatches area_error lap_identity_error"
    np.savetxt(args.output, data, header=header, fmt=["%d", "%d", "%d", "%.16e", "%.16e"])
    print(f"wrote {args.output}")

    if args.plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6.5, 4.6))
        plt.semilogy(data[:, 0], data[:, 4], "-o", linewidth=2.0, markersize=7)
        plt.xlabel("Polynomial degree")
        plt.ylabel(r"$\|\Delta_\Gamma u + 12u\|_\infty$")
        plt.grid(True, which="both", alpha=0.35)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
