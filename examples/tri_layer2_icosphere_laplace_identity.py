"""Layer 2: triangular icosphere geometry and surface differentiation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pysurfacefun as psf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=9, help="nodes per triangle edge")
    parser.add_argument("--nref", type=int, default=1, help="icosphere refinement level")
    parser.add_argument("--plot", action="store_true", help="plot the sampled solution")
    parser.add_argument("--vtu", default="", help="optional VTU output path")
    args = parser.parse_args()

    dom = psf.icosphere_tri(n=args.n, nref=args.nref)
    u = psf.tri_surfacefun(lambda x, y, z: x * y * z, dom)
    residual = psf.tri_lap(u) + 12 * u

    print(f"patches = {dom.npatches}")
    print(f"degree = {dom.degree}")
    print(f"nodes per patch = {dom.x[0].size}")
    print(f"surface area = {psf.tri_surfacearea(dom):.12f}")
    print(f"Delta_Gamma(xyz) + 12 xyz error = {residual.norm_inf():.3e}")

    if args.vtu:
        psf.write_tri_vtu(args.vtu, u)
        print(f"wrote {args.vtu}")

    if args.plot:
        psf.plot_tri_surface(u, title="Triangular icosphere: x y z")


if __name__ == "__main__":
    main()
