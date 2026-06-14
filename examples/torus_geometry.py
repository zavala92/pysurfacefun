"""Geometry checks for the Fourier torus."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pysurfacefun as psf


def main() -> None:
    print("Fourier torus geometry")
    print("----------------------")
    for n in (5, 7, 9, 11):
        dom = psf.torus(n=n, nu=2, nv=4)
        print(f"n = {n:2d}, patches = {dom.npatches:2d}, surface area = {psf.surfacearea(dom):.12f}")

    dom = psf.torus(n=9, nu=2, nv=4)
    f = psf.field(lambda x, y, z: x + y + z, dom)
    print(f"mean2(x+y+z) = {psf.mean2(f): .6e}")
    print(f"norm_inf(x+y+z) = {psf.norm(f, 'inf'):.6e}")

    try:
        psf.plot_surface(f, title="Fourier torus: x+y+z")
    except ImportError:
        print("plot skipped: install matplotlib")


if __name__ == "__main__":
    main()
