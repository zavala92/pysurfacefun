"""Small checks for scalar surface-function utilities."""

from pathlib import Path
import math
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pysurfacefun as psf


def main() -> None:
    for n in (5, 7, 9, 13):
        dom = psf.sphere(n=n, nref=0)
        area = psf.surfacearea(dom)
        print(f"n = {n:2d}, surface area = {area:.12f}, error = {abs(area - 4*math.pi):.3e}")

    dom = psf.sphere(n=7, nref=0)
    f = psf.field(lambda x, y, z: x * y * z, dom)

    print(f"mean2(x*y*z)       = {psf.mean2(f): .3e}")
    print(f"norm_inf(x*y*z)    = {psf.norm(f, 'inf'):.6e}")
    print(f"maxEst(x*y*z)      = {psf.maxEst(f): .6e}")
    print(f"minEst(x*y*z)      = {psf.minEst(f): .6e}")
    print(f"norm_inf(lap(f)+12f) = {psf.norm(psf.lap(f) + 12*f, 'inf'):.3e}")

    g = psf.resample(f, 11)
    print(f"resampled grid size = {g.domain.n} x {g.domain.n}")


if __name__ == "__main__":
    main()
