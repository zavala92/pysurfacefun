"""Solve a manufactured Laplace--Beltrami problem on S^2."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pysurfacefun as psf


def main() -> None:
    l, m = 3, 2
    dom = psf.sphere(n=7, nref=0)

    u_exact = psf.surfacefun(
        lambda x, y, z: psf.real_spherical_harmonic(l, m, x, y, z),
        dom,
    )

    rhs = -l * (l + 1) * u_exact

    L = psf.surfaceop(dom, {"lap": 1.0}, rhs)
    L.rankdef = True
    u_h = L.solve().remove_mean()

    relerr = psf.norm(u_h - u_exact.remove_mean(), "inf") / psf.norm(u_exact, "inf")
    print(f"relative L_inf error = {relerr:.3e}")

    try:
        psf.write_vtu("laplace_beltrami_sphere_solution.vtu", u_h)
        print("wrote laplace_beltrami_sphere_solution.vtu")
    except ImportError:
        print("VTU skipped: install meshio for ParaView export")


if __name__ == "__main__":
    main()
