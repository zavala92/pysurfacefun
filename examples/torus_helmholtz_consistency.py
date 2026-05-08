"""Self-consistency solve for a Helmholtz-type PDE on the Fourier torus."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pysurfacefun as psf


def solve_once(n: int) -> float:
    dom = psf.torus(n=n, nu=2, nv=4)
    exact = psf.surfacefun(lambda x, y, z: x + y + z, dom)

    alpha = 20.0
    rhs = psf.lap(exact) + alpha * exact

    op = psf.surfaceop(dom, {"lap": 1.0, "c": alpha}, rhs)
    numerical = op.solve()

    return psf.norm(numerical - exact, "inf") / psf.norm(exact, "inf")


def main() -> None:
    print("Solve (Delta_Gamma + alpha)u = f on the Fourier torus")
    print("where f is generated from a smooth exact function u = x+y+z.")
    for n in (7, 9, 11):
        print(f"n = {n:2d}, relative L_inf error = {solve_once(n):.3e}")


if __name__ == "__main__":
    main()
