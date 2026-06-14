"""p-refinement convergence check for the sphere Laplace--Beltrami solver."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pysurfacefun as psf


def solve_once(n: int, nref: int, l: int, m: int) -> float:
    dom = psf.sphere(n=n, nref=nref)
    u_exact = psf.field(
        lambda x, y, z: psf.real_spherical_harmonic(l, m, x, y, z),
        dom,
    )
    rhs = -l * (l + 1) * u_exact

    problem = psf.SurfaceProblem(dom, variables="u", namespace={"rhs": rhs})
    problem.add_equation("lap(u) = rhs")
    u_h = problem.build_solver(rankdef=True).solve().remove_mean()

    return psf.norm(u_h - u_exact.remove_mean(), "inf") / psf.norm(u_exact, "inf")


def main() -> None:
    l, m = 3, 2
    nref = 0
    ns = np.array([5, 7, 9, 11])
    errors = np.array([solve_once(int(n), nref, l, m) for n in ns])

    out = np.column_stack((ns, ns - 1, errors))
    np.savetxt(
        "laplace_beltrami_sphere_convergence.txt",
        out,
        header="n polynomial_degree relative_Linf_error",
        fmt=["%d", "%d", "%.16e"],
    )

    for n, err in zip(ns, errors):
        print(f"n = {n:2d}, p = {n-1:2d}, relative L_inf error = {err:.3e}")

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5.8, 4.2))
        ax.semilogy(ns - 1, errors, "-o", linewidth=2)
        ax.set_xlabel("polynomial degree")
        ax.set_ylabel(r"relative $L_\infty$ error")
        ax.grid(True, which="both", alpha=0.35)
        fig.tight_layout()
        fig.savefig("laplace_beltrami_sphere_convergence.png", dpi=250)
        print("wrote laplace_beltrami_sphere_convergence.png")
    except ImportError:
        print("plot skipped: install matplotlib")

    print(f"wrote {Path('laplace_beltrami_sphere_convergence.txt').resolve()}")


if __name__ == "__main__":
    main()
