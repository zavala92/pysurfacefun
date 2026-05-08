"""Layer 1: Recursive triangle nodes and Koornwinder differentiation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pysurfacefun as psf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=6, help="nodes per triangle edge")
    parser.add_argument("--plot", action="store_true", help="plot the reference nodes")
    args = parser.parse_args()

    degree = args.n - 1
    x, y = psf.tri_reference_nodes(args.n)
    Du, Dv, K = psf.tri_strong_diffmat(degree, x, y, basis="pkd")

    f = x**3 * y + 2 * x * y**2 + y
    fx = 3 * x**2 * y + 2 * y**2
    fy = x**3 + 4 * x * y + 1

    err_x = np.max(np.abs(Du @ f - fx))
    err_y = np.max(np.abs(Dv @ f - fy))

    print(f"degree = {degree}")
    print(f"nodes  = {x.size}")
    print(f"cond(K) = {np.linalg.cond(K):.3e}")
    print(f"reference derivative error dx = {err_x:.3e}")
    print(f"reference derivative error dy = {err_y:.3e}")

    if args.plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(5, 5))
        plt.plot([0, 1, 0, 0], [0, 0, 1, 0], "k-", linewidth=1.2)
        plt.scatter(x, y, s=42, c=np.arange(x.size), cmap="turbo")
        plt.gca().set_aspect("equal")
        plt.axis("off")
        plt.title(f"Recursive Chebyshev triangle nodes, degree {degree}")
        plt.show()


if __name__ == "__main__":
    main()
