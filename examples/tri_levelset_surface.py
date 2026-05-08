"""Build a triangular high-order surface from a coarse mesh and a level set."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pysurfacefun as psf


def octahedron_mesh(scale: float = 0.82):
    vertices = scale * np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    faces = np.array(
        [
            [0, 2, 4],
            [2, 1, 4],
            [1, 3, 4],
            [3, 0, 4],
            [2, 0, 5],
            [1, 2, 5],
            [3, 1, 5],
            [0, 3, 5],
        ]
    )
    return vertices, faces


def main() -> None:
    vertices, faces = octahedron_mesh()

    phi = lambda p: p[0] ** 2 + p[1] ** 2 + p[2] ** 2 - 1.0
    grad_phi = lambda p: np.array([2.0 * p[0], 2.0 * p[1], 2.0 * p[2]])

    dom = psf.LevelSetSurface((vertices, faces), phi, grad_phi, n=9)
    u = psf.tri_surfacefun(lambda x, y, z: x * y * z, dom)

    residual = max(
        np.max(np.abs(x * x + y * y + z * z - 1.0))
        for x, y, z in zip(dom.x, dom.y, dom.z)
    )
    print(f"patches = {dom.npatches}")
    print(f"degree = {dom.degree}")
    print(f"surface area = {psf.tri_surfacearea(dom):.12f}")
    print(f"max |phi| = {residual:.3e}")

    psf.write_tri_vtu("tri_levelset_sphere.vtu", u, point_name="u")
    print("wrote tri_levelset_sphere.vtu")

    try:
        psf.plot_tri_surface(u, title="Level-set projected triangular sphere")
    except Exception as exc:
        print("plot skipped:", exc)


if __name__ == "__main__":
    main()
