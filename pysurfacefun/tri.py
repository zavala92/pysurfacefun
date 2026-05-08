"""
Triangular-patch utilities for pysurfacefun.

The triangular formulation and fast direct solver framework are introduced in:
A High-Order Fast Direct Solver for Surface PDEs on Triangles
https://arxiv.org/pdf/2604.03097

The reference triangle is the unit simplex with vertices ``(0, 0)``,
``(1, 0)``, and ``(0, 1)``.  Nodes are RecursiveNodes-style simplex nodes
generated from Chebyshev points of the second kind by default.  The polynomial
basis is the Proriol-Koornwinder-Dubiner basis used for strong-form
differentiation on a triangle.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import struct
from typing import Callable, Iterable
import zlib

import numpy as np

from .core import (
    Parent,
    Patch,
    SurfaceMesh,
    barymat,
    chebpts,
    chebpts2,
    default_merge_idx,
    get_work_array,
    inverse_dense,
    merge_patches,
    parse_pdo,
    quadwts,
    rowscale,
    solve_with_inverse,
)
from .recursivenodes_polynomials import proriolkoornwinderdubinervandermondegrad


Array = np.ndarray


def shifted_lobatto_nodes(degree: int) -> Array:
    """Shifted Lobatto-Gauss-Legendre nodes on ``[0, 1]``."""
    if degree < 0:
        raise ValueError("degree must be nonnegative")
    if degree == 0:
        return np.array([0.5])
    if degree == 1:
        return np.array([0.0, 1.0])

    poly = np.polynomial.legendre.Legendre.basis(degree)
    roots = np.sort(poly.deriv().roots())
    nodes = np.concatenate(([-1.0], roots, [1.0]))
    return 0.5 * (nodes + 1.0)


def _node_family(max_degree: int, family: str) -> list[Array]:
    return [node_family_points(degree + 1, family) for degree in range(max_degree + 1)]


def _canonical_family_name(family: str) -> str:
    family = family.lower()
    aliases = {
        "cheb": "cheb2",
        "lgc": "cheb2",
        "chebyshev2": "cheb2",
        "gc": "cheb1",
        "chebyshev1": "cheb1",
        "leg": "gl",
        "legendre": "gl",
        "lob": "lgl",
        "lobatto": "lgl",
        "uni": "equi",
        "uniform": "equi",
        "lin": "equi",
        "linspace": "equi",
        "equispaced": "equi",
    }
    family = aliases.get(family, family)
    if family not in {"cheb2", "cheb1", "gl", "lgl", "equi"}:
        raise ValueError("unknown triangle node family")
    return family


def node_family_points(count: int, family: str = "cheb2") -> Array:
    """One-dimensional node family on ``[0, 1]`` used by RecursiveNodes."""
    if count <= 0:
        return np.empty(0)
    family = _canonical_family_name(family)
    if family == "cheb2":
        return chebpts(count, 2, (0.0, 1.0))
    if family == "cheb1":
        return chebpts(count, 1, (0.0, 1.0))
    if family == "gl":
        x, _ = np.polynomial.legendre.leggauss(count)
        return 0.5 * (x + 1.0)
    if family == "lgl":
        return shifted_lobatto_nodes(count - 1)
    if count == 1:
        return np.array([0.5])
    return np.linspace(0.0, 1.0, count)


def _multi_indices(length: int, total: int) -> Iterable[tuple[int, ...]]:
    if length == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _multi_indices(length - 1, total - first):
            yield (first,) + rest


def _insert_zero(v: Array, index: int) -> Array:
    out = np.zeros(v.size + 1)
    out[:index] = v[:index]
    out[index + 1 :] = v[index:]
    return out


def _recursive_barycentric(dim: int, degree: int, alpha: tuple[int, ...], family: list[Array]) -> Array:
    x_degree = family[degree]
    if dim == 1:
        return x_degree[list(alpha)]

    b = np.zeros(dim + 1)
    weight = 0.0
    for i in range(dim + 1):
        alpha_without_i = alpha[:i] + alpha[i + 1 :]
        degree_without_i = degree - alpha[i]
        w = x_degree[degree_without_i]
        br = _recursive_barycentric(dim - 1, degree_without_i, alpha_without_i, family)
        b += w * _insert_zero(br, i)
        weight += w
    return b / weight


def recursive_nodes(
    dim: int,
    degree: int,
    family: str = "cheb2",
    domain: str = "barycentric",
    interior: int = 0,
) -> Array:
    """
    Recursive interpolation nodes on a simplex.

    This follows the definition in the RecursiveNodes documentation.  For
    ``dim=2`` and ``domain='unit'`` the returned columns are the coordinates
    ``(x, y)`` on the unit reference triangle.
    """
    if dim < 1:
        raise ValueError("dim must be at least 1")
    if degree < 0:
        raise ValueError("degree must be nonnegative")
    if interior < 0:
        raise ValueError("interior must be nonnegative")

    family_nodes = _node_family(degree, family)
    bary = []
    for alpha in _multi_indices(dim + 1, degree):
        if interior and any(a < interior for a in alpha):
            continue
        bary.append(_recursive_barycentric(dim, degree, alpha, family_nodes))
    bary_arr = np.asarray(bary)

    domain = domain.lower()
    if domain == "barycentric":
        return bary_arr
    if domain == "unit":
        return bary_arr[:, :dim]
    if domain == "equilateral" and dim == 2:
        x = bary_arr[:, 0] + 0.5 * bary_arr[:, 1]
        y = (np.sqrt(3.0) / 2.0) * bary_arr[:, 1]
        return np.column_stack((x, y))
    raise ValueError("domain must be 'barycentric', 'unit', or 'equilateral'")


def tri_reference_nodes(n: int, family: str = "cheb2") -> tuple[Array, Array]:
    """
    Reference triangle nodes with ``n`` points on each edge.

    The polynomial degree is ``n - 1`` and the total number of nodes is
    ``n * (n + 1) / 2``.
    """
    if n < 1:
        raise ValueError("n must be positive")
    nodes = recursive_nodes(2, n - 1, family=family, domain="unit")
    return nodes[:, 0], nodes[:, 1]


def koornwinder_pkd(degree: int, x: Array | None = None, y: Array | None = None) -> tuple[Array, Array, Array]:
    """
    RecursiveNodes PKD Vandermonde matrix and unit-triangle derivatives.

    The underlying RecursiveNodes routines evaluate orthonormal
    Proriol-Koornwinder-Dubiner polynomials on the biunit triangle with
    coordinates ``(r, s)``. This package uses the unit triangle ``(x, y)``, so
    this function applies

    ``r = 2*x - 1,  s = 2*y - 1``

    and multiplies the returned gradients by ``2``.  Thus ``Kx`` and ``Ky``
    differentiate with respect to the unit reference coordinates.
    """
    if degree < 0:
        raise ValueError("degree must be nonnegative")
    if x is None or y is None:
        x, y = tri_reference_nodes(degree + 1)

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    xy_biunit = np.column_stack((2.0 * x - 1.0, 2.0 * y - 1.0))
    K, grad = proriolkoornwinderdubinervandermondegrad(2, degree, xy_biunit, both=True)
    Kx = 2.0 * grad[:, :, 0]
    Ky = 2.0 * grad[:, :, 1]
    return K, Kx, Ky


def tri_strong_diffmat(
    degree: int,
    x: Array | None = None,
    y: Array | None = None,
    basis: str = "pkd",
) -> tuple[Array, Array, Array]:
    """
    Strong-form reference-triangle differentiation matrices.

    ``Du @ f`` and ``Dv @ f`` differentiate nodal values with respect to the
    reference coordinates ``u=x`` and ``v=y``.
    """
    if x is None or y is None:
        x, y = tri_reference_nodes(degree + 1)
    basis = basis.lower()
    if basis not in {"pkd", "recursive", "recursivenodes"}:
        raise ValueError("only the RecursiveNodes PKD basis is supported")
    K, Ku, Kv = koornwinder_pkd(degree, x, y)
    col_scale = np.linalg.norm(K, axis=0)
    col_scale[col_scale == 0.0] = 1.0
    Ks = K / col_scale
    Kus = Ku / col_scale
    Kvs = Kv / col_scale
    invKs = np.linalg.solve(Ks, np.eye(Ks.shape[0]))
    return Kus @ invKs, Kvs @ invKs, K


def trilattice(n: int) -> Array:
    """Triangle connectivity for the regular lattice ordering."""
    if n < 2:
        return np.zeros((0, 3), dtype=int)

    triangles: list[list[int]] = []
    colstart = 0
    for i in range(n - 1):
        h = n - i - 1
        triangles.append([colstart, colstart + 1, colstart + 1 + h])
        for ss in range(colstart + 1, colstart + h):
            triangles.append([ss, ss + h, ss + h + 1])
        for ss in range(colstart + 1, colstart + h):
            triangles.append([ss, ss + 1, ss + h + 1])
        colstart += h + 1
    return np.asarray(triangles, dtype=int)


@lru_cache(maxsize=64)
def reference_triangle_quadrature_weights(n: int, family: str = "cheb2") -> Array:
    """Nodal quadrature weights on the reference triangle."""
    degree = n - 1
    x, y = tri_reference_nodes(n, family=family)
    K, _, _ = koornwinder_pkd(degree, x, y)
    basis_integrals = np.zeros(K.shape[1])
    basis_integrals[0] = np.sqrt(2.0) / 4.0
    return np.linalg.solve(K.T, basis_integrals)


def _triangular_n_from_count(count: int) -> int:
    n = int((np.sqrt(8 * count + 1) - 1) / 2)
    if n * (n + 1) // 2 != count:
        raise ValueError("triangle patches must have n*(n+1)/2 nodes")
    return n


@dataclass
class TriangleSurfaceMesh:
    """High-order triangular surface patch mesh."""

    x: list[Array]
    y: list[Array]
    z: list[Array]
    family: str = "cheb2"

    def __post_init__(self) -> None:
        self.family = _canonical_family_name(self.family)
        self.x = [np.asarray(v, dtype=float).ravel() for v in self.x]
        self.y = [np.asarray(v, dtype=float).ravel() for v in self.y]
        self.z = [np.asarray(v, dtype=float).ravel() for v in self.z]
        if not (len(self.x) == len(self.y) == len(self.z)):
            raise ValueError("x, y, z must have the same number of patches")

        self.n = _triangular_n_from_count(self.x[0].size)
        self.degree = self.n - 1
        self.u, self.v = tri_reference_nodes(self.n, family=self.family)
        self.Du, self.Dv, self.K = tri_strong_diffmat(self.degree, self.u, self.v, basis="pkd")
        self.triangles = trilattice(self.n)
        self.ref_weights = reference_triangle_quadrature_weights(self.n, self.family)

        self.xu: list[Array] = []
        self.xv: list[Array] = []
        self.yu: list[Array] = []
        self.yv: list[Array] = []
        self.zu: list[Array] = []
        self.zv: list[Array] = []

        self.ux: list[Array] = []
        self.vx: list[Array] = []
        self.uy: list[Array] = []
        self.vy: list[Array] = []
        self.uz: list[Array] = []
        self.vz: list[Array] = []

        self.E: list[Array] = []
        self.F: list[Array] = []
        self.G: list[Array] = []
        self.J: list[Array] = []
        self.singular: list[bool] = []

        for x, y, z in zip(self.x, self.y, self.z):
            xu = self.Du @ x
            xv = self.Dv @ x
            yu = self.Du @ y
            yv = self.Dv @ y
            zu = self.Du @ z
            zv = self.Dv @ z

            E = xu * xu + yu * yu + zu * zu
            G = xv * xv + yv * yv + zv * zv
            F = xu * xv + yu * yv + zu * zv
            J = E * G - F * F

            scl = float(np.max(np.abs(G * xu - F * xv)))
            singular = bool(np.any(np.abs(J) < 1e-10 * max(scl, 1.0)))

            self.xu.append(xu)
            self.xv.append(xv)
            self.yu.append(yu)
            self.yv.append(yv)
            self.zu.append(zu)
            self.zv.append(zv)
            if singular:
                self.ux.append(G * xu - F * xv)
                self.vx.append(E * xv - F * xu)
                self.uy.append(G * yu - F * yv)
                self.vy.append(E * yv - F * yu)
                self.uz.append(G * zu - F * zv)
                self.vz.append(E * zv - F * zu)
            else:
                self.ux.append((G * xu - F * xv) / J)
                self.vx.append((E * xv - F * xu) / J)
                self.uy.append((G * yu - F * yv) / J)
                self.vy.append((E * yv - F * yu) / J)
                self.uz.append((G * zu - F * zv) / J)
                self.vz.append((E * zv - F * zu) / J)
            self.E.append(E)
            self.F.append(F)
            self.G.append(G)
            self.J.append(J)
            self.singular.append(singular)

    @property
    def npatches(self) -> int:
        return len(self.x)

    @staticmethod
    def icosphere(n: int, nref: int = 0, family: str = "cheb2") -> "TriangleSurfaceMesh":
        vertices, faces = _icosahedron()
        for _ in range(nref):
            vertices, faces = _subdivide_icosphere(vertices, faces)
        faces = _orient_faces_outward(vertices, faces)

        u, v = tri_reference_nodes(n, family=family)
        xpatches: list[Array] = []
        ypatches: list[Array] = []
        zpatches: list[Array] = []

        for tri in faces:
            p0, p1, p2 = vertices[tri]
            xyz = (1.0 - u - v)[:, None] * p0 + u[:, None] * p1 + v[:, None] * p2
            xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)
            xpatches.append(xyz[:, 0])
            ypatches.append(xyz[:, 1])
            zpatches.append(xyz[:, 2])

        return TriangleSurfaceMesh(xpatches, ypatches, zpatches, family=family)


def levelset_surface_tri(
    vertices: Array,
    faces: Array,
    phi: Callable,
    grad_phi: Callable,
    n: int,
    family: str = "cheb2",
    index_base: str | int = "auto",
    max_iter: int = 30,
    tol: float = 1.0e-13,
    damping: float = 1.0,
) -> TriangleSurfaceMesh:
    """
    Build high-order triangular patches by projecting a coarse mesh to ``phi=0``.

    Parameters
    ----------
    vertices, faces:
        Coarse triangular mesh.  ``vertices`` has shape ``(nv, 3)`` and
        ``faces`` has shape ``(nf, 3)``.  Faces may be zero-based or one-based
        when ``index_base='auto'``.
    phi, grad_phi:
        Level-set function and gradient.  Each may accept either one point
        ``p`` with shape ``(3,)`` or three scalars ``x, y, z``.
    n:
        Number of high-order nodes on each triangle edge.

    Returns
    -------
    TriangleSurfaceMesh
        Curved triangular patch mesh with every node projected to the smooth
        implicit surface.
    """
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (nv, 3)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape (nf, 3)")
    if n < 2:
        raise ValueError("n must be at least 2")

    faces0 = normalize_face_indices(faces, vertices.shape[0], index_base)
    u, v = tri_reference_nodes(n, family=family)
    weights = np.column_stack((1.0 - u - v, u, v))

    xpatches: list[Array] = []
    ypatches: list[Array] = []
    zpatches: list[Array] = []
    for face in faces0:
        flat_points = weights @ vertices[face, :]
        projected = project_to_levelset(
            flat_points,
            phi,
            grad_phi,
            max_iter=max_iter,
            tol=tol,
            damping=damping,
        )
        xpatches.append(projected[:, 0])
        ypatches.append(projected[:, 1])
        zpatches.append(projected[:, 2])

    return TriangleSurfaceMesh(xpatches, ypatches, zpatches, family=family)


def levelset_surface_quad(
    vertices: Array,
    cells: Array,
    phi: Callable,
    grad_phi: Callable,
    n: int,
    index_base: str | int = "auto",
    max_iter: int = 30,
    tol: float = 1.0e-13,
    damping: float = 1.0,
) -> SurfaceMesh:
    """
    Build tensor-product Chebyshev patches by projecting quad cells to ``phi=0``.

    Each coarse quad is first mapped bilinearly from ``[0,1]^2`` and then every
    Chebyshev node is projected to the implicit surface.
    """
    vertices = np.asarray(vertices, dtype=float)
    cells = np.asarray(cells, dtype=int)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (nv, 3)")
    if cells.ndim != 2 or cells.shape[1] != 4:
        raise ValueError("quad cells must have shape (nq, 4)")
    if n < 2:
        raise ValueError("n must be at least 2")

    cells0 = normalize_face_indices(cells, vertices.shape[0], index_base)
    u, v = chebpts2(n, n, [0.0, 1.0, 0.0, 1.0])
    weights = np.stack(
        (
            (1.0 - u) * (1.0 - v),
            u * (1.0 - v),
            u * v,
            (1.0 - u) * v,
        ),
        axis=-1,
    )

    xpatches: list[Array] = []
    ypatches: list[Array] = []
    zpatches: list[Array] = []
    for cell in cells0:
        flat_points = np.einsum("ijk,kl->ijl", weights, vertices[cell, :])
        projected = project_to_levelset(
            flat_points,
            phi,
            grad_phi,
            max_iter=max_iter,
            tol=tol,
            damping=damping,
        )
        xpatches.append(projected[:, :, 0])
        ypatches.append(projected[:, :, 1])
        zpatches.append(projected[:, :, 2])

    return SurfaceMesh(xpatches, ypatches, zpatches)


def levelset_surface(
    mesh,
    phi: Callable,
    grad_phi: Callable,
    n: int,
    *,
    family: str = "cheb2",
    cell_type: str = "auto",
    vertex_name: str | None = None,
    face_name: str | None = None,
    mesh_name: str | None = None,
    index_base: str | int = "auto",
    nref: int = 0,
    orient_outward: bool = False,
    orientation_center: Array | None = None,
    max_iter: int = 30,
    tol: float = 1.0e-13,
    damping: float = 1.0,
) -> TriangleSurfaceMesh | SurfaceMesh:
    """
    Level-set surface constructor for triangular or quadrilateral coarse meshes.

    ``mesh`` may be a MAT-file path, a ``(vertices, cells)`` tuple, a
    dictionary/struct-like loaded mesh, or an object with vertex/face fields.
    Three-node cells create a ``TriangleSurfaceMesh``; four-node cells create
    a quadrilateral ``SurfaceMesh``. ``nref`` refines the coarse mesh
    connectivity before high-order nodes are mapped and projected to the level
    set.
    """
    vertices, cells = surface_mesh_arrays(
        mesh,
        vertex_name=vertex_name,
        face_name=face_name,
        mesh_name=mesh_name,
        index_base=index_base,
    )
    cell_type = _infer_levelset_cell_type(cells, cell_type)

    if cell_type == "tri":
        tri_cells = cells if cells.shape[1] == 3 else triangulate_faces(cells)
        if nref:
            vertices, tri_cells = refine_tri_mesh(vertices, tri_cells, nref=nref)
        if orient_outward:
            tri_cells = orient_tri_faces_outward(vertices, tri_cells, center=orientation_center)
        return levelset_surface_tri(
            vertices,
            tri_cells,
            phi,
            grad_phi,
            n,
            family=family,
            index_base=0,
            max_iter=max_iter,
            tol=tol,
            damping=damping,
        )

    if cells.shape[1] != 4:
        raise ValueError("quad level-set surfaces require four-node cells")
    if nref:
        vertices, cells = refine_quad_mesh(vertices, cells, nref=nref)
    return levelset_surface_quad(
        vertices,
        cells,
        phi,
        grad_phi,
        n,
        index_base=0,
        max_iter=max_iter,
        tol=tol,
        damping=damping,
    )


def levelset_surface_tri_from_mat(
    filename: str | Path,
    phi: Callable,
    grad_phi: Callable,
    n: int,
    *,
    vertex_name: str | None = None,
    face_name: str | None = None,
    mesh_name: str | None = None,
    family: str = "cheb2",
    index_base: str | int = "auto",
    nref: int = 0,
    triangulate: bool = True,
    orient_outward: bool = False,
    orientation_center: Array | None = None,
    max_iter: int = 30,
    tol: float = 1.0e-13,
    damping: float = 1.0,
) -> TriangleSurfaceMesh:
    """
    Build a level-set triangular surface directly from a MAT-file mesh.

    The mesh file may store arrays as ``vertices/faces``, ``xs/surfs``,
    ``V/F``, or inside a struct. Use ``vertex_name`` and ``face_name`` when
    the file uses different variable names.
    """
    vertices, faces = load_mat_tri_mesh(
        filename,
        vertex_name=vertex_name,
        face_name=face_name,
        mesh_name=mesh_name,
        index_base=index_base,
    )
    if triangulate:
        faces = triangulate_faces(
            faces,
            vertices=vertices if orient_outward else None,
            orient_outward=orient_outward,
            center=orientation_center,
        )
    if nref:
        vertices, faces = refine_tri_mesh(vertices, faces, nref=nref)
        if orient_outward:
            faces = orient_tri_faces_outward(vertices, faces, center=orientation_center)
    return levelset_surface_tri(
        vertices,
        faces,
        phi,
        grad_phi,
        n,
        family=family,
        index_base=0,
        max_iter=max_iter,
        tol=tol,
        damping=damping,
    )


def LevelSetSurface(
    mesh: str | Path | Array,
    faces_or_phi=None,
    phi: Callable | None = None,
    grad_phi: Callable | None = None,
    n: int | None = None,
    **kwargs,
) -> TriangleSurfaceMesh | SurfaceMesh:
    """
    Notebook-friendly level-set surface constructor.

    Examples
    --------
    ``LevelSetSurface("mesh.mat", phi, grad_phi, n=12)``

    ``LevelSetSurface((vertices, cells), phi, grad_phi, n=12)``

    ``LevelSetSurface(vertices, faces, phi, grad_phi, n=12)``
    """
    if callable(faces_or_phi):
        levelset_phi = faces_or_phi
        levelset_grad = phi
        degree_n = n
        if degree_n is None and grad_phi is not None and not callable(grad_phi):
            degree_n = int(grad_phi)
        if levelset_phi is None or levelset_grad is None or degree_n is None:
            raise ValueError("use LevelSetSurface(mesh, phi, grad_phi, n=...)")
        return levelset_surface(mesh, levelset_phi, levelset_grad, degree_n, **kwargs)

    if faces_or_phi is None or phi is None or grad_phi is None or n is None:
        raise ValueError(
            "use LevelSetSurface(mesh, phi, grad_phi, n=...) or "
            "LevelSetSurface(vertices, faces, phi, grad_phi, n=...)"
        )
    return levelset_surface((mesh, faces_or_phi), phi, grad_phi, n, **kwargs)


def load_mat_tri_mesh(
    filename: str | Path,
    *,
    vertex_name: str | None = None,
    face_name: str | None = None,
    mesh_name: str | None = None,
    index_base: str | int = "auto",
) -> tuple[Array, Array]:
    """Backward-compatible alias for ``load_mat_surface_mesh``."""
    return load_mat_surface_mesh(
        filename,
        vertex_name=vertex_name,
        face_name=face_name,
        mesh_name=mesh_name,
        index_base=index_base,
    )


def load_mat_surface_mesh(
    filename: str | Path,
    *,
    vertex_name: str | None = None,
    face_name: str | None = None,
    mesh_name: str | None = None,
    index_base: str | int = "auto",
) -> tuple[Array, Array]:
    """
    Extract vertices and surface cells from a MAT-file.

    The returned cell array is zero-based for direct Python use.  Supported
    common names include ``vertices/faces``, ``xs/surfs``, ``V/F``,
    ``nodes/elements``, and ``p/t``.
    """
    data = _load_mat_file(filename)
    vertices, cells = extract_tri_mesh_arrays(
        data,
        vertex_name=vertex_name,
        face_name=face_name,
        mesh_name=mesh_name,
    )
    cells = normalize_face_indices(cells, vertices.shape[0], index_base=index_base)
    return vertices, cells


def surface_mesh_arrays(
    mesh,
    *,
    vertex_name: str | None = None,
    face_name: str | None = None,
    mesh_name: str | None = None,
    index_base: str | int = "auto",
) -> tuple[Array, Array]:
    """Return ``(vertices, cells)`` from a path, tuple, dict, or mesh-like object."""
    if isinstance(mesh, (str, Path)):
        return load_mat_surface_mesh(
            mesh,
            vertex_name=vertex_name,
            face_name=face_name,
            mesh_name=mesh_name,
            index_base=index_base,
        )

    if isinstance(mesh, (tuple, list)) and len(mesh) == 2:
        vertices = _coerce_vertices(mesh[0], "vertices")
        cells = _coerce_faces(mesh[1], "faces")
        cells = normalize_face_indices(cells, vertices.shape[0], index_base=index_base)
        return vertices, cells

    vertices, cells = extract_tri_mesh_arrays(
        mesh,
        vertex_name=vertex_name,
        face_name=face_name,
        mesh_name=mesh_name,
    )
    cells = normalize_face_indices(cells, vertices.shape[0], index_base=index_base)
    return vertices, cells


def _infer_levelset_cell_type(cells: Array, cell_type: str = "auto") -> str:
    cell_type = cell_type.lower()
    if cell_type in {"triangle", "triangles"}:
        cell_type = "tri"
    if cell_type in {"quadrilateral", "quadrilaterals", "quad", "quads"}:
        cell_type = "quad"
    if cell_type not in {"auto", "tri", "quad"}:
        raise ValueError("cell_type must be 'auto', 'tri', or 'quad'")
    if cell_type != "auto":
        return cell_type
    if cells.shape[1] == 3:
        return "tri"
    if cells.shape[1] == 4:
        return "quad"
    return "tri"


def triangulate_faces(
    faces: Array,
    *,
    vertices: Array | None = None,
    orient_outward: bool = False,
    center: Array | None = None,
) -> Array:
    """
    Convert triangular or polygonal surface cells to triangle connectivity.

    Rows with three entries are kept.  Rows with more entries are split by a
    fan from the first vertex.  When ``orient_outward=True``, the triangles are
    flipped so their normals point away from ``center``.  If ``center`` is not
    supplied, the origin is used.
    """
    faces = np.asarray(faces, dtype=int)
    if faces.ndim != 2 or faces.shape[1] < 3:
        raise ValueError("faces must have shape (nf, m) with m >= 3")

    triangles: list[list[int]] = []
    for row in faces:
        valid = row[row >= 0]
        if valid.size < 3:
            continue
        for j in range(1, valid.size - 1):
            triangles.append([int(valid[0]), int(valid[j]), int(valid[j + 1])])
    out = np.asarray(triangles, dtype=int)

    if orient_outward:
        if vertices is None:
            raise ValueError("vertices are required when orient_outward=True")
        out = orient_tri_faces_outward(vertices, out, center=center)
    return out


def orient_tri_faces_outward(vertices: Array, faces: Array, center: Array | None = None) -> Array:
    """
    Orient triangular faces so normals point away from ``center``.

    This is intended for closed, star-shaped surfaces such as the sphere mesh
    used in the triangular notebook.
    """
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int)
    if center is None:
        center_arr = np.zeros(3)
    else:
        center_arr = np.asarray(center, dtype=float).reshape(3)

    out = faces.copy()
    for k, (a, b, c) in enumerate(out):
        p0, p1, p2 = vertices[[a, b, c]]
        normal = np.cross(p1 - p0, p2 - p0)
        face_center = (p0 + p1 + p2) / 3.0
        if np.dot(normal, face_center - center_arr) < 0.0:
            out[k, 1], out[k, 2] = out[k, 2], out[k, 1]
    return out


def refine_surface_mesh(
    vertices: Array,
    cells: Array,
    nref: int = 1,
    cell_type: str = "auto",
) -> tuple[Array, Array]:
    """
    Uniformly refine triangular or quadrilateral surface connectivity.

    This refinement is purely geometric and linear.  It is intended to be used
    before level-set projection: new edge/center vertices are introduced in the
    coarse mesh, and the subsequent ``LevelSetSurface`` call projects the
    high-order patch nodes to the smooth surface.
    """
    cell_type = _infer_levelset_cell_type(np.asarray(cells), cell_type)
    if cell_type == "tri":
        tri_cells = cells if np.asarray(cells).shape[1] == 3 else triangulate_faces(cells)
        return refine_tri_mesh(vertices, tri_cells, nref=nref)
    return refine_quad_mesh(vertices, cells, nref=nref)


def refine_tri_mesh(vertices: Array, faces: Array, nref: int = 1) -> tuple[Array, Array]:
    """Split each triangle into four triangles ``nref`` times."""
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int)
    if nref < 0:
        raise ValueError("nref must be nonnegative")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("triangular refinement requires faces with shape (nf, 3)")

    verts = vertices.copy()
    tris = faces.copy()
    for _ in range(nref):
        vert_list = [v.copy() for v in verts]
        edge_cache: dict[tuple[int, int], int] = {}

        def midpoint(i: int, j: int) -> int:
            key = tuple(sorted((int(i), int(j))))
            if key not in edge_cache:
                edge_cache[key] = len(vert_list)
                vert_list.append(0.5 * (verts[key[0]] + verts[key[1]]))
            return edge_cache[key]

        refined: list[list[int]] = []
        for a, b, c in tris:
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            refined.extend(
                (
                    [int(a), ab, ca],
                    [ab, int(b), bc],
                    [ca, bc, int(c)],
                    [ab, bc, ca],
                )
            )
        verts = np.asarray(vert_list, dtype=float)
        tris = np.asarray(refined, dtype=int)
    return verts, tris


def refine_quad_mesh(vertices: Array, cells: Array, nref: int = 1) -> tuple[Array, Array]:
    """Split each quadrilateral into four quadrilaterals ``nref`` times."""
    vertices = np.asarray(vertices, dtype=float)
    cells = np.asarray(cells, dtype=int)
    if nref < 0:
        raise ValueError("nref must be nonnegative")
    if cells.ndim != 2 or cells.shape[1] != 4:
        raise ValueError("quadrilateral refinement requires cells with shape (nq, 4)")

    verts = vertices.copy()
    quads = cells.copy()
    for _ in range(nref):
        vert_list = [v.copy() for v in verts]
        edge_cache: dict[tuple[int, int], int] = {}

        def midpoint(i: int, j: int) -> int:
            key = tuple(sorted((int(i), int(j))))
            if key not in edge_cache:
                edge_cache[key] = len(vert_list)
                vert_list.append(0.5 * (verts[key[0]] + verts[key[1]]))
            return edge_cache[key]

        refined: list[list[int]] = []
        for a, b, c, d in quads:
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            cd = midpoint(c, d)
            da = midpoint(d, a)
            center = len(vert_list)
            vert_list.append(0.25 * (verts[int(a)] + verts[int(b)] + verts[int(c)] + verts[int(d)]))
            refined.extend(
                (
                    [int(a), ab, center, da],
                    [ab, int(b), bc, center],
                    [center, bc, int(c), cd],
                    [da, center, cd, int(d)],
                )
            )
        verts = np.asarray(vert_list, dtype=float)
        quads = np.asarray(refined, dtype=int)
    return verts, quads


def extract_tri_mesh_arrays(
    data,
    *,
    vertex_name: str | None = None,
    face_name: str | None = None,
    mesh_name: str | None = None,
) -> tuple[Array, Array]:
    """Extract vertex and face arrays from a loaded mesh dictionary/struct."""
    if mesh_name is not None:
        data = _mat_get(data, mesh_name)

    if vertex_name is not None:
        vertices = _coerce_vertices(_mat_get(data, vertex_name), vertex_name)
    else:
        vertices = _find_mat_array(data, _VERTEX_FIELD_NAMES, _coerce_vertices, "vertices")

    if face_name is not None:
        faces = _coerce_faces(_mat_get(data, face_name), face_name)
    else:
        faces = _find_mat_array(data, _FACE_FIELD_NAMES, _coerce_faces, "faces")

    return vertices, faces


_VERTEX_FIELD_NAMES = (
    "vertices",
    "verts",
    "xs",
    "V",
    "v",
    "nodes",
    "Nodes",
    "points",
    "Points",
    "p",
)
_FACE_FIELD_NAMES = (
    "faces",
    "surfs",
    "F",
    "f",
    "tri",
    "tris",
    "triangles",
    "Triangles",
    "elements",
    "Elements",
    "cells",
    "Cells",
    "t",
)


def _load_mat_file(filename: str | Path):
    filename = Path(filename)
    try:
        from scipy.io import loadmat
    except ImportError as scipy_error:
        loadmat = None
    else:
        try:
            return loadmat(filename, squeeze_me=True, struct_as_record=False)
        except (NotImplementedError, ValueError, OSError) as scipy_error:
            pass

    try:
        return _load_mat_v5_numeric(filename)
    except (OSError, ValueError, zlib.error):
        pass

    try:
        import h5py
    except ImportError as h5py_error:
        if loadmat is None:
            raise ImportError(
                "Reading MAT-files requires scipy for v7 files or h5py "
                "for v7.3/HDF5 files. Install one of them, for example "
                "`python -m pip install scipy`."
            ) from h5py_error
        raise ImportError(
            "scipy could not read this file and h5py is not installed for "
            "v7.3/HDF5 MAT-files. Install h5py with "
            "`python -m pip install h5py`."
        ) from h5py_error

    try:
        with h5py.File(filename, "r") as h5:
            return {key: _read_hdf5_mat_object(h5[key]) for key in h5.keys()}
    except OSError as h5_error:
        if loadmat is None:
            raise ImportError(
                "Reading v7 MAT-files requires scipy. Install it with "
                "`python -m pip install scipy`."
            ) from h5_error
        try:
            return loadmat(filename, squeeze_me=True, struct_as_record=False)
        except Exception as retry_error:
            raise ImportError(
                "could not read the mesh file as a scipy or HDF5 MAT-file"
            ) from retry_error


def _load_mat_v5_numeric(filename: Path) -> dict[str, Array]:
    """
    Minimal MAT-file v5 reader for dense real numeric arrays.

    This keeps mesh loading usable in a small NumPy-only environment.  It is
    deliberately narrow: it is meant for arrays like ``xs`` and ``surfs``.
    """
    buf = filename.read_bytes()
    if len(buf) < 128 or b"MATLAB 5.0 MAT-file" not in buf[:128]:
        raise ValueError("not a MAT-file v5 file")
    endian = buf[126:128]
    if endian == b"IM":
        order = "<"
    elif endian == b"MI":
        order = ">"
    else:
        raise ValueError("unrecognized MAT-file v5 endian marker")

    arrays: dict[str, Array] = {}
    pos = 128
    while pos + 8 <= len(buf):
        dtype, nbytes, data_pos, _ = _mat_v5_read_tag(buf, pos, order)
        if dtype == 0 and nbytes == 0:
            break
        if dtype == _MI_COMPRESSED:
            decompressed = zlib.decompress(buf[data_pos : data_pos + nbytes])
            arrays.update(_mat_v5_read_elements(decompressed, order))
            pos = data_pos + nbytes
        elif dtype == _MI_MATRIX:
            name, value = _mat_v5_read_matrix(buf[pos : data_pos + nbytes], order)
            arrays[name] = value
            pos = data_pos + nbytes
        else:
            break
    if not arrays:
        raise ValueError("no supported MAT-file v5 numeric arrays found")
    return arrays


_MI_INT8 = 1
_MI_UINT8 = 2
_MI_INT16 = 3
_MI_UINT16 = 4
_MI_INT32 = 5
_MI_UINT32 = 6
_MI_SINGLE = 7
_MI_DOUBLE = 9
_MI_INT64 = 12
_MI_UINT64 = 13
_MI_MATRIX = 14
_MI_COMPRESSED = 15
_MAT_V5_DTYPES = {
    _MI_INT8: "i1",
    _MI_UINT8: "u1",
    _MI_INT16: "i2",
    _MI_UINT16: "u2",
    _MI_INT32: "i4",
    _MI_UINT32: "u4",
    _MI_SINGLE: "f4",
    _MI_DOUBLE: "f8",
    _MI_INT64: "i8",
    _MI_UINT64: "u8",
}


def _mat_v5_read_elements(buf: bytes, order: str) -> dict[str, Array]:
    arrays: dict[str, Array] = {}
    pos = 0
    while pos + 8 <= len(buf):
        dtype, nbytes, data_pos, _ = _mat_v5_read_tag(buf, pos, order)
        if dtype == _MI_MATRIX:
            name, value = _mat_v5_read_matrix(buf[pos : data_pos + nbytes], order)
            arrays[name] = value
            pos = data_pos + nbytes
        elif dtype == _MI_COMPRESSED:
            arrays.update(_mat_v5_read_elements(zlib.decompress(buf[data_pos : data_pos + nbytes]), order))
            pos = data_pos + nbytes
        elif dtype == 0 and nbytes == 0:
            break
        else:
            break
    return arrays


def _mat_v5_read_matrix(buf: bytes, order: str) -> tuple[str, Array]:
    dtype, nbytes, data_pos, _ = _mat_v5_read_tag(buf, 0, order)
    if dtype != _MI_MATRIX:
        raise ValueError("MAT-file element is not a matrix")
    pos = data_pos

    _, nb, dpos, small = _mat_v5_read_tag(buf, pos, order)
    pos = _mat_v5_next_pos(dpos, nb, small)

    _, nb, dpos, small = _mat_v5_read_tag(buf, pos, order)
    dims = struct.unpack(order + "i" * (nb // 4), buf[dpos : dpos + nb])
    pos = _mat_v5_next_pos(dpos, nb, small)

    _, nb, dpos, small = _mat_v5_read_tag(buf, pos, order)
    name = buf[dpos : dpos + nb].decode("latin1")
    pos = _mat_v5_next_pos(dpos, nb, small)

    dtype, nb, dpos, _ = _mat_v5_read_tag(buf, pos, order)
    if dtype not in _MAT_V5_DTYPES:
        raise ValueError(f"unsupported MAT-file numeric data type {dtype}")
    np_dtype = np.dtype(order + _MAT_V5_DTYPES[dtype])
    values = np.frombuffer(buf[dpos : dpos + nb], dtype=np_dtype).copy()
    return name, values.reshape(dims, order="F")


def _mat_v5_read_tag(buf: bytes, pos: int, order: str) -> tuple[int, int, int, bool]:
    raw = struct.unpack(order + "I", buf[pos : pos + 4])[0]
    dtype = raw & 0xFFFF
    nbytes = (raw >> 16) & 0xFFFF
    if nbytes:
        return dtype, nbytes, pos + 4, True
    dtype, nbytes = struct.unpack(order + "II", buf[pos : pos + 8])
    return dtype, nbytes, pos + 8, False


def _mat_v5_next_pos(data_pos: int, nbytes: int, small: bool) -> int:
    if small:
        return data_pos + 4
    return data_pos + ((nbytes + 7) & ~7)


def _read_hdf5_mat_object(obj):
    if hasattr(obj, "keys"):
        return {key: _read_hdf5_mat_object(obj[key]) for key in obj.keys()}
    arr = np.asarray(obj)
    return arr.T if arr.ndim == 2 else arr


def _mat_public_items(obj):
    if isinstance(obj, dict):
        return [(k, v) for k, v in obj.items() if not str(k).startswith("__")]
    if hasattr(obj, "_fieldnames"):
        return [(name, getattr(obj, name)) for name in obj._fieldnames]
    if isinstance(obj, np.ndarray) and obj.dtype.names:
        squeezed = np.squeeze(obj)
        if squeezed.shape == ():
            return [(name, squeezed[name].item()) for name in obj.dtype.names]
    if isinstance(obj, np.void) and obj.dtype.names:
        return [(name, obj[name].item() if np.asarray(obj[name]).shape == () else obj[name]) for name in obj.dtype.names]
    return []


def _mat_get(obj, name: str):
    if isinstance(obj, dict):
        if name in obj:
            return obj[name]
        lower = {str(k).lower(): k for k in obj.keys()}
        if name.lower() in lower:
            return obj[lower[name.lower()]]
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, np.ndarray) and obj.dtype.names:
        squeezed = np.squeeze(obj)
        if name in obj.dtype.names:
            return squeezed[name].item() if squeezed.shape == () else squeezed[name]
    if isinstance(obj, np.void) and obj.dtype.names and name in obj.dtype.names:
        value = obj[name]
        return value.item() if np.asarray(value).shape == () else value
    raise KeyError(f"could not find `{name}` in mesh data")


def _find_mat_array(data, names: tuple[str, ...], coercer: Callable, label: str):
    for name in names:
        try:
            return coercer(_mat_get(data, name), name)
        except (KeyError, ValueError, TypeError):
            pass

    for _, value in _mat_public_items(data):
        try:
            return _find_mat_array(value, names, coercer, label)
        except ValueError:
            pass

    raise ValueError(f"could not find a valid {label} array in mesh data")


def _coerce_vertices(value, name: str = "vertices") -> Array:
    arr = np.asarray(value, dtype=float)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"`{name}` is not a 2D vertex array")
    if arr.shape[1] == 3:
        out = arr
    elif arr.shape[0] == 3:
        out = arr.T
    else:
        raise ValueError(f"`{name}` must have shape (nv,3) or (3,nv)")
    return np.asarray(out, dtype=float)


def _coerce_faces(value, name: str = "faces") -> Array:
    arr = np.asarray(value)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.ndim == 1 and arr.size >= 3:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"`{name}` is not a 2D face array")
    if arr.shape[1] >= 3:
        out = arr
    elif arr.shape[0] >= 3:
        out = arr.T
    else:
        raise ValueError(f"`{name}` must contain at least three vertex indices per face")
    out = np.asarray(out, dtype=float)
    if not np.all(np.isfinite(out)) or not np.allclose(out, np.round(out)):
        raise ValueError(f"`{name}` must contain integer vertex indices")
    return np.asarray(np.round(out), dtype=int)


def project_to_levelset(
    points: Array,
    phi: Callable,
    grad_phi: Callable,
    max_iter: int = 30,
    tol: float = 1.0e-13,
    damping: float = 1.0,
) -> Array:
    """
    Project points to an implicit surface using normal Newton correction.

    The update is

    ``p <- p - damping * phi(p) * grad_phi(p) / |grad_phi(p)|^2``.
    """
    pts = np.asarray(points, dtype=float)
    original_shape = pts.shape
    pts = pts.reshape(-1, 3).copy()
    if not (0.0 < damping <= 1.0):
        raise ValueError("damping must satisfy 0 < damping <= 1")

    for row in range(pts.shape[0]):
        p = pts[row, :].copy()
        converged = False
        for _ in range(max_iter):
            value = float(_call_levelset_scalar(phi, p))
            grad = np.asarray(_call_levelset_vector(grad_phi, p), dtype=float).reshape(3)
            denom = float(np.dot(grad, grad))
            if denom <= np.finfo(float).eps:
                raise RuntimeError("level-set projection encountered a near-zero gradient")
            correction = value * grad / denom
            if float(np.linalg.norm(correction)) <= tol:
                converged = True
                break
            p = p - damping * correction
        final_value = float(_call_levelset_scalar(phi, p))
        final_grad = np.asarray(_call_levelset_vector(grad_phi, p), dtype=float).reshape(3)
        final_denom = float(np.dot(final_grad, final_grad))
        final_step = abs(final_value) / np.sqrt(max(final_denom, np.finfo(float).eps))
        if not converged and final_step > 10 * tol:
            raise RuntimeError(f"level-set projection did not converge; correction={final_step:.3e}")
        pts[row, :] = p

    return pts.reshape(original_shape)


def normalize_face_indices(faces: Array, nvertices: int, index_base: str | int = "auto") -> Array:
    """Convert zero- or one-based triangle indices to zero-based indices."""
    faces = np.asarray(faces, dtype=int)
    if index_base == "auto":
        if faces.size == 0:
            return faces.copy()
        if np.min(faces) == 1 and np.max(faces) == nvertices:
            out = faces - 1
        else:
            out = faces.copy()
    elif index_base in (0, "zero", "0"):
        out = faces.copy()
    elif index_base in (1, "one", "1"):
        out = faces - 1
    else:
        raise ValueError("index_base must be 'auto', 0, or 1")

    if np.any(out < 0) or np.any(out >= nvertices):
        raise ValueError("faces contain vertex indices outside the valid range")
    return out


def _call_levelset_scalar(func: Callable, p: Array):
    try:
        return func(p)
    except TypeError:
        return func(p[0], p[1], p[2])


def _call_levelset_vector(func: Callable, p: Array):
    try:
        return func(p)
    except TypeError:
        return func(p[0], p[1], p[2])


@dataclass
class TriangleSurfaceFunction:
    """Scalar function sampled on a ``TriangleSurfaceMesh``."""

    domain: TriangleSurfaceMesh
    vals: list[Array]

    @staticmethod
    def from_callable(domain: TriangleSurfaceMesh, func: Callable[[Array, Array, Array], Array]) -> "TriangleSurfaceFunction":
        vals = [np.asarray(func(x, y, z)).ravel() for x, y, z in zip(domain.x, domain.y, domain.z)]
        return TriangleSurfaceFunction(domain, vals)

    @staticmethod
    def constant(domain: TriangleSurfaceMesh, value: float | complex) -> "TriangleSurfaceFunction":
        return TriangleSurfaceFunction(domain, [value * np.ones_like(x) for x in domain.x])

    def norm_inf(self) -> float:
        return float(max(np.max(np.abs(v)) for v in self.vals))

    def mean2(self) -> float:
        return tri_integral2(self) / tri_surfacearea(self.domain)

    def remove_mean(self) -> "TriangleSurfaceFunction":
        return self - self.mean2()

    def copy(self) -> "TriangleSurfaceFunction":
        return TriangleSurfaceFunction(self.domain, [v.copy() for v in self.vals])

    def __add__(self, other: float | "TriangleSurfaceFunction") -> "TriangleSurfaceFunction":
        if isinstance(other, TriangleSurfaceFunction):
            return TriangleSurfaceFunction(self.domain, [a + b for a, b in zip(self.vals, other.vals)])
        return TriangleSurfaceFunction(self.domain, [a + other for a in self.vals])

    __radd__ = __add__

    def __sub__(self, other: float | "TriangleSurfaceFunction") -> "TriangleSurfaceFunction":
        if isinstance(other, TriangleSurfaceFunction):
            return TriangleSurfaceFunction(self.domain, [a - b for a, b in zip(self.vals, other.vals)])
        return TriangleSurfaceFunction(self.domain, [a - other for a in self.vals])

    def __rsub__(self, other: float) -> "TriangleSurfaceFunction":
        return TriangleSurfaceFunction(self.domain, [other - a for a in self.vals])

    def __mul__(self, other: float | "TriangleSurfaceFunction") -> "TriangleSurfaceFunction":
        if isinstance(other, TriangleSurfaceFunction):
            return TriangleSurfaceFunction(self.domain, [a * b for a, b in zip(self.vals, other.vals)])
        return TriangleSurfaceFunction(self.domain, [a * other for a in self.vals])

    __rmul__ = __mul__

    def __truediv__(self, other: float | "TriangleSurfaceFunction") -> "TriangleSurfaceFunction":
        if isinstance(other, TriangleSurfaceFunction):
            return TriangleSurfaceFunction(self.domain, [a / b for a, b in zip(self.vals, other.vals)])
        return TriangleSurfaceFunction(self.domain, [a / other for a in self.vals])

    def __rtruediv__(self, other: float) -> "TriangleSurfaceFunction":
        return TriangleSurfaceFunction(self.domain, [other / a for a in self.vals])

    def __pow__(self, power: float) -> "TriangleSurfaceFunction":
        return TriangleSurfaceFunction(self.domain, [a**power for a in self.vals])

    def __abs__(self) -> "TriangleSurfaceFunction":
        return TriangleSurfaceFunction(self.domain, [np.abs(a) for a in self.vals])

    def __neg__(self) -> "TriangleSurfaceFunction":
        return TriangleSurfaceFunction(self.domain, [-a for a in self.vals])


@dataclass
class TriangleLeaf(Patch):
    """Triangular leaf solution operator."""

    Aii_inv: Array | None = None
    ii_idx: Array | None = None
    normal_d: Array | None = None
    rhs_scale: Array | None = None
    _bc_work: Array | None = None

    def solve(self, bc: Array | Callable | None = None) -> list[Array]:
        if bc is None:
            self._bc_work = get_work_array(
                self._bc_work,
                (self.S.shape[1], self.u_part.shape[1]),
                self.u_part.dtype,
            )
            bc_arr = self._bc_work
        elif callable(bc):
            vals = bc(self.xyz[:, 0], self.xyz[:, 1], self.xyz[:, 2])
            bc_arr = np.asarray(vals).reshape(-1, 1)
        else:
            bc_arr = np.asarray(bc)
            if bc_arr.ndim == 0:
                bc_arr = np.full((self.S.shape[1], self.u_part.shape[1]), bc_arr.item())
            elif bc_arr.ndim == 1:
                bc_arr = bc_arr.reshape(-1, 1)

        u = self.S @ bc_arr + self.u_part
        return [u[:, j].copy() for j in range(u.shape[1])]


class TriangleSurfaceOp:
    """Scalar HPS-style operator for triangular surface patches."""

    def __init__(
        self,
        domain: TriangleSurfaceMesh,
        op: dict,
        rhs: TriangleSurfaceFunction | Callable | float = 0.0,
        merge_idx: list[list[tuple[int, int | None]]] | None = None,
        merge_strategy: str = "default",
    ):
        self.domain = domain
        self.op = parse_pdo(op)
        self.rhs = rhs
        self.rankdef = False
        self.merge_strategy = merge_strategy.lower()
        self.merge_idx = default_merge_idx(domain.npatches) if merge_idx is None else merge_idx
        self._ii_idx = np.where(np.all(np.asarray(list(_multi_indices(3, domain.degree)), dtype=int) > 0, axis=1))[0]
        self._num_int = int(self._ii_idx.size)
        self._rhs_work: Array | None = None
        self.patches: list[Patch] = initialize_triangle_leaves(self.op, self.domain, self.rhs)
        self._built = False

    def build(self) -> None:
        if self._built:
            return

        patches = self.patches
        if self.merge_strategy == "adjacent":
            while len(patches) > 1:
                pairs = greedy_adjacent_pairs(patches)
                next_patches: list[Patch] = []
                merged_any = False
                for a_idx, b_idx in pairs:
                    a = patches[a_idx]
                    if b_idx is None:
                        next_patches.append(a)
                    else:
                        top = len(patches) == 2
                        next_patches.append(merge_patches(a, patches[b_idx], rankdef=(self.rankdef and top)))
                        merged_any = True
                if not merged_any:
                    raise RuntimeError("could not find adjacent triangular patches to merge")
                patches = next_patches
        elif self.merge_strategy == "default":
            for level, idx in enumerate(self.merge_idx):
                top = level == len(self.merge_idx) - 1
                next_patches = []
                for a_idx, b_idx in idx:
                    a = patches[a_idx]
                    b = None if b_idx is None else patches[b_idx]
                    if b is None:
                        next_patches.append(a)
                    else:
                        next_patches.append(merge_patches(a, b, rankdef=(self.rankdef and top)))
                patches = next_patches
        else:
            raise ValueError("merge_strategy must be 'default' or 'adjacent'")

        self.patches = patches
        self._built = True

    def solve(self) -> TriangleSurfaceFunction:
        self.build()
        vals = self.patches[0].solve(None)
        return TriangleSurfaceFunction(self.domain, vals)

    def apply(self, rhs: TriangleSurfaceFunction | Callable | float) -> TriangleSurfaceFunction:
        """Update the right-hand side and immediately solve."""
        self.update_rhs(rhs)
        return self.solve()

    def update_rhs(self, rhs: TriangleSurfaceFunction | Callable | float) -> "TriangleSurfaceOp":
        """
        Update only the right-hand side of an already built triangular operator.

        The local inverses, Dirichlet-to-Neumann maps, and merge-tree Schur
        complements are reused.  Only the particular solution is recomputed and
        propagated through the merge tree.
        """
        if not self._built:
            self.rhs = rhs
            self.patches = initialize_triangle_leaves(self.op, self.domain, self.rhs)
            return self

        rhs_int = evaluate_triangle_rhs(rhs, self.domain, self._ii_idx, self._num_int, self._rhs_work)
        self._rhs_work = rhs_int
        for patch in self.patches:
            update_triangle_patch_rhs(patch, rhs_int)
        self.rhs = rhs
        return self


def tri_surfaceop(
    dom: TriangleSurfaceMesh,
    op: dict,
    rhs: TriangleSurfaceFunction | Callable | float = 0.0,
    merge_idx: list[list[tuple[int, int | None]]] | None = None,
    merge_strategy: str = "default",
) -> TriangleSurfaceOp:
    """Construct a triangular scalar surface operator."""
    return TriangleSurfaceOp(dom, op, rhs, merge_idx=merge_idx, merge_strategy=merge_strategy)


def initialize_triangle_leaves(op, dom: TriangleSurfaceMesh, rhs) -> list[Patch]:
    n = dom.n
    degree = dom.degree
    npatch = dom.npatches
    num_nodes = n * (n + 1) // 2

    alpha = np.asarray(list(_multi_indices(3, degree)), dtype=int)
    ii_mask = np.all(alpha > 0, axis=1)
    ee_mask = ~ii_mask
    ii_idx = np.where(ii_mask)[0]
    ee_idx = np.where(ee_mask)[0]
    num_int = ii_idx.size
    num_bdy = ee_idx.size

    edge_nodes = triangle_edge_node_indices(n)
    nskel = max(n - 2, 0)
    S2L, L2S = triangle_edge_interpolation_matrices(n, ee_idx, edge_nodes, dom.family)

    rhs_int = evaluate_triangle_rhs(rhs, dom, ii_idx, num_int)
    coeffs, flags = evaluate_triangle_operator_coefficients(op, dom)
    matrix_dtype = np.result_type(rhs_int, *(c.dtype for c in coeffs.values()))

    I = np.eye(num_nodes, dtype=matrix_dtype)

    leaves: list[Patch] = []
    tmp_template = np.zeros((num_nodes, num_bdy + rhs_int.shape[1]), dtype=matrix_dtype)
    tmp_template[np.ix_(ee_idx, np.arange(num_bdy))] = np.eye(num_bdy)
    left_skel = np.arange(0, nskel)
    down_skel = np.arange(nskel, 2 * nskel)
    hypot_skel = np.arange(2 * nskel, 3 * nskel)

    for k in range(npatch):
        Dx = rowscale(dom.ux[k], dom.Du) + rowscale(dom.vx[k], dom.Dv)
        Dy = rowscale(dom.uy[k], dom.Du) + rowscale(dom.vy[k], dom.Dv)
        Dz = rowscale(dom.uz[k], dom.Du) + rowscale(dom.vz[k], dom.Dv)

        A = np.zeros((num_nodes, num_nodes), dtype=matrix_dtype)
        J = dom.J[k].ravel()
        if dom.singular[k]:
            DxJ = Dx @ J
            DyJ = Dy @ J
            DzJ = Dz @ J
            if flags["dxx"]:
                A += rowscale(coeffs["dxx"][:, k], rowscale(J, Dx @ Dx) - rowscale(DxJ, Dx))
            if flags["dyy"]:
                A += rowscale(coeffs["dyy"][:, k], rowscale(J, Dy @ Dy) - rowscale(DyJ, Dy))
            if flags["dzz"]:
                A += rowscale(coeffs["dzz"][:, k], rowscale(J, Dz @ Dz) - rowscale(DzJ, Dz))
            if flags["dxy"]:
                A += rowscale(coeffs["dxy"][:, k], rowscale(J, Dx @ Dy) - rowscale(DxJ, Dy))
            if flags["dyx"]:
                A += rowscale(coeffs["dyx"][:, k], rowscale(J, Dy @ Dx) - rowscale(DyJ, Dx))
            if flags["dyz"]:
                A += rowscale(coeffs["dyz"][:, k], rowscale(J, Dy @ Dz) - rowscale(DyJ, Dz))
            if flags["dzy"]:
                A += rowscale(coeffs["dzy"][:, k], rowscale(J, Dz @ Dy) - rowscale(DzJ, Dy))
            if flags["dxz"]:
                A += rowscale(coeffs["dxz"][:, k], rowscale(J, Dx @ Dz) - rowscale(DxJ, Dz))
            if flags["dzx"]:
                A += rowscale(coeffs["dzx"][:, k], rowscale(J, Dz @ Dx) - rowscale(DzJ, Dx))
            if flags["dx"]:
                A += rowscale(coeffs["dx"][:, k] * J**2, Dx)
            if flags["dy"]:
                A += rowscale(coeffs["dy"][:, k] * J**2, Dy)
            if flags["dz"]:
                A += rowscale(coeffs["dz"][:, k] * J**2, Dz)
            if flags["b"]:
                A += rowscale(coeffs["b"][:, k] * J**3, I)
            local_rhs_data = (J[ii_idx] ** 3).reshape(-1, 1) * rhs_int[:, :, k]
        else:
            if flags["dxx"]:
                A += rowscale(coeffs["dxx"][:, k], Dx @ Dx)
            if flags["dyy"]:
                A += rowscale(coeffs["dyy"][:, k], Dy @ Dy)
            if flags["dzz"]:
                A += rowscale(coeffs["dzz"][:, k], Dz @ Dz)
            if flags["dxy"]:
                A += rowscale(coeffs["dxy"][:, k], Dx @ Dy)
            if flags["dyx"]:
                A += rowscale(coeffs["dyx"][:, k], Dy @ Dx)
            if flags["dyz"]:
                A += rowscale(coeffs["dyz"][:, k], Dy @ Dz)
            if flags["dzy"]:
                A += rowscale(coeffs["dzy"][:, k], Dz @ Dy)
            if flags["dxz"]:
                A += rowscale(coeffs["dxz"][:, k], Dx @ Dz)
            if flags["dzx"]:
                A += rowscale(coeffs["dzx"][:, k], Dz @ Dx)
            if flags["dx"]:
                A += rowscale(coeffs["dx"][:, k], Dx)
            if flags["dy"]:
                A += rowscale(coeffs["dy"][:, k], Dy)
            if flags["dz"]:
                A += rowscale(coeffs["dz"][:, k], Dz)
            if flags["b"]:
                A += rowscale(coeffs["b"][:, k], I)
            local_rhs_data = rhs_int[:, :, k]

        Aii = A[np.ix_(ii_idx, ii_idx)]
        Aie = A[np.ix_(ii_idx, ee_idx)]
        local_rhs = np.hstack((-Aie, local_rhs_data))
        Aii_inv = inverse_dense(Aii)
        S_int = solve_with_inverse(Aii_inv, local_rhs)

        tmp = tmp_template.copy()
        tmp[ii_idx, :] = S_int
        S = tmp[:, :num_bdy] @ S2L
        u_part = tmp[:, num_bdy:]

        if dom.singular[k]:
            dx = L2S @ rowscale(J[ee_idx] ** 2, Dx[ee_idx, :])
            dy = L2S @ rowscale(J[ee_idx] ** 2, Dy[ee_idx, :])
            dz = L2S @ rowscale(J[ee_idx] ** 2, Dz[ee_idx, :])
            Jss = L2S @ (J[ee_idx] ** 3)
            D2N_scl = [Jss[left_skel], Jss[down_skel], Jss[hypot_skel]]
        else:
            dx = L2S @ Dx[ee_idx, :]
            dy = L2S @ Dy[ee_idx, :]
            dz = L2S @ Dz[ee_idx, :]
            D2N_scl = [np.ones(nskel), np.ones(nskel), np.ones(nskel)]

        NN = triangle_conormals_on_skeleton(dom, k, edge_nodes)
        normal_d = rowscale(NN[:, 0], dx) + rowscale(NN[:, 1], dy) + rowscale(NN[:, 2], dz)
        D2N = normal_d @ S
        du_part = normal_d @ u_part

        edges = triangle_patch_edges(dom, k, edge_nodes, nskel)
        xyz = L2S @ np.column_stack((dom.x[k][ee_idx], dom.y[k][ee_idx], dom.z[k][ee_idx]))
        w = triangle_skeleton_weights(dom, k, L2S, ee_idx)

        leaves.append(
            TriangleLeaf(
                domain=dom,
                ids=[k],
                S=S,
                D2N=D2N,
                D2N_scl=D2N_scl,
                u_part=u_part,
                du_part=du_part,
                edges=edges,
                xyz=xyz,
                w=w,
                Aii_inv=Aii_inv,
                ii_idx=ii_idx,
                normal_d=normal_d,
                rhs_scale=(J[ii_idx] ** 3 if dom.singular[k] else np.ones_like(J[ii_idx])),
            )
        )

    return leaves


def evaluate_triangle_rhs(rhs, dom: TriangleSurfaceMesh, ii_idx: Array, num_int: int, out: Array | None = None) -> Array:
    if isinstance(rhs, TriangleSurfaceFunction):
        vals = [v[ii_idx] for v in rhs.vals]
        dtype = np.result_type(*vals) if vals else float
        out = get_work_array(out, (num_int, 1, dom.npatches), dtype)
        for k, v in enumerate(vals):
            out[:, 0, k] = v
        return out

    if callable(rhs):
        vals = []
        for k in range(dom.npatches):
            vals.append(np.asarray(rhs(dom.x[k][ii_idx], dom.y[k][ii_idx], dom.z[k][ii_idx])).ravel())
        dtype = np.result_type(*vals) if vals else float
        out = get_work_array(out, (num_int, 1, dom.npatches), dtype)
        for k, v in enumerate(vals):
            out[:, 0, k] = v
        return out

    out = get_work_array(out, (num_int, 1, dom.npatches), np.asarray(rhs).dtype)
    out.fill(rhs)
    return out


def update_triangle_patch_rhs(patch: Patch, rhs_int: Array) -> None:
    """Recursive right-hand-side update for a triangular HPS merge tree."""
    if isinstance(patch, TriangleLeaf):
        patch_id = patch.ids[0]
        rhs_k = rhs_int[:, :, patch_id]
        assert patch.Aii_inv is not None
        assert patch.ii_idx is not None
        assert patch.normal_d is not None
        assert patch.rhs_scale is not None

        local_rhs = patch.rhs_scale.reshape(-1, 1) * rhs_k
        num_nodes = patch.domain.x[patch_id].size
        u_part = np.zeros((num_nodes, local_rhs.shape[1]), dtype=np.result_type(patch.Aii_inv, local_rhs))
        u_part[patch.ii_idx, :] = solve_with_inverse(patch.Aii_inv, local_rhs)
        patch.u_part = u_part
        patch.du_part = patch.normal_d @ patch.u_part
        return

    if not isinstance(patch, Parent):
        raise TypeError("unknown patch type")

    assert patch.child1 is not None and patch.child2 is not None
    assert patch.idx1 is not None and patch.idx2 is not None
    assert patch.flip1 is not None and patch.flip2 is not None
    assert patch.scl1 is not None and patch.scl2 is not None
    assert patch.A_inv is not None and patch.M is not None

    update_triangle_patch_rhs(patch.child1, rhs_int)
    update_triangle_patch_rhs(patch.child2, rhs_int)

    i1, s1 = patch.idx1
    i2, s2 = patch.idx2
    nrhs = patch.child1.u_part.shape[1]

    if s1.size:
        z_part = rowscale(patch.scl2, patch.flip1 @ patch.child1.du_part[s1, :]) + rowscale(
            patch.scl1,
            patch.flip2 @ patch.child2.du_part[s2, :],
        )
        patch.u_part = solve_with_inverse(patch.A_inv, z_part)
    else:
        patch.u_part = np.zeros((0, nrhs), dtype=patch.child1.u_part.dtype)

    patch.du_part = (
        np.vstack((patch.child1.du_part[i1, :], patch.child2.du_part[i2, :]))
        + patch.M @ patch.u_part
    )


def evaluate_triangle_operator_coefficients(op, dom: TriangleSurfaceMesh) -> tuple[dict[str, Array], dict[str, bool]]:
    """Evaluate scalar or spatially varying PDE coefficients on all triangle nodes."""
    coeffs: dict[str, Array] = {}
    flags: dict[str, bool] = {}
    num_nodes = dom.n * (dom.n + 1) // 2

    for name in op.__dataclass_fields__:
        value = getattr(op, name)
        vals = triangle_coefficient_values(value, dom, num_nodes)
        coeffs[name] = vals
        flags[name] = bool(np.any(vals != 0))
    return coeffs, flags


def triangle_coefficient_values(value, dom: TriangleSurfaceMesh, num_nodes: int) -> Array:
    if isinstance(value, TriangleSurfaceFunction):
        return np.column_stack(value.vals)

    if callable(value):
        cols = []
        for k in range(dom.npatches):
            cols.append(np.asarray(value(dom.x[k], dom.y[k], dom.z[k])).ravel())
        return np.column_stack(cols)

    if np.isscalar(value):
        dtype = np.asarray(value).dtype
        out = np.empty((num_nodes, dom.npatches), dtype=dtype)
        out.fill(value)
        return out

    arr = np.asarray(value)
    if arr.shape == (num_nodes, dom.npatches):
        return arr
    if arr.shape == (dom.npatches, num_nodes):
        return arr.T
    if arr.size == num_nodes:
        return np.tile(arr.ravel().reshape(-1, 1), (1, dom.npatches))
    raise ValueError("coefficient arrays must be nodal values on the triangular mesh")


def triangle_edge_node_indices(n: int) -> list[Array]:
    degree = n - 1
    alpha = np.asarray(list(_multi_indices(3, degree)), dtype=int)
    edge0 = np.where(alpha[:, 0] == 0)[0]
    edge1 = np.where(alpha[:, 1] == 0)[0]
    edge2 = np.where(alpha[:, 2] == 0)[0]

    edge0 = edge0[np.argsort(alpha[edge0, 1])]
    edge1 = edge1[np.argsort(alpha[edge1, 0])]
    edge2 = edge2[np.argsort(alpha[edge2, 0])]
    return [edge0, edge1, edge2]


def triangle_edge_interpolation_matrices(n: int, ee_idx: Array, edge_nodes: list[Array], family: str) -> tuple[Array, Array]:
    nskel = max(n - 2, 0)
    if nskel == 0:
        return np.zeros((ee_idx.size, 0)), np.zeros((0, ee_idx.size))

    t_full = node_family_points(n, family)
    t_skel = 0.5 * (chebpts(nskel, 1) + 1.0)
    B_full_from_skel = barymat(t_full, t_skel)
    B_skel_from_full = barymat(t_skel, t_full)

    node_to_boundary = {int(node): row for row, node in enumerate(ee_idx)}
    S2L = np.zeros((ee_idx.size, 3 * nskel))
    counts = np.zeros(ee_idx.size)
    L2S = np.zeros((3 * nskel, ee_idx.size))

    for e, nodes in enumerate(edge_nodes):
        rows = np.asarray([node_to_boundary[int(node)] for node in nodes])
        cols = np.arange(e * nskel, (e + 1) * nskel)
        S2L[np.ix_(rows, cols)] += B_full_from_skel
        counts[rows] += 1.0
        L2S[np.ix_(cols, rows)] = B_skel_from_full

    counts[counts == 0.0] = 1.0
    S2L /= counts[:, None]
    return S2L, L2S


def triangle_conormals_on_skeleton(dom: TriangleSurfaceMesh, patch_id: int, edge_nodes: list[Array]) -> Array:
    n = dom.n
    nskel = max(n - 2, 0)
    if nskel == 0:
        return np.zeros((0, 3))

    t_full = node_family_points(dom.n, dom.family)
    t_skel = 0.5 * (chebpts(nskel, 1) + 1.0)
    B = barymat(t_skel, t_full)

    xu = np.column_stack((dom.xu[patch_id], dom.yu[patch_id], dom.zu[patch_id]))
    xv = np.column_stack((dom.xv[patch_id], dom.yv[patch_id], dom.zv[patch_id]))

    conormals = []
    for e, nodes in enumerate(edge_nodes):
        if e == 0:
            tangent = xv[nodes, :]
            raw = -xu[nodes, :]
        elif e == 1:
            tangent = xu[nodes, :]
            raw = -xv[nodes, :]
        else:
            tangent = xu[nodes, :] - xv[nodes, :]
            raw = xu[nodes, :] + xv[nodes, :]
        tangent = normalize_rows(tangent)
        normal = raw - tangent * np.sum(raw * tangent, axis=1, keepdims=True)
        normal = normalize_rows(normal)
        conormals.append(normalize_rows(B @ normal))

    return np.vstack(conormals)


def triangle_patch_edges(dom: TriangleSurfaceMesh, patch_id: int, edge_nodes: list[Array], nskel: int) -> Array:
    rows = []
    for nodes in edge_nodes:
        a = nodes[0]
        b = nodes[-1]
        rows.append(
            [
                dom.x[patch_id][a],
                dom.y[patch_id][a],
                dom.z[patch_id][a],
                dom.x[patch_id][b],
                dom.y[patch_id][b],
                dom.z[patch_id][b],
                nskel,
            ]
        )
    return np.asarray(rows, dtype=float)


def triangle_skeleton_weights(dom: TriangleSurfaceMesh, patch_id: int, L2S: Array, ee_idx: Array) -> Array:
    nskel = max(dom.n - 2, 0)
    if nskel == 0:
        return np.zeros(0)

    w1d = quadwts(nskel, 1)
    wskel = np.concatenate((0.5 * w1d, 0.5 * w1d, (np.sqrt(2.0) / 2.0) * w1d))
    JJ = L2S @ np.sqrt(np.maximum(dom.J[patch_id][ee_idx], 0.0))
    return wskel * JJ


def normalize_rows(v: Array) -> Array:
    nrm = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.where(nrm > 0.0, nrm, 1.0)


def greedy_adjacent_pairs(patches: list[Patch]) -> list[tuple[int, int | None]]:
    remaining = set(range(len(patches)))
    pairs: list[tuple[int, int | None]] = []

    while remaining:
        i = min(remaining)
        remaining.remove(i)
        partner = None
        for j in sorted(remaining):
            if patches_share_edge(patches[i], patches[j]):
                partner = j
                break
        if partner is None:
            pairs.append((i, None))
        else:
            remaining.remove(partner)
            pairs.append((i, partner))
    return pairs


def patches_share_edge(a: Patch, b: Patch) -> bool:
    if a.edges.size == 0 or b.edges.size == 0:
        return False
    scl = max(
        float(np.max(np.abs(a.edges[:, :6]))),
        float(np.max(np.abs(b.edges[:, :6]))),
        1.0,
    )
    tol = 1e-8 * scl
    for ea in a.edges:
        direct = np.sum(np.abs(b.edges[:, :6] - ea[:6]), axis=1)
        flipped = np.sum(np.abs(b.edges[:, :6] - ea[[3, 4, 5, 0, 1, 2]]), axis=1)
        if np.any(direct < tol) or np.any(flipped < tol):
            return True
    return False


def icosphere_tri(n: int, nref: int = 0, family: str = "cheb2") -> TriangleSurfaceMesh:
    """Convenience wrapper for a high-order triangular icosphere."""
    return TriangleSurfaceMesh.icosphere(n, nref=nref, family=family)


def tri_surfacefun(
    func: Callable[[Array, Array, Array], Array] | float | list[Array],
    dom: TriangleSurfaceMesh,
) -> TriangleSurfaceFunction:
    """Construct a scalar triangular surface function."""
    if callable(func):
        return TriangleSurfaceFunction.from_callable(dom, func)
    if np.isscalar(func):
        return TriangleSurfaceFunction.constant(dom, func)
    return TriangleSurfaceFunction(dom, [np.asarray(v).ravel() for v in func])


def tri_resample_mesh(dom: TriangleSurfaceMesh, n: int, family: str | None = None) -> TriangleSurfaceMesh:
    """Resample every triangular patch to order ``n``."""
    family_name = dom.family if family is None else _canonical_family_name(family)
    xpatches, ypatches, zpatches, _ = tri_resampled_patch_geometry(dom, nvis=int(n), family=family_name)
    return TriangleSurfaceMesh(
        [x.copy() for x in xpatches],
        [y.copy() for y in ypatches],
        [z.copy() for z in zpatches],
        family=family_name,
    )


def tri_resample(u: TriangleSurfaceFunction, n: int, family: str | None = None) -> TriangleSurfaceFunction:
    """Resample a triangular surface function using the PKD interpolation basis."""
    n = int(n)
    family_name = u.domain.family if family is None else _canonical_family_name(family)
    dom = tri_resample_mesh(u.domain, n, family=family_name)
    if n == u.domain.n and family_name == u.domain.family:
        vals = [val.copy() for val in u.vals]
    else:
        B = tri_resample_matrix(u.domain.n, n, u.domain.family, family_name)
        vals = [B @ val for val in u.vals]
    return TriangleSurfaceFunction(dom, vals)


def tri_diff(f: TriangleSurfaceFunction, dim: int = 1) -> TriangleSurfaceFunction:
    """Tangential Cartesian derivative on triangular patches."""
    if dim not in {1, 2, 3}:
        raise ValueError("dim must be 1, 2, or 3")

    vals: list[Array] = []
    for k, uvals in enumerate(f.vals):
        dfdu = f.domain.Du @ uvals
        dfdv = f.domain.Dv @ uvals
        if dim == 1:
            du, dv = f.domain.ux[k], f.domain.vx[k]
        elif dim == 2:
            du, dv = f.domain.uy[k], f.domain.vy[k]
        else:
            du, dv = f.domain.uz[k], f.domain.vz[k]
        vals.append(du * dfdu + dv * dfdv)
    return TriangleSurfaceFunction(f.domain, vals)


def tri_lap(f: TriangleSurfaceFunction) -> TriangleSurfaceFunction:
    """Surface Laplacian by repeated tangential Cartesian differentiation."""
    return tri_diff(tri_diff(f, 1), 1) + tri_diff(tri_diff(f, 2), 2) + tri_diff(tri_diff(f, 3), 3)


def tri_integral2(f: TriangleSurfaceFunction, reduce: bool = True) -> float | Array:
    """Surface integral on triangular patches."""
    dtype = np.result_type(*f.vals) if f.vals else float
    per_patch = np.zeros(f.domain.npatches, dtype=dtype)
    w = f.domain.ref_weights
    for k, vals in enumerate(f.vals):
        jac = np.sqrt(np.maximum(f.domain.J[k], 0.0))
        per_patch[k] = np.sum(w * jac * vals)
    total = np.sum(per_patch)
    return float(np.real(total)) if reduce and np.isrealobj(total) else total if reduce else per_patch


def tri_surfacearea(dom: TriangleSurfaceMesh) -> float:
    """Area of a triangular surface mesh."""
    return tri_integral2(TriangleSurfaceFunction.constant(dom, 1.0))


def write_tri_vtu(filename: str, u: TriangleSurfaceFunction, point_name: str = "u", nvis: int | None = None) -> None:
    """Write a triangular surface function to a ParaView VTU file."""
    points_arr, faces_arr, values_arr = triangle_vtk_arrays(u, nvis=nvis)
    write_ascii_vtu(filename, points_arr, faces_arr, {point_name: values_arr})


def write_tri_vtp(filename: str, u: TriangleSurfaceFunction, point_name: str = "u", nvis: int | None = None) -> None:
    """Write a triangular surface function to a ParaView VTP PolyData file."""
    points_arr, faces_arr, values_arr = triangle_vtk_arrays(u, nvis=nvis)
    write_ascii_vtp(filename, points_arr, faces_arr, {point_name: values_arr})


def write_tri_patch_boundaries_vtp(filename: str, dom: TriangleSurfaceMesh, nvis: int | None = None) -> None:
    """Write high-order triangular patch boundaries as ParaView PolyData lines."""
    segments = tri_patch_boundary_segments(dom, nvis=nvis)
    write_ascii_vtp_lines(filename, segments)


def triangle_vtk_arrays(u: TriangleSurfaceFunction, nvis: int | None = None) -> tuple[Array, Array, Array]:
    """Return concatenated points, triangle connectivity, and nodal values."""
    if nvis is None:
        xpatches, ypatches, zpatches = u.domain.x, u.domain.y, u.domain.z
        vals = u.vals
        triangles = u.domain.triangles
    else:
        xpatches, ypatches, zpatches, vals, nplot = tri_resampled_patch_values(u, nvis=nvis)
        triangles = trilattice(nplot)

    points = []
    values = []
    faces = []
    offset = 0
    for x, y, z, val in zip(xpatches, ypatches, zpatches, vals):
        points.append(np.column_stack((x, y, z)))
        values.append(val)
        faces.append(triangles + offset)
        offset += x.size

    points_arr = np.vstack(points)
    faces_arr = np.vstack(faces).astype(int)
    values_arr = np.concatenate(values)
    return points_arr, faces_arr, values_arr


def triangle_surface_mesh_arrays(dom: TriangleSurfaceMesh, nvis: int | None = None) -> tuple[Array, Array]:
    """Return concatenated points and triangle connectivity for a triangular surface mesh."""
    if nvis is None:
        xpatches, ypatches, zpatches = dom.x, dom.y, dom.z
        triangles = dom.triangles
    else:
        xpatches, ypatches, zpatches, nplot = tri_resampled_patch_geometry(dom, nvis=nvis)
        triangles = trilattice(nplot)

    points = []
    faces = []
    offset = 0
    for x, y, z in zip(xpatches, ypatches, zpatches):
        points.append(np.column_stack((x, y, z)))
        faces.append(triangles + offset)
        offset += x.size

    return np.vstack(points), np.vstack(faces).astype(int)


def triangle_meshio_mesh(
    obj: TriangleSurfaceMesh | TriangleSurfaceFunction,
    point_name: str = "u",
    nvis: int | None = None,
):
    """Convert a triangular surface mesh or function to a ``meshio.Mesh``."""
    import meshio

    if isinstance(obj, TriangleSurfaceFunction):
        points_arr, faces_arr, values_arr = triangle_vtk_arrays(obj, nvis=nvis)
        return meshio.Mesh(points_arr, [("triangle", faces_arr)], point_data={point_name: values_arr})

    if isinstance(obj, TriangleSurfaceMesh):
        points_arr, faces_arr = triangle_surface_mesh_arrays(obj, nvis=nvis)
        return meshio.Mesh(points_arr, [("triangle", faces_arr)])

    raise TypeError("triangle_meshio_mesh expects a TriangleSurfaceMesh or TriangleSurfaceFunction")


def triangle_boundary_meshio_mesh(
    dom: TriangleSurfaceMesh,
    nvis: int | None = None,
    deduplicate: bool = True,
):
    """Convert high-order triangular patch boundaries to a ``meshio.Mesh`` with line cells."""
    import meshio

    segments = tri_patch_boundary_segments(dom, nvis=nvis, deduplicate=deduplicate)
    if not segments:
        return meshio.Mesh(np.zeros((0, 3)), [("line", np.zeros((0, 2), dtype=int))])

    points = np.vstack(segments)
    lines = np.arange(points.shape[0], dtype=int).reshape(-1, 2)
    return meshio.Mesh(points, [("line", lines)])


def write_triangle_meshio(
    filename: str | Path,
    obj: TriangleSurfaceMesh | TriangleSurfaceFunction,
    point_name: str = "u",
    nvis: int | None = None,
) -> None:
    """Write a triangular surface mesh or function with ``meshio``."""
    mesh = triangle_meshio_mesh(obj, point_name=point_name, nvis=nvis)
    mesh.write(filename)


def tri_resampled_patch_values(
    u: TriangleSurfaceFunction,
    nvis: int | None = None,
    family: str | None = None,
) -> tuple[list[Array], list[Array], list[Array], list[Array], int]:
    """Return patch geometry and values sampled on a display triangle grid."""
    dom = u.domain
    xpatches, ypatches, zpatches, nplot = tri_resampled_patch_geometry(dom, nvis=nvis, family=family)
    if nplot == dom.n and (family is None or _canonical_family_name(family) == dom.family):
        vals = u.vals
    else:
        family_name = dom.family if family is None else _canonical_family_name(family)
        B = tri_resample_matrix(dom.n, nplot, dom.family, family_name)
        vals = [B @ val for val in u.vals]
    return xpatches, ypatches, zpatches, vals, nplot


def tri_resampled_patch_geometry(
    dom: TriangleSurfaceMesh,
    nvis: int | None = None,
    family: str | None = None,
) -> tuple[list[Array], list[Array], list[Array], int]:
    """Return patch geometry sampled on a display triangle grid."""
    nplot = dom.n if nvis is None else int(nvis)
    if nplot < 2:
        raise ValueError("nvis must be at least 2")
    family = dom.family if family is None else _canonical_family_name(family)

    if nplot == dom.n and family == dom.family:
        return dom.x, dom.y, dom.z, dom.n

    B = tri_resample_matrix(dom.n, nplot, dom.family, family)
    xpatches = [B @ x for x in dom.x]
    ypatches = [B @ y for y in dom.y]
    zpatches = [B @ z for z in dom.z]
    return xpatches, ypatches, zpatches, nplot


@lru_cache(maxsize=64)
def tri_resample_matrix(nsource: int, ntarget: int, source_family: str = "cheb2", target_family: str = "cheb2") -> Array:
    """Interpolation matrix from one triangular node set to another."""
    source_family = _canonical_family_name(source_family)
    target_family = _canonical_family_name(target_family)
    usrc, vsrc = tri_reference_nodes(nsource, family=source_family)
    utgt, vtgt = tri_reference_nodes(ntarget, family=target_family)
    degree = nsource - 1
    Ksrc, _, _ = koornwinder_pkd(degree, usrc, vsrc)
    Ktgt, _, _ = koornwinder_pkd(degree, utgt, vtgt)
    return Ktgt @ np.linalg.solve(Ksrc, np.eye(Ksrc.shape[0]))


def tri_patch_boundary_segments(dom: TriangleSurfaceMesh, nvis: int | None = None, deduplicate: bool = True) -> list[Array]:
    """Return line segments following the curved high-order patch boundaries."""
    nplot = dom.n if nvis is None else int(nvis)
    if nplot < 2:
        raise ValueError("nvis must be at least 2")
    if nplot == dom.n:
        xpatches, ypatches, zpatches = dom.x, dom.y, dom.z
        uref, vref = dom.u, dom.v
    else:
        B = tri_resample_matrix(dom.n, nplot, dom.family, dom.family)
        xpatches = [B @ x for x in dom.x]
        ypatches = [B @ y for y in dom.y]
        zpatches = [B @ z for z in dom.z]
    edge_indices = tri_edge_indices(nplot, dom.family)

    segments: list[Array] = []
    seen: set[tuple[tuple[float, ...], tuple[float, ...]]] = set()
    for x, y, z in zip(xpatches, ypatches, zpatches):
        points = np.column_stack((x, y, z))
        for idx in edge_indices:
            for a, b in zip(idx[:-1], idx[1:]):
                segment = points[[a, b], :]
                if deduplicate:
                    p0 = tuple(np.round(segment[0], 12))
                    p1 = tuple(np.round(segment[1], 12))
                    key = tuple(sorted((p0, p1)))
                    if key in seen:
                        continue
                    seen.add(key)
                segments.append(segment)
    return segments


def tri_edge_indices(n: int, family: str = "cheb2") -> tuple[Array, Array, Array]:
    """Triangle edge node indices determined from reference coordinates."""
    u, v = tri_reference_nodes(n, family=family)
    tol = 100.0 * np.finfo(float).eps

    e0 = np.flatnonzero(np.abs(u) <= tol)
    e1 = np.flatnonzero(np.abs(v) <= tol)
    e2 = np.flatnonzero(np.abs(u + v - 1.0) <= tol)

    e0 = e0[np.argsort(v[e0])]
    e1 = e1[np.argsort(u[e1])]
    e2 = e2[np.argsort(u[e2])]
    return e0.astype(int), e1.astype(int), e2.astype(int)


def tri_wireframe_edge_indices(n: int) -> tuple[Array, Array, Array]:
    """Triangle edge node indices for patch-boundary wireframes."""
    e1 = np.arange(n, dtype=int)
    e2 = np.cumsum(np.r_[1, np.arange(n, 1, -1)]) - 1
    e3 = np.cumsum(np.r_[n, np.arange(n - 1, 0, -1)]) - 1
    return e1, e2.astype(int), e3.astype(int)


def write_ascii_vtu(filename: str, points: Array, triangles: Array, point_data: dict[str, Array] | None = None) -> None:
    """Write a minimal ASCII VTK UnstructuredGrid with triangle cells."""
    point_data = {} if point_data is None else point_data
    points = np.asarray(points, dtype=float)
    triangles = np.asarray(triangles, dtype=int)
    npoints = points.shape[0]
    ncells = triangles.shape[0]
    offsets = 3 * np.arange(1, ncells + 1)
    cell_types = np.full(ncells, 5, dtype=int)

    with open(filename, "w", encoding="utf-8") as fid:
        fid.write('<?xml version="1.0"?>\n')
        fid.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        fid.write("  <UnstructuredGrid>\n")
        fid.write(f'    <Piece NumberOfPoints="{npoints}" NumberOfCells="{ncells}">\n')

        scalar_name = next(iter(point_data), "")
        fid.write(f'      <PointData Scalars="{scalar_name}">\n')
        for name, data in point_data.items():
            data = np.asarray(data).ravel()
            if np.iscomplexobj(data):
                _write_data_array(fid, f"{name}_real", np.real(data))
                _write_data_array(fid, f"{name}_imag", np.imag(data))
                _write_data_array(fid, f"{name}_abs", np.abs(data))
            else:
                _write_data_array(fid, name, data)
        fid.write("      </PointData>\n")

        fid.write("      <CellData>\n")
        fid.write("      </CellData>\n")

        fid.write("      <Points>\n")
        _write_data_array(fid, "Points", points, ncomp=3)
        fid.write("      </Points>\n")

        fid.write("      <Cells>\n")
        _write_data_array(fid, "connectivity", triangles.ravel(), dtype="Int64")
        _write_data_array(fid, "offsets", offsets, dtype="Int64")
        _write_data_array(fid, "types", cell_types, dtype="UInt8")
        fid.write("      </Cells>\n")

        fid.write("    </Piece>\n")
        fid.write("  </UnstructuredGrid>\n")
        fid.write("</VTKFile>\n")


def write_ascii_vtp(filename: str, points: Array, triangles: Array, point_data: dict[str, Array] | None = None) -> None:
    """Write a minimal ASCII VTK PolyData surface with triangle polygons."""
    point_data = {} if point_data is None else point_data
    points = np.asarray(points, dtype=float)
    triangles = np.asarray(triangles, dtype=int)
    npoints = points.shape[0]
    npolys = triangles.shape[0]
    offsets = 3 * np.arange(1, npolys + 1)

    with open(filename, "w", encoding="utf-8") as fid:
        fid.write('<?xml version="1.0"?>\n')
        fid.write('<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n')
        fid.write("  <PolyData>\n")
        fid.write(f'    <Piece NumberOfPoints="{npoints}" NumberOfVerts="0" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="{npolys}">\n')

        scalar_name = next(iter(point_data), "")
        fid.write(f'      <PointData Scalars="{scalar_name}">\n')
        for name, data in point_data.items():
            data = np.asarray(data).ravel()
            if np.iscomplexobj(data):
                _write_data_array(fid, f"{name}_real", np.real(data))
                _write_data_array(fid, f"{name}_imag", np.imag(data))
                _write_data_array(fid, f"{name}_abs", np.abs(data))
            else:
                _write_data_array(fid, name, data)
        fid.write("      </PointData>\n")

        fid.write("      <CellData>\n")
        fid.write("      </CellData>\n")

        fid.write("      <Points>\n")
        _write_data_array(fid, "Points", points, ncomp=3)
        fid.write("      </Points>\n")

        fid.write("      <Polys>\n")
        _write_data_array(fid, "connectivity", triangles.ravel(), dtype="Int64")
        _write_data_array(fid, "offsets", offsets, dtype="Int64")
        fid.write("      </Polys>\n")

        fid.write("    </Piece>\n")
        fid.write("  </PolyData>\n")
        fid.write("</VTKFile>\n")


def write_ascii_vtp_lines(filename: str, segments: list[Array]) -> None:
    """Write line segments to a minimal ASCII VTK PolyData file."""
    if segments:
        points = np.vstack(segments)
        nlines = len(segments)
    else:
        points = np.zeros((0, 3))
        nlines = 0
    connectivity = np.arange(points.shape[0], dtype=int)
    offsets = 2 * np.arange(1, nlines + 1)

    with open(filename, "w", encoding="utf-8") as fid:
        fid.write('<?xml version="1.0"?>\n')
        fid.write('<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n')
        fid.write("  <PolyData>\n")
        fid.write(f'    <Piece NumberOfPoints="{points.shape[0]}" NumberOfVerts="0" NumberOfLines="{nlines}" NumberOfStrips="0" NumberOfPolys="0">\n')

        fid.write("      <PointData>\n")
        fid.write("      </PointData>\n")
        fid.write("      <CellData>\n")
        fid.write("      </CellData>\n")

        fid.write("      <Points>\n")
        _write_data_array(fid, "Points", points, ncomp=3)
        fid.write("      </Points>\n")

        fid.write("      <Lines>\n")
        _write_data_array(fid, "connectivity", connectivity, dtype="Int64")
        _write_data_array(fid, "offsets", offsets, dtype="Int64")
        fid.write("      </Lines>\n")

        fid.write("    </Piece>\n")
        fid.write("  </PolyData>\n")
        fid.write("</VTKFile>\n")


def _write_data_array(fid, name: str, data: Array, ncomp: int | None = None, dtype: str = "Float64") -> None:
    data = np.asarray(data)
    ncomp_attr = "" if ncomp is None else f' NumberOfComponents="{ncomp}"'
    fid.write(f'        <DataArray type="{dtype}" Name="{name}" format="ascii"{ncomp_attr}>\n')
    if data.ndim == 2:
        for row in data:
            fid.write("          " + " ".join(_format_vtu_value(v) for v in row) + "\n")
    else:
        flat = data.ravel()
        width = 9 if dtype == "Float64" else 16
        for start in range(0, flat.size, width):
            chunk = flat[start : start + width]
            fid.write("          " + " ".join(_format_vtu_value(v) for v in chunk) + "\n")
    fid.write("        </DataArray>\n")


def _format_vtu_value(value) -> str:
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    return f"{float(value):.17e}"


def wireframe(
    dom: TriangleSurfaceMesh | SurfaceMesh,
    surface: str = "auto",
    edges: str = "auto",
    ax=None,
    color: str = "k",
    linestyle: str = "-",
    linewidth: float = 1.0,
    surface_color: str = "w",
    shrink: float = 0.005,
    nvis: int | None = None,
    line_offset: float = 0.0,
    deduplicate: bool = True,
    visible_only: bool = False,
    view_vector: Array | None = None,
    **line_kwargs,
):
    """Plot high-order patch boundaries."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    show_surface = _normalize_show_option(surface, "surface")
    show_edges = _normalize_show_option(edges, "edges")

    created_axes = ax is None
    if ax is None:
        fig = plt.figure(figsize=(7, 6), facecolor="w")
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    if isinstance(dom, TriangleSurfaceMesh):
        xpatches, ypatches, zpatches, nplot = tri_resampled_patch_geometry(dom, nvis=nvis)
        triangles = trilattice(nplot)
        normals = _triangle_patch_normals(dom, xpatches, ypatches, zpatches)
        segments = []
        seen: set[tuple[tuple[float, ...], tuple[float, ...]]] = set()
        edge_indices = tri_edge_indices(nplot, dom.family)
        if view_vector is not None:
            view = np.asarray(view_vector, dtype=float).ravel()
            view_norm = np.linalg.norm(view)
            if view_norm == 0.0:
                raise ValueError("view_vector must be nonzero")
            view = view / view_norm
        else:
            view = None
        for x, y, z, normal_arr in zip(xpatches, ypatches, zpatches, normals):
            base_points = np.column_stack((x, y, z))
            points = base_points + line_offset * normal_arr
            for idx in edge_indices:
                for a, b in zip(idx[:-1], idx[1:]):
                    if visible_only and view is not None:
                        normal_mid = 0.5 * (normal_arr[a] + normal_arr[b])
                        if float(np.dot(normal_mid, view)) <= -0.05:
                            continue
                    if deduplicate:
                        base_segment = base_points[[a, b], :]
                        p0 = tuple(np.round(base_segment[0], 12))
                        p1 = tuple(np.round(base_segment[1], 12))
                        key = tuple(sorted((p0, p1)))
                        if key in seen:
                            continue
                        seen.add(key)
                    segments.append(points[[a, b], :])

        if show_edges != "off" and segments:
            style = dict(colors=color, linewidths=linewidth, linestyles=linestyle, zorder=10)
            style.update(line_kwargs)
            ax.add_collection3d(Line3DCollection(segments, **style))

        if show_surface == "on" or (show_surface == "auto" and created_axes):
            faces = []
            for x, y, z, normal_arr in zip(xpatches, ypatches, zpatches, normals):
                points = np.column_stack((x, y, z)) - shrink * normal_arr
                faces.extend(points[triangles])
            ax.add_collection3d(
                Poly3DCollection(
                    faces,
                    facecolors=surface_color,
                    edgecolors="none",
                    linewidths=0.0,
                    alpha=1.0,
                )
            )

        _set_equal_3d_axes(ax, np.concatenate(xpatches), np.concatenate(ypatches), np.concatenate(zpatches))

    elif isinstance(dom, SurfaceMesh):
        segments = _quad_patch_boundary_segments(dom)
        normals = _quad_patch_normals(dom)

        if show_edges != "off" and segments:
            style = dict(colors=color, linewidths=linewidth, linestyles=linestyle)
            style.update(line_kwargs)
            ax.add_collection3d(Line3DCollection(segments, **style))

        if show_surface == "on" or (show_surface == "auto" and created_axes):
            for x, y, z, normal_arr in zip(dom.x, dom.y, dom.z, normals):
                ax.plot_surface(
                    x - shrink * normal_arr[:, :, 0],
                    y - shrink * normal_arr[:, :, 1],
                    z - shrink * normal_arr[:, :, 2],
                    color=surface_color,
                    edgecolor="none",
                    linewidth=0.0,
                    shade=True,
                    antialiased=True,
                )

        _set_equal_3d_axes(ax, np.concatenate([x.ravel() for x in dom.x]), np.concatenate([y.ravel() for y in dom.y]), np.concatenate([z.ravel() for z in dom.z]))

    else:
        raise TypeError("dom must be a SurfaceMesh or TriangleSurfaceMesh")

    if created_axes:
        ax.view_init(elev=24, azim=-45)
        ax.grid(True)
    return fig, ax


def plot_wireframe(dom: TriangleSurfaceMesh | SurfaceMesh, *args, **kwargs):
    """Alias for ``wireframe``."""
    return wireframe(dom, *args, **kwargs)


def _normalize_show_option(value: str, name: str) -> str:
    value = str(value).lower()
    if value not in {"auto", "on", "off"}:
        raise ValueError(f"{name} must be 'auto', 'on', or 'off'")
    return value


def _triangle_patch_normals(
    dom: TriangleSurfaceMesh,
    xpatches: list[Array],
    ypatches: list[Array],
    zpatches: list[Array],
) -> list[Array]:
    if xpatches is dom.x and ypatches is dom.y and zpatches is dom.z:
        xu, xv = dom.xu, dom.xv
        yu, yv = dom.yu, dom.yv
        zu, zv = dom.zu, dom.zv
        J = dom.J
    else:
        nplot = _triangular_n_from_count(xpatches[0].size)
        uref, vref = tri_reference_nodes(nplot, family=dom.family)
        Du, Dv, _ = tri_strong_diffmat(nplot - 1, uref, vref, basis="pkd")
        xu = [Du @ x for x in xpatches]
        xv = [Dv @ x for x in xpatches]
        yu = [Du @ y for y in ypatches]
        yv = [Dv @ y for y in ypatches]
        zu = [Du @ z for z in zpatches]
        zv = [Dv @ z for z in zpatches]
        J = [(a * a + c * c + e * e) * (b * b + d * d + f * f) - (a * b + c * d + e * f) ** 2 for a, b, c, d, e, f in zip(xu, xv, yu, yv, zu, zv)]

    normals = []
    volume = 0.0
    weights = reference_triangle_quadrature_weights(_triangular_n_from_count(xpatches[0].size), dom.family)
    for x, y, z, a, b, c, d, e, f, jac in zip(xpatches, ypatches, zpatches, xu, xv, yu, yv, zu, zv, J):
        n0 = c * f - e * d
        n1 = e * b - a * f
        n2 = a * d - c * b
        nrm = np.sqrt(np.maximum(n0 * n0 + n1 * n1 + n2 * n2, 1e-300))
        normal_arr = np.column_stack((n0 / nrm, n1 / nrm, n2 / nrm))
        normals.append(normal_arr)
        volume += float(np.sum(((x * normal_arr[:, 0] + y * normal_arr[:, 1] + z * normal_arr[:, 2]) / 3.0) * weights * np.sqrt(np.maximum(jac, 0.0))))
    if volume < 0.0:
        normals = [-normal_arr for normal_arr in normals]
    return normals


def _quad_patch_normals(dom: SurfaceMesh) -> list[Array]:
    normals = []
    volume = 0.0
    weights = quadwts(dom.n)
    W = np.outer(weights, weights)
    for x, y, z, xu, xv, yu, yv, zu, zv, J in zip(dom.x, dom.y, dom.z, dom.xu, dom.xv, dom.yu, dom.yv, dom.zu, dom.zv, dom.J):
        n0 = yu * zv - zu * yv
        n1 = zu * xv - xu * zv
        n2 = xu * yv - yu * xv
        nrm = np.sqrt(np.maximum(n0 * n0 + n1 * n1 + n2 * n2, 1e-300))
        normal_arr = np.stack((n0 / nrm, n1 / nrm, n2 / nrm), axis=2)
        normals.append(normal_arr)
        volume += float(np.sum(((x * normal_arr[:, :, 0] + y * normal_arr[:, :, 1] + z * normal_arr[:, :, 2]) / 3.0) * W * np.sqrt(np.maximum(J, 0.0))))
    if volume < 0.0:
        normals = [-normal_arr for normal_arr in normals]
    return normals


def _quad_patch_boundary_segments(dom: SurfaceMesh) -> list[Array]:
    segments = []
    for x, y, z in zip(dom.x, dom.y, dom.z):
        edge_points = (
            np.column_stack((x[:, 0], y[:, 0], z[:, 0])),
            np.column_stack((x[:, -1], y[:, -1], z[:, -1])),
            np.column_stack((x[0, :], y[0, :], z[0, :])),
            np.column_stack((x[-1, :], y[-1, :], z[-1, :])),
        )
        for points in edge_points:
            segments.extend(points[[i, i + 1], :] for i in range(points.shape[0] - 1))
    return segments


def _set_equal_3d_axes(ax, x: Array, y: Array, z: Array) -> None:
    mins = np.array([np.min(x), np.min(y), np.min(z)], dtype=float)
    maxs = np.array([np.max(x), np.max(y), np.max(z)], dtype=float)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    if radius == 0.0:
        radius = 1.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def plot_tri_surface(
    u: TriangleSurfaceFunction,
    title: str = "",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = True,
) -> None:
    """Basic Matplotlib visualization of a triangular surface function."""
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    plot_vals = [np.real(v) if np.iscomplexobj(v) else v for v in u.vals]
    if vmin is None:
        vmin = min(float(np.min(v)) for v in plot_vals)
    if vmax is None:
        vmax = max(float(np.max(v)) for v in plot_vals)
    norm_obj = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.turbo

    for x, y, z, val in zip(u.domain.x, u.domain.y, u.domain.z, plot_vals):
        points = np.column_stack((x, y, z))
        faces = points[u.domain.triangles]
        face_values = np.mean(val[u.domain.triangles], axis=1)
        collection = Poly3DCollection(
            faces,
            facecolors=cmap(norm_obj(face_values)),
            edgecolors=(0, 0, 0, 0.18),
            linewidths=0.12,
        )
        ax.add_collection3d(collection)

    allx = np.concatenate(u.domain.x)
    ally = np.concatenate(u.domain.y)
    allz = np.concatenate(u.domain.z)
    ax.set_xlim(float(np.min(allx)), float(np.max(allx)))
    ax.set_ylim(float(np.min(ally)), float(np.max(ally)))
    ax.set_zlim(float(np.min(allz)), float(np.max(allz)))
    ax.set_title(title)
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1))
    if colorbar:
        mappable = ScalarMappable(norm=norm_obj, cmap=cmap)
        mappable.set_array([])
        fig.colorbar(mappable, ax=ax, shrink=0.75, pad=0.02)
    plt.show()


def _icosahedron() -> tuple[Array, Array]:
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.array(
        [
            [-1, phi, 0],
            [1, phi, 0],
            [-1, -phi, 0],
            [1, -phi, 0],
            [0, -1, phi],
            [0, 1, phi],
            [0, -1, -phi],
            [0, 1, -phi],
            [phi, 0, -1],
            [phi, 0, 1],
            [-phi, 0, -1],
            [-phi, 0, 1],
        ],
        dtype=float,
    )
    vertices /= np.linalg.norm(vertices[0])
    faces = np.array(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ],
        dtype=int,
    )
    return vertices, faces


def _subdivide_icosphere(vertices: Array, faces: Array) -> tuple[Array, Array]:
    vertices_list = [v.copy() for v in vertices]
    midpoint_cache: dict[tuple[int, int], int] = {}

    def midpoint(i: int, j: int) -> int:
        key = tuple(sorted((int(i), int(j))))
        if key in midpoint_cache:
            return midpoint_cache[key]
        p = 0.5 * (vertices_list[key[0]] + vertices_list[key[1]])
        p /= np.linalg.norm(p)
        midpoint_cache[key] = len(vertices_list)
        vertices_list.append(p)
        return midpoint_cache[key]

    new_faces = []
    for a, b, c in faces:
        ab = midpoint(a, b)
        bc = midpoint(b, c)
        ca = midpoint(c, a)
        new_faces.extend(([a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]))

    return np.asarray(vertices_list), np.asarray(new_faces, dtype=int)


def _orient_faces_outward(vertices: Array, faces: Array) -> Array:
    out = faces.copy()
    for k, (a, b, c) in enumerate(out):
        p0, p1, p2 = vertices[[a, b, c]]
        normal = np.cross(p1 - p0, p2 - p0)
        if np.dot(normal, p0 + p1 + p2) < 0.0:
            out[k, 1], out[k, 2] = out[k, 2], out[k, 1]
    return out
