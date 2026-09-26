"""
Triangular-patch discretizations for pysurfacefun.

The triangular formulation and fast direct solver framework are introduced in:
A High-Order Fast Direct Solver for Surface PDEs on Triangles
https://arxiv.org/pdf/2604.03097

The reference triangle is the unit simplex with vertices ``(0, 0)``,
``(1, 0)``, and ``(0, 1)``.  Nodes are RecursiveNodes-style simplex nodes
generated from Chebyshev points of the second kind by default.  The polynomial
basis is the Proriol-Koornwinder-Dubiner basis used for strong-form
differentiation on a triangle.

A :class:`TriangleSurfaceMesh` with ``P`` patches of ``N = n (n + 1) / 2``
nodes stores coordinates and metric terms as contiguous ``(P, N)`` arrays;
:class:`TriangleSurfaceFunction` stores its values as one ``(P, N)`` array.
"""

from __future__ import annotations

import itertools
import struct
import zlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .core import (
    SurfaceMesh,
    _collect_coefficients,
    _singular_scaling,
    barymat,
    chebpts,
    chebpts2,
    quadwts,
)
from .fields import PatchField, PatchVectorField
from .hps import Leaf, LeafOperators
from .operators import PDO, HPSOperator, coefficient_arrays, evaluate_rhs_nodes, parse_pdo
from .recursivenodes_polynomials import proriolkoornwinderdubinervandermondegrad

Array = np.ndarray


def shifted_lobatto_nodes(degree: int) -> Array:
    """Shifted Lobatto-Gauss-Legendre nodes on ``[0, 1]``.

    The interior nodes are the roots of ``P'_degree``, i.e. the Gauss-Jacobi nodes for the
    weight ``1 - x**2``, computed as eigenvalues of the symmetric Jacobi matrix (Golub-Welsch).
    Unlike companion-matrix roots they are real by construction (``np.linalg.eigvals`` returns
    complex arrays from NumPy 2.5 on) and accurate to rounding for any degree.
    """
    if degree < 0:
        raise ValueError("degree must be nonnegative")
    if degree == 0:
        return np.array([0.5])
    if degree == 1:
        return np.array([0.0, 1.0])

    k = np.arange(1.0, degree - 1)
    beta = np.sqrt(k * (k + 2) / ((2 * k + 1) * (2 * k + 3)))
    roots = np.linalg.eigvalsh(np.diag(beta, -1) + np.diag(beta, 1))
    roots = 0.5 * (roots - roots[::-1])  # exactly symmetric about 0
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
            yield (first, *rest)


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


@lru_cache(maxsize=64)
def _reference_operators(n: int, family: str) -> tuple[Array, Array, Array, Array, Array]:
    """Cached reference nodes ``u, v`` and strong-form matrices ``Du, Dv, K``."""
    u, v = tri_reference_nodes(n, family=family)
    Du, Dv, K = tri_strong_diffmat(n - 1, u, v, basis="pkd")
    for arr in (u, v, Du, Dv, K):
        arr.setflags(write=False)
    return u, v, Du, Dv, K


def _stack_tri_patches(values: Any, name: str) -> Array:
    if isinstance(values, np.ndarray) and values.ndim == 2:
        return np.ascontiguousarray(values, dtype=float)
    arrays = [np.asarray(v, dtype=float).ravel() for v in values]
    if not arrays:
        raise ValueError(f"{name} must contain at least one patch")
    if any(a.size != arrays[0].size for a in arrays):
        raise ValueError(f"all {name} patches must have the same number of nodes")
    return np.stack(arrays)


class TriangleSurfaceMesh:
    """High-order triangular surface patch mesh.

    Parameters
    ----------
    x, y, z:
        Patch coordinates: sequences of length-``N`` arrays or ``(P, N)``
        arrays, ``N = n (n + 1) / 2`` nodes in the ordering of
        :func:`tri_reference_nodes`.
    family:
        One-dimensional node family used to build the simplex nodes.
    """

    def __init__(self, x: Any, y: Any, z: Any, family: str = "cheb2"):
        self.family = _canonical_family_name(family)
        X = _stack_tri_patches(x, "x")
        Y = _stack_tri_patches(y, "y")
        Z = _stack_tri_patches(z, "z")
        if not (X.shape == Y.shape == Z.shape):
            raise ValueError("x, y, z must have the same number of patches")
        self._X, self._Y, self._Z = X, Y, Z
        self.n = _triangular_n_from_count(X.shape[1])
        self.degree = self.n - 1
        self.u, self.v, self.Du, self.Dv, self.K = _reference_operators(self.n, self.family)
        self.triangles = trilattice(self.n)
        self.ref_weights = reference_triangle_quadrature_weights(self.n, self.family)

        DuT, DvT = self.Du.T, self.Dv.T
        XU, XV = X @ DuT, X @ DvT
        YU, YV = Y @ DuT, Y @ DvT
        ZU, ZV = Z @ DuT, Z @ DvT
        E = XU * XU + YU * YU + ZU * ZU
        G = XV * XV + YV * YV + ZV * ZV
        F = XU * XV + YU * YV + ZU * ZV
        J = E * G - F * F
        num_ux = G * XU - F * XV
        scl = np.max(np.abs(num_ux), axis=1)
        singular = np.any(np.abs(J) < 1e-10 * np.maximum(scl, 1.0)[:, None], axis=1)
        regular = np.broadcast_to(~singular[:, None], J.shape)

        def inverse_metric(numerator: Array) -> Array:
            out = numerator.copy()
            np.divide(numerator, J, out=out, where=regular)
            return out

        self._XU, self._XV, self._YU, self._YV, self._ZU, self._ZV = XU, XV, YU, YV, ZU, ZV
        self._UX = inverse_metric(num_ux)
        self._VX = inverse_metric(E * XV - F * XU)
        self._UY = inverse_metric(G * YU - F * YV)
        self._VY = inverse_metric(E * YV - F * YU)
        self._UZ = inverse_metric(G * ZU - F * ZV)
        self._VZ = inverse_metric(E * ZV - F * ZU)
        self._E, self._F, self._G, self._J = E, F, G, J
        self._singular = singular
        self._weights: Array | None = None

        for name in ("x", "y", "z", "xu", "xv", "yu", "yv", "zu", "zv", "ux", "vx", "uy", "vy", "uz", "vz"):
            setattr(self, name, list(getattr(self, "_" + name.upper())))
        self.E, self.F, self.G, self.J = list(E), list(F), list(G), list(J)
        self.singular = [bool(s) for s in singular]

    def __repr__(self) -> str:
        return f"TriangleSurfaceMesh(npatches={self.npatches}, n={self.n}, family={self.family!r})"

    @property
    def npatches(self) -> int:
        """Number of triangular patches."""
        return int(self._X.shape[0])

    @property
    def nnodes(self) -> int:
        """Nodes per patch, ``n (n + 1) / 2``."""
        return int(self._X.shape[1])

    @property
    def coords(self) -> Array:
        """Coordinates as a ``(3, P, N)`` array."""
        return np.stack((self._X, self._Y, self._Z))

    @property
    def quadrature_weights(self) -> Array:
        """Nodal reference weights times the surface Jacobian, shape ``(P, N)``."""
        if self._weights is None:
            self._weights = self.ref_weights[None, :] * np.sqrt(np.maximum(self._J, 0.0))
            self._weights.setflags(write=False)
        return self._weights

    @staticmethod
    def icosphere(n: int, nref: int = 0, family: str = "cheb2") -> TriangleSurfaceMesh:
        """Curved triangular patches on the unit sphere from a refined icosahedron."""
        vertices, faces = _icosahedron()
        for _ in range(nref):
            vertices, faces = _subdivide_icosphere(vertices, faces)
        faces = _orient_faces_outward(vertices, faces)
        u, v = tri_reference_nodes(n, family=family)
        weights = np.column_stack((1.0 - u - v, u, v))
        xyz = np.einsum("nk,fkd->fnd", weights, vertices[faces])
        xyz /= np.linalg.norm(xyz, axis=2, keepdims=True)
        return TriangleSurfaceMesh(xyz[:, :, 0], xyz[:, :, 1], xyz[:, :, 2], family=family)


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
    flat_points = np.einsum("nk,fkd->fnd", weights, vertices[faces0])
    projected = project_to_levelset(flat_points, phi, grad_phi, max_iter=max_iter, tol=tol, damping=damping)
    return TriangleSurfaceMesh(projected[:, :, 0], projected[:, :, 1], projected[:, :, 2], family=family)


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

    flat_points = np.einsum("ijk,ckl->cijl", weights, vertices[cells0])
    projected = project_to_levelset(flat_points, phi, grad_phi, max_iter=max_iter, tol=tol, damping=damping)
    return SurfaceMesh(projected[..., 0], projected[..., 1], projected[..., 2])


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
            "use LevelSetSurface(mesh, phi, grad_phi, n=...) or LevelSetSurface(vertices, faces, phi, grad_phi, n=...)"
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


class _EdgeMidpoints:
    """Vertex list with one shared midpoint per (unordered) edge."""

    def __init__(self, vertices: Array):
        self.vertices = vertices
        self.points = [v.copy() for v in vertices]
        self.cache: dict[tuple[int, int], int] = {}

    def __call__(self, i: int, j: int) -> int:
        key = (int(i), int(j)) if i < j else (int(j), int(i))
        if key not in self.cache:
            self.cache[key] = len(self.points)
            self.points.append(0.5 * (self.vertices[key[0]] + self.vertices[key[1]]))
        return self.cache[key]

    def add(self, point: Array) -> int:
        self.points.append(point)
        return len(self.points) - 1


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
        midpoint = _EdgeMidpoints(verts)
        refined: list[list[int]] = []
        for a, b, c in tris.tolist():
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            refined.extend(([a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]))
        verts = np.asarray(midpoint.points, dtype=float)
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
        midpoint = _EdgeMidpoints(verts)
        refined: list[list[int]] = []
        for a, b, c, d in quads.tolist():
            ab, bc, cd, da = midpoint(a, b), midpoint(b, c), midpoint(c, d), midpoint(d, a)
            center = midpoint.add(0.25 * (verts[a] + verts[b] + verts[c] + verts[d]))
            refined.extend(([a, ab, center, da], [ab, b, bc, center], [center, bc, c, cd], [da, center, cd, d]))
        verts = np.asarray(midpoint.points, dtype=float)
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
    except ImportError:
        loadmat = None
    else:
        try:
            return loadmat(filename, squeeze_me=True, struct_as_record=False)
        except (NotImplementedError, ValueError, OSError):
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
            return {key: _read_hdf5_mat_object(h5[key]) for key in h5}
    except OSError as h5_error:
        if loadmat is None:
            raise ImportError(
                "Reading v7 MAT-files requires scipy. Install it with `python -m pip install scipy`."
            ) from h5_error
        try:
            return loadmat(filename, squeeze_me=True, struct_as_record=False)
        except Exception as retry_error:
            raise ImportError("could not read the mesh file as a scipy or HDF5 MAT-file") from retry_error


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
    dtype, _nbytes, data_pos, _ = _mat_v5_read_tag(buf, 0, order)
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
        return {key: _read_hdf5_mat_object(obj[key]) for key in obj}
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
        return [
            (name, obj[name].item() if np.asarray(obj[name]).shape == () else obj[name]) for name in obj.dtype.names
        ]
    return []


def _mat_get(obj, name: str):
    if isinstance(obj, dict):
        if name in obj:
            return obj[name]
        lower = {str(k).lower(): k for k in obj}
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


def _vectorized_levelset(phi: Callable, grad_phi: Callable, pts: Array) -> tuple[Callable, Callable] | None:
    """Vectorized ``(value, gradient)`` evaluators when ``phi`` and ``grad_phi`` broadcast.

    The callables are tried on a ``(3, M)`` array ``p`` (so ``p[0]`` is the
    x-row); the results must have the expected shapes and agree with pointwise
    evaluation on a few probe points.  Returns ``None`` otherwise.
    """
    probe = pts[: min(3, pts.shape[0])]
    try:
        values = np.asarray(phi(pts.T), dtype=float)
        grads = np.asarray(grad_phi(pts.T), dtype=float)
    except Exception:
        return None
    m = pts.shape[0]
    if values.shape != (m,) or grads.shape != (3, m):
        return None
    try:
        for i, p in enumerate(probe):
            if not np.isclose(float(_call_levelset_scalar(phi, p)), values[i], rtol=1e-12, atol=1e-14):
                return None
            g = np.asarray(_call_levelset_vector(grad_phi, p), dtype=float).reshape(3)
            if not np.allclose(g, grads[:, i], rtol=1e-12, atol=1e-14):
                return None
    except Exception:
        return None
    return (lambda q: np.asarray(phi(q.T), dtype=float)), (lambda q: np.asarray(grad_phi(q.T), dtype=float).T)


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

    ``phi`` and ``grad_phi`` may take one point ``p`` of shape ``(3,)`` or three
    scalars ``x, y, z``.  When they also broadcast over a ``(3, M)`` array (as
    expressions like ``p[0]**2 + p[1]**2`` do), all points are projected
    simultaneously; otherwise the points are processed one at a time.
    """
    pts = np.asarray(points, dtype=float)
    original_shape = pts.shape
    pts = pts.reshape(-1, 3).copy()
    if not (0.0 < damping <= 1.0):
        raise ValueError("damping must satisfy 0 < damping <= 1")
    if pts.shape[0] == 0:
        return pts.reshape(original_shape)

    vectorized = _vectorized_levelset(phi, grad_phi, pts)
    if vectorized is None:
        for row in range(pts.shape[0]):
            pts[row] = _project_point(pts[row], phi, grad_phi, max_iter, tol, damping)
        return pts.reshape(original_shape)

    value_fn, grad_fn = vectorized
    active = np.ones(pts.shape[0], dtype=bool)
    eps = np.finfo(float).eps
    for _ in range(max_iter):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        p = pts[idx]
        value = value_fn(p)
        grad = grad_fn(p)
        denom = np.sum(grad * grad, axis=1)
        if np.any(denom <= eps):
            raise RuntimeError("level-set projection encountered a near-zero gradient")
        correction = (value / denom)[:, None] * grad
        done = np.linalg.norm(correction, axis=1) <= tol
        step = ~done
        pts[idx[step]] = p[step] - damping * correction[step]
        active[idx[done]] = False

    if np.any(active):
        idx = np.flatnonzero(active)
        p = pts[idx]
        grad = grad_fn(p)
        final_step = np.abs(value_fn(p)) / np.sqrt(np.maximum(np.sum(grad * grad, axis=1), eps))
        if np.any(final_step > 10 * tol):
            raise RuntimeError(f"level-set projection did not converge; correction={np.max(final_step):.3e}")
    return pts.reshape(original_shape)


def _project_point(p: Array, phi: Callable, grad_phi: Callable, max_iter: int, tol: float, damping: float) -> Array:
    p = p.copy()
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
    final_step = abs(final_value) / np.sqrt(max(float(np.dot(final_grad, final_grad)), np.finfo(float).eps))
    if not converged and final_step > 10 * tol:
        raise RuntimeError(f"level-set projection did not converge; correction={final_step:.3e}")
    return p


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


class TriangleSurfaceFunction(PatchField):
    """Scalar function sampled on the nodes of a :class:`TriangleSurfaceMesh`.

    ``f.data`` holds the values as a ``(P, N)`` array; ``f.vals`` is the list
    of per-patch views.
    """

    _family = "tri"

    @staticmethod
    def patch_shape(domain: TriangleSurfaceMesh) -> tuple[int, ...]:
        return (domain.nnodes,)


class TriangleSurfaceVectorFunction(PatchVectorField):
    """Three-component vector field sampled on a triangular surface mesh."""

    _scalar_family = "tri"


TriangleLeaf = Leaf
"""Triangular leaf node of the HPS merge tree (alias of :class:`pysurfacefun.hps.Leaf`)."""


class TriangleSurfaceOp(HPSOperator):
    """Fast direct solver for a scalar elliptic operator on triangular patches.

    Accepts the same coefficients as :class:`pysurfacefun.SurfaceOp`, including
    variable coefficients given as callables or triangular surface functions.
    """

    def _build_leaf_operators(self) -> LeafOperators:
        return build_triangle_leaves(self.op, self.domain)

    def _rhs_values(self, rhs: Any) -> Array:
        return evaluate_rhs_nodes(
            rhs, self.domain, _triangle_reference(self.domain.n, self.domain.family).interior, self.domain.nnodes
        )

    def _field_from_nodal(self, U: Array) -> TriangleSurfaceFunction:
        return TriangleSurfaceFunction(self.domain, U)


def tri_surfaceop(
    dom: TriangleSurfaceMesh,
    op: dict,
    rhs: TriangleSurfaceFunction | Callable | float = 0.0,
    merge_idx: list[list[tuple[int, int | None]]] | None = None,
    merge_strategy: str = "default",
    **kwargs,
) -> TriangleSurfaceOp:
    """Construct a triangular scalar surface operator."""
    return TriangleSurfaceOp(dom, op, rhs, merge_idx=merge_idx, merge_strategy=merge_strategy, **kwargs)


@dataclass(frozen=True)
class _TriangleReference:
    n: int
    interior: Array
    boundary: Array
    edge_nodes: tuple[Array, Array, Array]
    S2L: Array
    L2S: Array
    B: Array
    wskel: Array


@lru_cache(maxsize=32)
def _triangle_reference(n: int, family: str) -> _TriangleReference:
    alpha = np.asarray(list(_multi_indices(3, n - 1)), dtype=int)
    interior_mask = np.all(alpha > 0, axis=1)
    interior = np.flatnonzero(interior_mask)
    boundary = np.flatnonzero(~interior_mask)
    edge_nodes = tuple(triangle_edge_node_indices(n))
    nskel = max(n - 2, 0)
    S2L, L2S = triangle_edge_interpolation_matrices(n, boundary, list(edge_nodes), family)
    t_full = node_family_points(n, family)
    t_skel = 0.5 * (chebpts(nskel, 1) + 1.0)
    w1d = quadwts(nskel, 1)
    wskel = np.concatenate((0.5 * w1d, 0.5 * w1d, (np.sqrt(2.0) / 2.0) * w1d))
    return _TriangleReference(n, interior, boundary, edge_nodes, S2L, L2S, barymat(t_skel, t_full), wskel)  # type: ignore[arg-type]


def build_triangle_leaves(op: PDO | dict, dom: TriangleSurfaceMesh, max_chunk_bytes: float = 1.5e8) -> LeafOperators:
    """Dense HPS leaf operators for every patch of a triangular mesh.

    The strong-form collocation operator ``sum_d M_d D_d`` with
    ``M_d = sum_c a_cd D_c`` needs three dense products per patch; patches are
    processed in batches with stacked matrix products.  Patches with a
    degenerate parametrization are multiplied through by powers of the
    Jacobian.
    """
    op = parse_pdo(op)
    n = dom.n
    if n < 3:
        raise ValueError("triangular HPS leaves need at least n = 3 nodes per edge")
    ref = _triangle_reference(n, dom.family)
    P, N = dom.npatches, dom.nnodes
    interior, boundary = ref.interior, ref.boundary
    m = interior.size
    nskel = n - 2
    b = 3 * nskel

    second, first, zeroth = _collect_coefficients(op, dom, (P, N))
    U = np.stack((dom._UX, dom._UY, dom._UZ))
    V = np.stack((dom._VX, dom._VY, dom._VZ))
    J = dom._J
    singular = dom._singular
    rhs_scale = None
    flux_nodes = np.ones((P, N))
    if np.any(singular):
        second, first, zeroth = _singular_scaling(second, first, zeroth, U, V, J, J @ dom.Du.T, J @ dom.Dv.T, singular)
        mask = singular[:, None]
        rhs_scale = np.where(mask, J**3, 1.0)[:, interior]
        flux_nodes = np.where(mask, J**2, 1.0)
    dtype = np.result_type(float, *second.values(), *first.values(), *([] if zeroth is None else [zeroth]))

    Du, Dv = dom.Du, dom.Dv
    S_all = np.empty((P, N, b), dtype=dtype)
    D2N_all = np.empty((P, b, b), dtype=dtype)
    Ainv_all = np.empty((P, m, m), dtype=dtype)
    G_all = np.empty((P, b, m), dtype=dtype)

    NN = triangle_skeleton_conormals(dom)
    U_ee = (U * flux_nodes[None])[:, :, boundary]
    V_ee = (V * flux_nodes[None])[:, :, boundary]
    Du_ee, Dv_ee = Du[boundary], Dv[boundary]

    per_patch = np.dtype(dtype).itemsize * (8 * N * N + 2 * m * m + 3 * N * b)
    chunk = int(max(1, min(P, max_chunk_bytes // max(per_patch, 1))))
    for start in range(0, P, chunk):
        stop = min(P, start + chunk)
        sl = slice(start, stop)
        c = stop - start
        A = np.zeros((c, N, N), dtype=dtype)
        for d in range(3):
            Pd = sum((second[(k, d)][sl] * U[k, sl] for k in range(3) if (k, d) in second), np.zeros((c, N)))
            Qd = sum((second[(k, d)][sl] * V[k, sl] for k in range(3) if (k, d) in second), np.zeros((c, N)))
            if any((k, d) in second for k in range(3)):
                Dd = U[d, sl][:, :, None] * Du + V[d, sl][:, :, None] * Dv
                Md = Pd[:, :, None] * Du + Qd[:, :, None] * Dv
                A += np.matmul(Md, Dd)
        if first:
            beta_u = sum(first[k][sl] * U[k, sl] for k in first)
            beta_v = sum(first[k][sl] * V[k, sl] for k in first)
            A += beta_u[:, :, None] * Du + beta_v[:, :, None] * Dv
        if zeroth is not None:
            A[:, np.arange(N), np.arange(N)] += zeroth[sl]
        Arows = A[:, interior, :]
        Ainv = np.linalg.inv(Arows[:, :, interior])
        S = np.empty((c, N, b), dtype=dtype)
        S[:, boundary, :] = ref.S2L
        S[:, interior, :] = np.matmul(np.matmul(Ainv, -Arows[:, :, boundary]), ref.S2L)

        alpha = np.einsum("csk,kcp->csp", NN[sl], U_ee[:, sl]) * ref.L2S
        beta = np.einsum("csk,kcp->csp", NN[sl], V_ee[:, sl]) * ref.L2S
        normal_d = np.matmul(alpha, Du_ee) + np.matmul(beta, Dv_ee)
        if rhs_scale is not None:
            Ainv = Ainv * rhs_scale[sl][:, None, :]
        S_all[sl] = S
        D2N_all[sl] = np.matmul(normal_d, S)
        Ainv_all[sl] = Ainv
        G_all[sl] = np.matmul(normal_d[:, :, interior], Ainv)

    coords = np.stack((dom._X, dom._Y, dom._Z), axis=-1)
    xyz = np.matmul(ref.L2S, coords[:, boundary, :])
    sqrtJ = np.sqrt(np.maximum(J, 0.0))[:, boundary]
    w = ref.wskel * np.matmul(sqrtJ, ref.L2S.T)
    scale = None
    if np.any(singular):
        scale = np.matmul(np.where(singular[:, None], J**3, 1.0)[:, boundary], ref.L2S.T)
    starts = np.stack([coords[:, nodes[0]] for nodes in ref.edge_nodes], axis=1)
    ends = np.stack([coords[:, nodes[-1]] for nodes in ref.edge_nodes], axis=1)
    mids = xyz.reshape(P, 3, nskel, 3).mean(axis=2)
    return LeafOperators(
        S=S_all,
        D2N=D2N_all,
        Ainv=Ainv_all,
        G=G_all,
        interior=interior,
        edge_sizes=np.full(3, nskel, dtype=np.int64),
        edge_points=np.stack((starts, ends, mids), axis=2),
        xyz=xyz,
        w=w,
        scale=scale,
        centroids=coords.mean(axis=1),
    )


def evaluate_triangle_rhs(
    rhs, dom: TriangleSurfaceMesh, ii_idx: Array, num_int: int, out: Array | None = None
) -> Array:
    """Right-hand side at interior nodes in the legacy ``(num_int, 1, P)`` layout."""
    values = evaluate_rhs_nodes(rhs, dom, np.asarray(ii_idx), dom.nnodes)
    return np.ascontiguousarray(values.T[:, None, :])


def evaluate_triangle_operator_coefficients(op, dom: TriangleSurfaceMesh) -> tuple[dict[str, Array], dict[str, bool]]:
    """Evaluate scalar or spatially varying PDE coefficients on all triangle nodes (``(N, P)`` layout)."""
    coeffs: dict[str, Array] = {}
    flags: dict[str, bool] = {}
    for name, value in parse_pdo(op).items():
        vals = triangle_coefficient_values(value, dom, dom.nnodes)
        coeffs[name] = vals
        flags[name] = bool(np.any(vals != 0))
    return coeffs, flags


def triangle_coefficient_values(value, dom: TriangleSurfaceMesh, num_nodes: int) -> Array:
    """Coefficient values on every node in the legacy ``(num_nodes, P)`` layout."""
    return coefficient_arrays(value, dom, (dom.npatches, num_nodes)).T


def triangle_edge_node_indices(n: int) -> list[Array]:
    """Node indices of the three reference edges ``u = 0``, ``v = 0``, and the hypotenuse."""
    degree = n - 1
    alpha = np.asarray(list(_multi_indices(3, degree)), dtype=int)
    edge0 = np.where(alpha[:, 0] == 0)[0]
    edge1 = np.where(alpha[:, 1] == 0)[0]
    edge2 = np.where(alpha[:, 2] == 0)[0]
    edge0 = edge0[np.argsort(alpha[edge0, 1])]
    edge1 = edge1[np.argsort(alpha[edge1, 0])]
    edge2 = edge2[np.argsort(alpha[edge2, 0])]
    return [edge0, edge1, edge2]


#: Counterclockwise traversal sign of the stored edges ``u = 0``, ``v = 0``, and hypotenuse.
TRI_EDGE_CCW = np.array([-1, 1, -1])


def _tri_edge_points(dom: TriangleSurfaceMesh) -> Array:
    """Start point, end point, and mean point of the three edges of every patch, ``(P, 3, 3, 3)``."""
    coords = np.stack((dom._X, dom._Y, dom._Z), axis=-1)
    out = []
    for nodes in triangle_edge_node_indices(dom.n):
        pts = coords[:, nodes]
        out.append(np.stack((pts[:, 0], pts[:, -1], pts.mean(axis=1)), axis=1))
    return np.stack(out, axis=1)


def triangle_edge_interpolation_matrices(
    n: int, ee_idx: Array, edge_nodes: list[Array], family: str
) -> tuple[Array, Array]:
    """Skeleton-to-boundary (corners averaged) and boundary-to-skeleton interpolation matrices."""
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


def _unit_rows(v: Array) -> Array:
    nrm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(nrm > 0.0, nrm, 1.0)


def triangle_skeleton_conormals(dom: TriangleSurfaceMesh) -> Array:
    """Unit outward conormals at the skeleton points of every patch, shape ``(P, 3 nskel, 3)``."""
    ref = _triangle_reference(dom.n, dom.family)
    Xu = np.stack((dom._XU, dom._YU, dom._ZU), axis=-1)
    Xv = np.stack((dom._XV, dom._YV, dom._ZV), axis=-1)
    out = []
    for e, nodes in enumerate(ref.edge_nodes):
        xu, xv = Xu[:, nodes], Xv[:, nodes]
        if e == 0:
            tangent, raw = xv, -xu
        elif e == 1:
            tangent, raw = xu, -xv
        else:
            tangent, raw = xu - xv, xu + xv
        tangent = _unit_rows(tangent)
        normal = _unit_rows(raw - tangent * np.sum(raw * tangent, axis=-1, keepdims=True))
        out.append(_unit_rows(np.matmul(ref.B, normal)))
    return np.concatenate(out, axis=1)


def triangle_conormals_on_skeleton(
    dom: TriangleSurfaceMesh, patch_id: int, edge_nodes: list[Array] | None = None
) -> Array:
    """Conormals at the skeleton points of one patch (legacy helper)."""
    return triangle_skeleton_conormals(dom)[patch_id]


def triangle_patch_edges(dom: TriangleSurfaceMesh, patch_id: int, edge_nodes: list[Array], nskel: int) -> Array:
    """Legacy ``(3, 7)`` edge table: start point, end point, and skeleton size of each edge."""
    rows = []
    for nodes in edge_nodes:
        a, b = nodes[0], nodes[-1]
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
    """Skeleton quadrature weights (times the surface Jacobian) of one patch."""
    ref = _triangle_reference(dom.n, dom.family)
    return ref.wskel * (L2S @ np.sqrt(np.maximum(dom._J[patch_id][ee_idx], 0.0)))


def normalize_rows(v: Array) -> Array:
    """Normalize rows; zero rows are left unchanged."""
    return _unit_rows(v)


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
        return TriangleSurfaceFunction.constant(dom, func)  # type: ignore[arg-type]
    return TriangleSurfaceFunction(dom, func)


def tri_resample_mesh(dom: TriangleSurfaceMesh, n: int, family: str | None = None) -> TriangleSurfaceMesh:
    """Resample every triangular patch to ``n`` nodes per edge."""
    family_name = dom.family if family is None else _canonical_family_name(family)
    n = int(n)
    if n == dom.n and family_name == dom.family:
        return TriangleSurfaceMesh(dom._X.copy(), dom._Y.copy(), dom._Z.copy(), family=family_name)
    B = tri_resample_matrix(dom.n, n, dom.family, family_name)
    return TriangleSurfaceMesh(dom._X @ B.T, dom._Y @ B.T, dom._Z @ B.T, family=family_name)


def tri_resample(u: TriangleSurfaceFunction, n: int, family: str | None = None) -> TriangleSurfaceFunction:
    """Resample a triangular surface function using the PKD interpolation basis."""
    n = int(n)
    family_name = u.domain.family if family is None else _canonical_family_name(family)
    dom = tri_resample_mesh(u.domain, n, family=family_name)
    if n == u.domain.n and family_name == u.domain.family:
        return TriangleSurfaceFunction(dom, u.data.copy())
    B = tri_resample_matrix(u.domain.n, n, u.domain.family, family_name)
    return TriangleSurfaceFunction(dom, u.data @ B.T)


def _tri_metric_pair(dom: TriangleSurfaceMesh, dim: int) -> tuple[Array, Array]:
    if dim == 1:
        return dom._UX, dom._VX
    if dim == 2:
        return dom._UY, dom._VY
    if dim == 3:
        return dom._UZ, dom._VZ
    raise ValueError("dim must be 1, 2, or 3")


def tri_diff(f: TriangleSurfaceFunction, dim: int = 1) -> TriangleSurfaceFunction:
    """Tangential Cartesian derivative on triangular patches (``dim = 1, 2, 3`` for ``x, y, z``)."""
    dom = f.domain
    du, dv = _tri_metric_pair(dom, dim)
    return TriangleSurfaceFunction(dom, du * (f.data @ dom.Du.T) + dv * (f.data @ dom.Dv.T))


def tri_lap(f: TriangleSurfaceFunction) -> TriangleSurfaceFunction:
    """Surface Laplacian by repeated tangential Cartesian differentiation."""
    return tri_diff(tri_diff(f, 1), 1) + tri_diff(tri_diff(f, 2), 2) + tri_diff(tri_diff(f, 3), 3)


def tri_integral2(f: TriangleSurfaceFunction, reduce: bool = True) -> float | Array:
    """Surface integral on triangular patches (per patch when ``reduce=False``)."""
    return f.integral(reduce=reduce)


def tri_surfacearea(dom: TriangleSurfaceMesh) -> float:
    """Area of a triangular surface mesh."""
    return float(np.sum(dom.quadrature_weights))


def write_tri_vtu(
    filename: str | Path,
    u: TriangleSurfaceFunction,
    point_name: str = "u",
    nvis: int | None = None,
    binary: bool = True,
) -> None:
    """Write a triangular surface function to a ParaView ``.vtu`` file."""
    from .vtk import write_vtu_fields

    write_vtu_fields(filename, {point_name: u}, nvis=nvis, binary=binary)


def write_tri_vtp(
    filename: str | Path,
    u: TriangleSurfaceFunction,
    point_name: str = "u",
    nvis: int | None = None,
    binary: bool = True,
) -> None:
    """Write a triangular surface function to a ParaView ``.vtp`` PolyData file."""
    from .vtk import write_vtp

    points_arr, faces_arr, values_arr = triangle_vtk_arrays(u, nvis=nvis)
    write_vtp(filename, points_arr, polys=faces_arr, point_data={point_name: values_arr}, binary=binary)


def write_tri_patch_boundaries_vtp(
    filename: str | Path, dom: TriangleSurfaceMesh, nvis: int | None = None, binary: bool = True
) -> None:
    """Write high-order triangular patch boundaries as ParaView PolyData lines."""
    _write_segments(filename, tri_patch_boundary_segments(dom, nvis=nvis), binary=binary)


def _write_segments(filename: str | Path, segments: list[Array], binary: bool = True) -> None:
    from .vtk import write_vtp

    points = np.vstack(segments) if segments else np.zeros((0, 3))
    lines = np.arange(points.shape[0], dtype=np.int64).reshape(-1, 2)
    write_vtp(filename, points, lines=lines, binary=binary)


def triangle_vtk_arrays(u: TriangleSurfaceFunction, nvis: int | None = None) -> tuple[Array, Array, Array]:
    """Concatenated points, triangle connectivity, and nodal values."""
    from .vtk import field_point_values, surface_grid

    points, cells, _, _ = surface_grid(u.domain, nvis)
    return points, cells, field_point_values(u, nvis)


def triangle_surface_mesh_arrays(dom: TriangleSurfaceMesh, nvis: int | None = None) -> tuple[Array, Array]:
    """Concatenated points and triangle connectivity for a triangular surface mesh."""
    from .vtk import surface_grid

    points, cells, _, _ = surface_grid(dom, nvis)
    return points, cells


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
        vals = list(u.data @ B.T)
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
    return list(dom._X @ B.T), list(dom._Y @ B.T), list(dom._Z @ B.T), nplot


@lru_cache(maxsize=64)
def tri_resample_matrix(
    nsource: int, ntarget: int, source_family: str = "cheb2", target_family: str = "cheb2"
) -> Array:
    """Interpolation matrix from one triangular node set to another."""
    source_family = _canonical_family_name(source_family)
    target_family = _canonical_family_name(target_family)
    usrc, vsrc = tri_reference_nodes(nsource, family=source_family)
    utgt, vtgt = tri_reference_nodes(ntarget, family=target_family)
    degree = nsource - 1
    Ksrc, _, _ = koornwinder_pkd(degree, usrc, vsrc)
    Ktgt, _, _ = koornwinder_pkd(degree, utgt, vtgt)
    return Ktgt @ np.linalg.solve(Ksrc, np.eye(Ksrc.shape[0]))


def tri_patch_boundary_segments(
    dom: TriangleSurfaceMesh, nvis: int | None = None, deduplicate: bool = True
) -> list[Array]:
    """Return line segments following the curved high-order patch boundaries."""
    nplot = dom.n if nvis is None else int(nvis)
    if nplot < 2:
        raise ValueError("nvis must be at least 2")
    if nplot == dom.n:
        xpatches, ypatches, zpatches = dom.x, dom.y, dom.z
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
            for a, b in itertools.pairwise(idx):
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


def write_ascii_vtu(
    filename: str | Path, points: Array, triangles: Array, point_data: dict[str, Array] | None = None
) -> None:
    """Write an ASCII VTK UnstructuredGrid with triangle cells."""
    from .vtk import VTK_TRIANGLE, write_vtu

    write_vtu(filename, points, triangles, VTK_TRIANGLE, point_data, binary=False)


def write_ascii_vtp(
    filename: str | Path, points: Array, triangles: Array, point_data: dict[str, Array] | None = None
) -> None:
    """Write an ASCII VTK PolyData surface with triangle polygons."""
    from .vtk import write_vtp

    write_vtp(filename, points, polys=triangles, point_data=point_data, binary=False)


def write_ascii_vtp_lines(filename: str | Path, segments: list[Array]) -> None:
    """Write line segments to an ASCII VTK PolyData file."""
    _write_segments(filename, segments, binary=False)


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
                for a, b in itertools.pairwise(idx):
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
            style = {"colors": color, "linewidths": linewidth, "linestyles": linestyle, "zorder": 10}
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
            style = {"colors": color, "linewidths": linewidth, "linestyles": linestyle}
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

        _set_equal_3d_axes(
            ax,
            np.concatenate([x.ravel() for x in dom.x]),
            np.concatenate([y.ravel() for y in dom.y]),
            np.concatenate([z.ravel() for z in dom.z]),
        )

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
        J = [
            (a * a + c * c + e * e) * (b * b + d * d + f * f) - (a * b + c * d + e * f) ** 2
            for a, b, c, d, e, f in zip(xu, xv, yu, yv, zu, zv)
        ]

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
        volume += float(
            np.sum(
                ((x * normal_arr[:, 0] + y * normal_arr[:, 1] + z * normal_arr[:, 2]) / 3.0)
                * weights
                * np.sqrt(np.maximum(jac, 0.0))
            )
        )
    if volume < 0.0:
        normals = [-normal_arr for normal_arr in normals]
    return normals


def _quad_patch_normals(dom: SurfaceMesh) -> list[Array]:
    normals = []
    volume = 0.0
    weights = quadwts(dom.n)
    W = np.outer(weights, weights)
    for x, y, z, xu, xv, yu, yv, zu, zv, J in zip(
        dom.x, dom.y, dom.z, dom.xu, dom.xv, dom.yu, dom.yv, dom.zu, dom.zv, dom.J
    ):
        n0 = yu * zv - zu * yv
        n1 = zu * xv - xu * zv
        n2 = xu * yv - yu * xv
        nrm = np.sqrt(np.maximum(n0 * n0 + n1 * n1 + n2 * n2, 1e-300))
        normal_arr = np.stack((n0 / nrm, n1 / nrm, n2 / nrm), axis=2)
        normals.append(normal_arr)
        volume += float(
            np.sum(
                ((x * normal_arr[:, :, 0] + y * normal_arr[:, :, 1] + z * normal_arr[:, :, 2]) / 3.0)
                * W
                * np.sqrt(np.maximum(J, 0.0))
            )
        )
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
    cmap: str = "turbo",
    ax=None,
    show: bool = True,
):
    """Matplotlib visualization of a triangular surface function.

    Returns ``(fig, ax)``.  Complex fields are plotted by their real part.
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if ax is None:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    data = np.real(u.data) if np.iscomplexobj(u.data) else u.data
    vmin = float(np.min(data)) if vmin is None else vmin
    vmax = float(np.max(data)) if vmax is None else vmax
    norm_obj = Normalize(vmin=vmin, vmax=vmax)
    colormap = plt.get_cmap(cmap)

    dom = u.domain
    tris = dom.triangles
    coords = np.stack((dom._X, dom._Y, dom._Z), axis=-1)
    faces = coords[:, tris].reshape(-1, 3, 3)
    face_values = np.mean(data[:, tris], axis=2).ravel()
    ax.add_collection3d(
        Poly3DCollection(faces, facecolors=colormap(norm_obj(face_values)), edgecolors=(0, 0, 0, 0.18), linewidths=0.12)
    )
    ax.set_xlim(float(np.min(dom._X)), float(np.max(dom._X)))
    ax.set_ylim(float(np.min(dom._Y)), float(np.max(dom._Y)))
    ax.set_zlim(float(np.min(dom._Z)), float(np.max(dom._Z)))
    ax.set_title(title)
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1))
    if colorbar:
        mappable = ScalarMappable(norm=norm_obj, cmap=colormap)
        mappable.set_array([])
        fig.colorbar(mappable, ax=ax, shrink=0.75, pad=0.02)
    if show:
        plt.show()
    return fig, ax


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
