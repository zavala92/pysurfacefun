"""
High-order fast direct solver for elliptic surface PDEs on quadrilateral Chebyshev patches.

This module is a Python version of the MATLAB Surfacefun package:
https://github.com/danfortunato/surfacefun

It provides geometry construction, strong-form surface differentiation, scalar
and vector surface functions, and fast direct solves of

.. math::

    \\sum_{c,d} a_{cd} D_c D_d u + \\sum_c b_c D_c u + b_0 u = f

with constant or variable (real or complex) coefficients.  For the pure
Laplace--Beltrami problem on a closed surface, set ``rankdef=True`` and use a
mean-zero right-hand side.

Storage
-------
A :class:`SurfaceMesh` with ``P`` patches of ``n x n`` Chebyshev points stores
its coordinates and metric terms as contiguous ``(P, n, n)`` arrays, and a
:class:`SurfaceFunction` stores its values as one ``(P, n, n)`` array
(``field.data``).  The list-valued attributes of the original API (``dom.x``,
``f.vals``, ...) are lists of views into that storage.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from .fields import PatchField, PatchVectorField
from .hps import Leaf, LeafOperators, Parent, Patch, merge_idx_from_tree, merge_patches, natural_tree
from .operators import (
    FIRST_ORDER_TERMS,
    PDO,
    SECOND_ORDER_TERMS,
    HPSOperator,
    coefficient_arrays,
    evaluate_rhs_nodes,
    parse_pdo,
)

Array = np.ndarray

__all__ = [
    "PDO",
    "SurfaceFunction",
    "SurfaceMesh",
    "SurfaceOp",
    "SurfaceVectorFunction",
    "build_quad_leaves",
    "parse_pdo",
]


# ---------------------------------------------------------------------------
# Chebyshev utilities.
# ---------------------------------------------------------------------------


def chebpts(n: int, kind: int = 2, interval: tuple[float, float] = (-1.0, 1.0)) -> Array:
    """Chebyshev points of the first or second kind, ordered from left to right."""
    if n <= 0:
        return np.empty(0)
    if n == 1:
        x = np.array([0.0])
    elif kind == 1:
        x = np.polynomial.chebyshev.chebpts1(n)
    elif kind == 2:
        x = np.polynomial.chebyshev.chebpts2(n)
    else:
        raise ValueError("kind must be 1 or 2")
    a, b = interval
    return 0.5 * ((b - a) * x + (a + b))


def chebpts2(
    nx: int,
    ny: int | None = None,
    D: Array | tuple[float, float, float, float] | list[float] | None = None,
    kind: int = 2,
) -> tuple[Array, Array]:
    """Tensor-product Chebyshev grid ``(X, Y)`` on the rectangle ``D = [x0, x1, y0, y1]``."""
    if ny is None:
        ny = nx
    if D is None:
        D = np.array([-1.0, 1.0, -1.0, 1.0])
    else:
        D = np.array(D, dtype=float).flatten()
        if D.size != 4:
            raise ValueError("Unrecognized domain.")
    if kind is None:
        kind = 2
    if kind not in (1, 2):
        raise ValueError("kind must be 1 or 2")
    x = chebpts(nx, kind)
    y = chebpts(ny, kind)
    x = 0.5 * ((D[1] - D[0]) * x + (D[0] + D[1]))
    y = 0.5 * ((D[3] - D[2]) * y + (D[2] + D[3]))
    return np.meshgrid(x, y)


@lru_cache(maxsize=64)
def _diffmat_cached(n: int) -> Array:
    if n == 0:
        return np.empty((0, 0))
    if n == 1:
        return np.zeros((1, 1))
    x = chebpts(n, 2)
    w = (-1.0) ** np.arange(n)
    w[0] *= 0.5
    w[-1] *= 0.5
    dx = x[:, None] - x[None, :]
    np.fill_diagonal(dx, 1.0)
    D = w[None, :] / (w[:, None] * dx)
    np.fill_diagonal(D, 0.0)
    D[np.diag_indices(n)] = -np.sum(D, axis=1)
    D.setflags(write=False)
    return D


def diffmat(n: int) -> Array:
    """Dense Chebyshev (second kind) nodal differentiation matrix."""
    return _diffmat_cached(int(n)).copy()


def barycentric_weights(x: Array) -> Array:
    """Barycentric interpolation weights for the nodes ``x``."""
    x = np.asarray(x, dtype=float).ravel()
    diff = x[:, None] - x[None, :]
    np.fill_diagonal(diff, 1.0)
    return 1.0 / np.prod(diff, axis=1)


def barymat(x_eval: Array, x_nodes: Array, weights: Array | None = None) -> Array:
    """Barycentric interpolation matrix from ``x_nodes`` to ``x_eval``."""
    x_eval = np.asarray(x_eval, dtype=float).ravel()
    x_nodes = np.asarray(x_nodes, dtype=float).ravel()
    if weights is None:
        weights = barycentric_weights(x_nodes)
    diff = x_eval[:, None] - x_nodes[None, :]
    hit = np.abs(diff) <= 1e-14
    with np.errstate(divide="ignore", invalid="ignore"):
        tmp = weights[None, :] / diff
        B = tmp / np.sum(tmp, axis=1, keepdims=True)
    rows = np.flatnonzero(np.any(hit, axis=1))
    if rows.size:
        B[rows] = 0.0
        B[rows, np.argmax(hit[rows], axis=1)] = 1.0
    return B


@lru_cache(maxsize=64)
def _quadwts_cached(n: int, kind: int) -> Array:
    if n <= 0:
        return np.empty(0)
    if n == 1:
        return np.array([2.0])
    if kind == 2:
        c = 2 / np.concatenate(([1], 1 - np.arange(2, n, 2) ** 2), axis=0)
        c = np.concatenate((c, c[int(n / 2) - 1 : 0 : -1]), axis=0)
        w = np.real(np.fft.ifft(c))
        w[0] = w[0] / 2
        w = np.concatenate((w, [w[0]]), axis=0)
    else:
        # Fejer's first rule on Chebyshev points of the first kind.
        theta = (2 * np.arange(n) + 1) * np.pi / (2 * n)
        k = np.arange(1, n // 2 + 1)
        w = (2.0 / n) * (1.0 - 2.0 * np.sum(np.cos(2 * np.outer(theta, k)) / (4 * k**2 - 1), axis=1))
        w = w[::-1]
    w.setflags(write=False)
    return w


def quadwts(n: int, kind: int = 2) -> Array:
    """Quadrature weights on ``n`` Chebyshev points of the given kind on ``[-1, 1]``.

    Kind 2 uses the Clenshaw--Curtis rule and kind 1 Fejer's first rule.
    """
    return _quadwts_cached(int(n), int(kind)).copy()


def rowscale(scale: Array, A: Array) -> Array:
    """Multiply the rows of ``A`` by ``scale``."""
    return np.asarray(scale).reshape(-1, 1) * A


def get_work_array(work: Array | None, shape: tuple[int, ...], dtype) -> Array:
    """Zeroed work array, reusing ``work`` when shape and dtype match."""
    dtype = np.dtype(dtype)
    if work is None or work.shape != shape or work.dtype != dtype:
        return np.zeros(shape, dtype=dtype)
    work.fill(0)
    return work


def block_diag(blocks: Iterable[Array]) -> Array:
    """Dense block-diagonal matrix."""
    blocks = [np.asarray(B) for B in blocks]
    if not blocks:
        return np.zeros((0, 0))
    nr = sum(B.shape[0] for B in blocks)
    nc = sum(B.shape[1] for B in blocks)
    out = np.zeros((nr, nc), dtype=np.result_type(*blocks))
    r = c = 0
    for B in blocks:
        out[r : r + B.shape[0], c : c + B.shape[1]] = B
        r += B.shape[0]
        c += B.shape[1]
    return out


# ---------------------------------------------------------------------------
# Surface meshes.
# ---------------------------------------------------------------------------


def _stack_patches(values: Any, name: str) -> Array:
    if isinstance(values, np.ndarray) and values.ndim == 3:
        return np.ascontiguousarray(values, dtype=float)
    arrays = [np.asarray(v, dtype=float) for v in values]
    if not arrays:
        raise ValueError(f"{name} must contain at least one patch")
    shape = arrays[0].shape
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(f"{name} patches must be square n x n arrays")
    if any(a.shape != shape for a in arrays):
        raise ValueError(f"all {name} patches must have the same shape")
    return np.stack(arrays)


class SurfaceMesh:
    """Mesh of ``P`` tensor-product Chebyshev patches with ``n x n`` points each.

    Parameters
    ----------
    x, y, z:
        Patch coordinates: sequences of ``(n, n)`` arrays or ``(P, n, n)``
        arrays.  Row index ``i`` follows the second parameter ``v`` and column
        index ``j`` the first parameter ``u``.

    Attributes
    ----------
    x, y, z, xu, xv, ..., ux, vx, ..., E, F, G, J:
        Per-patch lists of views into contiguous ``(P, n, n)`` arrays:
        coordinates, parametric derivatives, inverse-metric terms
        (``ux = du/dx`` along the surface), first fundamental form, and its
        determinant.  For patches flagged in ``singular`` the inverse-metric
        terms are *not* divided by ``J`` and the solver scales the equations
        accordingly.
    """

    def __init__(self, x: Any, y: Any, z: Any):
        X = _stack_patches(x, "x")
        Y = _stack_patches(y, "y")
        Z = _stack_patches(z, "z")
        if not (X.shape == Y.shape == Z.shape):
            raise ValueError("x, y, z must have the same number and shape of patches")
        self._X, self._Y, self._Z = X, Y, Z
        n = X.shape[1]
        D = _diffmat_cached(n)

        XU, XV = X @ D.T, np.matmul(D, X)
        YU, YV = Y @ D.T, np.matmul(D, Y)
        ZU, ZV = Z @ D.T, np.matmul(D, Z)
        E = XU * XU + YU * YU + ZU * ZU
        G = XV * XV + YV * YV + ZV * ZV
        F = XU * XV + YU * YV + ZU * ZV
        J = E * G - F * F

        num_ux = G * XU - F * XV
        scl = np.max(np.abs(num_ux).reshape(X.shape[0], -1), axis=1)
        singular = np.any((np.abs(J) < 1e-10 * np.maximum(scl, 1.0)[:, None, None]).reshape(X.shape[0], -1), axis=1)
        regular = ~singular[:, None, None]

        def inverse_metric(numerator: Array) -> Array:
            out = numerator.copy()
            np.divide(numerator, J, out=out, where=np.broadcast_to(regular, J.shape))
            return out

        self._XU, self._XV, self._YU, self._YV, self._ZU, self._ZV = XU, XV, YU, YV, ZU, ZV
        self._UX = inverse_metric(num_ux)
        self._UY = inverse_metric(G * YU - F * YV)
        self._UZ = inverse_metric(G * ZU - F * ZV)
        self._VX = inverse_metric(E * XV - F * XU)
        self._VY = inverse_metric(E * YV - F * YU)
        self._VZ = inverse_metric(E * ZV - F * ZU)
        self._E, self._F, self._G, self._J = E, F, G, J
        self._singular = singular
        self._weights: Array | None = None

        for name in ("x", "y", "z", "xu", "xv", "yu", "yv", "zu", "zv", "ux", "vx", "uy", "vy", "uz", "vz"):
            setattr(self, name, list(getattr(self, "_" + name.upper())))
        self.E, self.F, self.G, self.J = list(E), list(F), list(G), list(J)
        self.singular = [bool(s) for s in singular]

    def __repr__(self) -> str:
        return f"SurfaceMesh(npatches={self.npatches}, n={self.n})"

    @property
    def npatches(self) -> int:
        """Number of patches."""
        return int(self._X.shape[0])

    @property
    def n(self) -> int:
        """Chebyshev points per patch direction."""
        return int(self._X.shape[1])

    @property
    def order(self) -> int:
        """Polynomial degree per direction."""
        return self.n - 1

    @property
    def coords(self) -> Array:
        """Coordinates as a ``(3, P, n, n)`` array."""
        return np.stack((self._X, self._Y, self._Z))

    @property
    def quadrature_weights(self) -> Array:
        """Clenshaw--Curtis weights times the surface Jacobian, shape ``(P, n, n)``."""
        if self._weights is None:
            w = _quadwts_cached(self.n, 2)
            self._weights = np.outer(w, w)[None, :, :] * np.sqrt(np.maximum(self._J, 0.0))
            self._weights.setflags(write=False)
        return self._weights

    @staticmethod
    def sphere(n: int, nref: int = 0, projection: str = "quasiuniform") -> SurfaceMesh:
        """Six-patch cubed-sphere discretization of the unit sphere with ``4**nref`` patches per face."""
        v = np.array([[-1.0, 1.0, -1.0, 1.0]])
        for _ in range(nref):
            vnew = np.zeros((4 * v.shape[0], 4))
            for k, vk in enumerate(v):
                mid = np.array([np.mean(vk[:2]), np.mean(vk[2:])])
                vnew[4 * k + 0] = [vk[0], mid[0], vk[2], mid[1]]
                vnew[4 * k + 1] = [mid[0], vk[1], vk[2], mid[1]]
                vnew[4 * k + 2] = [vk[0], mid[0], mid[1], vk[3]]
                vnew[4 * k + 3] = [mid[0], vk[1], mid[1], vk[3]]
            v = vnew

        xx0, yy0 = chebpts2(n, n, [0.0, 1.0, 0.0, 1.0])
        uu = np.stack([(vk[1] - vk[0]) * xx0 + vk[0] for vk in v], axis=2)
        vv = np.stack([(vk[3] - vk[2]) * yy0 + vk[2] for vk in v], axis=2)
        ones = np.ones_like(uu)
        xx = np.concatenate((-ones, ones, vv, vv, uu, uu), axis=2)
        yy = np.concatenate((uu, uu, -ones, ones, vv, vv), axis=2)
        zz = np.concatenate((vv, vv, uu, uu, -ones, ones), axis=2)

        if projection.lower() == "naive":
            xx, yy, zz = project_naive(xx, yy, zz)
        elif projection.lower() == "quasiuniform":
            xx, yy, zz = project_quasiuniform(xx, yy, zz)
        else:
            raise ValueError("projection must be 'naive' or 'quasiuniform'")
        return SurfaceMesh(np.moveaxis(xx, 2, 0), np.moveaxis(yy, 2, 0), np.moveaxis(zz, 2, 0))

    @staticmethod
    def torus(n: int, nu: int = 8, nv: int | None = None) -> SurfaceMesh:
        """Fourier-parametrized torus with ``nu x nv`` patches."""
        if nv is None:
            nv = nu
        ubreaks = np.linspace(0.0, 2.0 * np.pi, nu + 1)
        vbreaks = np.linspace(0.0, 2.0 * np.pi, nv + 1)
        x, y, z = [], [], []
        for ku in range(nu):
            for kv in range(nv):
                uu, vv = chebpts2(n, n, [ubreaks[ku], ubreaks[ku + 1], vbreaks[kv], vbreaks[kv + 1]])
                xx, yy, zz = eval_torus(uu, vv)
                x.append(xx)
                y.append(yy)
                z.append(zz)
        return SurfaceMesh(x, y, z)

    @staticmethod
    def stellarator(n: int, nu: int = 8, nv: int | None = None) -> SurfaceMesh:
        """Fourier-parametrized stellarator with ``nu x nv`` patches."""
        if nv is None:
            nv = nu
        ubreaks = np.linspace(0.0, 2.0 * np.pi, nu + 1)
        vbreaks = np.linspace(0.0, 2.0 * np.pi, nv + 1)
        x, y, z = [], [], []
        for ku in range(nu):
            for kv in range(nv):
                uu, vv = chebpts2(n, n, [ubreaks[ku], ubreaks[ku + 1], vbreaks[kv], vbreaks[kv + 1]])
                xx, yy, zz = eval_stellarator(uu, vv)
                x.append(xx)
                y.append(yy)
                z.append(zz)
        if is_power_of_two(nv) and is_power_of_two(nu):
            ordering = morton(nv, nu).ravel(order="F") - 1
            x, y, z = reorder_cells_by_destination(ordering, x, y, z)
        return SurfaceMesh(x, y, z)

    @staticmethod
    def from_rhino(filename: str, n: int) -> SurfaceMesh:
        """Import a Rhino CSV patch mesh with columns ``x,y,z`` and ``n*n`` rows per patch."""
        data = np.loadtxt(filename, delimiter=",")
        if data.ndim != 2 or data.shape[1] < 3:
            raise ValueError("Rhino CSV must contain at least three numeric columns")
        rows_per_patch = n * n
        if data.shape[0] % rows_per_patch != 0:
            raise ValueError(f"row count {data.shape[0]} is not divisible by n*n={rows_per_patch}")
        nelem = data.shape[0] // rows_per_patch
        coords = [data[:, c].reshape((nelem, n, n)).transpose(0, 2, 1) for c in range(3)]
        return SurfaceMesh(*coords)


def project_naive(x: Array, y: Array, z: Array) -> tuple[Array, Array, Array]:
    """Radial projection of cube points onto the unit sphere."""
    nrm = np.sqrt(x * x + y * y + z * z)
    return x / nrm, y / nrm, z / nrm


def project_quasiuniform(x: Array, y: Array, z: Array) -> tuple[Array, Array, Array]:
    """Equal-area-like projection of cube points onto the unit sphere."""
    xp = x * np.sqrt(1 - y * y / 2 - z * z / 2 + (y * y * z * z) / 3)
    yp = y * np.sqrt(1 - x * x / 2 - z * z / 2 + (x * x * z * z) / 3)
    zp = z * np.sqrt(1 - x * x / 2 - y * y / 2 + (x * x * y * y) / 3)
    return xp, yp, zp


def eval_torus(u: Array, v: Array) -> tuple[Array, Array, Array]:
    """Evaluate the Fourier torus parametrization."""
    d = np.array(
        [
            [0.17, 0.11, 0.0, 0.0],
            [0.0, 1.0, 0.01, 0.0],
            [0.0, 4.5, 0.0, 0.0],
            [0.0, -0.25, -0.45, 0.0],
        ]
    )
    x = np.zeros_like(u)
    y = np.zeros_like(u)
    z = np.zeros_like(u)
    for i in range(-1, 3):
        for j in range(-1, 3):
            coeff = d[i + 1, j + 1]
            phase = (1 - i) * u + j * v
            x = x + coeff * np.cos(v) * np.cos(phase)
            y = y + coeff * np.sin(v) * np.cos(phase)
            z = z + coeff * np.sin(phase)
    return x, y, z


def eval_stellarator(u: Array, v: Array) -> tuple[Array, Array, Array]:
    """Evaluate the Fourier stellarator parametrization."""
    Q = 3
    R = 1.0
    r0 = 0.0
    z0 = 0.0
    d = np.array(
        [
            [0.15, 0.09, 0.00, 0.00, 0.00],
            [0.00, 1.00, 0.03, -0.01, 0.00],
            [0.08, 4.00, -0.01, -0.02, 0.00],
            [0.01, -0.28, -0.28, 0.03, 0.02],
            [0.00, 0.09, -0.03, 0.06, 0.00],
            [0.01, -0.02, 0.02, 0.00, -0.02],
        ]
    )
    rz = np.zeros_like(u, dtype=complex)
    for j in range(-1, 5):
        for k in range(-1, 4):
            rz = rz + d[j + 1, k + 1] * np.exp(-1j * j * u + 1j * k * Q * v)
    rz = np.exp(1j * u) * rz
    r = r0 + R * (np.real(rz) - r0)
    z = z0 + R * (np.imag(rz) - z0)
    return r * np.cos(v), r * np.sin(v), z


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def morton(m: int, n: int | None = None) -> Array:
    """Morton (Z-order) numbering matrix, as in Surfacefun's ``tools/morton.m``."""
    if n is None:
        n = m
    if m == 0 or n == 0:
        return np.empty((0, 0), dtype=int)
    if not (is_power_of_two(m) and is_power_of_two(n)):
        raise ValueError("m and n must be powers of two")
    if m == 1:
        return np.arange(1, n + 1).reshape(1, n)
    if n == 1:
        return np.arange(1, m + 1).reshape(m, 1)
    if m == 2 and n == 2:
        return np.array([[1, 2], [3, 4]])
    B = morton(m // 2, n // 2)
    shift = (m // 2) * (n // 2)
    return np.block([[B, B + shift], [B + 2 * shift, B + 3 * shift]])


def reorder_cells_by_destination(
    ordering: Array, x: list[Array], y: list[Array], z: list[Array]
) -> tuple[list[Array], list[Array], list[Array]]:
    newx: list[Array | None] = [None] * len(x)
    newy: list[Array | None] = [None] * len(y)
    newz: list[Array | None] = [None] * len(z)
    for dest, xx, yy, zz in zip(ordering, x, y, z):
        newx[int(dest)] = xx
        newy[int(dest)] = yy
        newz[int(dest)] = zz
    return list(newx), list(newy), list(newz)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Scalar and vector surface functions.
# ---------------------------------------------------------------------------


class SurfaceFunction(PatchField):
    """Scalar function sampled on the Chebyshev nodes of a :class:`SurfaceMesh`.

    ``f.data`` holds the values as a ``(P, n, n)`` array; ``f.vals`` is the
    list of per-patch views.  Fields support arithmetic with numbers, arrays,
    and other fields on the same mesh, and NumPy ufuncs (``np.sin(f)``).
    """

    _family = "quad"

    @staticmethod
    def patch_shape(domain: SurfaceMesh) -> tuple[int, ...]:
        return (domain.n, domain.n)

    def vec(self) -> Array:
        """Values stacked column-major per patch (legacy layout)."""
        return np.column_stack([v.ravel(order="F") for v in self.vals]).ravel(order="F")


class SurfaceVectorFunction(PatchVectorField):
    """Three-component vector field on a :class:`SurfaceMesh`."""

    _scalar_family = "quad"


def sphere(n: int, nref: int = 0, projection: str = "quasiuniform") -> SurfaceMesh:
    """Six-patch cubed-sphere discretization of the unit sphere."""
    return SurfaceMesh.sphere(n, nref, projection)


def torus(n: int, nu: int = 8, nv: int | None = None) -> SurfaceMesh:
    """Fourier-parametrized torus patch mesh."""
    return SurfaceMesh.torus(n, nu, nv)


def stellarator(n: int, nu: int = 8, nv: int | None = None) -> SurfaceMesh:
    """Fourier-parametrized stellarator patch mesh."""
    return SurfaceMesh.stellarator(n, nu, nv)


def from_rhino(filename: str, n: int) -> SurfaceMesh:
    """Load a Rhino-style CSV patch mesh."""
    return SurfaceMesh.from_rhino(filename, n)


def surfacefun(func: Callable[[Array, Array, Array], Array] | float | list[Array], dom: SurfaceMesh) -> SurfaceFunction:
    """Scalar surface function from a callable ``f(x, y, z)``, a number, or per-patch values."""
    if callable(func):
        return SurfaceFunction.from_callable(dom, func)
    if np.isscalar(func):
        return SurfaceFunction.constant(dom, func)  # type: ignore[arg-type]
    return SurfaceFunction(dom, func)


def surfacefunv(
    fx: Any,
    fy: Any = None,
    fz: Any = None,
    dom: SurfaceMesh | None = None,
) -> SurfaceVectorFunction:
    """Three-component vector surface function."""
    if isinstance(fx, SurfaceMesh) and fy is None and fz is None:
        zero = surfacefun(0.0, fx)
        return SurfaceVectorFunction((zero, zero.copy(), zero.copy()))
    if isinstance(fx, SurfaceFunction) and isinstance(fy, SurfaceFunction) and isinstance(fz, SurfaceFunction):
        return SurfaceVectorFunction((fx, fy, fz))
    if dom is None:
        raise ValueError("dom is required when components are not SurfaceFunction objects")
    if fy is None or fz is None:
        raise ValueError("all three vector components are required")
    return SurfaceVectorFunction((surfacefun(fx, dom), surfacefun(fy, dom), surfacefun(fz, dom)))


def surfaceop(dom: SurfaceMesh, op: dict, rhs: Any = 0.0, **kwargs) -> SurfaceOp:
    """Scalar elliptic surface operator with a fast direct solver."""
    return SurfaceOp(dom, op, rhs, **kwargs)


def compose(op: Callable, f: Any, g: Any = None) -> Any:
    """Apply a pointwise function to one or two surface functions."""
    if g is None:
        if not isinstance(f, PatchField):
            raise TypeError("single-argument compose expects a surface function")
        return f._new(np.asarray(op(f.data)))
    if isinstance(f, PatchField) and isinstance(g, PatchField):
        return f._new(np.asarray(op(f.data, g.data)))
    if isinstance(f, PatchField):
        return f._new(np.asarray(op(f.data, g)))
    if isinstance(g, PatchField):
        return g._new(np.asarray(op(f, g.data)))
    raise TypeError("at least one argument must be a surface function")


def exp(f: Any) -> Any:
    return compose(np.exp, f)


def log(f: Any) -> Any:
    return compose(np.log, f)


def log10(f: Any) -> Any:
    return compose(np.log10, f)


def sqrt(f: Any) -> Any:
    return compose(np.sqrt, f)


def sin(f: Any) -> Any:
    return compose(np.sin, f)


def cos(f: Any) -> Any:
    return compose(np.cos, f)


def real(f: Any) -> Any:
    return compose(np.real, f)


def imag(f: Any) -> Any:
    return compose(np.imag, f)


def conj(f: Any) -> Any:
    return compose(np.conj, f)


def maxEst(f: Any) -> float:
    """Largest sampled value."""
    return float(np.max(f.data))


def minEst(f: Any) -> float:
    """Smallest sampled value."""
    return float(np.min(f.data))


def resample_mesh(dom: SurfaceMesh, n: int) -> SurfaceMesh:
    """Resample every patch of a surface mesh to an ``n x n`` Chebyshev grid."""
    B = barymat(chebpts(n, 2), chebpts(dom.n, 2))
    return SurfaceMesh(*(np.matmul(B, np.matmul(C, B.T)) for C in (dom._X, dom._Y, dom._Z)))


def resample(f: SurfaceFunction, n: int) -> SurfaceFunction:
    """Resample a surface function to an ``n x n`` grid on each patch."""
    B = barymat(chebpts(n, 2), chebpts(f.domain.n, 2))
    return SurfaceFunction(resample_mesh(f.domain, n), np.matmul(B, np.matmul(f.data, B.T)))


def prolong(f: SurfaceFunction, n: int | None = None) -> SurfaceFunction:
    """Prolong or restrict a surface function to a new polynomial grid."""
    if n is None:
        return f.copy()
    return resample(f, n)


def _metric_pair(dom: SurfaceMesh, dim: int) -> tuple[Array, Array]:
    if dim == 1:
        return dom._UX, dom._VX
    if dim == 2:
        return dom._UY, dom._VY
    if dim == 3:
        return dom._UZ, dom._VZ
    raise ValueError("dim must be 1, 2, or 3")


def _diff_data(data: Array, dom: SurfaceMesh, dim: int) -> Array:
    D = _diffmat_cached(dom.n)
    du, dv = _metric_pair(dom, dim)
    return du * (data @ D.T) + dv * np.matmul(D, data)


def diff(f: SurfaceFunction, n: int | tuple[int, int, int] = 1, dim: int = 1) -> SurfaceFunction:
    """Tangential Cartesian derivative of a surface function.

    ``dim=1, 2, 3`` differentiates in ``x, y, z``.  ``n=(nx, ny, nz)`` applies
    mixed repeated derivatives (``x`` first).
    """
    if isinstance(n, tuple):
        nx, ny, nz = n
    elif dim in (1, 2, 3):
        counts = [0, 0, 0]
        counts[dim - 1] = int(n)
        nx, ny, nz = counts
    else:
        raise ValueError("dim must be 1, 2, or 3")
    data = f.data
    for d, count in ((1, nx), (2, ny), (3, nz)):
        for _ in range(count):
            data = _diff_data(data, f.domain, d)
    return SurfaceFunction(f.domain, data if data is not f.data else data.copy())


def mapped_vdiff(vals: Array, dom: SurfaceMesh, k: int, dim: int, D: Array) -> Array:
    """Tangential derivative of one patch's values (legacy helper)."""
    du, dv = _metric_pair(dom, dim)
    return du[k] * (vals @ D.T) + dv[k] * (D @ vals)


def diffx(f: SurfaceFunction, n: int = 1) -> SurfaceFunction:
    return diff(f, n, 1)


def diffy(f: SurfaceFunction, n: int = 1) -> SurfaceFunction:
    return diff(f, n, 2)


def diffz(f: SurfaceFunction, n: int = 1) -> SurfaceFunction:
    return diff(f, n, 3)


def gradient(f: SurfaceFunction) -> SurfaceVectorFunction:
    """Surface gradient as a three-component vector function."""
    return SurfaceVectorFunction((diffx(f), diffy(f), diffz(f)))


def grad(f: SurfaceFunction) -> SurfaceVectorFunction:
    return gradient(f)


def laplacian(f: SurfaceFunction) -> SurfaceFunction:
    """Laplace--Beltrami operator ``D_x D_x f + D_y D_y f + D_z D_z f``."""
    dom = f.domain
    out = _diff_data(_diff_data(f.data, dom, 1), dom, 1)
    out += _diff_data(_diff_data(f.data, dom, 2), dom, 2)
    out += _diff_data(_diff_data(f.data, dom, 3), dom, 3)
    return SurfaceFunction(dom, out)


def lap(f: SurfaceFunction) -> SurfaceFunction:
    return laplacian(f)


def _quad_edge_points(dom: SurfaceMesh) -> Array:
    """Start point, end point, and mean point of the four edges of every patch, ``(P, 4, 3, 3)``."""
    grid = np.stack((dom._X, dom._Y, dom._Z), axis=-1)
    edges = (grid[:, :, 0], grid[:, :, -1], grid[:, 0, :], grid[:, -1, :])  # left, right, down, up
    return np.stack([np.stack((e[:, 0], e[:, -1], e.mean(axis=1)), axis=1) for e in edges], axis=1)


#: Counterclockwise traversal sign of the stored left, right, down, up edges.
QUAD_EDGE_CCW = np.array([-1, 1, 1, -1])


def patch_orientation(dom: Any) -> Array:
    """Per-patch signs (``+1``/``-1``) that orient all patch normals consistently."""
    cached = getattr(dom, "_orientation", None)
    if cached is None:
        from .hps import edge_keys_from_points, orientation_signs

        if isinstance(dom, SurfaceMesh):
            points, ccw = _quad_edge_points(dom), QUAD_EDGE_CCW
        else:
            from .tri import TRI_EDGE_CCW, _tri_edge_points

            points, ccw = _tri_edge_points(dom), TRI_EDGE_CCW
        cached = orientation_signs(edge_keys_from_points(points), ccw)
        dom._orientation = cached
    return cached


def _unit_normals(dom: Any) -> tuple[Array, Array, Array]:
    """Unit normals oriented consistently and outward (positive enclosed volume)."""
    n0 = dom._YU * dom._ZV - dom._ZU * dom._YV
    n1 = dom._ZU * dom._XV - dom._XU * dom._ZV
    n2 = dom._XU * dom._YV - dom._YU * dom._XV
    signs = patch_orientation(dom).reshape((-1,) + (1,) * (n0.ndim - 1))
    scl = np.sqrt(n0 * n0 + n1 * n1 + n2 * n2) * signs
    n0, n1, n2 = n0 / scl, n1 / scl, n2 / scl
    volume = float(np.sum((dom._X * n0 + dom._Y * n1 + dom._Z * n2) / 3.0 * dom.quadrature_weights))
    if volume < 0:
        n0, n1, n2 = -n0, -n1, -n2
    return n0, n1, n2


def normal(dom: Any) -> PatchVectorField:
    """Unit normal, oriented consistently across patches and outward on closed surfaces."""
    n0, n1, n2 = _unit_normals(dom)
    if isinstance(dom, SurfaceMesh):
        return SurfaceVectorFunction((SurfaceFunction(dom, n0), SurfaceFunction(dom, n1), SurfaceFunction(dom, n2)))
    from .tri import TriangleSurfaceFunction, TriangleSurfaceVectorFunction

    return TriangleSurfaceVectorFunction(
        (TriangleSurfaceFunction(dom, n0), TriangleSurfaceFunction(dom, n1), TriangleSurfaceFunction(dom, n2))
    )


def dot(f: PatchVectorField, g: PatchVectorField) -> Any:
    """Pointwise dot product of two vector fields."""
    fc, gc = f.components, g.components
    return fc[0] * gc[0] + fc[1] * gc[1] + fc[2] * gc[2]


def cross(f: Any, g: Any, *args) -> Any:
    """Pointwise cross product of vector fields and/or constant 3-vectors."""
    if args:
        return cross(f, cross(g, *args))
    f_is_vec = isinstance(f, PatchVectorField)
    g_is_vec = isinstance(g, PatchVectorField)
    if f_is_vec and g_is_vec:
        fc, gc = f.components, g.components
        return type(f)((fc[1] * gc[2] - fc[2] * gc[1], fc[2] * gc[0] - fc[0] * gc[2], fc[0] * gc[1] - fc[1] * gc[0]))
    if f_is_vec:
        arr = np.asarray(g, dtype=float).ravel()
        if arr.size != 3:
            raise ValueError("constant vector must have length 3")
        fc = f.components
        return type(f)(
            (fc[1] * arr[2] - fc[2] * arr[1], fc[2] * arr[0] - fc[0] * arr[2], fc[0] * arr[1] - fc[1] * arr[0])
        )
    if g_is_vec:
        arr = np.asarray(f, dtype=float).ravel()
        if arr.size != 3:
            raise ValueError("constant vector must have length 3")
        gc = g.components
        return type(g)(
            (arr[1] * gc[2] - arr[2] * gc[1], arr[2] * gc[0] - arr[0] * gc[2], arr[0] * gc[1] - arr[1] * gc[0])
        )
    raise TypeError("at least one argument must be a vector surface function")


def divergence(f: SurfaceVectorFunction) -> SurfaceFunction:
    """Surface divergence of a vector field."""
    fc = f.components
    dom = fc[0].domain
    out = _diff_data(fc[0].data, dom, 1) + _diff_data(fc[1].data, dom, 2) + _diff_data(fc[2].data, dom, 3)
    return SurfaceFunction(dom, out)


def div(f: SurfaceVectorFunction) -> SurfaceFunction:
    return divergence(f)


def vector_norm(f: PatchVectorField) -> Any:
    """Pointwise Euclidean length of a vector field."""
    fc = f.components
    return sqrt(fc[0] * fc[0] + fc[1] * fc[1] + fc[2] * fc[2])


def normalize(f: PatchVectorField) -> PatchVectorField:
    """Unit vector field (zero vectors stay zero)."""
    mag = vector_norm(f)
    safe = mag._new(np.where(np.abs(mag.data) > 10 * np.finfo(float).eps, mag.data, 1.0))
    return f / safe


def hodge(
    f: SurfaceVectorFunction,
) -> tuple[SurfaceFunction, SurfaceFunction, SurfaceVectorFunction, SurfaceVectorFunction, SurfaceVectorFunction]:
    """Hodge decomposition of a tangential vector field.

    Returns ``u, v, w, curlfree, divfree`` with
    ``f = grad(u) + normal x grad(v) + w``.  Both scalar potentials reuse one
    factorization of the Laplace--Beltrami operator.
    """
    dom = f.domain
    nvec = normal(dom)
    L = SurfaceOp(dom, {"lap": 1.0}, rankdef=True)
    u = L.solve(div(f)).remove_mean()
    v = L.solve(-div(cross(nvec, f))).remove_mean()
    curlfree = grad(u)
    divfree = cross(nvec, grad(v))
    w = f - curlfree - divfree
    return u, v, w, curlfree, divfree


def integral2(f: Any, reduce: bool = True) -> Any:
    """Surface integral of a scalar surface function (per patch if ``reduce=False``)."""
    return f.integral(reduce=reduce)


def integral(f: Any, reduce: bool = True) -> Any:
    return integral2(f, reduce)


def sum2(f: Any, reduce: bool = True) -> Any:
    return integral2(f, reduce)


def mean2(f: Any) -> Any:
    """Surface average of a scalar function."""
    return f.mean2()


def surfacearea(dom: Any) -> float:
    """Total surface area."""
    return float(np.sum(dom.quadrature_weights))


def boundingbox(dom: Any) -> Array:
    """Bounding box ``[xmin, xmax, ymin, ymax, zmin, zmax]``."""
    out = []
    for coord in (dom.x, dom.y, dom.z):
        arr = np.concatenate([np.ravel(c) for c in coord])
        out.extend((float(np.min(arr)), float(np.max(arr))))
    return np.array(out)


def randnfun3(
    length_scale: float,
    bbox: Array | tuple[float, float, float, float, float, float],
    seed: int | None = None,
    nmodes: int = 64,
) -> Callable[[Array, Array, Array], Array]:
    """Smooth random function on a 3D bounding box (random Fourier features).

    Returns a callable ``f(x, y, z)``; the same ``seed`` reproduces the field.
    """
    bbox = np.asarray(bbox, dtype=float).ravel()
    if bbox.size != 6:
        raise ValueError("bbox must be [xmin, xmax, ymin, ymax, zmin, zmax]")
    if length_scale <= 0:
        raise ValueError("length_scale must be positive")

    rng = np.random.default_rng(seed)
    center = np.array([bbox[0] + bbox[1], bbox[2] + bbox[3], bbox[4] + bbox[5]]) / 2.0
    wavevectors = rng.normal(scale=1.0 / length_scale, size=(nmodes, 3))
    phases = rng.uniform(0.0, 2.0 * np.pi, size=nmodes)
    acos = rng.normal(size=nmodes) / np.sqrt(nmodes)
    asin = rng.normal(size=nmodes) / np.sqrt(nmodes)

    def field(x: Array, y: Array, z: Array) -> Array:
        shape = np.shape(x)
        pts = np.column_stack(
            (
                np.asarray(x).ravel() - center[0],
                np.asarray(y).ravel() - center[1],
                np.asarray(z).ravel() - center[2],
            )
        )
        theta = pts @ wavevectors.T + phases
        vals = np.cos(theta) @ acos + np.sin(theta) @ asin
        return vals.reshape(shape)

    return field


def smooth_random_function_3d(
    length_scale: float,
    bbox: Array | tuple[float, float, float, float, float, float],
    seed: int | None = None,
    nmodes: int = 64,
) -> Callable[[Array, Array, Array], Array]:
    """Descriptive alias for :func:`randnfun3`."""
    return randnfun3(length_scale, bbox, seed=seed, nmodes=nmodes)


def _gradient_components(f: Any) -> tuple[Any, Any, Any]:
    if isinstance(f, SurfaceFunction):
        return gradient(f).components
    from .tri import tri_diff  # local import: tri depends on core

    return tri_diff(f, 1), tri_diff(f, 2), tri_diff(f, 3)


def _laplacian_any(f: Any) -> Any:
    if isinstance(f, SurfaceFunction):
        return laplacian(f)
    from .tri import tri_lap

    return tri_lap(f)


def norm(f: Any, p: int | float | str = 2, reduce: bool = True) -> Any:
    """Norm of a scalar or vector surface field.

    Supported: ``1``, ``2``, any positive ``p``, ``"inf"``/``"max"``, ``"H1"``
    (``L2`` norm of the field and its gradient) and ``"lap"`` (``L2`` norm of
    the field and its Laplacian).  With ``reduce=False`` the per-patch
    contributions are returned.
    """
    if isinstance(f, PatchVectorField):
        mag = vector_norm(f)
        if p == 2 and reduce:
            return float(np.sqrt(np.real(integral2(mag * mag))))
        return norm(mag, p, reduce)
    if not isinstance(f, PatchField):
        raise TypeError("norm expects a scalar or vector surface field")

    npatches = f.npatches
    if p in (np.inf, "inf", "max"):
        per_patch = np.max(np.abs(f.data).reshape(npatches, -1), axis=1)
        return float(np.max(per_patch)) if reduce else per_patch
    if p == "H1":
        parts = [norm(f, 2, False)] + [norm(comp, 2, False) for comp in _gradient_components(f)]
        per_patch = np.sqrt(sum(part * part for part in parts))
        return float(np.sqrt(np.sum(per_patch * per_patch))) if reduce else per_patch
    if p == "lap":
        n0 = norm(f, 2, False)
        n1 = norm(_laplacian_any(f), 2, False)
        per_patch = np.sqrt(n0 * n0 + n1 * n1)
        return float(np.sqrt(np.sum(per_patch * per_patch))) if reduce else per_patch

    p_float = float(p)
    powered = f._new(np.abs(f.data) ** p_float)
    per_patch = np.real(powered.integral(reduce=False)) ** (1.0 / p_float)
    if reduce:
        return float(np.sum(per_patch**p_float) ** (1.0 / p_float))
    return per_patch


# ---------------------------------------------------------------------------
# Quadrilateral HPS leaves.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _QuadReference:
    """Reference data for ``n x n`` Chebyshev leaves (C-order node numbering)."""

    n: int
    D: Array
    interior: Array
    boundary: Array
    S2L: Array
    L2S: Array
    wskel: Array
    B: Array
    Du_boundary: Array
    Dv_boundary: Array


@lru_cache(maxsize=16)
def _quad_reference(n: int) -> _QuadReference:
    nskel = n - 2
    Xref, Yref = chebpts2(n)
    interior_f = (np.abs(Xref) < 1.0) & (np.abs(Yref) < 1.0)
    boundary_f = np.flatnonzero(~interior_f.ravel(order="F"))
    # Legacy (column-major) boundary sequence expressed in row-major node numbers.
    boundary = (boundary_f % n) * n + boundary_f // n
    idx = np.arange(1, n - 1)
    interior = (idx[:, None] * n + idx[None, :]).ravel()
    D = _diffmat_cached(n)
    eye = np.eye(n)
    return _QuadReference(
        n=n,
        D=D,
        interior=interior,
        boundary=boundary,
        S2L=skel2leaf(n, nskel),
        L2S=leaf2skel(nskel, n),
        wskel=np.tile(quadwts(nskel, 1), 4),
        B=barymat(chebpts(nskel, 1), chebpts(n, 2)),
        Du_boundary=np.kron(eye, D)[boundary],
        Dv_boundary=np.kron(D, eye)[boundary],
    )


def interior_mask_flat(n: int) -> Array:
    """Interior-node mask in column-major patch order (legacy helper)."""
    Xref, Yref = chebpts2(n)
    return ((np.abs(Xref) < 1.0) & (np.abs(Yref) < 1.0)).ravel(order="F")


def skel2leaf(nleaf: int, nskel: int) -> Array:
    """Interpolation from the four edge skeletons to the leaf boundary nodes (corners averaged)."""
    xskel = chebpts(nskel, 1)
    xleaf = chebpts(nleaf, 2)
    B = barymat(xleaf, xskel)
    left_skel = np.arange(0, nskel)
    right_skel = np.arange(nskel, 2 * nskel)
    down_skel = np.arange(2 * nskel, 3 * nskel)
    up_skel = np.arange(3 * nskel, 4 * nskel)
    left_leaf = np.arange(0, nleaf)
    right_leaf = np.arange(3 * nleaf - 4, 4 * nleaf - 4)
    up_leaf = np.r_[np.arange(nleaf - 1, 3 * nleaf - 4, 2), 4 * nleaf - 5]
    down_leaf = np.r_[0, np.arange(nleaf, 3 * nleaf - 3, 2)]
    P = np.zeros((4 * nleaf - 4, 4 * nskel))
    P[np.ix_(left_leaf, left_skel)] = B
    P[np.ix_(right_leaf, right_skel)] = B
    P[np.ix_(down_leaf, down_skel)] = B
    P[np.ix_(up_leaf, up_skel)] = B
    corners = np.array([0, nleaf - 1, 3 * nleaf - 4, 4 * nleaf - 5])
    P[corners, :] *= 0.5
    return P


def leaf2skel(nskel: int, nleaf: int) -> Array:
    """Interpolation from the leaf boundary nodes to the four edge skeletons."""
    xskel = chebpts(nskel, 1)
    xleaf = chebpts(nleaf, 2)
    B = barymat(xskel, xleaf)
    left_skel = np.arange(0, nskel)
    right_skel = np.arange(nskel, 2 * nskel)
    down_skel = np.arange(2 * nskel, 3 * nskel)
    up_skel = np.arange(3 * nskel, 4 * nskel)
    left_leaf = np.arange(0, nleaf)
    right_leaf = np.arange(3 * nleaf - 4, 4 * nleaf - 4)
    up_leaf = np.r_[np.arange(nleaf - 1, 3 * nleaf - 4, 2), 4 * nleaf - 5]
    down_leaf = np.r_[0, np.arange(nleaf, 3 * nleaf - 3, 2)]
    P = np.zeros((4 * nskel, 4 * nleaf - 4))
    P[np.ix_(left_skel, left_leaf)] = B
    P[np.ix_(right_skel, right_leaf)] = B
    P[np.ix_(down_skel, down_leaf)] = B
    P[np.ix_(up_skel, up_leaf)] = B
    return P


def normalize_rows(v: Array) -> Array:
    """Normalize vectors along the last axis; (near-)zero vectors are left unchanged."""
    nrm = np.linalg.norm(v, axis=-1, keepdims=True)
    nrm = np.where(nrm > 1e-14, nrm, 1.0)
    return v / nrm


def _edge_conormals(dom: SurfaceMesh) -> Array:
    """Unit outward conormals at the nodes of the four edges, shape ``(P, 4, n, 3)``."""
    Xu = np.stack((dom._XU, dom._YU, dom._ZU), axis=-1)
    Xv = np.stack((dom._XV, dom._YV, dom._ZV), axis=-1)
    raw = np.stack((-Xu[:, :, 0], Xu[:, :, -1], -Xv[:, 0, :], Xv[:, -1, :]), axis=1)
    tangent = normalize_rows(np.stack((Xv[:, :, 0], Xv[:, :, -1], Xu[:, 0, :], Xu[:, -1, :]), axis=1))
    return normalize_rows(raw - tangent * np.sum(raw * tangent, axis=-1, keepdims=True))


def binormals(dom: SurfaceMesh) -> tuple[Array, Array, Array, Array]:
    """Edge conormals ``NL, NR, ND, NU`` with shape ``(n, 3, P)`` (legacy layout)."""
    N = _edge_conormals(dom)
    return tuple(np.moveaxis(N[:, e], 0, -1) for e in range(4))  # type: ignore[return-value]


def _collect_coefficients(op: PDO, dom: Any, shape: tuple[int, ...]):
    second: dict[tuple[int, int], Array] = {}
    first: dict[int, Array] = {}
    zeroth: Array | None = None
    for name, value in op.nonzero().items():
        arr = coefficient_arrays(value, dom, shape)
        if name in SECOND_ORDER_TERMS:
            second[SECOND_ORDER_TERMS[name]] = arr
        elif name in FIRST_ORDER_TERMS:
            first[FIRST_ORDER_TERMS[name]] = arr
        else:
            zeroth = arr
    return second, first, zeroth


def _singular_scaling(second, first, zeroth, U, V, J, dJdu, dJdv, singular):
    """Multiply the equations on singular patches by ``J**3``.

    With unnormalized metric terms ``D~_c = J D_c`` the scaled operator is
    ``J a_cd D~_c D~_d + (J**2 b_d - a_cd D~_c J) D~_d + J**3 b0`` and the
    right-hand side is multiplied by ``J**3``.  ``dJdu``/``dJdv`` are the
    parametric derivatives of ``J``; other patches are left unchanged.
    """
    mask = singular.reshape((-1,) + (1,) * (J.ndim - 1))
    sJ = np.where(mask, J, 1.0)
    dJ = U * dJdu + V * dJdv
    new_first: dict[int, Array] = {}
    for d in range(3):
        value = sJ**2 * first[d] if d in first else None
        for c in range(3):
            if (c, d) in second:
                correction = np.where(mask, second[(c, d)] * dJ[c], 0.0)
                value = -correction if value is None else value - correction
        if value is not None:
            new_first[d] = value
    new_second = {key: sJ * value for key, value in second.items()}
    new_zeroth = None if zeroth is None else sJ**3 * zeroth
    return new_second, new_first, new_zeroth


def build_quad_leaves(op: PDO | dict, dom: SurfaceMesh, max_chunk_bytes: float = 1.5e8) -> LeafOperators:
    """Dense HPS leaf operators for every patch of a quadrilateral mesh.

    The collocation operator on an ``n x n`` Chebyshev leaf is assembled from
    the tensor-product structure ``D_u = I (x) D``, ``D_v = D (x) I``: every
    term of ``D_c D_d`` has at most four nonzero index patterns, so only the
    interior rows are formed, in ``O(n**4)`` work per patch instead of the
    ``O(n**6)`` dense products.  Leaves are processed in batches.
    """
    op = parse_pdo(op)
    n = dom.n
    if n < 3:
        raise ValueError("quadrilateral HPS leaves need at least n = 3 points per direction")
    ref = _quad_reference(n)
    P = dom.npatches
    N = n * n
    q = n - 2
    m = q * q
    nskel = n - 2
    b = 4 * nskel
    D = ref.D
    DI = D[1:-1]
    interior, boundary = ref.interior, ref.boundary
    shape = (P, n, n)

    second, first, zeroth = _collect_coefficients(op, dom, shape)
    U = np.stack((dom._UX, dom._UY, dom._UZ))
    V = np.stack((dom._VX, dom._VY, dom._VZ))
    J = dom._J
    singular = dom._singular
    rhs_scale = None
    flux_nodes = np.ones((P, N))
    if np.any(singular):
        second, first, zeroth = _singular_scaling(second, first, zeroth, U, V, J, J @ D.T, np.matmul(D, J), singular)
        J3 = np.where(singular[:, None, None], J**3, 1.0).reshape(P, N)
        rhs_scale = J3[:, interior]
        flux_nodes = np.where(singular[:, None, None], J**2, 1.0).reshape(P, N)
    dtype = np.result_type(float, *second.values(), *first.values(), *([] if zeroth is None else [zeroth]))

    inner = (slice(None), slice(1, -1), slice(1, -1))
    zeros_inner = np.zeros((P, q, q), dtype=dtype)

    def combine(coeffs: dict, metric: Array, keys) -> Array:
        """Sum of coeffs[key] * metric[axis] over (key, axis) pairs, on interior nodes."""
        out = zeros_inner.copy()
        for key, axis in keys:
            if key in coeffs:
                out = out + coeffs[key][inner] * metric[axis][inner]
        return out

    Pd = Qd = None
    if second:
        # Left multipliers of the D_u (Pd) and D_v (Qd) parts of D_c D_d, summed over c.
        Pd = np.stack([combine(second, U, [((c, d), c) for c in range(3)]) for d in range(3)])
        Qd = np.stack([combine(second, V, [((c, d), c) for c in range(3)]) for d in range(3)])
    beta_u = combine(first, U, [(c, c) for c in range(3)]) if first else None
    beta_v = combine(first, V, [(c, c) for c in range(3)]) if first else None
    b0 = None if zeroth is None else zeroth[inner]

    S_all = np.empty((P, N, b), dtype=dtype)
    D2N_all = np.empty((P, b, b), dtype=dtype)
    Ainv_all = np.empty((P, m, m), dtype=dtype)
    G_all = np.empty((P, b, m), dtype=dtype)

    # Conormals at skeleton points and the flux operator pieces.
    NN = np.einsum("sn,pend->pesd", ref.B, _edge_conormals(dom)).reshape(P, b, 3)
    U_ee = (U * flux_nodes.reshape(shape)[None]).reshape(3, P, N)[:, :, boundary]
    V_ee = (V * flux_nodes.reshape(shape)[None]).reshape(3, P, N)[:, :, boundary]

    itemsize = np.dtype(dtype).itemsize
    per_patch = itemsize * (4 * q * q * n * n + 2 * m * m + 4 * N * b)
    chunk = int(max(1, min(P, max_chunk_bytes // max(per_patch, 1))))
    jj = np.arange(q)
    for start in range(0, P, chunk):
        stop = min(P, start + chunk)
        sl = slice(start, stop)
        c = stop - start
        A4 = np.zeros((c, q, q, n, n), dtype=dtype)
        T1 = np.zeros((c, q, q, n), dtype=dtype)
        T4 = np.zeros((c, q, q, n), dtype=dtype)
        if second:
            assert Pd is not None and Qd is not None
            PI, QI = Pd[:, sl], Qd[:, sl]
            # Mixed D_u D_v and D_v D_u terms: A4[c,i,j,i',j'] = D[i,i'] X2[c,i,j,j'] + X3[c,i,j,i'] D[j,j'].
            X2 = np.einsum("dcij,dcil->cijl", PI, V[:, sl, 1:-1, :]) * DI[None, None, :, :]
            X3 = np.einsum("dcij,dckj->cijk", QI, U[:, sl, :, 1:-1]) * DI[None, :, None, :]
            for a in range(q):  # one interior row index at a time keeps temporaries small
                np.multiply(DI[a][None, None, :, None], X2[:, a][:, :, None, :], out=A4[:, a])
                A4[:, a] += X3[:, a][:, :, :, None] * DI[None, :, None, :]
            W_uu = np.einsum("dcij,dcik->cijk", PI, U[:, sl, 1:-1, :])
            T1 += np.matmul(W_uu * DI[None, None, :, :], D)
            W_vv = np.einsum("dcij,dckj->cijk", QI, V[:, sl, :, 1:-1])
            T4 += np.matmul(W_vv * DI[None, :, None, :], D)
        if beta_u is not None:
            T1 += beta_u[sl][..., None] * DI[None, None, :, :]
            T4 += beta_v[sl][..., None] * DI[None, :, None, :]
        if b0 is not None:
            T1[:, :, jj, jj + 1] += b0[sl]
        for a in range(q):
            A4[:, a, :, a + 1, :] += T1[:, a]
            A4[:, :, a, :, a + 1] += T4[:, :, a]
        Ainv = np.linalg.inv(A4[:, :, :, 1:-1, 1:-1].reshape(c, m, m))
        Aie = A4.reshape(c, m, N)[:, :, boundary]
        S = np.empty((c, N, b), dtype=dtype)
        S[:, boundary, :] = ref.S2L
        S[:, interior, :] = np.matmul(np.matmul(Ainv, -Aie), ref.S2L)

        alpha = np.einsum("csk,kcp->csp", NN[sl], U_ee[:, sl]) * ref.L2S
        beta = np.einsum("csk,kcp->csp", NN[sl], V_ee[:, sl]) * ref.L2S
        normal_d = np.matmul(alpha, ref.Du_boundary) + np.matmul(beta, ref.Dv_boundary)
        if rhs_scale is not None:
            Ainv = Ainv * rhs_scale[sl][:, None, :]
        S_all[sl] = S
        D2N_all[sl] = np.matmul(normal_d, S)
        Ainv_all[sl] = Ainv
        G_all[sl] = np.matmul(normal_d[:, :, interior], Ainv)

    coords = np.stack((dom._X, dom._Y, dom._Z), axis=-1).reshape(P, N, 3)
    xyz = np.matmul(ref.L2S, coords[:, boundary, :])
    sqrtJ = np.sqrt(np.maximum(J, 0.0)).reshape(P, N)[:, boundary]
    w = ref.wskel * np.matmul(sqrtJ, ref.L2S.T)
    scale = None
    if np.any(singular):
        J3_boundary = np.where(singular[:, None, None], J**3, 1.0).reshape(P, N)[:, boundary]
        scale = np.matmul(J3_boundary, ref.L2S.T)

    grid = np.stack((dom._X, dom._Y, dom._Z), axis=-1)
    starts = np.stack((grid[:, 0, 0], grid[:, 0, -1], grid[:, 0, 0], grid[:, -1, 0]), axis=1)
    ends = np.stack((grid[:, -1, 0], grid[:, -1, -1], grid[:, 0, -1], grid[:, -1, -1]), axis=1)
    mids = xyz.reshape(P, 4, nskel, 3).mean(axis=2)
    return LeafOperators(
        S=S_all,
        D2N=D2N_all,
        Ainv=Ainv_all,
        G=G_all,
        interior=interior,
        edge_sizes=np.full(4, nskel, dtype=np.int64),
        edge_points=np.stack((starts, ends, mids), axis=2),
        xyz=xyz,
        w=w,
        scale=scale,
        centroids=coords.mean(axis=1),
    )


class SurfaceOp(HPSOperator):
    """Fast direct solver for a scalar elliptic operator on a :class:`SurfaceMesh`.

    Examples
    --------
    >>> dom = sphere(9, 1)
    >>> f = surfacefun(lambda x, y, z: x * y * z, dom)
    >>> L = SurfaceOp(dom, {"lap": 1.0, "c": 2.0}, (2.0 - 12.0) * f)
    >>> u = L.solve()                       # factor + solve
    >>> u2 = L.solve(3.0 * f)               # reuse the factorization
    """

    def _build_leaf_operators(self) -> LeafOperators:
        return build_quad_leaves(self.op, self.domain)

    def _rhs_values(self, rhs: Any) -> Array:
        n = self.domain.n
        return evaluate_rhs_nodes(rhs, self.domain, _quad_reference(n).interior, n * n)

    def _field_from_nodal(self, U: Array) -> SurfaceFunction:
        n = self.domain.n
        return SurfaceFunction(self.domain, U.reshape(self.domain.npatches, n, n))


# ---------------------------------------------------------------------------
# Legacy helpers kept for compatibility.
# ---------------------------------------------------------------------------


def default_merge_idx(npatches: int) -> list[list[tuple[int, int | None]]]:
    """Merge schedule that pairs consecutive patch indices (the original ordering)."""
    if npatches <= 1:
        return []
    return merge_idx_from_tree(natural_tree(npatches), npatches)


def stack_patch_vectors(cells: list[Array]) -> Array:
    return np.column_stack([np.asarray(a).ravel(order="F") for a in cells])


def solve_dense(A: Array, B: Array) -> Array:
    if A.shape[0] == 0:
        return np.zeros_like(B)
    return np.linalg.solve(A, B)


def inverse_dense(A: Array) -> Array:
    if A.shape[0] == 0:
        return np.zeros_like(A)
    return np.linalg.inv(A)


def solve_with_inverse(A_inv: Array, B: Array) -> Array:
    if A_inv.shape[0] == 0:
        return np.zeros((0, B.shape[1]), dtype=B.dtype)
    return A_inv @ B


def intersect_tol(A: Array, B: Array, tol: float) -> tuple[Array, Array]:
    """Rows of ``A`` matching a row of ``B`` within an L1 tolerance."""
    AI, BI = [], []
    for k in range(A.shape[0]):
        hit = np.flatnonzero(np.sum(np.abs(B - A[k, :]), axis=1) < tol)
        if hit.size:
            AI.append(k)
            BI.append(int(hit[0]))
    return np.asarray(AI, dtype=int), np.asarray(BI, dtype=int)


# ---------------------------------------------------------------------------
# Sphere test functions, plotting, and VTU export.
# ---------------------------------------------------------------------------


def associated_legendre(l: int, m: int, x: Array) -> Array:
    """Associated Legendre function ``P_l^m(x)`` (unnormalized, Condon--Shortley phase), ``m >= 0``."""
    if m < 0 or m > l:
        raise ValueError("require 0 <= m <= l")
    x = np.asarray(x, dtype=float)
    pmm = np.ones_like(x)
    if m > 0:
        somx2 = np.sqrt(np.maximum(1.0 - x * x, 0.0))
        fact = 1.0
        for _ in range(1, m + 1):
            pmm = pmm * (-fact * somx2)
            fact += 2.0
    if l == m:
        return pmm
    pmmp1 = x * (2 * m + 1) * pmm
    if l == m + 1:
        return pmmp1
    p_lm2, p_lm1 = pmm, pmmp1
    pll = np.zeros_like(x)
    for ll in range(m + 2, l + 1):
        pll = ((2 * ll - 1) * x * p_lm1 - (ll + m - 1) * p_lm2) / (ll - m)
        p_lm2, p_lm1 = p_lm1, pll
    return pll


def real_spherical_harmonic(l: int, m: int, x: Array, y: Array, z: Array) -> Array:
    """Real, unnormalized spherical harmonic ``Y_l^m`` evaluated on the unit sphere."""
    phi = np.arctan2(y, x)
    Plm = associated_legendre(l, abs(m), np.clip(z, -1.0, 1.0))
    if m >= 0:
        return Plm * np.cos(m * phi)
    return Plm * np.sin(abs(m) * phi)


def solve_laplace_beltrami_sphere(
    f: SurfaceFunction | Callable,
    n: int = 9,
    nref: int = 0,
    c: float = 0.0,
    rankdef: bool = True,
) -> SurfaceFunction:
    """Solve ``Delta_Gamma u + c u = f`` on the unit sphere.

    For ``c = 0`` the problem is rank deficient; use a mean-zero right-hand
    side and ``rankdef=True``.
    """
    dom = f.domain if isinstance(f, SurfaceFunction) else SurfaceMesh.sphere(n, nref)
    L = SurfaceOp(dom, {"lap": 1.0, "c": c}, f, rankdef=rankdef)
    u = L.solve()
    return u.remove_mean() if rankdef and c == 0.0 else u


def write_vtu(filename: Any, u: Any, point_name: str = "u", nvis: int | None = None, **kwargs) -> None:
    """Write a surface field to a ParaView ``.vtu`` file (no third-party dependency).

    Accepts quadrilateral or triangular scalar or vector fields.  ``nvis``
    resamples each patch for smoother visualization.  See
    :func:`pysurfacefun.vtk.write_vtu_fields` for all options.
    """
    from .vtk import write_vtu_fields

    write_vtu_fields(filename, {point_name: u}, nvis=nvis, **kwargs)


def _patch_values_for_plot(u: Any) -> list[Array]:
    return [np.real(v) if np.iscomplexobj(v) else v for v in u.vals]


def plot_surface(
    u: SurfaceFunction,
    title: str = "",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = True,
    cmap: str = "turbo",
    ax=None,
    show: bool = True,
):
    """Matplotlib visualization of a quadrilateral surface function.

    Returns ``(fig, ax)``.  Complex fields are plotted by their real part.
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    if ax is None:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure
    plot_vals = _patch_values_for_plot(u)
    vmin = float(np.min(plot_vals)) if vmin is None else vmin
    vmax = float(np.max(plot_vals)) if vmax is None else vmax
    norm_obj = Normalize(vmin=vmin, vmax=vmax)
    colormap = plt.get_cmap(cmap)
    for x, y, z, val in zip(u.domain.x, u.domain.y, u.domain.z, plot_vals):
        ax.plot_surface(
            x,
            y,
            z,
            facecolors=colormap(norm_obj(val)),
            linewidth=0.15,
            edgecolor=(0, 0, 0, 0.25),
            antialiased=True,
            shade=False,
        )
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


def plot_vector_field(
    f: SurfaceVectorFunction,
    title: str = "",
    stride: int = 2,
    scale: float = 0.6,
    ax=None,
    show: bool = True,
):
    """Plot the vector magnitude with sparse 3D quiver arrows.  Returns ``(fig, ax)``."""
    import matplotlib.pyplot as plt

    mag = vector_norm(f)
    if ax is None:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure
    vmin = float(np.min(mag.data))
    vmax = float(np.max(mag.data))
    fx, fy, fz = f.components
    for x, y, z, mval, vx, vy, vz in zip(f.domain.x, f.domain.y, f.domain.z, mag.vals, fx.vals, fy.vals, fz.vals):
        ax.plot_surface(
            x,
            y,
            z,
            facecolors=plt.cm.turbo((mval - vmin) / (vmax - vmin + 1e-15)),
            linewidth=0.1,
            edgecolor=(0, 0, 0, 0.12),
            antialiased=True,
            shade=False,
        )
        sl = (slice(None, None, stride), slice(None, None, stride))
        ax.quiver(x[sl], y[sl], z[sl], vx[sl], vy[sl], vz[sl], length=scale, normalize=True, color="k", linewidth=0.45)
    ax.set_title(title)
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1))
    if show:
        plt.show()
    return fig, ax


def demo_laplace_beltrami_sphere() -> None:
    """Solve a small Laplace--Beltrami problem on the unit sphere."""
    l, m, n = 3, 2, 7
    dom = SurfaceMesh.sphere(n, 0)
    exact = SurfaceFunction.from_callable(dom, lambda x, y, z: real_spherical_harmonic(l, m, x, y, z))
    L = SurfaceOp(dom, {"lap": 1.0}, -l * (l + 1) * exact, rankdef=True)
    uh = L.solve().remove_mean()
    relerr = (uh - exact.remove_mean()).norm_inf() / exact.norm_inf()
    print(f"relative L_inf error = {relerr:.3e}")


# Names re-exported for code that imported the solver internals from this module.
_LEGACY_HPS_NAMES = (Leaf, Parent, Patch, merge_patches)

if __name__ == "__main__":
    demo_laplace_beltrami_sphere()
