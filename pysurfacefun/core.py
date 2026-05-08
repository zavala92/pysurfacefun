"""
High-order fast direct solver for elliptic surface PDEs on quadrilateral Chebyshev patches.

This module is a reduced Python version of the MATLAB Surfacefun package:
https://github.com/danfortunato/surfacefun

It provides geometry construction, strong-form surface differentiation, scalar
surface functions, and direct patch-based solves for

    lapcoef * Delta_Gamma u + ccoef * u = f.

For pure Laplace-Beltrami problems on closed surfaces, set ``rankdef=True`` and
use a mean-zero right-hand side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np


Array = np.ndarray


# ---------------------------------------------------------------------------
# Chebyshev utilities.
# ---------------------------------------------------------------------------


def chebpts(n: int, kind: int = 2, interval: tuple[float, float] = (-1.0, 1.0)) -> Array:
    """Chebyshev points, ordered from left to right."""
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
    D: Array | tuple[float, float, float, float] | None = None,
    kind: int = 2,
) -> tuple[Array, Array]:
    """
    Tensor-product Chebyshev grid.

    The domain mapping is included here because the sphere constructor calls
    ``chebpts2(n, n, [0, 1, 0, 1])``.
    """
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

    if kind == 2:
        x = np.polynomial.chebyshev.chebpts2(nx)
        y = np.polynomial.chebyshev.chebpts2(ny)
    elif kind == 1:
        x = np.polynomial.chebyshev.chebpts1(nx)
        y = np.polynomial.chebyshev.chebpts1(ny)
    else:
        raise ValueError("kind must be 1 or 2")

    x = 0.5 * ((D[1] - D[0]) * x + (D[0] + D[1]))
    y = 0.5 * ((D[3] - D[2]) * y + (D[2] + D[3]))
    return np.meshgrid(x, y)


def diffmat(n: int) -> Array:
    """Dense Chebyshev nodal differentiation matrix."""
    x = chebpts(n, 2)
    if n == 0:
        return np.empty((0, 0))
    if n == 1:
        return np.zeros((1, 1))

    w = (-1.0) ** np.arange(n)
    w[0] *= 0.5
    w[-1] *= 0.5

    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = w[j] / (w[i] * (x[i] - x[j]))
    D[np.diag_indices(n)] = -np.sum(D, axis=1)
    return D


def barycentric_weights(x: Array) -> Array:
    """Generic barycentric weights for the small Chebyshev grids used here."""
    x = np.asarray(x, dtype=float)
    w = np.ones_like(x)
    for j in range(x.size):
        diff = x[j] - np.delete(x, j)
        w[j] = 1.0 / np.prod(diff)
    return w


def barymat(x_eval: Array, x_nodes: Array, weights: Array | None = None) -> Array:
    """Barycentric interpolation matrix."""
    x_eval = np.asarray(x_eval, dtype=float).ravel()
    x_nodes = np.asarray(x_nodes, dtype=float).ravel()
    if weights is None:
        weights = barycentric_weights(x_nodes)

    B = np.zeros((x_eval.size, x_nodes.size))
    for i, x in enumerate(x_eval):
        hit = np.where(np.isclose(x, x_nodes, rtol=0.0, atol=1e-14))[0]
        if hit.size:
            B[i, hit[0]] = 1.0
        else:
            tmp = weights / (x - x_nodes)
            B[i, :] = tmp / np.sum(tmp)
    return B


def quadwts(n: int, kind: int = 2) -> Array:
    """
    Quadrature weights on Chebyshev nodes.

    The second-kind formula is an FFT-based Clenshaw--Curtis routine.
    """
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
        return w
    return np.full(n, 2.0 / n)


def rowscale(scale: Array, A: Array) -> Array:
    return np.asarray(scale).reshape(-1, 1) * A


def get_work_array(work: Array | None, shape: tuple[int, ...], dtype) -> Array:
    dtype = np.dtype(dtype)
    if work is None or work.shape != shape or work.dtype != dtype:
        return np.zeros(shape, dtype=dtype)
    work.fill(0)
    return work


def block_diag(blocks: Iterable[Array]) -> Array:
    blocks = [np.asarray(B) for B in blocks]
    if not blocks:
        return np.zeros((0, 0))
    nr = sum(B.shape[0] for B in blocks)
    nc = sum(B.shape[1] for B in blocks)
    out = np.zeros((nr, nc))
    r = 0
    c = 0
    for B in blocks:
        out[r : r + B.shape[0], c : c + B.shape[1]] = B
        r += B.shape[0]
        c += B.shape[1]
    return out


# ---------------------------------------------------------------------------
# surfacemesh.sphere and metric data.
# ---------------------------------------------------------------------------


@dataclass
class SurfaceMesh:
    x: list[Array]
    y: list[Array]
    z: list[Array]

    def __post_init__(self) -> None:
        n = self.x[0].shape[0]
        D = diffmat(n)

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
            xu = x @ D.T
            xv = D @ x
            yu = y @ D.T
            yv = D @ y
            zu = z @ D.T
            zv = D @ z

            E = xu * xu + yu * yu + zu * zu
            G = xv * xv + yv * yv + zv * zv
            F = xu * xv + yu * yv + zu * zv
            J = E * G - F * F

            scl = np.max(np.abs(G * xu - F * xv))
            singular = bool(np.any(np.abs(J) < 1e-10 * max(scl, 1.0)))

            if singular:
                ux = G * xu - F * xv
                uy = G * yu - F * yv
                uz = G * zu - F * zv
                vx = E * xv - F * xu
                vy = E * yv - F * yu
                vz = E * zv - F * zu
            else:
                ux = (G * xu - F * xv) / J
                uy = (G * yu - F * yv) / J
                uz = (G * zu - F * zv) / J
                vx = (E * xv - F * xu) / J
                vy = (E * yv - F * yu) / J
                vz = (E * zv - F * zu) / J

            self.xu.append(xu)
            self.xv.append(xv)
            self.yu.append(yu)
            self.yv.append(yv)
            self.zu.append(zu)
            self.zv.append(zv)
            self.ux.append(ux)
            self.vx.append(vx)
            self.uy.append(uy)
            self.vy.append(vy)
            self.uz.append(uz)
            self.vz.append(vz)
            self.E.append(E)
            self.F.append(F)
            self.G.append(G)
            self.J.append(J)
            self.singular.append(singular)

    @property
    def npatches(self) -> int:
        return len(self.x)

    @property
    def n(self) -> int:
        return self.x[0].shape[0]

    @property
    def order(self) -> int:
        return self.n - 1

    @staticmethod
    def sphere(n: int, nref: int = 0, projection: str = "quasiuniform") -> "SurfaceMesh":
        """Build a six-patch Chebyshev discretization of the unit sphere."""
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
        uu = []
        vv = []
        for vk in v:
            sclx = vk[1] - vk[0]
            scly = vk[3] - vk[2]
            uu.append(sclx * xx0 + vk[0])
            vv.append(scly * yy0 + vk[2])
        uu = np.stack(uu, axis=2)
        vv = np.stack(vv, axis=2)

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

        x = [xx[:, :, k] for k in range(xx.shape[2])]
        y = [yy[:, :, k] for k in range(yy.shape[2])]
        z = [zz[:, :, k] for k in range(zz.shape[2])]
        return SurfaceMesh(x, y, z)

    @staticmethod
    def torus(n: int, nu: int = 8, nv: int | None = None) -> "SurfaceMesh":
        """Build a Fourier-parametrized torus patch mesh."""
        if nv is None:
            nv = nu

        x: list[Array] = []
        y: list[Array] = []
        z: list[Array] = []
        ubreaks = np.linspace(0.0, 2.0 * np.pi, nu + 1)
        vbreaks = np.linspace(0.0, 2.0 * np.pi, nv + 1)

        for ku in range(nu):
            for kv in range(nv):
                uu, vv = chebpts2(n, n, [ubreaks[ku], ubreaks[ku + 1], vbreaks[kv], vbreaks[kv + 1]])
                xx, yy, zz = eval_torus(uu, vv)
                x.append(xx)
                y.append(yy)
                z.append(zz)

        return SurfaceMesh(x, y, z)

    @staticmethod
    def stellarator(n: int, nu: int = 8, nv: int | None = None) -> "SurfaceMesh":
        """Build a Fourier-parametrized stellarator patch mesh."""
        if nv is None:
            nv = nu

        x: list[Array] = []
        y: list[Array] = []
        z: list[Array] = []
        ubreaks = np.linspace(0.0, 2.0 * np.pi, nu + 1)
        vbreaks = np.linspace(0.0, 2.0 * np.pi, nv + 1)

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
    def from_rhino(filename: str, n: int) -> "SurfaceMesh":
        """
        Import a Rhino CSV patch mesh.

        The file must have three numeric columns ``x,y,z`` and ``n*n``
        consecutive rows per patch.
        """
        data = np.loadtxt(filename, delimiter=",")
        if data.ndim != 2 or data.shape[1] < 3:
            raise ValueError("Rhino CSV must contain at least three numeric columns")

        rows_per_patch = n * n
        if data.shape[0] % rows_per_patch != 0:
            raise ValueError(
                f"row count {data.shape[0]} is not divisible by n*n={rows_per_patch}"
            )

        nelem = data.shape[0] // rows_per_patch
        X = data[:, 0].reshape((n, n * nelem), order="F")
        Y = data[:, 1].reshape((n, n * nelem), order="F")
        Z = data[:, 2].reshape((n, n * nelem), order="F")

        x = [X[:, k * n : (k + 1) * n].copy() for k in range(nelem)]
        y = [Y[:, k * n : (k + 1) * n].copy() for k in range(nelem)]
        z = [Z[:, k * n : (k + 1) * n].copy() for k in range(nelem)]
        return SurfaceMesh(x, y, z)


def project_naive(x: Array, y: Array, z: Array) -> tuple[Array, Array, Array]:
    nrm = np.sqrt(x * x + y * y + z * z)
    return x / nrm, y / nrm, z / nrm


def project_quasiuniform(x: Array, y: Array, z: Array) -> tuple[Array, Array, Array]:
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
    r1 = np.real(rz)
    z1 = np.imag(rz)
    r = r0 + R * (r1 - r0)
    z = z0 + R * (z1 - z0)

    x = r * np.cos(v)
    y = r * np.sin(v)
    return x, y, z


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def morton(m: int, n: int | None = None) -> Array:
    """Create the same Morton ordering matrix as ``tools/morton.m``."""
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


def reorder_cells_by_destination(ordering: Array, x: list[Array], y: list[Array], z: list[Array]) -> tuple[list[Array], list[Array], list[Array]]:
    newx: list[Array | None] = [None] * len(x)
    newy: list[Array | None] = [None] * len(y)
    newz: list[Array | None] = [None] * len(z)
    for dest, xx, yy, zz in zip(ordering, x, y, z):
        newx[int(dest)] = xx
        newy[int(dest)] = yy
        newz[int(dest)] = zz
    return list(newx), list(newy), list(newz)


# ---------------------------------------------------------------------------
# Scalar and vector surface functions.
# ---------------------------------------------------------------------------


@dataclass
class SurfaceFunction:
    domain: SurfaceMesh
    vals: list[Array]

    @staticmethod
    def from_callable(domain: SurfaceMesh, func: Callable[[Array, Array, Array], Array]) -> "SurfaceFunction":
        vals = [np.asarray(func(x, y, z)) for x, y, z in zip(domain.x, domain.y, domain.z)]
        return SurfaceFunction(domain, vals)

    @staticmethod
    def constant(domain: SurfaceMesh, value: float | complex) -> "SurfaceFunction":
        return SurfaceFunction(domain, [value * np.ones_like(x) for x in domain.x])

    def vec(self) -> Array:
        return np.column_stack([v.ravel(order="F") for v in self.vals]).ravel(order="F")

    def norm_inf(self) -> float:
        return float(max(np.max(np.abs(v)) for v in self.vals))

    def mean2(self) -> float:
        total = 0.0 + 0.0j
        area = 0.0
        wu = quadwts(self.domain.n, 2)
        W = np.outer(wu, wu)
        for v, J in zip(self.vals, self.domain.J):
            jac = np.sqrt(np.maximum(J, 0.0))
            total += np.sum(W * jac * v)
            area += float(np.sum(W * jac))
        out = total / area
        return float(np.real(out)) if np.isrealobj(out) else out

    def remove_mean(self) -> "SurfaceFunction":
        return self - self.mean2()

    def copy(self) -> "SurfaceFunction":
        return SurfaceFunction(self.domain, [v.copy() for v in self.vals])

    def __add__(self, other: float | "SurfaceFunction") -> "SurfaceFunction":
        if isinstance(other, SurfaceFunction):
            return SurfaceFunction(self.domain, [a + b for a, b in zip(self.vals, other.vals)])
        return SurfaceFunction(self.domain, [a + other for a in self.vals])

    __radd__ = __add__

    def __sub__(self, other: float | "SurfaceFunction") -> "SurfaceFunction":
        if isinstance(other, SurfaceFunction):
            return SurfaceFunction(self.domain, [a - b for a, b in zip(self.vals, other.vals)])
        return SurfaceFunction(self.domain, [a - other for a in self.vals])

    def __rsub__(self, other: float) -> "SurfaceFunction":
        return SurfaceFunction(self.domain, [other - a for a in self.vals])

    def __mul__(self, other: float | "SurfaceFunction") -> "SurfaceFunction":
        if isinstance(other, SurfaceFunction):
            return SurfaceFunction(self.domain, [a * b for a, b in zip(self.vals, other.vals)])
        return SurfaceFunction(self.domain, [a * other for a in self.vals])

    __rmul__ = __mul__

    def __truediv__(self, other: float | "SurfaceFunction") -> "SurfaceFunction":
        if isinstance(other, SurfaceFunction):
            return SurfaceFunction(self.domain, [a / b for a, b in zip(self.vals, other.vals)])
        return SurfaceFunction(self.domain, [a / other for a in self.vals])

    def __rtruediv__(self, other: float) -> "SurfaceFunction":
        return SurfaceFunction(self.domain, [other / a for a in self.vals])

    def __pow__(self, power: float) -> "SurfaceFunction":
        return SurfaceFunction(self.domain, [a**power for a in self.vals])

    def __abs__(self) -> "SurfaceFunction":
        return SurfaceFunction(self.domain, [np.abs(a) for a in self.vals])

    def __neg__(self) -> "SurfaceFunction":
        return SurfaceFunction(self.domain, [-a for a in self.vals])


@dataclass
class SurfaceVectorFunction:
    """Three-component vector field sampled on a surface mesh."""

    components: tuple[SurfaceFunction, SurfaceFunction, SurfaceFunction]

    @property
    def domain(self) -> SurfaceMesh:
        return self.components[0].domain

    def norm_inf(self) -> float:
        vals = zip(*(component.vals for component in self.components))
        return float(max(np.max(np.sqrt(a * a + b * b + c * c)) for a, b, c in vals))

    def __add__(self, other: float | Array | "SurfaceVectorFunction") -> "SurfaceVectorFunction":
        if isinstance(other, SurfaceVectorFunction):
            return SurfaceVectorFunction(tuple(a + b for a, b in zip(self.components, other.components)))
        arr = np.asarray(other)
        if arr.ndim == 0:
            return SurfaceVectorFunction(tuple(a + float(arr) for a in self.components))
        if arr.size == 3:
            return SurfaceVectorFunction(tuple(a + arr[i] for i, a in enumerate(self.components)))
        raise ValueError("can only add scalars, length-3 vectors, or SurfaceVectorFunction")

    __radd__ = __add__

    def __sub__(self, other: float | Array | "SurfaceVectorFunction") -> "SurfaceVectorFunction":
        return self + (-other)

    def __rsub__(self, other: float | Array) -> "SurfaceVectorFunction":
        return (-self) + other

    def __neg__(self) -> "SurfaceVectorFunction":
        return SurfaceVectorFunction(tuple(-a for a in self.components))

    def __mul__(self, other: float | SurfaceFunction) -> "SurfaceVectorFunction":
        return SurfaceVectorFunction(tuple(a * other for a in self.components))

    __rmul__ = __mul__

    def __truediv__(self, other: float | SurfaceFunction) -> "SurfaceVectorFunction":
        return SurfaceVectorFunction(tuple(a / other for a in self.components))


def sphere(n: int, nref: int = 0, projection: str = "quasiuniform") -> SurfaceMesh:
    """Build a six-patch Chebyshev discretization of the unit sphere."""
    return SurfaceMesh.sphere(n, nref, projection)


def torus(n: int, nu: int = 8, nv: int | None = None) -> SurfaceMesh:
    """Build a Fourier-parametrized torus patch mesh."""
    return SurfaceMesh.torus(n, nu, nv)


def stellarator(n: int, nu: int = 8, nv: int | None = None) -> SurfaceMesh:
    """Build a Fourier-parametrized stellarator patch mesh."""
    return SurfaceMesh.stellarator(n, nu, nv)


def from_rhino(filename: str, n: int) -> SurfaceMesh:
    """Load a Rhino-style CSV patch mesh."""
    return SurfaceMesh.from_rhino(filename, n)


def surfacefun(func: Callable[[Array, Array, Array], Array] | float | list[Array], dom: SurfaceMesh) -> SurfaceFunction:
    """Construct a scalar surface function on a ``SurfaceMesh``."""
    if callable(func):
        return SurfaceFunction.from_callable(dom, func)
    if np.isscalar(func):
        return SurfaceFunction.constant(dom, func)
    return SurfaceFunction(dom, [np.asarray(v) for v in func])


def surfacefunv(
    fx: Callable[[Array, Array, Array], Array] | SurfaceFunction | float | list[Array],
    fy: Callable[[Array, Array, Array], Array] | SurfaceFunction | float | list[Array] | None = None,
    fz: Callable[[Array, Array, Array], Array] | SurfaceFunction | float | list[Array] | None = None,
    dom: SurfaceMesh | None = None,
) -> SurfaceVectorFunction:
    """Construct a three-component vector surface function."""
    if isinstance(fx, SurfaceMesh) and fy is None and fz is None:
        zero = surfacefun(0.0, fx)
        return SurfaceVectorFunction((zero, zero.copy(), zero.copy()))

    if isinstance(fx, SurfaceFunction) and isinstance(fy, SurfaceFunction) and isinstance(fz, SurfaceFunction):
        return SurfaceVectorFunction((fx, fy, fz))

    if dom is None:
        raise ValueError("dom is required when components are not SurfaceFunction objects")

    assert fy is not None and fz is not None
    return SurfaceVectorFunction((surfacefun(fx, dom), surfacefun(fy, dom), surfacefun(fz, dom)))


def surfaceop(dom: SurfaceMesh, op: dict, rhs: SurfaceFunction | Callable | float = 0.0) -> "SurfaceOp":
    """Construct a scalar elliptic surface operator."""
    return SurfaceOp(dom, op, rhs)


def compose(op: Callable, f: SurfaceFunction | float, g: SurfaceFunction | float | None = None) -> SurfaceFunction:
    """
    Compose a scalar function with one or two surface functions.
    """
    if g is None:
        if not isinstance(f, SurfaceFunction):
            raise TypeError("single-argument compose expects a SurfaceFunction")
        return SurfaceFunction(f.domain, [op(v) for v in f.vals])

    if isinstance(f, SurfaceFunction) and isinstance(g, SurfaceFunction):
        return SurfaceFunction(f.domain, [op(a, b) for a, b in zip(f.vals, g.vals)])
    if isinstance(f, SurfaceFunction):
        return SurfaceFunction(f.domain, [op(a, g) for a in f.vals])
    if isinstance(g, SurfaceFunction):
        return SurfaceFunction(g.domain, [op(f, b) for b in g.vals])
    raise TypeError("at least one argument must be a SurfaceFunction")


def exp(f: SurfaceFunction) -> SurfaceFunction:
    return compose(np.exp, f)


def log(f: SurfaceFunction) -> SurfaceFunction:
    return compose(np.log, f)


def log10(f: SurfaceFunction) -> SurfaceFunction:
    return compose(np.log10, f)


def sqrt(f: SurfaceFunction) -> SurfaceFunction:
    return compose(np.sqrt, f)


def sin(f: SurfaceFunction) -> SurfaceFunction:
    return compose(np.sin, f)


def cos(f: SurfaceFunction) -> SurfaceFunction:
    return compose(np.cos, f)


def real(f: SurfaceFunction) -> SurfaceFunction:
    return compose(np.real, f)


def imag(f: SurfaceFunction) -> SurfaceFunction:
    return compose(np.imag, f)


def conj(f: SurfaceFunction) -> SurfaceFunction:
    return compose(np.conj, f)


def maxEst(f: SurfaceFunction) -> float:
    """Estimate the maximum sampled value."""
    return float(max(np.max(v) for v in f.vals))


def minEst(f: SurfaceFunction) -> float:
    """Estimate the minimum sampled value."""
    return float(min(np.min(v) for v in f.vals))


def resample_mesh(dom: SurfaceMesh, n: int) -> SurfaceMesh:
    """Resample every patch of a surface mesh to an ``n`` by ``n`` grid."""
    m = dom.n
    B = barymat(chebpts(n, 2), chebpts(m, 2))
    x = [B @ patch @ B.T for patch in dom.x]
    y = [B @ patch @ B.T for patch in dom.y]
    z = [B @ patch @ B.T for patch in dom.z]
    return SurfaceMesh(x, y, z)


def resample(f: SurfaceFunction, n: int) -> SurfaceFunction:
    """Resample a surface function to an ``n`` by ``n`` grid on each patch."""
    m = f.domain.n
    B = barymat(chebpts(n, 2), chebpts(m, 2))
    vals = [B @ v @ B.T for v in f.vals]
    return SurfaceFunction(resample_mesh(f.domain, n), vals)


def prolong(f: SurfaceFunction, n: int | None = None) -> SurfaceFunction:
    """
    Prolong/restrict a surface function to a new polynomial grid.

    This implementation uses barycentric patch interpolation, which is stable
    for the smooth resolved functions used in the examples.
    """
    if n is None:
        return f.copy()
    return resample(f, n)


def diff(f: SurfaceFunction, n: int | tuple[int, int, int] = 1, dim: int = 1) -> SurfaceFunction:
    """
    Differentiate a surface function in Cartesian surface directions.

    ``dim=1`` gives the tangential x derivative, ``dim=2`` gives y, and
    ``dim=3`` gives z.  Passing ``n=(nx, ny, nz)`` applies mixed repeated
    derivatives.
    """
    if isinstance(n, tuple):
        nx, ny, nz = n
    elif dim == 1:
        nx, ny, nz = int(n), 0, 0
    elif dim == 2:
        nx, ny, nz = 0, int(n), 0
    elif dim == 3:
        nx, ny, nz = 0, 0, int(n)
    else:
        raise ValueError("dim must be 1, 2, or 3")

    D = diffmat(f.domain.n)
    vals = [v.copy() for v in f.vals]
    out = SurfaceFunction(f.domain, vals)

    for k in range(f.domain.npatches):
        v = out.vals[k]
        for _ in range(nx):
            v = mapped_vdiff(v, f.domain, k, 1, D)
        for _ in range(ny):
            v = mapped_vdiff(v, f.domain, k, 2, D)
        for _ in range(nz):
            v = mapped_vdiff(v, f.domain, k, 3, D)
        out.vals[k] = v
    return out


def mapped_vdiff(vals: Array, dom: SurfaceMesh, k: int, dim: int, D: Array) -> Array:
    dfdu = vals @ D.T
    dfdv = D @ vals
    if dim == 1:
        du, dv = dom.ux[k], dom.vx[k]
    elif dim == 2:
        du, dv = dom.uy[k], dom.vy[k]
    elif dim == 3:
        du, dv = dom.uz[k], dom.vz[k]
    else:
        raise ValueError("dim must be 1, 2, or 3")
    return du * dfdu + dv * dfdv


def diffx(f: SurfaceFunction, n: int = 1) -> SurfaceFunction:
    return diff(f, n, 1)


def diffy(f: SurfaceFunction, n: int = 1) -> SurfaceFunction:
    return diff(f, n, 2)


def diffz(f: SurfaceFunction, n: int = 1) -> SurfaceFunction:
    return diff(f, n, 3)


def gradient(f: SurfaceFunction) -> SurfaceVectorFunction:
    """Return the surface gradient as a three-component vector function."""
    return SurfaceVectorFunction((diffx(f), diffy(f), diffz(f)))


def grad(f: SurfaceFunction) -> SurfaceVectorFunction:
    return gradient(f)


def laplacian(f: SurfaceFunction) -> SurfaceFunction:
    """Surface Laplacian."""
    return diffx(f, 2) + diffy(f, 2) + diffz(f, 2)


def lap(f: SurfaceFunction) -> SurfaceFunction:
    return laplacian(f)


def normal(dom: SurfaceMesh) -> SurfaceVectorFunction:
    """Surface unit normal."""
    nx: list[Array] = []
    ny: list[Array] = []
    nz: list[Array] = []
    volume = 0.0
    wu = quadwts(dom.n, 2)
    W = np.outer(wu, wu)

    for x, y, z, xu, xv, yu, yv, zu, zv, J in zip(
        dom.x, dom.y, dom.z, dom.xu, dom.xv, dom.yu, dom.yv, dom.zu, dom.zv, dom.J
    ):
        n0 = yu * zv - zu * yv
        n1 = zu * xv - xu * zv
        n2 = xu * yv - yu * xv
        scl = np.sqrt(n0 * n0 + n1 * n1 + n2 * n2)
        n0 = n0 / scl
        n1 = n1 / scl
        n2 = n2 / scl
        nx.append(n0)
        ny.append(n1)
        nz.append(n2)
        volume += float(np.sum(((x * n0 + y * n1 + z * n2) / 3.0) * W * np.sqrt(np.maximum(J, 0.0))))

    if volume < 0:
        nx = [-v for v in nx]
        ny = [-v for v in ny]
        nz = [-v for v in nz]

    return surfacefunv(nx, ny, nz, dom)


def dot(f: SurfaceVectorFunction, g: SurfaceVectorFunction) -> SurfaceFunction:
    fc = f.components
    gc = g.components
    return fc[0] * gc[0] + fc[1] * gc[1] + fc[2] * gc[2]


def cross(f: SurfaceVectorFunction | Array | list[float], g: SurfaceVectorFunction | Array | list[float], *args) -> SurfaceVectorFunction:
    """Vector cross product for arrays or surface vector functions."""
    if args:
        return cross(f, cross(g, *args))

    f_is_vec = isinstance(f, SurfaceVectorFunction)
    g_is_vec = isinstance(g, SurfaceVectorFunction)

    if f_is_vec and g_is_vec:
        fc = f.components
        gc = g.components
        return surfacefunv(
            fc[1] * gc[2] - fc[2] * gc[1],
            fc[2] * gc[0] - fc[0] * gc[2],
            fc[0] * gc[1] - fc[1] * gc[0],
        )

    if f_is_vec:
        arr = np.asarray(g, dtype=float).ravel()
        if arr.size != 3:
            raise ValueError("constant vector must have length 3")
        fc = f.components
        return surfacefunv(
            fc[1] * arr[2] - fc[2] * arr[1],
            fc[2] * arr[0] - fc[0] * arr[2],
            fc[0] * arr[1] - fc[1] * arr[0],
        )

    if g_is_vec:
        arr = np.asarray(f, dtype=float).ravel()
        if arr.size != 3:
            raise ValueError("constant vector must have length 3")
        gc = g.components
        return surfacefunv(
            arr[1] * gc[2] - arr[2] * gc[1],
            arr[2] * gc[0] - arr[0] * gc[2],
            arr[0] * gc[1] - arr[1] * gc[0],
        )

    raise TypeError("at least one argument must be a SurfaceVectorFunction")


def divergence(f: SurfaceVectorFunction) -> SurfaceFunction:
    fc = f.components
    return diffx(fc[0]) + diffy(fc[1]) + diffz(fc[2])


def div(f: SurfaceVectorFunction) -> SurfaceFunction:
    return divergence(f)


def vector_norm(f: SurfaceVectorFunction) -> SurfaceFunction:
    fc = f.components
    return sqrt(fc[0] * fc[0] + fc[1] * fc[1] + fc[2] * fc[2])


def normalize(f: SurfaceVectorFunction) -> SurfaceVectorFunction:
    mag = vector_norm(f)
    safe = SurfaceFunction(
        f.domain,
        [np.where(np.abs(v) > 10 * np.finfo(float).eps, v, 1.0) for v in mag.vals],
    )
    return f / safe


def hodge(f: SurfaceVectorFunction) -> tuple[SurfaceFunction, SurfaceFunction, SurfaceVectorFunction, SurfaceVectorFunction, SurfaceVectorFunction]:
    """
    Hodge decomposition of a vector surface function.

    Returns ``u, v, w, curlfree, divfree`` such that

        f = grad(u) + normal x grad(v) + w.
    """
    dom = f.domain
    nvec = normal(dom)

    L_u = SurfaceOp(dom, {"lap": 1.0}, div(f))
    L_u.rankdef = True
    u = L_u.solve().remove_mean()

    L_v = SurfaceOp(dom, {"lap": 1.0}, -div(cross(nvec, f)))
    L_v.rankdef = True
    v = L_v.solve().remove_mean()

    curlfree = grad(u)
    divfree = cross(nvec, grad(v))
    w = f - curlfree - divfree
    return u, v, w, curlfree, divfree


def integral2(f: SurfaceFunction, reduce: bool = True) -> float | Array:
    """Surface integral of a scalar surface function."""
    wu = quadwts(f.domain.n, 2)
    W = np.outer(wu, wu)
    dtype = np.result_type(*f.vals) if f.vals else float
    per_patch = np.zeros(f.domain.npatches, dtype=dtype)
    for k, (vals, J) in enumerate(zip(f.vals, f.domain.J)):
        per_patch[k] = np.sum(vals * W * np.sqrt(np.maximum(J, 0.0)))
    total = np.sum(per_patch)
    return float(np.real(total)) if reduce and np.isrealobj(total) else total if reduce else per_patch


def integral(f: SurfaceFunction, reduce: bool = True) -> float | Array:
    return integral2(f, reduce)


def sum2(f: SurfaceFunction, reduce: bool = True) -> float | Array:
    return integral2(f, reduce)


def mean2(f: SurfaceFunction) -> float:
    return integral2(f) / surfacearea(f.domain)


def surfacearea(dom: SurfaceMesh) -> float:
    return integral2(SurfaceFunction.constant(dom, 1.0))


def boundingbox(dom: SurfaceMesh) -> Array:
    """Bounding box ``[xmin, xmax, ymin, ymax, zmin, zmax]``."""
    xmin = min(float(np.min(x)) for x in dom.x)
    xmax = max(float(np.max(x)) for x in dom.x)
    ymin = min(float(np.min(y)) for y in dom.y)
    ymax = max(float(np.max(y)) for y in dom.y)
    zmin = min(float(np.min(z)) for z in dom.z)
    zmax = max(float(np.max(z)) for z in dom.z)
    return np.array([xmin, xmax, ymin, ymax, zmin, zmax])


def randnfun3(
    length_scale: float,
    bbox: Array | tuple[float, float, float, float, float, float],
    seed: int | None = None,
    nmodes: int = 64,
) -> Callable[[Array, Array, Array], Array]:
    """
    Smooth deterministic 3D random field.

    This returns a finite smooth random Fourier field on the supplied physical
    bounding box. It is meant for reproducible initial data in time-dependent
    examples.
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
    """Descriptive alias for ``randnfun3``."""
    return randnfun3(length_scale, bbox, seed=seed, nmodes=nmodes)


def norm(f: SurfaceFunction | SurfaceVectorFunction, p: int | float | str = 2, reduce: bool = True) -> float | Array | SurfaceFunction:
    """
    Surface function norms.

    Supported: 1, 2, positive integer p, ``"inf"``, ``"max"``, ``"H1"``,
    and ``"lap"``.
    """
    if isinstance(f, SurfaceVectorFunction):
        mag = vector_norm(f)
        if p == 2 and reduce:
            return mag
        return norm(mag, p, reduce)

    if p in (np.inf, "inf", "max"):
        per_patch = np.asarray([np.max(np.abs(v)) for v in f.vals])
        return float(np.max(per_patch)) if reduce else per_patch

    if p == "H1":
        g = gradient(f)
        parts = [norm(f, 2, False)]
        parts += [norm(comp, 2, False) for comp in g.components]
        per_patch = np.sqrt(sum(part * part for part in parts))
        return float(np.sqrt(np.sum(per_patch * per_patch))) if reduce else per_patch

    if p == "lap":
        n0 = norm(f, 2, False)
        n1 = norm(lap(f), 2, False)
        per_patch = np.sqrt(n0 * n0 + n1 * n1)
        return float(np.sqrt(np.sum(per_patch * per_patch))) if reduce else per_patch

    p_float = float(p)
    per_patch = np.zeros(f.domain.npatches)
    for k, vals in enumerate(f.vals):
        per_patch[k] = integral2(SurfaceFunction(f.domain, [np.abs(v) ** p_float if j == k else np.zeros_like(v) for j, v in enumerate(f.vals)]))
    per_patch = per_patch ** (1.0 / p_float)
    if reduce:
        return float(np.sum(per_patch ** p_float) ** (1.0 / p_float))
    return per_patch


# ---------------------------------------------------------------------------
# surfaceop subset: leaf operators, Schur complements, recursive solve.
# ---------------------------------------------------------------------------


@dataclass
class PDO:
    dxx: float = 0.0
    dyy: float = 0.0
    dzz: float = 0.0
    dxy: float = 0.0
    dyx: float = 0.0
    dyz: float = 0.0
    dzy: float = 0.0
    dxz: float = 0.0
    dzx: float = 0.0
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    b: float = 0.0


def parse_pdo(op: dict) -> PDO:
    out = PDO()
    if "lap" in op:
        out.dxx = op["lap"]
        out.dyy = op["lap"]
        out.dzz = op["lap"]
    if "grad" in op:
        out.dx = op["grad"]
        out.dy = op["grad"]
        out.dz = op["grad"]
    for name in out.__dataclass_fields__:
        if name in op:
            setattr(out, name, op[name])
    if "c" in op:
        out.b = op["c"]
    return out


@dataclass
class Patch:
    domain: SurfaceMesh
    ids: list[int]
    S: Array
    D2N: Array
    D2N_scl: list[Array]
    u_part: Array
    du_part: Array
    edges: Array
    xyz: Array
    w: Array

    def solve(self, bc: Array | Callable | None = None) -> list[Array]:
        raise NotImplementedError


@dataclass
class Leaf(Patch):
    n: int = 0
    Aii: Array | None = None
    Aii_inv: Array | None = None
    ii_flat: Array | None = None
    normal_d: Array | None = None
    _bc_work: Array | None = field(default=None, init=False, repr=False)

    def solve(self, bc: Array | Callable | None = None) -> list[Array]:
        if bc is None:
            self._bc_work = get_work_array(self._bc_work, (self.S.shape[1], self.u_part.shape[1]), self.u_part.dtype)
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
        return [u[:, j].reshape((self.n, self.n), order="F") for j in range(u.shape[1])]


@dataclass
class Parent(Patch):
    child1: Patch | None = None
    child2: Patch | None = None
    idx1: tuple[Array, Array] | None = None
    idx2: tuple[Array, Array] | None = None
    flip1: Array | None = None
    flip2: Array | None = None
    scl1: Array | None = None
    scl2: Array | None = None
    A_inv: Array | None = None
    M: Array | None = None
    rankdef_merged: bool = False
    _bc_work: Array | None = field(default=None, init=False, repr=False)
    _bc1_work: Array | None = field(default=None, init=False, repr=False)
    _bc2_work: Array | None = field(default=None, init=False, repr=False)

    def solve(self, bc: Array | Callable | None = None) -> list[Array]:
        nrhs = self.u_part.shape[1]
        if bc is None:
            self._bc_work = get_work_array(self._bc_work, (self.xyz.shape[0], nrhs), self.u_part.dtype)
            bc_arr = self._bc_work
        elif callable(bc):
            vals = bc(self.xyz[:, 0], self.xyz[:, 1], self.xyz[:, 2])
            bc_arr = np.asarray(vals).reshape(-1, 1)
        else:
            bc_arr = np.asarray(bc)
            if bc_arr.ndim == 0:
                bc_arr = np.full((self.xyz.shape[0], nrhs), bc_arr.item())
            elif bc_arr.ndim == 1:
                bc_arr = bc_arr.reshape(-1, 1)

        u_interface = self.u_part.copy()
        if self.S.size and bc_arr.size:
            u_interface = u_interface + self.S @ bc_arr

        assert self.child1 is not None and self.child2 is not None
        assert self.idx1 is not None and self.idx2 is not None
        assert self.flip1 is not None and self.flip2 is not None

        ext1, glue1 = self.idx1
        ext2, glue2 = self.idx2

        bc_dtype = np.result_type(bc_arr, u_interface)
        self._bc1_work = get_work_array(self._bc1_work, (self.child1.S.shape[1], nrhs), bc_dtype)
        self._bc2_work = get_work_array(self._bc2_work, (self.child2.S.shape[1], nrhs), bc_dtype)
        bc1 = self._bc1_work
        bc2 = self._bc2_work

        if ext1.size:
            bc1[ext1, :] = bc_arr[: ext1.size, :]
        if glue1.size:
            bc1[glue1, :] = self.flip1.T @ u_interface

        offset = ext1.size
        if ext2.size:
            bc2[ext2, :] = bc_arr[offset : offset + ext2.size, :]
        if glue2.size:
            bc2[glue2, :] = self.flip2.T @ u_interface

        return self.child1.solve(bc1) + self.child2.solve(bc2)


class SurfaceOp:
    def __init__(self, domain: SurfaceMesh, op: dict, rhs: SurfaceFunction | Callable | float = 0.0):
        self.domain = domain
        self.op = parse_pdo(op)
        self.rhs = rhs
        self.rankdef = False
        self.merge_idx = default_merge_idx(domain.npatches)
        self._ii_flat = interior_mask_flat(domain.n)
        self._num_int = int(np.sum(self._ii_flat))
        self._rhs_work: Array | None = None
        self.patches: list[Patch] = initialize_leaves(self.op, self.domain, self.rhs)
        self._built = False

    def build(self) -> None:
        if self._built:
            return
        patches = self.patches
        for level, idx in enumerate(self.merge_idx):
            top = level == len(self.merge_idx) - 1
            q: list[Patch] = []
            for a_idx, b_idx in idx:
                a = patches[a_idx]
                b = None if b_idx is None else patches[b_idx]
                if b is None:
                    q.append(a)
                else:
                    q.append(merge_patches(a, b, rankdef=(self.rankdef and top)))
            patches = q
        self.patches = patches
        self._built = True

    def solve(self) -> SurfaceFunction:
        self.build()
        vals = self.patches[0].solve(None)
        return SurfaceFunction(self.domain, vals)

    def apply(self, rhs: SurfaceFunction | Callable | float) -> SurfaceFunction:
        """Update the right-hand side and immediately solve."""
        self.update_rhs(rhs)
        return self.solve()

    def update_rhs(self, rhs: SurfaceFunction | Callable | float) -> "SurfaceOp":
        """
        Update only the right-hand side data of an already assembled operator.

        The geometry, local differential matrices, and Schur-complement
        solution operators are reused.  Only the particular solution data are
        recomputed and pushed up the merge tree.
        """
        if not self._built:
            self.rhs = rhs
            self.patches = initialize_leaves(self.op, self.domain, self.rhs)
            return self

        rhs_int = evaluate_rhs(rhs, self.domain, self._ii_flat, self._num_int, self._rhs_work)
        self._rhs_work = rhs_int
        for patch in self.patches:
            update_patch_rhs(patch, rhs_int)
        self.rhs = rhs
        return self


def default_merge_idx(npatches: int) -> list[list[tuple[int, int | None]]]:
    """Binary merge schedule for patch-local solution operators."""
    out: list[list[tuple[int, int | None]]] = []
    np_now = npatches
    ids: list[int | None] = list(range(npatches))
    while np_now > 1:
        if len(ids) % 2:
            ids.append(None)
        level = []
        next_ids: list[int | None] = []
        for k in range(0, len(ids), 2):
            level.append((ids[k], ids[k + 1]))
            next_ids.append(k // 2)
        out.append(level)
        ids = next_ids
        np_now = len(ids)
    return out


def initialize_leaves(op: PDO, dom: SurfaceMesh, rhs: SurfaceFunction | Callable | float) -> list[Patch]:
    n = dom.n
    num_patches = dom.npatches

    Xref, Yref = chebpts2(n)
    ii = (np.abs(Xref) < 1.0) & (np.abs(Yref) < 1.0)
    ee = ~ii
    ii_flat = ii.ravel(order="F")
    ee_flat = ee.ravel(order="F")
    num_bdy = int(np.sum(ee_flat))
    num_int = int(np.sum(ii_flat))

    nskel = n - 2
    num_skel = 4 * nskel
    S2L = skel2leaf(n, nskel)
    L2S = leaf2skel(nskel, n)
    xskel = chebpts(nskel, 1)
    xleaf = chebpts(n, 2)
    B = barymat(xskel, xleaf)
    wskel_1d = quadwts(nskel, 1).reshape(-1, 1)
    wskel = np.vstack((wskel_1d, wskel_1d, wskel_1d, wskel_1d)).ravel()

    left_skel = np.arange(0, nskel)
    right_skel = np.arange(nskel, 2 * nskel)
    down_skel = np.arange(2 * nskel, 3 * nskel)
    up_skel = np.arange(3 * nskel, 4 * nskel)

    NL, NR, ND, NU = binormals(dom)
    NN_list = []
    for k in range(num_patches):
        NN_list.append(np.vstack((B @ NL[:, :, k], B @ NR[:, :, k], B @ ND[:, :, k], B @ NU[:, :, k])))

    ux = stack_patch_vectors(dom.ux)
    vx = stack_patch_vectors(dom.vx)
    uy = stack_patch_vectors(dom.uy)
    vy = stack_patch_vectors(dom.vy)
    uz = stack_patch_vectors(dom.uz)
    vz = stack_patch_vectors(dom.vz)

    D = diffmat(n)
    I = np.eye(n)
    II = np.eye(n * n)
    Du = np.kron(D, I)
    Dv = np.kron(I, D)

    rhs_int = evaluate_rhs(rhs, dom, ii_flat, num_int)

    leaves: list[Patch] = []
    tmpS_template = np.zeros((n * n, num_bdy + rhs_int.shape[1]), dtype=rhs_int.dtype)
    tmpS_template[np.ix_(ee_flat, np.arange(num_bdy))] = np.eye(num_bdy)

    for k in range(num_patches):
        x = dom.x[k]
        y = dom.y[k]
        z = dom.z[k]

        edges = np.array(
            [
                [x[0, 0], y[0, 0], z[0, 0], x[-1, 0], y[-1, 0], z[-1, 0], nskel],
                [x[0, -1], y[0, -1], z[0, -1], x[-1, -1], y[-1, -1], z[-1, -1], nskel],
                [x[0, 0], y[0, 0], z[0, 0], x[0, -1], y[0, -1], z[0, -1], nskel],
                [x[-1, 0], y[-1, 0], z[-1, 0], x[-1, -1], y[-1, -1], z[-1, -1], nskel],
            ],
            dtype=float,
        )

        Dx = ux[:, [k]] * Du + vx[:, [k]] * Dv
        Dy = uy[:, [k]] * Du + vy[:, [k]] * Dv
        Dz = uz[:, [k]] * Du + vz[:, [k]] * Dv

        A = np.zeros((n * n, n * n))
        if op.dxx:
            A += op.dxx * (Dx @ Dx)
        if op.dyy:
            A += op.dyy * (Dy @ Dy)
        if op.dzz:
            A += op.dzz * (Dz @ Dz)
        if op.dxy:
            A += op.dxy * (Dx @ Dy)
        if op.dyx:
            A += op.dyx * (Dy @ Dx)
        if op.dyz:
            A += op.dyz * (Dy @ Dz)
        if op.dzy:
            A += op.dzy * (Dz @ Dy)
        if op.dxz:
            A += op.dxz * (Dx @ Dz)
        if op.dzx:
            A += op.dzx * (Dz @ Dx)
        if op.dx:
            A += op.dx * Dx
        if op.dy:
            A += op.dy * Dy
        if op.dz:
            A += op.dz * Dz
        if op.b:
            A += op.b * II

        Aii = A[np.ix_(ii_flat, ii_flat)]
        Aie = A[np.ix_(ii_flat, ee_flat)]
        rhs_k = rhs_int[:, :, k]
        local_rhs = np.hstack((-Aie, rhs_k))
        Aii_inv = inverse_dense(Aii)
        S_int = solve_with_inverse(Aii_inv, local_rhs)

        tmpS = tmpS_template.copy()
        tmpS[ii_flat, :] = S_int
        S = tmpS[:, :num_bdy] @ S2L
        u_part = tmpS[:, num_bdy:]

        dx = L2S @ Dx[ee_flat, :]
        dy = L2S @ Dy[ee_flat, :]
        dz = L2S @ Dz[ee_flat, :]

        NN = NN_list[k]
        normal_d = rowscale(NN[:, 0], dx) + rowscale(NN[:, 1], dy) + rowscale(NN[:, 2], dz)

        D2N = normal_d @ S
        du_part = normal_d @ u_part

        D2N_scl = [
            np.ones(nskel),
            np.ones(nskel),
            np.ones(nskel),
            np.ones(nskel),
        ]

        J = dom.J[k].ravel(order="F")
        xyz = L2S @ np.column_stack(
            (
                x.ravel(order="F")[ee_flat],
                y.ravel(order="F")[ee_flat],
                z.ravel(order="F")[ee_flat],
            )
        )
        JJ = L2S @ np.sqrt(np.maximum(J[ee_flat], 0.0))
        w = wskel * JJ

        leaves.append(
            Leaf(
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
                n=n,
                Aii=Aii,
                Aii_inv=Aii_inv,
                ii_flat=ii_flat,
                normal_d=normal_d,
            )
        )

    return leaves


def interior_mask_flat(n: int) -> Array:
    """Interior-node mask in column-major patch order."""
    Xref, Yref = chebpts2(n)
    return ((np.abs(Xref) < 1.0) & (np.abs(Yref) < 1.0)).ravel(order="F")


def update_patch_rhs(patch: Patch, rhs_int: Array) -> None:
    """Recursive right-hand-side update for the HPS merge tree."""
    if isinstance(patch, Leaf):
        patch_id = patch.ids[0]
        rhs_k = rhs_int[:, :, patch_id]
        assert patch.Aii_inv is not None and patch.ii_flat is not None and patch.normal_d is not None
        ii_flat = patch.ii_flat
        u_part = np.zeros((patch.n * patch.n, rhs_k.shape[1]), dtype=rhs_k.dtype)
        u_part[ii_flat, :] = solve_with_inverse(patch.Aii_inv, rhs_k)
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

    update_patch_rhs(patch.child1, rhs_int)
    update_patch_rhs(patch.child2, rhs_int)

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


def evaluate_rhs(
    rhs: SurfaceFunction | Callable | float,
    dom: SurfaceMesh,
    ii_flat: Array,
    num_int: int,
    out: Array | None = None,
) -> Array:
    if isinstance(rhs, SurfaceFunction):
        vals = [v.ravel(order="F")[ii_flat] for v in rhs.vals]
        dtype = np.result_type(*vals) if vals else float
        out = get_work_array(out, (num_int, 1, dom.npatches), dtype)
        for k, v in enumerate(vals):
            out[:, 0, k] = v
        return out

    if callable(rhs):
        vals = []
        for k in range(dom.npatches):
            x = dom.x[k].ravel(order="F")[ii_flat]
            y = dom.y[k].ravel(order="F")[ii_flat]
            z = dom.z[k].ravel(order="F")[ii_flat]
            vals.append(np.asarray(rhs(x, y, z)).ravel())
        dtype = np.result_type(*vals) if vals else float
        out = get_work_array(out, (num_int, 1, dom.npatches), dtype)
        for k, v in enumerate(vals):
            out[:, 0, k] = v
        return out

    out = get_work_array(out, (num_int, 1, dom.npatches), np.asarray(rhs).dtype)
    out.fill(rhs)
    return out


def stack_patch_vectors(cells: list[Array]) -> Array:
    return np.column_stack([a.ravel(order="F") for a in cells])


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


def skel2leaf(nleaf: int, nskel: int) -> Array:
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


def binormals(dom: SurfaceMesh) -> tuple[Array, Array, Array, Array]:
    n = dom.n
    P = dom.npatches
    NL = np.zeros((n, 3, P))
    NR = np.zeros((n, 3, P))
    ND = np.zeros((n, 3, P))
    NU = np.zeros((n, 3, P))

    for k in range(P):
        xu, xv = dom.xu[k], dom.xv[k]
        yu, yv = dom.yu[k], dom.yv[k]
        zu, zv = dom.zu[k], dom.zv[k]

        nl = -np.column_stack((xu[:, 0], yu[:, 0], zu[:, 0]))
        nr = np.column_stack((xu[:, -1], yu[:, -1], zu[:, -1]))
        nd = -np.column_stack((xv[0, :], yv[0, :], zv[0, :]))
        nu = np.column_stack((xv[-1, :], yv[-1, :], zv[-1, :]))

        tl = normalize_rows(np.column_stack((xv[:, 0], yv[:, 0], zv[:, 0])))
        tr = normalize_rows(np.column_stack((xv[:, -1], yv[:, -1], zv[:, -1])))
        td = normalize_rows(np.column_stack((xu[0, :], yu[0, :], zu[0, :])))
        tu = normalize_rows(np.column_stack((xu[-1, :], yu[-1, :], zu[-1, :])))

        NL[:, :, k] = normalize_rows(nl - tl * np.sum(nl * tl, axis=1, keepdims=True))
        NR[:, :, k] = normalize_rows(nr - tr * np.sum(nr * tr, axis=1, keepdims=True))
        ND[:, :, k] = normalize_rows(nd - td * np.sum(nd * td, axis=1, keepdims=True))
        NU[:, :, k] = normalize_rows(nu - tu * np.sum(nu * tu, axis=1, keepdims=True))

    return NL, NR, ND, NU


def normalize_rows(v: Array) -> Array:
    nrm = np.linalg.norm(v, axis=1, keepdims=True)
    return v / nrm


def merge_patches(a: Patch, b: Patch, rankdef: bool = False) -> Patch:
    i1, i2, s1, s2, flip1, flip2, scl1, scl2, sclAB, edgesAB = patch_intersect(a, b)

    D2Na = a.D2N
    D2Nb = b.D2N

    A = -(
        rowscale(scl2, flip1 @ D2Na[np.ix_(s1, s1)] @ flip1.T)
        + rowscale(scl1, flip2 @ D2Nb[np.ix_(s2, s2)] @ flip2.T)
    )
    z = np.hstack(
        (
            rowscale(scl2, flip1 @ D2Na[np.ix_(s1, i1)]),
            rowscale(scl1, flip2 @ D2Nb[np.ix_(s2, i2)]),
        )
    )
    z_part = rowscale(scl2, flip1 @ a.du_part[s1, :]) + rowscale(scl1, flip2 @ b.du_part[s2, :])

    if rankdef and i1.size == 0 and i2.size == 0 and A.size:
        w = a.w[s1].reshape(-1, 1)
        A = A + w @ w.T

    A_inv = inverse_dense(A)
    S = solve_with_inverse(A_inv, z)
    u_part = solve_with_inverse(A_inv, z_part)

    M = np.vstack((D2Na[np.ix_(i1, s1)] @ flip1.T, D2Nb[np.ix_(i2, s2)] @ flip2.T))
    D2N = M @ S if S.size else np.zeros((i1.size + i2.size, i1.size + i2.size))

    b1 = np.arange(i1.size)
    b2 = np.arange(i2.size) + i1.size
    if i1.size:
        D2N[np.ix_(b1, b1)] += D2Na[np.ix_(i1, i1)]
    if i2.size:
        D2N[np.ix_(b2, b2)] += D2Nb[np.ix_(i2, i2)]

    du_part = np.vstack((a.du_part[i1, :], b.du_part[i2, :])) + M @ u_part
    xyz = np.vstack((a.xyz[i1, :], b.xyz[i2, :]))
    w = np.concatenate((a.w[i1], b.w[i2]))

    return Parent(
        domain=a.domain,
        ids=a.ids + b.ids,
        S=S,
        D2N=D2N,
        D2N_scl=sclAB,
        u_part=u_part,
        du_part=du_part,
        edges=edgesAB,
        xyz=xyz,
        w=w,
        child1=a,
        child2=b,
        idx1=(i1, s1),
        idx2=(i2, s2),
        flip1=flip1,
        flip2=flip2,
        scl1=scl1,
        scl2=scl2,
        A_inv=A_inv,
        M=M,
        rankdef_merged=rankdef,
    )


def patch_intersect(a: Patch, b: Patch) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array, list[Array], Array]:
    edgesA = a.edges.copy()
    edgesB = b.edges.copy()

    sclA = float(np.max(np.abs(edgesA[:, :6]))) if edgesA.size else 0.0
    sclB = float(np.max(np.abs(edgesB[:, :6]))) if edgesB.size else 0.0
    scl = max(sclA, sclB, 1.0)
    tol = 1e-8 * scl

    iA1, iB1 = intersect_tol(edgesA[:, :6], edgesB[:, :6], tol)
    iA2, iB2 = intersect_tol(edgesA[:, [3, 4, 5, 0, 1, 2]], edgesB[:, :6], tol)

    iA = np.concatenate((iA1, iA2)).astype(int)
    iB = np.concatenate((iB1, iB2)).astype(int)

    pA = edgesA[:, 6].astype(int)
    pB = edgesB[:, 6].astype(int)
    ppA = np.r_[0, np.cumsum(pA)]
    ppB = np.r_[0, np.cumsum(pB)]

    s1 = np.concatenate([np.arange(ppA[e], ppA[e + 1]) for e in iA]) if iA.size else np.empty(0, dtype=int)
    s2 = np.concatenate([np.arange(ppB[e], ppB[e + 1]) for e in iB]) if iB.size else np.empty(0, dtype=int)

    flip1 = np.eye(s1.size)
    blocks = []
    for e in iB1:
        blocks.append(np.eye(pB[e]))
    for e in iB2:
        blocks.append(np.fliplr(np.eye(pB[e])))
    flip2 = block_diag(blocks) if blocks else np.eye(s1.size)

    allA = np.arange(int(np.sum(pA)))
    allB = np.arange(int(np.sum(pB)))
    i1 = np.setdiff1d(allA, s1, assume_unique=False)
    i2 = np.setdiff1d(allB, s2, assume_unique=False)

    scl1 = np.concatenate([a.D2N_scl[e] for e in iA]) if iA.size else np.empty(0)
    scl2 = np.concatenate([b.D2N_scl[e] for e in iB]) if iB.size else np.empty(0)

    sclA = list(a.D2N_scl)
    sclB = list(b.D2N_scl)
    for e in sorted(iA.tolist(), reverse=True):
        del sclA[e]
    for e in sorted(iB.tolist(), reverse=True):
        del sclB[e]
    sclAB = sclA + sclB

    edgesA_remaining = np.delete(edgesA, iA, axis=0) if iA.size else edgesA
    edgesB_remaining = np.delete(edgesB, iB, axis=0) if iB.size else edgesB
    edgesAB = np.vstack((edgesA_remaining, edgesB_remaining))

    return i1, i2, s1, s2, flip1, flip2, scl1, scl2, sclAB, edgesAB


def intersect_tol(A: Array, B: Array, tol: float) -> tuple[Array, Array]:
    AI = []
    BI = []
    for k in range(A.shape[0]):
        dist = np.sum(np.abs(B - A[k, :]), axis=1)
        hit = np.where(dist < tol)[0]
        if hit.size:
            AI.append(k)
            BI.append(int(hit[0]))
    return np.asarray(AI, dtype=int), np.asarray(BI, dtype=int)


# ---------------------------------------------------------------------------
# Sphere test functions, plotting, and VTU export.
# ---------------------------------------------------------------------------


def associated_legendre(l: int, m: int, x: Array) -> Array:
    """Associated Legendre P_l^m(x), unnormalized, for m >= 0."""
    if m < 0 or m > l:
        raise ValueError("require 0 <= m <= l")
    x = np.asarray(x)
    pmm = np.ones_like(x)
    if m > 0:
        somx2 = np.sqrt(np.maximum(1.0 - x * x, 0.0))
        fact = 1.0
        for _ in range(1, m + 1):
            pmm *= -fact * somx2
            fact += 2.0
    if l == m:
        return pmm
    pmmp1 = x * (2 * m + 1) * pmm
    if l == m + 1:
        return pmmp1
    pll = np.zeros_like(x)
    p_lm2 = pmm
    p_lm1 = pmmp1
    for ll in range(m + 2, l + 1):
        pll = ((2 * ll - 1) * x * p_lm1 - (ll + m - 1) * p_lm2) / (ll - m)
        p_lm2, p_lm1 = p_lm1, pll
    return pll


def real_spherical_harmonic(l: int, m: int, x: Array, y: Array, z: Array) -> Array:
    """
    Real, unnormalized spherical harmonic Y_l^m on the unit sphere.

    This avoids a scipy dependency.  Normalization is irrelevant for
    convergence tests because the error is relative.
    """
    phi = np.arctan2(y, x)
    P = associated_legendre(l, abs(m), np.clip(z, -1.0, 1.0))
    if m >= 0:
        return P * np.cos(m * phi)
    return P * np.sin(abs(m) * phi)


def solve_laplace_beltrami_sphere(
    f: SurfaceFunction | Callable,
    n: int = 9,
    nref: int = 0,
    c: float = 0.0,
    rankdef: bool = True,
) -> SurfaceFunction:
    """
    Solve Delta_Gamma u + c u = f on S^2.

    For c=0 this is rank deficient; use a mean-zero RHS and rankdef=True.
    """
    dom = f.domain if isinstance(f, SurfaceFunction) else SurfaceMesh.sphere(n, nref)
    L = SurfaceOp(dom, {"lap": 1.0, "c": c}, f)
    L.rankdef = rankdef
    u = L.solve()
    return u.remove_mean() if rankdef and c == 0.0 else u


def write_vtu(filename: str, u: SurfaceFunction, point_name: str = "u") -> None:
    """
    Write a VTU file with triangulated patch grids.

    Requires meshio.  Install with: pip install meshio
    """
    import meshio

    points = []
    values = []
    faces = []
    offset = 0
    n = u.domain.n
    for x, y, z, val in zip(u.domain.x, u.domain.y, u.domain.z, u.vals):
        pts = np.column_stack((x.ravel(order="F"), y.ravel(order="F"), z.ravel(order="F")))
        points.append(pts)
        values.append(val.ravel(order="F"))
        for j in range(n - 1):
            for i in range(n - 1):
                a = offset + i + j * n
                b = offset + (i + 1) + j * n
                c = offset + i + (j + 1) * n
                d = offset + (i + 1) + (j + 1) * n
                faces.append([a, b, d])
                faces.append([a, d, c])
        offset += n * n

    meshio.write_points_cells(
        filename,
        np.vstack(points),
        [("triangle", np.asarray(faces, dtype=int))],
        point_data={point_name: np.concatenate(values)},
    )


def plot_surface(
    u: SurfaceFunction,
    title: str = "",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = True,
) -> None:
    """Basic Matplotlib visualization of the patch values."""
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    plot_vals = [np.real(v) if np.iscomplexobj(v) else v for v in u.vals]
    if vmin is None:
        vmin = min(float(np.min(v)) for v in plot_vals)
    if vmax is None:
        vmax = max(float(np.max(v)) for v in plot_vals)
    norm_obj = Normalize(vmin=vmin, vmax=vmax)

    for x, y, z, val in zip(u.domain.x, u.domain.y, u.domain.z, plot_vals):
        ax.plot_surface(
            x,
            y,
            z,
            facecolors=plt.cm.turbo(norm_obj(val)),
            linewidth=0.15,
            edgecolor=(0, 0, 0, 0.25),
            antialiased=True,
            shade=False,
        )
    ax.set_title(title)
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1))
    if colorbar:
        mappable = ScalarMappable(norm=norm_obj, cmap=plt.cm.turbo)
        mappable.set_array([])
        fig.colorbar(mappable, ax=ax, shrink=0.75, pad=0.02)
    plt.show()


def plot_vector_field(f: SurfaceVectorFunction, title: str = "", stride: int = 2, scale: float = 0.6) -> None:
    """Plot vector magnitude on the surface with sparse 3D quiver arrows."""
    import matplotlib.pyplot as plt

    mag = vector_norm(f)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    vmin = min(float(np.min(v)) for v in mag.vals)
    vmax = max(float(np.max(v)) for v in mag.vals)

    fx, fy, fz = f.components
    for x, y, z, mval, vx, vy, vz in zip(
        f.domain.x, f.domain.y, f.domain.z, mag.vals, fx.vals, fy.vals, fz.vals
    ):
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
        ax.quiver(
            x[sl],
            y[sl],
            z[sl],
            vx[sl],
            vy[sl],
            vz[sl],
            length=scale,
            normalize=True,
            color="k",
            linewidth=0.45,
        )

    ax.set_title(title)
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1))
    plt.show()


def demo_laplace_beltrami_sphere() -> None:
    """Solve a small Laplace-Beltrami problem on the unit sphere."""
    l = 3
    m = 2
    n = 7
    nref = 0

    dom = SurfaceMesh.sphere(n, nref)
    exact = SurfaceFunction.from_callable(dom, lambda x, y, z: real_spherical_harmonic(l, m, x, y, z))
    rhs = -l * (l + 1) * exact

    L = SurfaceOp(dom, {"lap": 1.0}, rhs)
    L.rankdef = True
    uh = L.solve().remove_mean()

    relerr = (uh - exact.remove_mean()).norm_inf() / exact.norm_inf()
    print(f"relative L_inf error = {relerr:.3e}")

    try:
        write_vtu("laplace_beltrami_sphere.vtu", uh)
        print("wrote laplace_beltrami_sphere.vtu")
    except ImportError:
        print("VTU skipped: install meshio to export ParaView files")

    try:
        plot_surface(uh, "Laplace-Beltrami on S^2")
    except ImportError:
        print("plot skipped: install matplotlib to plot")


if __name__ == "__main__":
    demo_laplace_beltrami_sphere()
