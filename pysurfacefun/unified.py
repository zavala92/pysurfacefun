"""
Discretization-independent API.

Every function here accepts quadrilateral (:class:`~pysurfacefun.SurfaceMesh`)
and triangular (:class:`~pysurfacefun.TriangleSurfaceMesh`) meshes and fields
and dispatches to the matching implementation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from . import core
from .core import SurfaceFunction, SurfaceMesh, SurfaceOp, SurfaceVectorFunction
from .fields import PatchField, PatchVectorField
from .operators import FIRST_ORDER_TERMS, PDO, SECOND_ORDER_TERMS, coefficient_arrays, parse_pdo
from .tri import (
    TriangleSurfaceFunction,
    TriangleSurfaceMesh,
    TriangleSurfaceOp,
    TriangleSurfaceVectorFunction,
    tri_diff,
    tri_lap,
    tri_resample,
    tri_resample_mesh,
    tri_surfacefun,
)

SurfaceField = SurfaceFunction | TriangleSurfaceFunction
SurfaceVectorField = SurfaceVectorFunction | TriangleSurfaceVectorFunction
SurfaceDomain = SurfaceMesh | TriangleSurfaceMesh

__all__ = [
    "SurfaceDomain",
    "SurfaceField",
    "SurfaceVectorField",
    "apply_operator",
    "diff",
    "div",
    "divergence",
    "field",
    "grad",
    "gradient",
    "hodge",
    "integral",
    "integral2",
    "lap",
    "laplacian",
    "norm",
    "operator",
    "resample",
    "resample_mesh",
    "surfaceop",
    "vector_field",
    "vector_norm",
]


def _require_domain(dom: Any, name: str) -> None:
    if not isinstance(dom, (SurfaceMesh, TriangleSurfaceMesh)):
        raise TypeError(f"{name} expects a SurfaceMesh or TriangleSurfaceMesh domain")


def field(func: Callable | float | Any, dom: SurfaceDomain) -> SurfaceField:
    """Scalar field on a quadrilateral or triangular mesh.

    ``func`` may be a callable ``f(x, y, z)``, a number, per-patch values, or
    an existing field on the mesh (copied).
    """
    _require_domain(dom, "field")
    if isinstance(func, PatchField):
        return func.copy()
    if isinstance(dom, TriangleSurfaceMesh):
        return tri_surfacefun(func, dom)
    return core.surfacefun(func, dom)


def vector_field(fx: Any, fy: Any = None, fz: Any = None, dom: SurfaceDomain | None = None) -> SurfaceVectorField:
    """Three-component vector field on a quadrilateral or triangular mesh.

    ``vector_field(dom)`` returns the zero field; ``vector_field(fx, fy, fz)``
    accepts scalar fields; ``vector_field(fx, fy, fz, dom)`` accepts callables
    or numbers.
    """
    if isinstance(fx, PatchVectorField) and fy is None and fz is None:
        return fx
    if isinstance(fx, (SurfaceMesh, TriangleSurfaceMesh)) and fy is None and fz is None:
        zero = field(0.0, fx)
        return _vector_type(fx)((zero, zero.copy(), zero.copy()))
    if all(isinstance(c, PatchField) for c in (fx, fy, fz)):
        return _vector_type(fx.domain)((fx, fy, fz))
    if dom is None:
        raise ValueError("dom is required when components are not already surface fields")
    if fy is None or fz is None:
        raise ValueError("all three vector components are required")
    _require_domain(dom, "vector_field")
    return _vector_type(dom)((field(fx, dom), field(fy, dom), field(fz, dom)))


def _vector_type(dom: Any):
    return TriangleSurfaceVectorFunction if isinstance(dom, TriangleSurfaceMesh) else SurfaceVectorFunction


def diff(f: SurfaceField, n: int | tuple[int, int, int] = 1, dim: int = 1) -> SurfaceField:
    """Tangential derivative ``D_x``, ``D_y``, or ``D_z`` (``dim = 1, 2, 3``) applied ``n`` times.

    ``n=(nx, ny, nz)`` applies mixed derivatives, ``x`` first (same signature
    as :func:`pysurfacefun.core.diff`).
    """
    if isinstance(f, SurfaceFunction):
        return core.diff(f, n, dim)
    if not isinstance(f, TriangleSurfaceFunction):
        raise TypeError("diff expects a scalar surface field")
    if isinstance(n, tuple):
        counts = list(n)
    else:
        if dim not in (1, 2, 3):
            raise ValueError("dim must be 1, 2, or 3")
        counts = [0, 0, 0]
        counts[dim - 1] = int(n)
    out = f.copy()
    for axis, count in enumerate(counts, start=1):
        for _ in range(count):
            out = tri_diff(out, axis)
    return out


def lap(f: SurfaceField) -> SurfaceField:
    """Laplace--Beltrami operator of a scalar field."""
    if isinstance(f, TriangleSurfaceFunction):
        return tri_lap(f)
    if isinstance(f, SurfaceFunction):
        return core.lap(f)
    raise TypeError("lap expects a scalar surface field")


def laplacian(f: SurfaceField) -> SurfaceField:
    """Alias for :func:`lap`."""
    return lap(f)


def grad(f: SurfaceField) -> SurfaceVectorField:
    """Surface gradient of a scalar field."""
    if isinstance(f, TriangleSurfaceFunction):
        return TriangleSurfaceVectorFunction((tri_diff(f, 1), tri_diff(f, 2), tri_diff(f, 3)))
    if isinstance(f, SurfaceFunction):
        return core.grad(f)
    raise TypeError("grad expects a scalar surface field")


def gradient(f: SurfaceField) -> SurfaceVectorField:
    """Alias for :func:`grad`."""
    return grad(f)


def div(f: SurfaceVectorField) -> SurfaceField:
    """Surface divergence of a vector field."""
    if isinstance(f, TriangleSurfaceVectorFunction):
        a, b, c = f.components
        return tri_diff(a, 1) + tri_diff(b, 2) + tri_diff(c, 3)
    if isinstance(f, SurfaceVectorFunction):
        return core.div(f)
    raise TypeError("div expects a vector surface field")


def divergence(f: SurfaceVectorField) -> SurfaceField:
    """Alias for :func:`div`."""
    return div(f)


def vector_norm(f: SurfaceVectorField) -> SurfaceField:
    """Pointwise magnitude of a vector field."""
    if not isinstance(f, PatchVectorField):
        raise TypeError("vector_norm expects a vector surface field")
    return core.vector_norm(f)


def integral(f: SurfaceField, reduce: bool = True):
    """Surface integral of a scalar field (per patch when ``reduce=False``)."""
    if not isinstance(f, PatchField):
        raise TypeError("integral expects a scalar surface field")
    return f.integral(reduce=reduce)


def integral2(f: SurfaceField, reduce: bool = True):
    """Alias for :func:`integral`."""
    return integral(f, reduce=reduce)


def norm(f: SurfaceField | SurfaceVectorField, p: int | float | str = 2, reduce: bool = True):
    """Norm of a scalar or vector field: ``p`` in ``1, 2, ..., "inf", "H1", "lap"``."""
    return core.norm(f, p=p, reduce=reduce)


def resample(obj: Any, n: int):
    """Resample a scalar field or a mesh to ``n`` points per patch edge."""
    if isinstance(obj, TriangleSurfaceFunction):
        return tri_resample(obj, n)
    if isinstance(obj, TriangleSurfaceMesh):
        return tri_resample_mesh(obj, n)
    if isinstance(obj, SurfaceFunction):
        return core.resample(obj, n)
    if isinstance(obj, SurfaceMesh):
        return core.resample_mesh(obj, n)
    raise TypeError("resample expects a surface function or surface mesh")


def resample_mesh(dom: SurfaceDomain, n: int) -> SurfaceDomain:
    """Resample a quadrilateral or triangular mesh."""
    if isinstance(dom, TriangleSurfaceMesh):
        return tri_resample_mesh(dom, n)
    if isinstance(dom, SurfaceMesh):
        return core.resample_mesh(dom, n)
    raise TypeError("resample_mesh expects a surface mesh")


def surfaceop(dom: SurfaceDomain, op: dict | PDO, rhs: Any = 0.0, **kwargs) -> SurfaceOp | TriangleSurfaceOp:
    """Fast direct solver for ``L u = f`` on a quadrilateral or triangular mesh.

    See :class:`pysurfacefun.operators.HPSOperator` for the options
    (``rankdef``, ``merge_strategy``, ``merge_idx``, ...).
    """
    _require_domain(dom, "surfaceop")
    if isinstance(dom, TriangleSurfaceMesh):
        return TriangleSurfaceOp(dom, op, rhs, **kwargs)
    return SurfaceOp(dom, op, rhs, **kwargs)


operator = surfaceop


def apply_operator(op: dict | PDO, u: SurfaceField) -> SurfaceField:
    """Apply ``L u = sum a_cd D_c D_d u + sum b_c D_c u + b u`` by spectral differentiation.

    At interior nodes this is exactly the collocation operator that the direct
    solver inverts, so ``apply_operator(op, surfaceop(dom, op, f).solve())``
    reproduces ``f`` there.
    """
    if not isinstance(u, PatchField):
        raise TypeError("apply_operator expects a scalar surface field")
    pdo = parse_pdo(op)
    dom = u.domain
    shape = u.data.shape
    first_derivs = {d: diff(u, dim=d + 1) for d in range(3)}
    out = u._new(np.zeros(shape, dtype=np.result_type(u.data, float)))
    for name, value in pdo.nonzero().items():
        coeff = coefficient_arrays(value, dom, shape)
        if name in SECOND_ORDER_TERMS:
            outer, inner = SECOND_ORDER_TERMS[name]
            term = diff(first_derivs[inner], dim=outer + 1).data
        elif name in FIRST_ORDER_TERMS:
            term = first_derivs[FIRST_ORDER_TERMS[name]].data
        else:
            term = u.data
        out = out._new(out.data + coeff * term)
    return out


def hodge(f: SurfaceVectorField):
    """Hodge decomposition ``f = grad(u) + n x grad(v) + w`` of a tangential field.

    Returns ``(u, v, w, curlfree, divfree)``.  Both potentials reuse one
    factorization of the Laplace--Beltrami operator.
    """
    dom = f.domain
    nvec = core.normal(dom)
    L = surfaceop(dom, {"lap": 1.0}, rankdef=True)
    u = L.solve(div(f)).remove_mean()
    v = L.solve(-div(core.cross(nvec, f))).remove_mean()
    curlfree = grad(u)
    divfree = core.cross(nvec, grad(v))
    w = f - curlfree - divfree
    return u, v, w, curlfree, divfree
