"""
pysurfacefun
============

High-order patch-based tools for solving variable-coefficient elliptic
partial differential equations on smooth surfaces.

The implementation uses Chebyshev surface patches, strong-form surface
differentiation, local patch solution operators, and hierarchical
Schur-complement merging for fast repeated elliptic solves.
"""

import numpy as np

from .core import (
    PDO,
    SurfaceFunction,
    SurfaceMesh,
    SurfaceOp,
    SurfaceVectorFunction,
    boundingbox,
    chebpts,
    chebpts2,
    compose,
    conj,
    cos,
    cross,
    div,
    divergence,
    diff,
    diffmat,
    diffx,
    diffy,
    diffz,
    dot,
    exp,
    from_rhino,
    grad as _quad_grad,
    hodge,
    imag,
    integral as _quad_integral,
    lap as _quad_lap,
    log,
    log10,
    maxEst,
    mean2,
    minEst,
    norm as _quad_norm,
    normal,
    normalize,
    plot_surface,
    plot_vector_field,
    prolong,
    real,
    real_spherical_harmonic,
    randnfun3,
    resample as _quad_resample,
    resample_mesh as _quad_resample_mesh,
    sin,
    solve_laplace_beltrami_sphere,
    sphere,
    sqrt,
    stellarator,
    surfacearea,
    surfacefun as _quad_surfacefun,
    surfacefunv as _quad_surfacefunv,
    surfaceop,
    sum2,
    torus,
    vector_norm as _quad_vector_norm,
    write_vtu,
    smooth_random_function_3d,
)
from .tri import (
    LevelSetSurface,
    TriangleSurfaceFunction,
    TriangleSurfaceMesh,
    TriangleSurfaceOp,
    TriangleSurfaceVectorFunction,
    extract_tri_mesh_arrays,
    icosphere_tri,
    koornwinder_pkd,
    levelset_surface,
    levelset_surface_quad,
    levelset_surface_tri,
    levelset_surface_tri_from_mat,
    load_mat_tri_mesh,
    load_mat_surface_mesh,
    node_family_points,
    orient_tri_faces_outward,
    plot_wireframe,
    plot_tri_surface,
    project_to_levelset,
    recursive_nodes,
    refine_quad_mesh,
    refine_surface_mesh,
    refine_tri_mesh,
    reference_triangle_quadrature_weights,
    shifted_lobatto_nodes,
    surface_mesh_arrays,
    triangle_boundary_meshio_mesh,
    triangle_meshio_mesh,
    triangle_surface_mesh_arrays,
    tri_edge_indices,
    tri_patch_boundary_segments,
    tri_wireframe_edge_indices,
    tri_resample,
    tri_resample_mesh,
    tri_resampled_patch_geometry,
    tri_resampled_patch_values,
    tri_diff,
    tri_integral2,
    tri_lap,
    tri_reference_nodes,
    tri_strong_diffmat,
    tri_surfacearea,
    tri_surfacefun,
    tri_surfaceop,
    triangulate_faces,
    trilattice,
    write_tri_patch_boundaries_vtp,
    write_tri_vtp,
    write_tri_vtu,
    write_triangle_meshio,
    wireframe,
)
from .problems import (
    SurfaceEquation,
    SurfaceIVP,
    SurfaceIVPSolver,
    SurfaceLBVP,
    SurfaceLBVPSolver,
    SurfaceProblem,
)
from .evaluator import (
    EvaluationRecord,
    EvaluationTask,
    Evaluator,
    JSONLinesOutputHandler,
    NPZOutputHandler,
    OutputHandler,
    VTKOutputHandler,
)


SurfaceField = SurfaceFunction | TriangleSurfaceFunction
SurfaceVectorField = SurfaceVectorFunction | TriangleSurfaceVectorFunction

surfacefun = _quad_surfacefun
surfacefunv = _quad_surfacefunv


def field(func, dom: SurfaceMesh | TriangleSurfaceMesh) -> SurfaceField:
    """Construct a scalar field on a quadrilateral or triangular surface mesh."""
    if isinstance(dom, TriangleSurfaceMesh):
        return tri_surfacefun(func, dom)
    if isinstance(dom, SurfaceMesh):
        return _quad_surfacefun(func, dom)
    raise TypeError("field expects a SurfaceMesh or TriangleSurfaceMesh domain")


def vector_field(
    fx,
    fy=None,
    fz=None,
    dom: SurfaceMesh | TriangleSurfaceMesh | None = None,
) -> SurfaceVectorField:
    """Construct a vector field on a quadrilateral or triangular surface mesh."""
    if isinstance(fx, (SurfaceVectorFunction, TriangleSurfaceVectorFunction)) and fy is None and fz is None:
        return fx

    if isinstance(fx, SurfaceMesh) and fy is None and fz is None:
        return _quad_surfacefunv(fx)

    if isinstance(fx, TriangleSurfaceMesh) and fy is None and fz is None:
        zero = field(0.0, fx)
        return TriangleSurfaceVectorFunction((zero, zero.copy(), zero.copy()))

    if isinstance(fx, SurfaceFunction) and isinstance(fy, SurfaceFunction) and isinstance(fz, SurfaceFunction):
        return _quad_surfacefunv(fx, fy, fz)

    if isinstance(fx, TriangleSurfaceFunction) and isinstance(fy, TriangleSurfaceFunction) and isinstance(fz, TriangleSurfaceFunction):
        return TriangleSurfaceVectorFunction((fx, fy, fz))

    if dom is None:
        raise ValueError("dom is required when components are not already surface fields")

    if fy is None or fz is None:
        raise ValueError("all three vector components are required")

    if isinstance(dom, SurfaceMesh):
        return _quad_surfacefunv(fx, fy, fz, dom)

    if isinstance(dom, TriangleSurfaceMesh):
        return TriangleSurfaceVectorFunction((field(fx, dom), field(fy, dom), field(fz, dom)))

    raise TypeError("vector_field expects a SurfaceMesh or TriangleSurfaceMesh domain")


def lap(f: SurfaceField) -> SurfaceField:
    """Surface Laplacian for quadrilateral or triangular scalar fields."""
    if isinstance(f, TriangleSurfaceFunction):
        return tri_lap(f)
    if isinstance(f, SurfaceFunction):
        return _quad_lap(f)
    raise TypeError("lap expects a scalar surface field")


def laplacian(f: SurfaceField) -> SurfaceField:
    """Alias for :func:`lap`."""
    return lap(f)


def grad(f: SurfaceField) -> SurfaceVectorField:
    """Surface gradient for quadrilateral or triangular scalar fields."""
    if isinstance(f, TriangleSurfaceFunction):
        return TriangleSurfaceVectorFunction((tri_diff(f, 1), tri_diff(f, 2), tri_diff(f, 3)))
    if isinstance(f, SurfaceFunction):
        return _quad_grad(f)
    raise TypeError("grad expects a scalar surface field")


def gradient(f: SurfaceField) -> SurfaceVectorField:
    """Alias for :func:`grad`."""
    return grad(f)


def vector_norm(f: SurfaceVectorField) -> SurfaceField:
    """Pointwise magnitude of a quadrilateral or triangular vector field."""
    if isinstance(f, TriangleSurfaceVectorFunction):
        a, b, c = f.components
        vals = [np.sqrt(x * x + y * y + z * z) for x, y, z in zip(a.vals, b.vals, c.vals)]
        return TriangleSurfaceFunction(f.domain, vals)
    if isinstance(f, SurfaceVectorFunction):
        return _quad_vector_norm(f)
    raise TypeError("vector_norm expects a vector surface field")


def integral(f: SurfaceField, reduce: bool = True):
    """Surface integral for quadrilateral or triangular scalar fields."""
    if isinstance(f, TriangleSurfaceFunction):
        return tri_integral2(f, reduce=reduce)
    if isinstance(f, SurfaceFunction):
        return _quad_integral(f, reduce=reduce)
    raise TypeError("integral expects a scalar surface field")


def integral2(f: SurfaceField, reduce: bool = True):
    """Alias for :func:`integral`."""
    return integral(f, reduce=reduce)


def norm(f: SurfaceField | SurfaceVectorField, p: int | float | str = 2, reduce: bool = True):
    """Norm of a quadrilateral or triangular scalar/vector surface field."""
    if isinstance(f, (SurfaceFunction, SurfaceVectorFunction)):
        return _quad_norm(f, p=p, reduce=reduce)

    if isinstance(f, TriangleSurfaceVectorFunction):
        mag = vector_norm(f)
        if p == 2 and reduce:
            return float(np.sqrt(integral(mag * mag)))
        return norm(mag, p=p, reduce=reduce)

    if not isinstance(f, TriangleSurfaceFunction):
        raise TypeError("norm expects a scalar or vector surface field")

    if p in (np.inf, "inf", "max"):
        per_patch = np.asarray([np.max(np.abs(v)) for v in f.vals])
        return float(np.max(per_patch)) if reduce else per_patch

    if p == "H1":
        g = grad(f)
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
    per_patch = tri_integral2(TriangleSurfaceFunction(f.domain, [np.abs(v) ** p_float for v in f.vals]), reduce=False)
    per_patch = per_patch ** (1.0 / p_float)
    if reduce:
        return float(np.sum(per_patch**p_float) ** (1.0 / p_float))
    return per_patch


def resample(obj, n: int):
    """Resample a scalar surface function or surface mesh."""
    if isinstance(obj, TriangleSurfaceFunction):
        return tri_resample(obj, n)
    if isinstance(obj, TriangleSurfaceMesh):
        return tri_resample_mesh(obj, n)
    if isinstance(obj, SurfaceFunction):
        return _quad_resample(obj, n)
    if isinstance(obj, SurfaceMesh):
        return _quad_resample_mesh(obj, n)
    raise TypeError("resample expects a surface function or surface mesh")


def resample_mesh(dom, n: int):
    """Resample a quadrilateral or triangular surface mesh."""
    if isinstance(dom, TriangleSurfaceMesh):
        return tri_resample_mesh(dom, n)
    if isinstance(dom, SurfaceMesh):
        return _quad_resample_mesh(dom, n)
    raise TypeError("resample_mesh expects a surface mesh")


__all__ = [
    "PDO",
    "SurfaceField",
    "SurfaceFunction",
    "SurfaceEquation",
    "SurfaceIVP",
    "SurfaceIVPSolver",
    "SurfaceLBVP",
    "SurfaceLBVPSolver",
    "SurfaceMesh",
    "SurfaceOp",
    "SurfaceProblem",
    "SurfaceVectorField",
    "SurfaceVectorFunction",
    "boundingbox",
    "chebpts",
    "chebpts2",
    "compose",
    "conj",
    "cos",
    "cross",
    "div",
    "divergence",
    "diff",
    "diffmat",
    "diffx",
    "diffy",
    "diffz",
    "dot",
    "exp",
    "field",
    "from_rhino",
    "grad",
    "gradient",
    "hodge",
    "imag",
    "integral",
    "integral2",
    "lap",
    "laplacian",
    "log",
    "log10",
    "maxEst",
    "mean2",
    "minEst",
    "norm",
    "normal",
    "normalize",
    "plot_surface",
    "plot_vector_field",
    "prolong",
    "real",
    "real_spherical_harmonic",
    "randnfun3",
    "resample",
    "resample_mesh",
    "sin",
    "solve_laplace_beltrami_sphere",
    "sphere",
    "sqrt",
    "stellarator",
    "smooth_random_function_3d",
    "surfacearea",
    "surfacefun",
    "surfacefunv",
    "surfaceop",
    "sum2",
    "torus",
    "vector_norm",
    "vector_field",
    "write_vtu",
    "LevelSetSurface",
    "TriangleSurfaceFunction",
    "TriangleSurfaceMesh",
    "TriangleSurfaceOp",
    "TriangleSurfaceVectorFunction",
    "EvaluationRecord",
    "EvaluationTask",
    "Evaluator",
    "JSONLinesOutputHandler",
    "NPZOutputHandler",
    "OutputHandler",
    "VTKOutputHandler",
    "extract_tri_mesh_arrays",
    "icosphere_tri",
    "koornwinder_pkd",
    "levelset_surface",
    "levelset_surface_quad",
    "levelset_surface_tri",
    "levelset_surface_tri_from_mat",
    "load_mat_tri_mesh",
    "load_mat_surface_mesh",
    "node_family_points",
    "orient_tri_faces_outward",
    "plot_wireframe",
    "plot_tri_surface",
    "project_to_levelset",
    "recursive_nodes",
    "refine_quad_mesh",
    "refine_surface_mesh",
    "refine_tri_mesh",
    "reference_triangle_quadrature_weights",
    "shifted_lobatto_nodes",
    "surface_mesh_arrays",
    "triangle_boundary_meshio_mesh",
    "triangle_meshio_mesh",
    "triangle_surface_mesh_arrays",
    "tri_edge_indices",
    "tri_patch_boundary_segments",
    "tri_wireframe_edge_indices",
    "tri_resample",
    "tri_resample_mesh",
    "tri_resampled_patch_geometry",
    "tri_resampled_patch_values",
    "tri_diff",
    "tri_integral2",
    "tri_lap",
    "tri_reference_nodes",
    "tri_strong_diffmat",
    "tri_surfacearea",
    "tri_surfacefun",
    "tri_surfaceop",
    "triangulate_faces",
    "trilattice",
    "write_tri_patch_boundaries_vtp",
    "write_tri_vtp",
    "write_tri_vtu",
    "write_triangle_meshio",
    "wireframe",
]
