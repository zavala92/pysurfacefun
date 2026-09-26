"""
Dependency-free VTK XML writers for surface meshes and fields.

The writers produce ParaView-compatible ``.vtu`` (UnstructuredGrid) and
``.vtp`` (PolyData) files using only NumPy and the standard library.  Data
arrays are written in the VTK *binary* inline format (base64 with a
``UInt64`` header), optionally zlib-compressed, or as ASCII text.

Quadrilateral patches become grids of ``VTK_QUAD`` cells and triangular
patches ``VTK_TRIANGLE`` cells; every cell carries its patch index in the
``patch`` cell array.  Complex scalar fields are written as ``<name>_real``,
``<name>_imag``, and ``<name>_abs``; vector fields as three-component arrays.
"""

from __future__ import annotations

import base64
import zlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

Array = np.ndarray

VTK_LINE = 3
VTK_TRIANGLE = 5
VTK_QUAD = 9

_VTK_TYPES = {
    np.dtype("<f8"): "Float64",
    np.dtype("<f4"): "Float32",
    np.dtype("<i8"): "Int64",
    np.dtype("<i4"): "Int32",
    np.dtype("<u1"): "UInt8",
}
_BLOCK_SIZE = 1 << 15


def _little_endian(arr: Array) -> Array:
    arr = np.ascontiguousarray(arr)
    if arr.dtype.kind == "f":
        return arr.astype("<f8", copy=False)
    if arr.dtype == np.uint8 or arr.dtype.kind == "b":
        return arr.astype("<u1", copy=False)
    if arr.dtype.kind in "iu":
        return arr.astype("<i4" if arr.dtype.kind == "i" and arr.dtype.itemsize <= 4 else "<i8", copy=False)
    raise TypeError(f"unsupported VTK data type {arr.dtype}")


def encode_binary(arr: Array, compress: bool = True) -> str:
    """Encode an array in VTK's inline binary format (``header_type="UInt64"``)."""
    raw = np.ascontiguousarray(arr).tobytes()
    if not compress:
        header = np.array([len(raw)], dtype="<u8").tobytes()
        return base64.b64encode(header + raw).decode("ascii")
    nblocks = (len(raw) + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    blocks = [zlib.compress(raw[i * _BLOCK_SIZE : (i + 1) * _BLOCK_SIZE], 6) for i in range(nblocks)]
    last = len(raw) - (nblocks - 1) * _BLOCK_SIZE if nblocks else 0
    header = np.array([nblocks, _BLOCK_SIZE, last, *(len(b) for b in blocks)], dtype="<u8")
    return base64.b64encode(header.tobytes()).decode("ascii") + base64.b64encode(b"".join(blocks)).decode("ascii")


def _format_ascii(arr: Array) -> str:
    flat = np.asarray(arr).ravel()
    if flat.dtype.kind == "f":
        items = [f"{float(v):.17e}" for v in flat]
    else:
        items = [str(int(v)) for v in flat]
    width = 9 if flat.dtype.kind == "f" else 16
    return "\n".join("          " + " ".join(items[i : i + width]) for i in range(0, len(items), width))


def _data_array(name: str | None, arr: Array, binary: bool, compress: bool, ncomp: int | None = None) -> str:
    arr = _little_endian(arr)
    vtk_type = _VTK_TYPES[arr.dtype]
    attrs = f'type="{vtk_type}"'
    if name is not None:
        attrs += f' Name="{name}"'
    if ncomp is not None:
        attrs += f' NumberOfComponents="{ncomp}"'
    if binary:
        return f'        <DataArray {attrs} format="binary">{encode_binary(arr, compress)}</DataArray>\n'
    return f'        <DataArray {attrs} format="ascii">\n{_format_ascii(arr)}\n        </DataArray>\n'


def _expand_point_data(point_data: Mapping[str, Any] | None) -> list[tuple[str, Array, int | None]]:
    """Split complex arrays into real/imag/abs parts and tag vector arrays."""
    out: list[tuple[str, Array, int | None]] = []
    for name, data in (point_data or {}).items():
        arr = np.asarray(data)
        ncomp = 3 if arr.ndim == 2 and arr.shape[1] == 3 else None
        if ncomp is None:
            arr = arr.ravel()
        if np.iscomplexobj(arr):
            if ncomp is None:
                out.append((f"{name}_real", np.real(arr), None))
                out.append((f"{name}_imag", np.imag(arr), None))
                out.append((f"{name}_abs", np.abs(arr), None))
            else:
                out.append((f"{name}_real", np.real(arr), 3))
                out.append((f"{name}_imag", np.imag(arr), 3))
        else:
            out.append((name, arr.astype(float), ncomp))
    return out


def _header(kind: str, binary: bool, compress: bool) -> str:
    compressor = ' compressor="vtkZLibDataCompressor"' if binary and compress else ""
    return (
        '<?xml version="1.0"?>\n'
        f'<VTKFile type="{kind}" version="1.0" byte_order="LittleEndian" header_type="UInt64"{compressor}>\n'
    )


def _attribute_block(tag: str, arrays: list[tuple[str, Array, int | None]], binary: bool, compress: bool) -> str:
    scalars = next((name for name, _, nc in arrays if nc is None), None)
    vectors = next((name for name, _, nc in arrays if nc == 3), None)
    attrs = ""
    if scalars is not None:
        attrs += f' Scalars="{scalars}"'
    if vectors is not None:
        attrs += f' Vectors="{vectors}"'
    body = "".join(_data_array(name, arr, binary, compress, nc) for name, arr, nc in arrays)
    return f"      <{tag}{attrs}>\n{body}      </{tag}>\n"


def write_vtu(
    filename: str | Path,
    points: Array,
    cells: Array,
    cell_type: int = VTK_TRIANGLE,
    point_data: Mapping[str, Any] | None = None,
    cell_data: Mapping[str, Any] | None = None,
    binary: bool = True,
    compress: bool = True,
) -> Path:
    """Write an unstructured grid with a single cell type to ``filename``."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    cells = np.asarray(cells, dtype=np.int64)
    ncells, nverts = cells.shape if cells.size else (0, 3)
    offsets = nverts * np.arange(1, ncells + 1, dtype=np.int64)
    types = np.full(ncells, cell_type, dtype=np.uint8)
    parts = [
        _header("UnstructuredGrid", binary, compress),
        "  <UnstructuredGrid>\n",
        f'    <Piece NumberOfPoints="{points.shape[0]}" NumberOfCells="{ncells}">\n',
        _attribute_block("PointData", _expand_point_data(point_data), binary, compress),
        _attribute_block("CellData", _expand_point_data(cell_data), binary, compress),
        "      <Points>\n",
        _data_array(None, points, binary, compress, 3),
        "      </Points>\n",
        "      <Cells>\n",
        _data_array("connectivity", cells.ravel(), binary, compress),
        _data_array("offsets", offsets, binary, compress),
        _data_array("types", types, binary, compress),
        "      </Cells>\n",
        "    </Piece>\n",
        "  </UnstructuredGrid>\n",
        "</VTKFile>\n",
    ]
    path = Path(filename)
    path.write_text("".join(parts), encoding="utf-8")
    return path


def write_vtp(
    filename: str | Path,
    points: Array,
    polys: Array | None = None,
    lines: Array | None = None,
    point_data: Mapping[str, Any] | None = None,
    cell_data: Mapping[str, Any] | None = None,
    binary: bool = True,
    compress: bool = True,
) -> Path:
    """Write PolyData with polygon cells (``polys``) and/or line cells (``lines``)."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    polys = np.zeros((0, 3), dtype=np.int64) if polys is None else np.asarray(polys, dtype=np.int64)
    lines = np.zeros((0, 2), dtype=np.int64) if lines is None else np.asarray(lines, dtype=np.int64)
    npolys = polys.shape[0]
    nlines = lines.shape[0]

    def cell_block(tag: str, conn: Array) -> str:
        if conn.shape[0] == 0:
            return ""
        offsets = conn.shape[1] * np.arange(1, conn.shape[0] + 1, dtype=np.int64)
        return (
            f"      <{tag}>\n"
            + _data_array("connectivity", conn.ravel(), binary, compress)
            + _data_array("offsets", offsets, binary, compress)
            + f"      </{tag}>\n"
        )

    parts = [
        _header("PolyData", binary, compress),
        "  <PolyData>\n",
        f'    <Piece NumberOfPoints="{points.shape[0]}" NumberOfVerts="0" NumberOfLines="{nlines}" '
        f'NumberOfStrips="0" NumberOfPolys="{npolys}">\n',
        _attribute_block("PointData", _expand_point_data(point_data), binary, compress),
        _attribute_block("CellData", _expand_point_data(cell_data), binary, compress),
        "      <Points>\n",
        _data_array(None, points, binary, compress, 3),
        "      </Points>\n",
        cell_block("Lines", lines),
        cell_block("Polys", polys),
        "    </Piece>\n",
        "  </PolyData>\n",
        "</VTKFile>\n",
    ]
    path = Path(filename)
    path.write_text("".join(parts), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Surface meshes and fields.
# ---------------------------------------------------------------------------


def _is_quad_domain(domain: Any) -> bool:
    from .core import SurfaceMesh

    return isinstance(domain, SurfaceMesh)


def _domain_of(value: Any) -> Any:
    domain = getattr(value, "domain", None)
    if domain is None:
        raise TypeError("expected a surface field or surface mesh")
    return domain


def _resampler(domain: Any, nvis: int | None):
    """Return ``(npoints_per_patch, grid_n, func)`` where ``func`` resamples ``(P, ...)`` patch data."""
    if _is_quad_domain(domain):
        from .core import barymat, chebpts

        n = domain.n
        nplot = n if nvis is None else int(nvis)
        if nplot < 2:
            raise ValueError("nvis must be at least 2")
        if nplot == n:
            return nplot * nplot, nplot, lambda data: np.asarray(data).reshape(domain.npatches, n * n)
        B = barymat(chebpts(nplot, 2), chebpts(n, 2))
        return (
            nplot * nplot,
            nplot,
            lambda data: np.matmul(B, np.matmul(np.asarray(data), B.T)).reshape(domain.npatches, -1),
        )

    from .tri import tri_resample_matrix

    n = domain.n
    nplot = n if nvis is None else int(nvis)
    if nplot < 2:
        raise ValueError("nvis must be at least 2")
    if nplot == n:
        return n * (n + 1) // 2, nplot, lambda data: np.asarray(data).reshape(domain.npatches, -1)
    B = tri_resample_matrix(n, nplot, domain.family, domain.family)
    return nplot * (nplot + 1) // 2, nplot, lambda data: np.asarray(data).reshape(domain.npatches, -1) @ B.T


def surface_grid(domain: Any, nvis: int | None = None) -> tuple[Array, Array, int, Array]:
    """Points, cells, VTK cell type, and per-cell patch ids for a surface mesh."""
    npts, nplot, resample = _resampler(domain, nvis)
    coords = np.stack(
        [resample(np.stack([np.asarray(c) for c in comp])) for comp in (domain.x, domain.y, domain.z)], axis=-1
    )
    P = domain.npatches
    points = coords.reshape(P * npts, 3)
    if _is_quad_domain(domain):
        i, j = np.meshgrid(np.arange(nplot - 1), np.arange(nplot - 1), indexing="ij")
        a = (i * nplot + j).ravel()
        local = np.stack((a, a + 1, a + nplot + 1, a + nplot), axis=1)
        cell_type = VTK_QUAD
    else:
        from .tri import trilattice

        local = trilattice(nplot)
        cell_type = VTK_TRIANGLE
    cells = (local[None, :, :] + (npts * np.arange(P))[:, None, None]).reshape(-1, local.shape[1])
    patch_ids = np.repeat(np.arange(P, dtype=np.int64), local.shape[0])
    return points, cells, cell_type, patch_ids


def field_point_values(value: Any, nvis: int | None = None) -> Array:
    """Point values of a scalar ``(Np,)`` or vector ``(Np, 3)`` field on :func:`surface_grid` points."""
    from .fields import PatchField, PatchVectorField

    domain = _domain_of(value)
    _npts, _, resample = _resampler(domain, nvis)
    if isinstance(value, PatchVectorField):
        return np.stack([resample(c.data).reshape(-1) for c in value.components], axis=1)
    if isinstance(value, PatchField):
        return resample(value.data).reshape(-1)
    raise TypeError("expected a scalar or vector surface field")


def write_vtu_fields(
    filename: str | Path,
    fields: Mapping[str, Any] | Any,
    nvis: int | None = None,
    binary: bool = True,
    compress: bool = True,
) -> Path:
    """Write one or more surface fields (or a bare mesh) to a ``.vtu`` file.

    Parameters
    ----------
    filename:
        Output path.
    fields:
        ``{name: field}`` mapping of scalar or vector fields on the same mesh,
        a single field (written as ``"u"``), or a surface mesh (geometry only).
    nvis:
        Optional number of points per patch edge for visualization
        (spectral resampling of geometry and data).
    binary, compress:
        Binary (base64) inline data, optionally zlib-compressed; ASCII when
        ``binary=False``.
    """
    if not isinstance(fields, Mapping):
        fields = {"u": fields}
    fields = dict(fields)
    if not fields:
        raise ValueError("no fields to write")
    domain = _domain_of(next(iter(fields.values())))
    points, cells, cell_type, patch_ids = surface_grid(domain, nvis)
    point_data = {name: field_point_values(value, nvis) for name, value in fields.items()}
    return write_vtu(filename, points, cells, cell_type, point_data, {"patch": patch_ids}, binary, compress)


def write_mesh_vtu(
    filename: str | Path, domain: Any, nvis: int | None = None, binary: bool = True, compress: bool = True
) -> Path:
    """Write the geometry of a surface mesh (with patch ids) to a ``.vtu`` file."""
    points, cells, cell_type, patch_ids = surface_grid(domain, nvis)
    return write_vtu(filename, points, cells, cell_type, None, {"patch": patch_ids}, binary, compress)
