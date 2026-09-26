"""VTK XML output: decode the files with the standard library and compare the data."""

import base64
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

import numpy as np
import pytest

import pysurfacefun as psf

_TYPES = {"Float64": "<f8", "Float32": "<f4", "Int64": "<i8", "Int32": "<i4", "UInt8": "<u1"}


def _decode(element, compressed):
    dtype = np.dtype(_TYPES[element.get("type")])
    text = element.text.strip()
    if element.get("format") == "ascii":
        return np.array(text.split(), dtype=dtype)
    if not compressed:
        raw = base64.b64decode(text)
        nbytes = int(np.frombuffer(raw[:8], "<u8")[0])
        return np.frombuffer(raw[8 : 8 + nbytes], dtype)
    nblocks = int(np.frombuffer(base64.b64decode(text[:12])[:8], "<u8")[0])
    header_chars = 4 * ((8 * (3 + nblocks) + 2) // 3)
    header = np.frombuffer(base64.b64decode(text[:header_chars]), "<u8")
    data = base64.b64decode(text[header_chars:])
    out, pos = [], 0
    for size in header[3:]:
        out.append(zlib.decompress(data[pos : pos + int(size)]))
        pos += int(size)
    return np.frombuffer(b"".join(out), dtype)


def _read(path):
    root = ET.parse(path).getroot()
    compressed = root.get("compressor") is not None
    arrays = {}
    for element in root.iter("DataArray"):
        name = element.get("Name") or "Points"
        values = _decode(element, compressed)
        ncomp = int(element.get("NumberOfComponents", "1"))
        arrays[name] = values.reshape(-1, ncomp) if ncomp > 1 else values
    return root, arrays


@pytest.mark.parametrize("binary,compress", [(True, True), (True, False), (False, False)])
def test_quad_scalar_and_vector_fields_round_trip(tmp_path, binary, compress):
    dom = psf.sphere(6, 1)
    u = psf.field(lambda x, y, z: x * y * z, dom)
    g = psf.grad(u)
    path = psf.write_vtu_fields(tmp_path / "quad.vtu", {"u": u, "g": g}, binary=binary, compress=compress)
    root, arrays = _read(path)
    piece = root.find("UnstructuredGrid/Piece")
    assert int(piece.get("NumberOfPoints")) == dom.npatches * 36
    assert int(piece.get("NumberOfCells")) == dom.npatches * 25
    assert np.array_equal(arrays["u"], u.data.reshape(-1))
    assert np.array_equal(arrays["g"][:, 2], g.components[2].data.reshape(-1))
    assert set(np.unique(arrays["types"])) == {9}
    assert np.array_equal(np.unique(arrays["patch"]), np.arange(dom.npatches))
    assert np.allclose(arrays["Points"][:, 0], dom.coords[0].reshape(-1))


def test_triangular_complex_field_with_resampling(tmp_path):
    dom = psf.icosphere_tri(5, 0)
    u = psf.field(lambda x, y, z: (1 + 2j) * x, dom)
    psf.write_tri_vtu(tmp_path / "tri.vtu", u, point_name="w", nvis=8)
    _, arrays = _read(tmp_path / "tri.vtu")
    assert {"w_real", "w_imag", "w_abs"} <= set(arrays)
    assert np.allclose(arrays["w_imag"], 2 * arrays["w_real"])
    assert np.allclose(arrays["w_real"], arrays["Points"][:, 0], atol=1e-12)
    assert arrays["Points"].shape[0] == dom.npatches * 36


def test_legacy_writers_and_polydata(tmp_path):
    dom = psf.icosphere_tri(5, 0)
    u = psf.field(lambda x, y, z: z, dom)
    psf.write_vtu(tmp_path / "a.vtu", psf.field(lambda x, y, z: z, psf.sphere(5, 0)))
    psf.write_tri_vtp(tmp_path / "b.vtp", u)
    psf.write_tri_patch_boundaries_vtp(tmp_path / "c.vtp", dom)
    for name in ("a.vtu", "b.vtp", "c.vtp"):
        root, arrays = _read(tmp_path / name)
        assert root.tag == "VTKFile" and "Points" in arrays
    assert _read(tmp_path / "c.vtp")[1]["connectivity"].size > 0


def test_meshio_can_read_the_output(tmp_path):
    meshio = pytest.importorskip("meshio")
    dom = psf.sphere(5, 0)
    psf.write_vtu(tmp_path / "m.vtu", psf.field(lambda x, y, z: x, dom), point_name="x")
    mesh = meshio.read(Path(tmp_path / "m.vtu"))
    assert np.allclose(mesh.point_data["x"], mesh.points[:, 0])
