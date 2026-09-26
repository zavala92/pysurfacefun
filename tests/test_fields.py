"""Field storage, arithmetic, geometry, and leaf assembly."""

import numpy as np
import pytest

import pysurfacefun as psf
from pysurfacefun.core import _quad_reference, build_quad_leaves
from pysurfacefun.operators import FIRST_ORDER_TERMS, SECOND_ORDER_TERMS, parse_pdo


def test_vals_are_write_through_views_of_contiguous_data():
    dom = psf.sphere(5, 0)
    f = psf.field(lambda x, y, z: x, dom)
    assert f.data.shape == (dom.npatches, 5, 5)
    assert np.shares_memory(f.vals[2], f.data)
    f.vals[2] = np.ones((5, 5))
    assert np.all(f.data[2] == 1.0)
    f.vals[3][0, 0] = 7.0
    assert f.data[3, 0, 0] == 7.0
    f.vals[1] = 2j * np.ones((5, 5))
    assert np.iscomplexobj(f.data) and f.data[1, 0, 0] == 2j
    with pytest.raises(TypeError):
        f.vals.append(np.zeros((5, 5)))


def test_arithmetic_and_numpy_ufuncs():
    dom = psf.icosphere_tri(5, 0)
    f = psf.field(lambda x, y, z: x + 2.0, dom)
    g = np.sin(f) * 2.0 + np.float64(1.0) * f**2 - abs(-f) / 2
    expected = 2 * np.sin(f.data) + f.data**2 - f.data / 2
    assert isinstance(g, psf.TriangleSurfaceFunction)
    assert np.allclose(g.data, expected)
    assert np.allclose((f - 1).real.data, f.data - 1)
    assert float(np.max(f)) == pytest.approx(float(np.max(f.data)))


def test_mixing_quadrilateral_and_triangular_fields_is_an_error():
    f = psf.field(1.0, psf.sphere(5, 0))
    g = psf.field(1.0, psf.icosphere_tri(5, 0))
    with pytest.raises(TypeError):
        f + g


def test_field_construction_validates_patch_count():
    dom = psf.sphere(5, 0)
    with pytest.raises(ValueError):
        psf.SurfaceFunction(dom, [np.zeros((5, 5))] * (dom.npatches - 1))
    const = psf.SurfaceFunction(dom, [np.asarray(1.5)] * dom.npatches)
    assert np.all(const.data == 1.5)


@pytest.mark.parametrize("make", [lambda: psf.sphere(8, 1), lambda: psf.icosphere_tri(8, 1)])
def test_normals_are_consistently_outward(make):
    # The cubed sphere has opposite faces with the same parametric orientation; the
    # orientation must be propagated across edges, not guessed from a global sign.
    dom = make()
    nvec = psf.normal(dom)
    radial = sum(c * x for c, x in zip(nvec.components, (dom.coords[0], dom.coords[1], dom.coords[2])))
    assert np.min(radial.data) > 0.99
    volume = psf.integral(radial) / 3.0
    assert volume == pytest.approx(4 * np.pi / 3, rel=1e-5)


def test_hodge_decomposition_recovers_known_potentials():
    dom = psf.sphere(12, 1)
    u = psf.field(lambda x, y, z: x * y, dom)
    v = psf.field(lambda x, y, z: z**2 + x, dom)
    exact_normal = psf.vector_field(lambda x, y, z: x, lambda x, y, z: y, lambda x, y, z: z, dom)
    f = psf.grad(u) + psf.cross(exact_normal, psf.grad(v))
    uh, vh, w, _curlfree, _divfree = psf.hodge(f)
    # With inconsistently oriented normals (as on the cubed sphere before the fix) these are O(1).
    assert psf.norm(uh - u.remove_mean(), "inf") < 1e-4
    assert psf.norm(vh - v.remove_mean(), "inf") < 1e-4
    assert psf.norm(w, "inf") < 1e-4


def _dense_quad_operator(dom, k, coeffs):
    n = dom.n
    D = psf.diffmat(n)
    eye = np.eye(n)
    Du, Dv = np.kron(eye, D), np.kron(D, eye)
    U = [dom.ux[k].ravel(), dom.uy[k].ravel(), dom.uz[k].ravel()]
    V = [dom.vx[k].ravel(), dom.vy[k].ravel(), dom.vz[k].ravel()]
    Dc = [U[c][:, None] * Du + V[c][:, None] * Dv for c in range(3)]
    A = np.zeros((n * n, n * n))
    for name, value in coeffs.items():
        a = np.broadcast_to(value[k] if np.ndim(value) == 3 else value, (n, n)).ravel()[:, None]
        if name in SECOND_ORDER_TERMS:
            c, d = SECOND_ORDER_TERMS[name]
            A += a * (Dc[c] @ Dc[d])
        elif name in FIRST_ORDER_TERMS:
            A += a * Dc[FIRST_ORDER_TERMS[name]]
        else:
            A += a * np.eye(n * n)
    return A


def test_structured_quad_leaf_assembly_matches_dense_products():
    dom = psf.sphere(7, 0)
    rng = np.random.default_rng(0)
    coeffs = {}
    for name in list(SECOND_ORDER_TERMS) + list(FIRST_ORDER_TERMS) + ["b"]:
        a, b, c = rng.normal(size=3)
        coeffs[name] = np.stack([0.3 * (a * x + b * y + c * z) for x, y, z in zip(dom.x, dom.y, dom.z)])
    for name in ("dxx", "dyy", "dzz"):
        coeffs[name] = coeffs[name] + 2.0
    leaves = build_quad_leaves(parse_pdo(coeffs), dom)
    ref = _quad_reference(dom.n)
    for k in range(dom.npatches):
        A = _dense_quad_operator(dom, k, coeffs)
        Aii = A[np.ix_(ref.interior, ref.interior)]
        Aie = A[np.ix_(ref.interior, ref.boundary)]
        assert np.allclose(leaves.Ainv[k] @ Aii, np.eye(Aii.shape[0]), atol=1e-8)
        S_int = -np.linalg.solve(Aii, Aie) @ ref.S2L
        assert np.allclose(leaves.S[k][ref.interior], S_int, atol=1e-9 * np.max(np.abs(S_int)))


def test_parse_pdo_accumulates_terms():
    pdo = parse_pdo({"lap": 1.0, "dxx": 2.0, "c": 3.0, "b": 1.0, "grad": 0.5, "dy": 1.0})
    assert pdo.dxx == 3.0 and pdo.dyy == 1.0 and pdo.b == 4.0 and pdo.dx == 0.5 and pdo.dy == 1.5
    with pytest.raises(ValueError):
        parse_pdo({"laplacian": 1.0})


def test_quadrature_weights_integrate_first_kind_points_exactly():
    from pysurfacefun.core import chebpts, quadwts

    for n in (2, 5, 8):
        x, w = chebpts(n, 1), quadwts(n, 1)
        for k in range(n):
            assert w @ x**k == pytest.approx(2 / (k + 1) if k % 2 == 0 else 0.0, abs=1e-13)


def test_levelset_projection_vectorized_and_pointwise_agree():
    phi_vec = lambda p: p[0] ** 2 + 2 * p[1] ** 2 + p[2] ** 2 - 1.0
    grad_vec = lambda p: np.array([2 * p[0], 4 * p[1], 2 * p[2]])
    phi_pt = lambda x, y, z: float(np.dot([x, 2 * y, z], [x, y, z])) - 1.0
    grad_pt = lambda x, y, z: [2 * x, 4 * y, 2 * z]
    pts = np.random.default_rng(1).normal(size=(40, 3))
    a = psf.project_to_levelset(pts, phi_vec, grad_vec)
    b = psf.project_to_levelset(pts, phi_pt, grad_pt)
    assert np.max(np.abs(a - b)) < 1e-12
    assert np.max(np.abs(a[:, 0] ** 2 + 2 * a[:, 1] ** 2 + a[:, 2] ** 2 - 1)) < 1e-12
