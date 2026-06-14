import numpy as np

import pysurfacefun as psf
from pysurfacefun.recursivenodes_polynomials import proriolkoornwinderdubiner


def test_recursive_triangle_nodes_match_reference_degree_four():
    nodes = psf.recursive_nodes(2, 4, family="lgl", domain="unit")
    expected = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.17267316],
            [0.0, 0.5],
            [0.0, 0.82732684],
            [0.0, 1.0],
            [0.17267316, 0.0],
            [0.22215520, 0.22215520],
            [0.22215520, 0.55568960],
            [0.17267316, 0.82732684],
            [0.5, 0.0],
            [0.55568960, 0.22215520],
            [0.5, 0.5],
            [0.82732684, 0.0],
            [0.82732684, 0.17267316],
            [1.0, 0.0],
        ]
    )
    assert np.max(np.abs(nodes - expected)) < 5e-8


def test_tri_strong_diffmat_exact_for_reference_polynomial():
    x, y = psf.tri_reference_nodes(6)
    Du, Dv, _ = psf.tri_strong_diffmat(5, x, y, basis="pkd")

    f = x**3 * y + 2 * x * y**2 + y
    fx = 3 * x**2 * y + 2 * y**2
    fy = x**3 + 4 * x * y + 1

    assert np.max(np.abs(Du @ f - fx)) < 2e-12
    assert np.max(np.abs(Dv @ f - fy)) < 2e-12


def test_recursive_pkd_basis_shapes_and_derivatives():
    x, y = psf.tri_reference_nodes(6)
    Du_pkd, Dv_pkd, K = psf.tri_strong_diffmat(5, x, y, basis="pkd")
    K2, Kx, Ky = psf.koornwinder_pkd(5, x, y)

    assert K.shape == K2.shape == Kx.shape == Ky.shape
    f = x**2 * y**2 + x
    fx = 2 * x * y**2 + 1
    fy = 2 * x**2 * y
    assert np.max(np.abs(Du_pkd @ f - fx)) < 2e-12
    assert np.max(np.abs(Dv_pkd @ f - fy)) < 2e-12


def test_pkd_polynomial_out_parameter_receives_values():
    x, y = psf.tri_reference_nodes(5)
    xy_biunit = np.column_stack((2.0 * x - 1.0, 2.0 * y - 1.0))
    expected = proriolkoornwinderdubiner(2, (1, 2), xy_biunit)
    out = np.empty_like(expected)

    result = proriolkoornwinderdubiner(2, (1, 2), xy_biunit, out=out)

    assert result is out
    assert np.max(np.abs(out - expected)) < 1e-14


def test_triangular_icosphere_area_converges_with_p():
    exact = 4 * np.pi
    area5 = psf.tri_surfacearea(psf.icosphere_tri(n=5, nref=0))
    area7 = psf.tri_surfacearea(psf.icosphere_tri(n=7, nref=0))
    area9 = psf.tri_surfacearea(psf.icosphere_tri(n=9, nref=0))

    assert abs(area9 - exact) < abs(area7 - exact) < abs(area5 - exact)


def test_triangular_surface_laplacian_identity_improves_with_p():
    errors = []
    for n in (5, 7, 9):
        dom = psf.icosphere_tri(n=n, nref=1)
        f = psf.tri_surfacefun(lambda x, y, z: x * y * z, dom)
        residual = psf.tri_lap(f) + 12 * f
        errors.append(residual.norm_inf())

    assert errors[2] < errors[1] < errors[0]


def test_unified_field_api_dispatches_triangular_operations():
    dom = psf.icosphere_tri(n=7, nref=1)
    f = psf.field(lambda x, y, z: x * y * z, dom)
    one = psf.field(1.0, dom)
    g = psf.grad(f)
    radial = psf.vector_field(lambda x, y, z: x, lambda x, y, z: y, lambda x, y, z: z, dom)

    assert isinstance(f, psf.TriangleSurfaceFunction)
    assert isinstance(g, psf.TriangleSurfaceVectorFunction)
    assert isinstance(radial, psf.TriangleSurfaceVectorFunction)
    assert all(isinstance(comp, psf.TriangleSurfaceFunction) for comp in g.components)
    assert all(np.allclose(a, b) for a, b in zip(psf.lap(f).vals, psf.tri_lap(f).vals))
    assert abs(psf.integral(one) - psf.tri_surfacearea(dom)) < 1e-12
    assert psf.norm(f, "inf") == f.norm_inf()
    assert psf.norm(g, 2) > 0.0
    assert abs(psf.norm(radial, 2) ** 2 - psf.integral(one)) < 1e-12


def test_triangular_surfaceop_helmholtz_improves_with_p():
    errors = []
    for n in (5, 7, 9):
        dom = psf.icosphere_tri(n=n, nref=1)
        exact = psf.tri_surfacefun(lambda x, y, z: x * y * z, dom)
        rhs = 8 * exact
        op = psf.tri_surfaceop(dom, {"lap": 1.0, "c": 20.0}, rhs)
        uh = op.solve()
        errors.append((uh - exact).norm_inf() / exact.norm_inf())

    assert errors[2] < errors[1] < errors[0]


def test_surface_problem_string_equation_solves_triangular_helmholtz():
    dom = psf.icosphere_tri(n=9, nref=1)
    exact = psf.tri_surfacefun(lambda x, y, z: x * y * z, dom)
    alpha = 20.0
    rhs = 8.0 * exact

    problem = psf.SurfaceProblem(dom, variables="u", namespace={"alpha": alpha, "rhs": rhs})
    problem.add_equation("lap(u) + alpha*u = rhs")
    sol = problem.solve()

    err = (sol - exact).norm_inf() / exact.norm_inf()
    assert err < 5e-3


def test_levelset_surface_tri_projects_nodes_to_sphere():
    vertices = 0.8 * np.array(
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

    phi = lambda p: p[0] ** 2 + p[1] ** 2 + p[2] ** 2 - 1.0
    grad_phi = lambda p: np.array([2.0 * p[0], 2.0 * p[1], 2.0 * p[2]])
    dom = psf.levelset_surface_tri(vertices, faces, phi, grad_phi, n=6)

    residual = max(
        np.max(np.abs(x * x + y * y + z * z - 1.0))
        for x, y, z in zip(dom.x, dom.y, dom.z)
    )
    assert residual < 1e-12


def test_extract_tri_mesh_arrays_accepts_legacy_names_and_transposes():
    vertices = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
        ]
    )
    faces_one_based = np.array(
        [
            [1, 2, 3],
            [2, 4, 3],
        ]
    )
    data = {"mesh": {"xs": vertices.T, "surfs": faces_one_based.T}}

    xs, surfs = psf.extract_tri_mesh_arrays(data, mesh_name="mesh")

    assert np.allclose(xs, vertices)
    assert np.array_equal(surfs, faces_one_based)


def test_triangulate_faces_splits_quad_and_orients_outward():
    vertices = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    quad_inward = np.array([[0, 1, 2, 3]])

    tri = psf.triangulate_faces(quad_inward, vertices=vertices, orient_outward=True)
    normals = np.cross(vertices[tri[:, 1]] - vertices[tri[:, 0]], vertices[tri[:, 2]] - vertices[tri[:, 0]])
    centers = np.mean(vertices[tri], axis=1)

    assert tri.shape == (2, 3)
    assert np.all(np.einsum("ij,ij->i", normals, centers) >= 0.0)


def test_levelset_surface_dispatches_triangle_and_quad_meshes():
    phi = lambda p: p[0] ** 2 + p[1] ** 2 + p[2] ** 2 - 1.0
    grad_phi = lambda p: np.array([2.0 * p[0], 2.0 * p[1], 2.0 * p[2]])

    tri_vertices = 0.8 * np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    tri_dom = psf.LevelSetSurface((tri_vertices, np.array([[0, 1, 2]])), phi, grad_phi, n=4)
    assert isinstance(tri_dom, psf.TriangleSurfaceMesh)

    quad_vertices = 0.8 * np.array(
        [
            [0.2, 0.2, 1.0],
            [0.6, 0.2, 1.0],
            [0.6, 0.6, 1.0],
            [0.2, 0.6, 1.0],
        ]
    )
    quad_dom = psf.LevelSetSurface((quad_vertices, np.array([[0, 1, 2, 3]])), phi, grad_phi, n=5)
    assert isinstance(quad_dom, psf.SurfaceMesh)
    residual = max(
        np.max(np.abs(x * x + y * y + z * z - 1.0))
        for x, y, z in zip(quad_dom.x, quad_dom.y, quad_dom.z)
    )
    assert residual < 1e-12


def test_refine_surface_mesh_before_levelset_projection():
    phi = lambda p: p[0] ** 2 + p[1] ** 2 + p[2] ** 2 - 1.0
    grad_phi = lambda p: np.array([2.0 * p[0], 2.0 * p[1], 2.0 * p[2]])

    tri_vertices = 0.8 * np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    tri_faces = np.array([[0, 1, 2]])
    verts_ref, faces_ref = psf.refine_surface_mesh(tri_vertices, tri_faces, nref=2)
    assert faces_ref.shape == (16, 3)

    tri_dom = psf.LevelSetSurface((tri_vertices, tri_faces), phi, grad_phi, n=4, nref=2)
    assert isinstance(tri_dom, psf.TriangleSurfaceMesh)
    assert tri_dom.npatches == 16

    quad_vertices = 0.8 * np.array(
        [
            [0.2, 0.2, 1.0],
            [0.6, 0.2, 1.0],
            [0.6, 0.6, 1.0],
            [0.2, 0.6, 1.0],
        ]
    )
    quad_cells = np.array([[0, 1, 2, 3]])
    verts_ref, cells_ref = psf.refine_surface_mesh(quad_vertices, quad_cells, nref=1)
    assert cells_ref.shape == (4, 4)

    quad_dom = psf.LevelSetSurface((quad_vertices, quad_cells), phi, grad_phi, n=4, nref=1)
    assert isinstance(quad_dom, psf.SurfaceMesh)
    assert quad_dom.npatches == 4
