import pysurfacefun as psf


def test_laplace_beltrami_error_decreases_with_p():
    l, m = 3, 2
    errors = []
    for n in (5, 7, 9):
        dom = psf.sphere(n=n, nref=0)
        exact = psf.surfacefun(
            lambda x, y, z: psf.real_spherical_harmonic(l, m, x, y, z),
            dom,
        )
        rhs = -l * (l + 1) * exact
        op = psf.surfaceop(dom, {"lap": 1.0}, rhs)
        op.rankdef = True
        sol = op.solve().remove_mean()
        errors.append(psf.norm(sol - exact.remove_mean(), "inf") / psf.norm(exact, "inf"))

    assert errors[2] < errors[1] < errors[0]


def test_surface_differentiation_identity_for_xyz():
    dom = psf.sphere(n=7, nref=0)
    f = psf.surfacefun(lambda x, y, z: x * y * z, dom)

    # x*y*z is a degree-3 spherical harmonic on S^2, so
    # Delta_Gamma f = -12 f.
    residual = psf.lap(f) + 12 * f
    assert psf.norm(residual, "inf") < 2e-2


def test_surface_area_converges_to_unit_sphere_area():
    area7 = psf.surfacearea(psf.sphere(n=7, nref=0))
    area9 = psf.surfacearea(psf.sphere(n=9, nref=0))
    exact = 4 * 3.141592653589793
    assert abs(area9 - exact) < abs(area7 - exact)


def test_resample_preserves_low_order_function():
    dom = psf.sphere(n=7, nref=0)
    f = psf.surfacefun(lambda x, y, z: x * y * z, dom)
    g = psf.resample(f, 11)
    assert g.domain.n == 11
    assert abs(psf.norm(g, "inf") - psf.norm(f, "inf")) < 1e-12


def test_torus_geometry_area_stabilizes_with_p():
    area7 = psf.surfacearea(psf.torus(n=7, nu=2, nv=4))
    area9 = psf.surfacearea(psf.torus(n=9, nu=2, nv=4))
    area11 = psf.surfacearea(psf.torus(n=11, nu=2, nv=4))

    assert abs(area11 - area9) < abs(area9 - area7)


def test_torus_helmholtz_self_consistency():
    dom = psf.torus(n=9, nu=2, nv=4)
    exact = psf.surfacefun(lambda x, y, z: x + y + z, dom)
    alpha = 20.0
    rhs = psf.lap(exact) + alpha * exact

    op = psf.surfaceop(dom, {"lap": 1.0, "c": alpha}, rhs)
    sol = op.solve()

    err = psf.norm(sol - exact, "inf") / psf.norm(exact, "inf")
    assert err < 5e-3


def test_torus_merge_tree_is_closed():
    dom = psf.torus(n=7, nu=2, nv=4)
    f = psf.surfacefun(lambda x, y, z: x + y + z, dom)
    op = psf.surfaceop(dom, {"lap": 1.0, "c": 20.0}, f)
    op.build()

    assert op.patches[0].xyz.shape[0] == 0


def test_stellarator_normal_is_unit_length():
    dom = psf.stellarator(n=5, nu=2, nv=4)
    vn = psf.normal(dom)
    assert psf.norm(psf.vector_norm(vn) - 1, "inf") < 1e-12


def test_hodge_reconstruction_smoke():
    dom = psf.stellarator(n=5, nu=2, nv=4)
    x0 = y0 = z0 = 0.2

    def denom(x, y, z):
        return ((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2) ** 1.5

    G = psf.surfacefunv(
        lambda x, y, z: (x - x0) / denom(x, y, z),
        lambda x, y, z: (y - y0) / denom(x, y, z),
        lambda x, y, z: (z - z0) / denom(x, y, z),
        dom,
    )
    V = psf.cross([0, 1, 1], G)
    vn = psf.normal(dom)
    f = -psf.cross(vn, vn, V)

    _, _, w, curlfree, divfree = psf.hodge(f)
    reconstruction = f - curlfree - divfree - w

    assert psf.norm(psf.dot(vn, f), "inf") < 1e-12
    assert psf.norm(reconstruction, "inf") < 1e-12
