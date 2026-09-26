"""Equation language, variable coefficients, and high-order time stepping."""

import numpy as np
import pytest

import pysurfacefun as psf

EXACT = lambda x, y, z: x * y * z
KAPPA = lambda x, y, z: 1.0 + 0.5 * x
# On the unit sphere div(kappa grad(xyz)) = kappa*(-12 xyz) + yz(1 - 3x^2)/2.
RHS = lambda x, y, z: -((1 + 0.5 * x) * (-12 * x * y * z) + 0.5 * y * z * (1 - 3 * x**2)) + x * y * z


def _relerr(a, b):
    return psf.norm(a - b, "inf") / psf.norm(b, "inf")


@pytest.mark.parametrize("make", [lambda n: psf.sphere(n, 1), lambda n: psf.icosphere_tri(n, 1)])
def test_divergence_form_variable_coefficients_converge(make):
    errors = []
    for n in (7, 10, 13):
        dom = make(n)
        problem = psf.SurfaceProblem(
            dom, variables="u", namespace={"k": psf.field(KAPPA, dom), "f": psf.field(RHS, dom)}
        )
        problem.add_equation("-div(k*grad(u)) + u = f")
        errors.append(_relerr(problem.solve(), psf.field(EXACT, dom)))
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 1e-5


def test_callable_coefficients_and_operator_dictionaries():
    dom = psf.sphere(11, 1)
    k = psf.field(KAPPA, dom)
    gk = psf.grad(k)
    op = {
        "lap": lambda x, y, z: -(1.0 + 0.5 * x),
        "dx": -gk.components[0],
        "dy": -gk.components[1],
        "dz": -gk.components[2],
        "c": 1.0,
    }
    assert _relerr(psf.surfaceop(dom, op, RHS).solve(), psf.field(EXACT, dom)) < 1e-6


@pytest.mark.parametrize("make", [lambda: psf.sphere(11, 1), lambda: psf.icosphere_tri(12, 1)])
def test_complex_coefficients(make):
    dom = make()
    c = 1.0 + 0.5j
    u = psf.surfaceop(dom, {"lap": c, "c": 3.0}, lambda x, y, z: (-12 * c + 3) * x * y * z).solve()
    assert np.iscomplexobj(u.data)
    assert _relerr(u, psf.field(EXACT, dom)) < 1e-4


def test_apply_operator_inverts_the_solver_at_collocation_nodes():
    for dom in (psf.sphere(8, 1), psf.icosphere_tri(8, 1)):
        op = {"lap": 1.0, "c": 5.0, "dx": 0.3, "dyz": 0.1}
        L = psf.surfaceop(dom, op)
        g = psf.field(lambda x, y, z: np.cos(x) + y, dom)
        residual = psf.apply_operator(op, L.solve(g)) - g
        interior = L.solver.leaf_ops.interior
        assert np.max(np.abs(residual.data.reshape(dom.npatches, -1)[:, interior])) < 1e-9


def test_triangular_fields_can_be_differentiated_in_equations():
    dom = psf.icosphere_tri(9, 1)
    exact = psf.field(EXACT, dom)
    problem = psf.SurfaceProblem(dom, variables="u", namespace={"g": exact})
    problem.add_equation("lap(u) - 2*u = lap(g) - 2*g + dx(g) - dx(g)")
    assert _relerr(problem.solve(), exact) < 1e-3


def test_equation_errors():
    dom = psf.sphere(5, 0)
    problem = psf.SurfaceProblem(dom, variables="u")
    with pytest.raises(ValueError):
        problem.add_equation("u*u = 1")
    with pytest.raises(ValueError):
        psf.SurfaceProblem(dom, variables="u").add_equation("dx(dx(dx(u))) = 1")
    with pytest.raises(ValueError):
        psf.SurfaceProblem(dom, variables="u").add_equation("grad(u) = 1")


def _run(dom, scheme, nsteps, T=0.5):
    u0 = psf.field(lambda x, y, z: 0.8 * x + 0.3 * y * z, dom)
    ivp = psf.SurfaceIVP(u0, diffusion=0.1, reaction=lambda u, t: np.sin(u))
    return ivp.build_solver(T / nsteps, scheme=scheme).run(nsteps)


@pytest.mark.parametrize("scheme", ["sbdf1", "sbdf2", "sbdf3", "sbdf4", "cnab1", "cnab2", "rk111", "rk222", "rk443"])
def test_time_steppers_reach_their_design_order(scheme):
    dom = psf.sphere(8, 0)
    reference = _run(dom, "rk443", 256)
    errors = [psf.norm(_run(dom, scheme, n) - reference, "inf") for n in (8, 16, 32)]
    rate = np.log2(errors[1] / errors[2])
    assert rate > psf.get_scheme(scheme).order - 0.25


def test_multifield_equations_callables_and_legacy_loop_agree():
    dom = psf.icosphere_tri(6, 1)
    bb = psf.boundingbox(dom)
    u0 = psf.field(psf.randnfun3(0.3, bb, seed=0), dom)
    v0 = psf.field(psf.randnfun3(0.3, bb, seed=1), dom)
    Du, Dv, dt = 2.6e-3, 5e-3, 0.1
    Nu = lambda u, v: 0.9 * u * (1 - 0.02 * v**2) + v * (1 - 0.2 * u)
    Nv = lambda u, v: -0.9 * v + u * (-0.9 + 0.2 * v)

    ivp = psf.SurfaceIVP(dom, variables={"u": u0, "v": v0}, namespace={"Du": Du, "Dv": Dv, "Nu": Nu, "Nv": Nv})
    ivp.add_equation("dt(u) - Du*lap(u) = Nu(u, v)")
    ivp.add_equation("dt(v) - Dv*lap(v) = Nv(u, v)")
    a = ivp.build_solver(dt).run(5)

    b = psf.SurfaceIVP(
        dom, variables={"u": u0, "v": v0}, diffusion={"u": Du, "v": Dv}, reaction=lambda u, v, t: (Nu(u, v), Nv(u, v))
    )
    b = b.build_solver(dt).run(5)

    us = psf.SurfaceIVP(u0, diffusion=Du).build_solver(dt).build()
    vs = psf.SurfaceIVP(v0, diffusion=Dv).build_solver(dt).build()
    u, v = u0, v0
    for _ in range(5):
        u, v = us.step(rhs=u + dt * Nu(u, v)), vs.step(rhs=v + dt * Nv(u, v))
    for name, legacy in (("u", u), ("v", v)):
        assert psf.norm(a[name] - b[name], "inf") < 1e-13
        assert psf.norm(a[name] - legacy, "inf") < 1e-13

    # Terms on the right-hand side are explicit: explicit diffusion differs from implicit diffusion.
    explicit = psf.SurfaceIVP(dom, variables={"u": u0, "v": v0}, namespace={"Du": Du, "Dv": Dv, "Nu": Nu, "Nv": Nv})
    explicit.add_equation("dt(u) - Du*lap(u) = Nu(u, v)")
    explicit.add_equation("dt(v) = Dv*lap(v) + Nv(u, v)")
    c = explicit.build_solver(dt).run(5)
    assert psf.norm(c["u"] - a["u"], "inf") > 1e-6


def test_fields_with_identical_operators_share_one_factorization():
    dom = psf.sphere(6, 0)
    ivp = psf.SurfaceIVP(dom, variables={"u": 1.0, "v": 0.0}, diffusion=1e-2, reaction=lambda u, v: (v, -u))
    solver = ivp.build_solver(0.05, scheme="rk222")
    out = solver.run(10)
    assert len(solver._operators) == 1
    # u'' = -u with spatially constant data: u = cos(t), v = -sin(t) (to second order).
    assert abs(float(np.mean(out["u"].data)) - np.cos(0.5)) < 1e-3


def test_ivp_errors_and_state_access():
    u0 = psf.field(lambda x, y, z: x, psf.sphere(6, 0))
    solver = psf.SurfaceIVP(u0, diffusion=0.1).build_solver(0.1, scheme="rk222")
    with pytest.raises(ValueError):
        solver.step(rhs=u0)
    with pytest.raises(ValueError):
        psf.get_scheme("rk999")
    solver.state = 2.0 * u0
    solver.step()
    assert solver.iteration == 1 and solver.t == pytest.approx(0.1)
    ivp = psf.SurfaceIVP(u0.domain, variables=["u", "v"])
    ivp.add_equation("dt(u) = lap(u)")
    with pytest.raises(ValueError):
        ivp.build_solver(0.1).step()
