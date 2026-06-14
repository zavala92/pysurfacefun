import json
from pathlib import Path
import tempfile

import pysurfacefun as psf


def test_surface_ivp_backward_euler_heat_step_on_quad_mesh():
    dom = psf.sphere(n=9, nref=0)
    u0 = psf.field(lambda x, y, z: x * y * z, dom)
    diffusion = 0.1
    dt = 0.05

    solver = psf.SurfaceIVP(u0, diffusion=diffusion).build_solver(dt).build()
    u1 = solver.step()
    expected = u0 / (1.0 + 12.0 * diffusion * dt)

    err = psf.norm(u1 - expected, "inf") / psf.norm(expected, "inf")
    assert err < 2e-3
    assert solver.iteration == 1
    assert solver.t == dt


def test_surface_ivp_reaction_diffusion_step_on_tri_mesh():
    dom = psf.icosphere_tri(n=9, nref=1)
    u0 = psf.field(lambda x, y, z: x * y * z, dom)
    diffusion = 0.1
    alpha = 0.4
    dt = 0.05

    problem = psf.SurfaceIVP(
        dom,
        initial=lambda x, y, z: x * y * z,
        diffusion=diffusion,
        reaction=lambda u, t: alpha * u,
    )
    solver = problem.build_solver(dt)
    u1 = solver.step()
    expected = ((1.0 + alpha * dt) / (1.0 + 12.0 * diffusion * dt)) * u0

    err = psf.norm(u1 - expected, "inf") / psf.norm(expected, "inf")
    assert err < 5e-3


def test_surface_ivp_step_accepts_explicit_rhs_override():
    dom = psf.icosphere_tri(n=9, nref=1)
    u0 = psf.field(lambda x, y, z: x * y * z, dom)
    dt = 0.05
    diffusion = 0.1
    forcing = 0.25 * u0

    solver = psf.SurfaceIVP(u0, diffusion=diffusion).build_solver(dt).build()
    u1 = solver.step(rhs=u0 + dt * forcing)
    expected = (1.0 + 0.25 * dt) * u0 / (1.0 + 12.0 * diffusion * dt)

    err = psf.norm(u1 - expected, "inf") / psf.norm(expected, "inf")
    assert err < 5e-3


def test_surface_ivp_run_writes_evaluator_diagnostics():
    with tempfile.TemporaryDirectory() as tmp:
        dom = psf.sphere(n=7, nref=0)
        u0 = psf.field(lambda x, y, z: x * y * z, dom)
        solver = psf.SurfaceIVP(u0, diffusion=0.05).build_solver(0.02)
        evaluator = psf.Evaluator(tmp, prefix="ivp", handlers=[psf.JSONLinesOutputHandler()])
        evaluator.add_task("linf", lambda state: psf.norm(state["u"], "inf"))

        solver.run(2, evaluator=evaluator, evaluate_initial=True)

        rows = [
            json.loads(line)
            for line in (Path(tmp) / "scalars.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert [row["iteration"] for row in rows] == [0, 1, 2]
        assert rows[-1]["time"] == solver.t
