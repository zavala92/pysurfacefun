# pysurfacefun

[![CI](https://github.com/zavala92/pysurfacefun/actions/workflows/ci.yml/badge.svg)](https://github.com/zavala92/pysurfacefun/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE)
[![Documentation Status](https://readthedocs.org/projects/pysurfacefun/badge/?version=latest)](https://pysurfacefun.readthedocs.io/en/latest/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

High-order fast direct solvers for partial differential equations on smooth surfaces.

`pysurfacefun` discretizes surfaces with curved high-order patches, either
tensor-product Chebyshev quadrilaterals or Proriol-Koornwinder-Dubiner
triangles. It inverts elliptic surface operators with a hierarchical
Poincaré–Steklov (HPS) fast direct solver and advances reaction–diffusion
systems with high-order IMEX schemes that reuse the factorization. It depends
only on NumPy.

## Highlights

- **Spectral accuracy** on quadrilateral and triangular patches, one API for both
- **Fast direct solver** with nested-dissection merge ordering, and repeated
  solves compiled into batched matrix products
- **Variable and complex coefficients**, written as equations:
  `"-div(k*grad(u)) + u = f"`
- **IMEX time stepping**: SBDF1–4, CNAB1–2, RK111/222/443, for single fields
  or coupled systems (`"dt(u) - D*lap(u) = N(u, v)"`)
- **Geometry**: cubed sphere, Fourier tori and stellarators, Rhino patch
  meshes, and level-set surfaces from coarse triangular or quadrilateral
  meshes. Open surfaces with Dirichlet data and degenerate patches are supported.
- **Output**: dependency-free binary VTU/VTP files for ParaView, plus evaluator
  handlers for repeatable runs

## Installation

```bash
git clone https://github.com/zavala92/pysurfacefun
cd pysurfacefun
python -m pip install -e .            # NumPy only
python -m pip install -e ".[dev]"     # plotting, MAT files, tests, docs, linting
```

## Quick start

```python
import pysurfacefun as psf

dom = psf.sphere(n=12, nref=1)                      # 24 curved Chebyshev patches
u = psf.field(lambda x, y, z: x * y * z, dom)
print(psf.norm(psf.lap(u) + 12 * u, "inf"))          # x y z is an l = 3 harmonic
```

Solve an elliptic problem with a variable coefficient:

```python
k = psf.field(lambda x, y, z: 1 + 0.5 * x, dom)
f = psf.field(lambda x, y, z: x * y * z, dom)

problem = psf.SurfaceProblem(dom, variables="u", namespace={"k": k, "f": f})
problem.add_equation("-div(k*grad(u)) + u = f")
uh = problem.solve()
```

Reuse one factorization for many right-hand sides:

```python
L = psf.surfaceop(dom, {"lap": -1.0, "c": 4.0})     # coefficient dictionary
u1 = L.solve(f)
u2 = L.solve(2 * f + 1)                             # no refactorization
```

Triangular patches, including level-set surfaces, use the same interface:

```python
dom = psf.icosphere_tri(n=10, nref=1)
u = psf.field(lambda x, y, z: x * y * z, dom)
print(psf.integral(psf.field(1.0, dom)))            # 4 pi
```

Advance a coupled reaction–diffusion system with a third-order IMEX scheme:

```python
bb = psf.boundingbox(dom)
u0 = psf.field(psf.randnfun3(0.3, bb, seed=0), dom)
v0 = psf.field(psf.randnfun3(0.3, bb, seed=1), dom)

ivp = psf.SurfaceIVP(dom, variables={"u": u0, "v": v0}, namespace={"Du": 1e-3, "Dv": 5e-3})
ivp.add_equation("dt(u) - Du*lap(u) = u - u**3 - v")
ivp.add_equation("dt(v) - Dv*lap(v) = 0.1*(u - v)")
state = ivp.build_solver(dt=0.05, scheme="rk443").run(200)   # {"u": ..., "v": ...}
psf.write_vtu("state.vtu", state["u"], point_name="u")
```

The single-field form is unchanged:

```python
solver = psf.SurfaceIVP(u, diffusion=1e-3, reaction=lambda u, t: u - u**3).build_solver(dt=0.05).build()
u = solver.run(20)
```

## Performance

`benchmarks/benchmark_solver.py` measures leaf assembly, factorization, and
repeated-solve times. A repeated solve is the cost of one implicit time step.
Compared with version 0.1.0 (4-core machine, OpenBLAS):

| mesh | patches | factorization | repeated solve |
|---|---:|---:|---:|
| cow (Rhino), n = 12 | 339 | 2.20 s → 0.15 s | 28 ms → 5.0 ms |
| torus 16 × 32, n = 9 | 512 | 1.05 s → 0.17 s | 39 ms → 2.8 ms |
| triangular sphere, n = 9 | 416 | 0.28 s → 0.15 s | 29 ms → 1.7 ms |
| Swiss-cheese set, n = 10 | 616 | 0.36 s → 0.23 s | 29 ms → 3.0 ms |

Nested dissection shrinks the largest interface system on the cow mesh from
1820 to about 200 unknowns. The 2000-step cow and Swiss-cheese notebooks now
finish in 10 s and 18 s. At a fixed step size, the high-order schemes are orders
of magnitude more accurate than IMEX Euler. For the Swiss-cheese system at
Δt = 0.1, the error drops from 2e-1 (SBDF1) to 5e-4 (SBDF4) and 2e-4 (RK443).

## Examples

```bash
python examples/laplace_beltrami_sphere.py
python examples/convergence_laplace_beltrami_sphere.py
python examples/variable_coefficient_elliptic.py
python examples/reaction_diffusion_system.py
python examples/tri_layer4_surfaceproblem_helmholtz.py
python examples/cow_reaction_diffusion_reduced.py --scheme rk222
python examples/evaluator_outputs.py
```

Notebooks in `notebooks/` reproduce the convergence studies and the cow,
Swiss-cheese, and stellarator examples. Generated figures, tables, and VTK files
are written to `notebook_outputs/`.

## Documentation and development

Online documentation: https://pysurfacefun.readthedocs.io/en/latest/

```bash
python -m pip install -e ".[dev]"
python -m pytest                                   # test suite
ruff check . && ruff format --check .              # lint and formatting
sphinx-build -W -b html docs docs/_build/html      # documentation
python benchmarks/benchmark_solver.py              # performance
```

See [CHANGELOG.md](CHANGELOG.md) for release notes.

## References

The quadrilateral HPS implementation is a Python version of the MATLAB
Surfacefun package:
https://github.com/danfortunato/surfacefun

Fortunato, D. (2024). *A high-order fast direct solver for surface PDEs*.
SIAM Journal on Scientific Computing, 46(4), A2582-A2606.

The triangular formulation and fast direct solver framework are introduced in:
*A High-Order Fast Direct Solver for Surface PDEs on Triangles*.
https://arxiv.org/pdf/2604.03097

The IMEX Runge–Kutta schemes are those of Ascher, U. M., Ruuth, S. J., &
Spiteri, R. J. (1997). *Implicit-explicit Runge-Kutta methods for
time-dependent partial differential equations*. Applied Numerical Mathematics,
25(2-3), 151-167.
