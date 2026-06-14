# pysurfacefun

[![License: MIT](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE)
[![Documentation Status](https://readthedocs.org/projects/pysurfacefun/badge/?version=latest)](https://pysurfacefun.readthedocs.io/en/latest/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

High-order solvers for partial differential equations on smooth surfaces.

`pysurfacefun` provides spectral patch discretizations, surface differential
operators, and direct solvers for scalar elliptic PDEs on embedded surfaces. It
supports both tensor-product quadrilateral patches and high-order triangular
patches, with tools for geometry construction, level-set projection,
resampling, plotting, and ParaView export.

## Highlights

- High-order Chebyshev discretizations on quadrilateral surface patches
- High-order triangular patches with a Proriol-Koornwinder-Dubiner basis
- Strong-form surface gradients, divergence, and Laplace-Beltrami operators
- Fast direct solvers based on patch-local solution operators and Schur complements
- Level-set construction from coarse triangular or quadrilateral meshes
- Reproducible examples for convergence studies and reaction-diffusion dynamics
- VTU/VTP export for ParaView

## Installation

From this directory:

```bash
python -m pip install -e .
```

Optional dependencies:

```bash
python -m pip install ".[dev]"
```

The core package depends only on NumPy. Optional extras enable plotting, MAT-file
mesh loading, notebook workflows, and ParaView export.

## Quick Start

```python
import pysurfacefun as psf

dom = psf.sphere(n=9, nref=1)
u = psf.field(lambda x, y, z: x * y * z, dom)

print(psf.surfacearea(dom))
print(psf.norm(psf.lap(u) + 12 * u, "inf"))
```

Solve a shifted Laplace-Beltrami problem:

```python
alpha = 2.0
rhs = (alpha - 12.0) * u

problem = psf.SurfaceProblem(dom, variables="u", namespace={"alpha": alpha, "rhs": rhs})
problem.add_equation("lap(u) + alpha*u = rhs")
uh = problem.solve()

relerr = psf.norm(uh - u, "inf") / psf.norm(u, "inf")
print(relerr)
```

Triangular patches use the same public interface:

```python
dom = psf.icosphere_tri(n=10, nref=1)
u = psf.field(lambda x, y, z: x * y * z, dom)

print(psf.integral(psf.field(1.0, dom)))
print(psf.norm(psf.lap(u) + 12 * u, "inf"))
```

Repeatable examples and notebooks can use an evaluator to collect diagnostics,
raw arrays, visualization snapshots, and a manifest:

```python
evaluator = psf.Evaluator(
    "notebook_outputs",
    prefix="sphere_demo",
    handlers=[
        psf.JSONLinesOutputHandler(),
        psf.NPZOutputHandler(),
        psf.VTKOutputHandler(nvis=16),
    ],
)
evaluator.add_task("linf_residual", lambda state: psf.norm(psf.lap(state["u"]) + 12 * state["u"], "inf"))
evaluator.add_task("u", lambda state: state["u"], every=10)
evaluator.evaluate(0, state={"u": u})
```

Reaction-diffusion time stepping uses ``SurfaceIVP``:

```python
def reaction(u, t):
    return u - 2.0 * u**3

solver = psf.SurfaceIVP(u, diffusion=1e-3, reaction=reaction).build_solver(dt=0.05).build()
u = solver.run(20)
```

## Examples

Run examples from this directory:

```bash
python examples/laplace_beltrami_sphere.py
python examples/convergence_laplace_beltrami_sphere.py
python examples/tri_layer4_surfaceproblem_helmholtz.py
python examples/evaluator_outputs.py
```

Notebook examples are available in `notebooks/`. Generated figures, tables, and
VTK files are written to `notebook_outputs/`.

## Documentation

Online documentation:
https://pysurfacefun.readthedocs.io/en/latest/

The Sphinx documentation source is in `docs/`:

```bash
python -m pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

## References

The quadrilateral HPS implementation is a reduced Python version of the MATLAB
Surfacefun package:
https://github.com/danfortunato/surfacefun

Fortunato, D. (2024). *A high-order fast direct solver for surface PDEs*.
SIAM Journal on Scientific Computing, 46(4), A2582-A2606.

The triangular formulation and fast direct solver framework are introduced in:
*A High-Order Fast Direct Solver for Surface PDEs on Triangles*.
https://arxiv.org/pdf/2604.03097
