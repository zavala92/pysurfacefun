# pysurfacefun

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
u = psf.surfacefun(lambda x, y, z: x * y * z, dom)

print(psf.surfacearea(dom))
print(psf.norm(psf.lap(u) + 12 * u, "inf"))
```

Solve a shifted Laplace-Beltrami problem:

```python
alpha = 2.0
rhs = (alpha - 12.0) * u

L = psf.surfaceop(dom, {"lap": 1.0, "b": alpha}, rhs)
uh = L.solve()

relerr = psf.norm(uh - u, "inf") / psf.norm(u, "inf")
print(relerr)
```

Triangular patches use the same public interface:

```python
dom = psf.icosphere_tri(n=10, nref=1)
u = psf.tri_surfacefun(lambda x, y, z: x * y * z, dom)

print(psf.tri_surfacearea(dom))
print(psf.tri_lap(u).norm_inf())
```

## Examples

Run examples from this directory:

```bash
python examples/laplace_beltrami_sphere.py
python examples/convergence_laplace_beltrami_sphere.py
python examples/tri_layer4_surfaceop_helmholtz.py
```

Notebook examples are available in `notebooks/`. Generated figures, tables, and
VTK files are written to `notebook_outputs/`.

## Documentation

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
