# Triangular Patches

The triangular solver provides high-order surface discretizations on curved
triangle patches. The implementation uses RecursiveNodes-style simplex nodes
and a Proriol-Koornwinder-Dubiner basis for strong-form differentiation.

The triangular formulation and fast direct solver framework are introduced in:
*A High-Order Fast Direct Solver for Surface PDEs on Triangles*.
https://arxiv.org/pdf/2604.03097

## Core Tools

- `tri_reference_nodes` and `tri_strong_diffmat` for reference-triangle nodes
  and differentiation matrices
- `TriangleSurfaceMesh` for curved high-order triangular patch geometry
- `LevelSetSurface` for projecting coarse meshes to smooth implicit surfaces
- `tri_surfacefun`, `tri_lap`, `tri_integral2`, and `tri_surfacearea` for scalar
  fields and surface operators
- `TriangleSurfaceOp` for shifted elliptic solves on triangular patch meshes
- `write_tri_vtu` and `write_tri_vtp` for ParaView output

## Examples

```bash
python examples/tri_layer1_nodes_basis.py --n 8 --plot
python examples/tri_layer2_icosphere_laplace_identity.py --n 9 --nref 1 --plot
python examples/tri_layer3_prefinement_convergence.py --n-values "5 7 9 11" --nref 1 --plot
python examples/tri_layer4_surfaceop_helmholtz.py --n-values "5 7 9 11" --nref 0 --plot
```

## Level-Set Surfaces

```python
import numpy as np
import pysurfacefun as psf

phi = lambda p: p[0]**2 + p[1]**2 + p[2]**2 - 1.0
dphi = lambda p: np.array([2*p[0], 2*p[1], 2*p[2]])

dom = psf.LevelSetSurface(
    "mesh.mat",
    phi,
    dphi,
    n=12,
    nref=1,
    vertex_name="vertices",
    face_name="faces",
)
```

`LevelSetSurface` accepts triangular or quadrilateral coarse cells. Refinement,
high-order node placement, and level-set projection are handled before the
surface operators are assembled.
