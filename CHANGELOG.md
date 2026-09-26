# Changelog

## 0.2.0

### Fast direct solver

- New shared HPS engine (`pysurfacefun.hps`) for quadrilateral and triangular
  patches.
- Merges are ordered by nested dissection of the patch adjacency graph:
  principal-axis bisection refined by Fiduccia–Mattheyses passes. On the Rhino
  cow mesh at n = 12, the largest interface system shrinks from 1820 to ~200
  unknowns and factorization work drops ~60×. `merge_strategy="natural"`
  restores the previous ordering.
- The factorization is independent of the right-hand side.
  `SurfaceOp.solve(rhs)`, `solve_many([...])`, and `apply(rhs)` reuse it.
  Repeated solves are compiled into level-batched gathers and stacked matrix
  products and are 5–15× faster.
- Complex right-hand sides and complex coefficients are supported.
- `solve(bc=g)` imposes Dirichlet data on open surfaces.
- `stats()` reports separator sizes and cost estimates.
- Quadrilateral leaves are assembled in O(n⁴) from the tensor-product structure
  instead of O(n⁶) dense products.
- Quadrilateral leaves support variable and complex coefficients and
  degenerate (singular) patches.

### Fields and geometry

- Meshes and fields store their values as contiguous `(P, ...)` arrays
  (`field.data`). `field.vals` remains a list of per-patch views with
  write-through assignment.
- Arithmetic, NumPy ufuncs (`np.sin(u)`, `np.max(u)`), differentiation,
  integration, and norms are vectorized over patches.
- `normal()` orients all patches consistently (and outward on closed surfaces).
  It now works on triangular meshes.
- `div`, `diff`, `surfaceop`, and `hodge` dispatch on quadrilateral and
  triangular meshes. `hodge` factors the Laplace–Beltrami operator once for
  both potentials.
- Level-set projection is vectorized (~30× faster surface construction).
  Pointwise-only level-set functions still work.
- New `apply_operator(op, u)` applies a coefficient dictionary by spectral
  differentiation.

### Problems and time stepping

- The equation language supports variable coefficients and the product rule:
  `"-div(k*grad(u)) + u = f"`, `grad`, `div`, `dot`, and `dx/dy/dz` of
  triangular fields.
- New IMEX schemes: SBDF1–4 (started with RK443), CNAB1–2, and RK111/222/443
  (Ascher–Ruuth–Spiteri). Select them with `build_solver(dt, scheme=...)`. All
  reach their design order.
- Multi-field initial-value problems can be written as equations
  (`"dt(u) - D*lap(u) = N(u, v)"`) or with callables. Fields with identical
  linear operators share one factorization.

### Output

- A dependency-free VTK XML writer (`pysurfacefun.vtk`) writes binary,
  zlib-compressed files: ~9× smaller than the previous ASCII output. It uses
  quad cells for quadrilateral patches, a per-cell `patch` id, and
  three-component vector arrays.
- `write_vtu` no longer requires meshio and accepts triangular and vector
  fields.
- The evaluator's VTK handler writes a vector field to a single file.

### Bug fixes

- `merge_strategy="adjacent"` and custom `merge_idx` returned the solution
  patches in the wrong order.
- On coarse periodic meshes (for example `torus(n, 2, 4)` or
  `stellarator(n, 2, 4)`), different edges sharing both end points were glued
  together, producing wrong solutions.
- `normal()` pointed inward on half of the cubed-sphere faces, which broke
  `hodge` on the sphere.
- Complex right-hand sides or coefficients crashed when a merge had no shared
  edges.
- Hodge components of real fields were returned as complex arrays.
- Flux scalings of singular patches were misaligned on reversed shared edges.
- `parse_pdo` overwrote repeated terms (`{"lap": 1, "dxx": 2}`) instead of
  summing them.
- `quadwts(n, kind=1)` returned uniform weights instead of Fejér weights.
- `shifted_lobatto_nodes` (used by `family="lgl"` triangle nodes) failed with
  NumPy 2.5, where `np.linalg.eigvals` returns complex arrays. The nodes are now
  eigenvalues of the symmetric Jacobi matrix: real, exactly symmetric, and
  more accurate at high degree.
- The cow example defaulted to a machine-specific file path.

### Project

- Added GitHub Actions CI with tests on Python 3.10–3.13, linting, examples,
  and a docs build with warnings as errors.
- Added ruff configuration, a `py.typed` marker, and dynamic versioning
  (`pysurfacefun.__version__`).
- Added benchmarks.
- The test suite grew from 33 to 84 tests.
- New documentation pages on the solver, equations, and time stepping.

## 0.1.0

- Initial release: quadrilateral and triangular patch discretizations,
  HPS solves, level-set surfaces, evaluators, and notebooks.
