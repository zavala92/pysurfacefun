The Fast Direct Solver
======================

Elliptic surface problems

.. math::

   \sum_{c,d} a_{cd}\, D_c D_d u + \sum_c b_c\, D_c u + b_0\, u = f,
   \qquad D_c \in \{D_x, D_y, D_z\}\ \text{(tangential derivatives)},

are discretized by strong-form spectral collocation on curved high-order
patches and inverted with a hierarchical Poincaré--Steklov (HPS) fast direct
solver.  The coefficients may be constants, callables ``f(x, y, z)``, or
surface fields, real or complex.

Leaves
------

Each patch (a *leaf*) is discretized on its own nodes: an ``n x n`` Chebyshev
grid for quadrilaterals or ``n (n + 1) / 2`` recursive simplex nodes with a
Proriol--Koornwinder--Dubiner basis for triangles.  For every leaf the solver
forms dense operators that map

* Dirichlet data on the patch boundary skeleton to the solution in the patch
  (the *solution operator*),
* the same data to the outward flux on the skeleton (the *Dirichlet-to-Neumann
  map*), and
* the right-hand side at the interior nodes to the particular solution and its
  flux.

On quadrilaterals the collocation operator is assembled from the
tensor-product structure :math:`D_u = I \otimes D`, :math:`D_v = D \otimes I`:
only the interior rows are formed, in :math:`O(n^4)` work per patch rather than
the :math:`O(n^6)` of dense matrix products.  Patches with a degenerate
parametrization (a vanishing Jacobian, as at a collapsed corner) are handled
by multiplying the equations through by powers of the Jacobian.

Merging and nested dissection
-----------------------------

Neighboring leaves are merged by eliminating the unknowns on their shared
edges with a dense Schur complement.  The cost is dominated by the largest
interfaces, so the *order* of the merges matters.  The solver

1. identifies shared edges topologically, by snapping edge end points (and,
   where two edges share both end points, their midpoints) to integer ids;
2. builds the weighted patch adjacency graph; and
3. orders the merges by **nested dissection**: every set of patches is split in
   two at the median of the centroids along the principal axis, and the cut is
   refined by Fiduccia--Mattheyses passes that minimize the number of shared
   skeleton unknowns while keeping the halves balanced.

On a closed surface the final merge closes the surface.  For the pure
Laplace--Beltrami operator (``rankdef=True``) it adds a rank-one term that
removes the constant null space.

The factorization depends only on the operator, never on the right-hand side,
so it is computed once and reused for any number of solves:

.. code-block:: python

   L = psf.surfaceop(dom, {"lap": -1.0, "c": 4.0})  # assemble the leaves
   u1 = L.solve(f1)                  # factor once, then solve
   u2 = L.solve(f2)                  # reuse the factorization
   us = L.solve_many([f1, f2, f3])   # one sweep for three right-hand sides
   print(L.stats())                  # separator sizes and cost estimates

``merge_strategy`` selects the ordering: ``"nested_dissection"`` (default),
``"matching"`` (greedy bottom-up pairing of adjacent patches), or ``"natural"``
(consecutive patch indices, the ordering of earlier versions).  A custom tree
can be passed as ``merge_tree=((0, 1), (2, 3))`` or in the level-by-level
``merge_idx`` format; every ordering gives the same solution up to rounding.

Compiled solves
---------------

After factorization the merge tree is compiled into a flat representation: all
nodes of the same height and interface sizes are grouped, their dense blocks
are stacked, and the upward (particular solutions) and downward (interface
values) sweeps become a handful of gathers and batched matrix products per
tree level.  Complex right-hand sides with real operators are processed as
real arrays with twice the columns.  This makes repeated solves -- the inner
loop of implicit time stepping -- cheap.

Open surfaces
-------------

Surfaces with a boundary (for example a single patch, or a spherical cap built
with :func:`pysurfacefun.LevelSetSurface`) are solved with Dirichlet data on
the boundary:

.. code-block:: python

   u = psf.surfaceop(cap, {"lap": 1.0}, f).solve(bc=lambda x, y, z: x * y)

Performance
-----------

``benchmarks/benchmark_solver.py`` reproduces the following measurements
(4-core machine, OpenBLAS; ``solve`` is the time of one repeated solve):

.. list-table::
   :header-rows: 1
   :widths: 40 10 25 25

   * - mesh
     - patches
     - factor time (natural → ND)
     - solve time (0.1.0 → 0.2.0)
   * - cow (Rhino), n = 12
     - 339
     - 2.20 s → 0.15 s
     - 28 ms → 5.0 ms
   * - torus 16 x 32, n = 9
     - 512
     - 1.05 s → 0.17 s
     - 39 ms → 2.8 ms
   * - triangular level-set sphere, n = 9
     - 416
     - 0.28 s → 0.15 s
     - 29 ms → 1.7 ms
   * - Swiss-cheese level set, n = 10
     - 616
     - 0.36 s → 0.23 s
     - 29 ms → 3.0 ms
