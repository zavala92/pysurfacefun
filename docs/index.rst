pysurfacefun
============

``pysurfacefun`` provides high-order discretizations and fast direct solvers for
partial differential equations on smooth surfaces. The package supports
quadrilateral Chebyshev patches and triangular high-order patches through one
public field/operator interface, with notebook examples for convergence studies
and time-dependent surface dynamics.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   quickstart
   equations
   timestepping
   solver
   examples
   api
   references

Highlights
----------

* High-order quadrilateral (Chebyshev) and triangular (PKD) surface patches
* Hierarchical Poincaré--Steklov fast direct solver with nested-dissection
  merge ordering and compiled, batched repeated solves
* Constant, variable, and complex coefficients; divergence-form equations such
  as ``-div(k*grad(u)) + u = f`` written as strings
* IMEX time stepping (SBDF1--4, CNAB1--2, RK111/222/443) for coupled
  reaction--diffusion systems, reusing one factorization per operator
* Level-set surfaces from coarse triangular or quadrilateral meshes, open
  surfaces with Dirichlet data, and degenerate patches
* Dependency-free VTU/VTP output for ParaView and evaluator output handlers
  for repeatable runs
