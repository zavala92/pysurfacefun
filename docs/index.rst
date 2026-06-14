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
   examples
   api
   references

Highlights
----------

* High-order quadrilateral and triangular surface patches
* Unified ``field``, ``lap``, ``grad``, ``integral``, and ``norm`` dispatch
* ``SurfaceProblem`` and ``SurfaceLBVP`` for elliptic solves
* ``SurfaceIVP`` for implicit reaction-diffusion stepping
* Evaluator output handlers for repeatable examples and notebooks
* VTU/VTP output for ParaView
