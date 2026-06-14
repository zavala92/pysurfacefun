API Overview
============

The public API is available from the top-level ``pysurfacefun`` namespace.

Unified fields and operators
----------------------------

.. autosummary::

   pysurfacefun.SurfaceField
   pysurfacefun.SurfaceVectorField
   pysurfacefun.field
   pysurfacefun.vector_field
   pysurfacefun.lap
   pysurfacefun.grad
   pysurfacefun.integral
   pysurfacefun.norm
   pysurfacefun.vector_norm
   pysurfacefun.resample

Surface geometries
------------------

.. autosummary::

   pysurfacefun.sphere
   pysurfacefun.torus
   pysurfacefun.stellarator
   pysurfacefun.icosphere_tri
   pysurfacefun.LevelSetSurface
   pysurfacefun.levelset_surface_tri
   pysurfacefun.normal
   pysurfacefun.surfacearea

Boundary-value and initial-value problems
-----------------------------------------

.. autosummary::

   pysurfacefun.SurfaceProblem
   pysurfacefun.SurfaceLBVP
   pysurfacefun.SurfaceLBVPSolver
   pysurfacefun.SurfaceIVP
   pysurfacefun.SurfaceIVPSolver

Evaluation and output
---------------------

.. autosummary::

   pysurfacefun.Evaluator
   pysurfacefun.JSONLinesOutputHandler
   pysurfacefun.NPZOutputHandler
   pysurfacefun.VTKOutputHandler
   pysurfacefun.write_vtu
   pysurfacefun.write_tri_vtu
   pysurfacefun.write_triangle_meshio

Compatibility and lower-level APIs
----------------------------------

These names remain public for users who need direct access to the older
quadrilateral or triangular solver layers.

.. autosummary::

   pysurfacefun.SurfaceFunction
   pysurfacefun.SurfaceVectorFunction
   pysurfacefun.surfacefun
   pysurfacefun.surfacefunv
   pysurfacefun.surfaceop
   pysurfacefun.TriangleSurfaceFunction
   pysurfacefun.TriangleSurfaceVectorFunction
   pysurfacefun.tri_surfacefun
   pysurfacefun.tri_surfaceop
   pysurfacefun.tri_lap
   pysurfacefun.tri_surfacearea

Module reference
----------------

.. automodule:: pysurfacefun
   :members:
   :undoc-members:
   :show-inheritance:
