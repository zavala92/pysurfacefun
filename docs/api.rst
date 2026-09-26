API Reference
=============

The public API is available from the top-level ``pysurfacefun`` namespace.
Functions dispatch on the mesh type, so the same code runs on quadrilateral
and triangular meshes.

Fields and operators
--------------------

.. autosummary::

   pysurfacefun.field
   pysurfacefun.vector_field
   pysurfacefun.lap
   pysurfacefun.grad
   pysurfacefun.div
   pysurfacefun.diff
   pysurfacefun.integral
   pysurfacefun.norm
   pysurfacefun.vector_norm
   pysurfacefun.dot
   pysurfacefun.cross
   pysurfacefun.normal
   pysurfacefun.hodge
   pysurfacefun.resample
   pysurfacefun.apply_operator

Surface geometries
------------------

.. autosummary::

   pysurfacefun.sphere
   pysurfacefun.torus
   pysurfacefun.stellarator
   pysurfacefun.from_rhino
   pysurfacefun.icosphere_tri
   pysurfacefun.LevelSetSurface
   pysurfacefun.levelset_surface_tri
   pysurfacefun.refine_surface_mesh
   pysurfacefun.surfacearea
   pysurfacefun.patch_orientation

Boundary-value and initial-value problems
-----------------------------------------

.. autosummary::

   pysurfacefun.SurfaceProblem
   pysurfacefun.SurfaceLBVP
   pysurfacefun.SurfaceLBVPSolver
   pysurfacefun.SurfaceIVP
   pysurfacefun.SurfaceIVPSolver
   pysurfacefun.get_scheme

Fast direct solver
------------------

.. autosummary::

   pysurfacefun.surfaceop
   pysurfacefun.SurfaceOp
   pysurfacefun.TriangleSurfaceOp
   pysurfacefun.HPSOperator
   pysurfacefun.HPSSolver
   pysurfacefun.LeafOperators
   pysurfacefun.nested_dissection_tree
   pysurfacefun.parse_pdo

Evaluation and output
---------------------

.. autosummary::

   pysurfacefun.Evaluator
   pysurfacefun.JSONLinesOutputHandler
   pysurfacefun.NPZOutputHandler
   pysurfacefun.VTKOutputHandler
   pysurfacefun.write_vtu
   pysurfacefun.write_vtu_fields
   pysurfacefun.write_mesh_vtu
   pysurfacefun.write_tri_vtu
   pysurfacefun.write_triangle_meshio

Lower-level APIs
----------------

These names remain public for users who need direct access to the
quadrilateral or triangular layers.

.. autosummary::

   pysurfacefun.SurfaceMesh
   pysurfacefun.SurfaceFunction
   pysurfacefun.SurfaceVectorFunction
   pysurfacefun.surfacefun
   pysurfacefun.surfacefunv
   pysurfacefun.TriangleSurfaceMesh
   pysurfacefun.TriangleSurfaceFunction
   pysurfacefun.TriangleSurfaceVectorFunction
   pysurfacefun.tri_surfacefun
   pysurfacefun.tri_surfaceop
   pysurfacefun.tri_lap
   pysurfacefun.tri_surfacearea

Modules
-------

.. automodule:: pysurfacefun.problems
   :members: SurfaceLBVP, SurfaceLBVPSolver, SurfaceIVP, SurfaceIVPSolver, SurfaceEquation

.. automodule:: pysurfacefun.timesteppers
   :members: get_scheme, SBDF, CNAB, IMEXRungeKutta

.. automodule:: pysurfacefun.operators
   :members: PDO, parse_pdo, HPSOperator

.. automodule:: pysurfacefun.hps
   :members: HPSSolver, LeafOperators, nested_dissection_tree, natural_tree, matching_tree, merge_patches, orientation_signs

.. automodule:: pysurfacefun.unified
   :members:

.. automodule:: pysurfacefun.vtk
   :members: write_vtu_fields, write_mesh_vtu, write_vtu, write_vtp

Package namespace
-----------------

.. automodule:: pysurfacefun
   :members:
   :undoc-members:
   :show-inheritance:
