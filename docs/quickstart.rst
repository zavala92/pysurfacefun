Quick Start
===========

Create a high-order quadrilateral sphere and check a Laplace-Beltrami identity:

.. code-block:: python

   import pysurfacefun as psf

   dom = psf.sphere(n=9, nref=1)
   u = psf.field(lambda x, y, z: x * y * z, dom)

   print(psf.surfacearea(dom))
   print(psf.norm(psf.lap(u) + 12*u, "inf"))

Solve a shifted surface Helmholtz problem:

.. code-block:: python

   alpha = 2.0
   rhs = (alpha - 12.0) * u

   problem = psf.SurfaceProblem(dom, variables="u", namespace={"alpha": alpha, "rhs": rhs})
   problem.add_equation("lap(u) + alpha*u = rhs")
   uh = problem.solve()

   relerr = psf.norm(uh - u, "inf") / psf.norm(u, "inf")
   print(relerr)

Triangular patches use the same style of interface:

.. code-block:: python

   dom = psf.icosphere_tri(n=10, nref=1)
   u = psf.field(lambda x, y, z: x * y * z, dom)

   print(psf.integral(psf.field(1.0, dom)))
   print(psf.norm(psf.lap(u) + 12*u, "inf"))

Repeatable output for examples and notebooks:

.. code-block:: python

   evaluator = psf.Evaluator(
       "notebook_outputs",
       prefix="sphere_demo",
       handlers=[
           psf.JSONLinesOutputHandler(),
           psf.NPZOutputHandler(),
           psf.VTKOutputHandler(nvis=16),
       ],
   )
   evaluator.add_task("linf_residual", lambda state: psf.norm(psf.lap(state["u"]) + 12*state["u"], "inf"))
   evaluator.add_task("u", lambda state: state["u"], every=10)
   evaluator.evaluate(0, state={"u": u})

Implicit reaction-diffusion time stepping:

.. code-block:: python

   def reaction(u, t):
       return u - 2.0*u**3

   solver = psf.SurfaceIVP(u, diffusion=1e-3, reaction=reaction).build_solver(dt=0.05).build()
   u = solver.run(20)
