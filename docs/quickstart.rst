Quick Start
===========

Create a high-order quadrilateral sphere and check a Laplace--Beltrami identity:

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

Variable coefficients in divergence form (see :doc:`equations`):

.. code-block:: python

   k = psf.field(lambda x, y, z: 1 + 0.5*x, dom)
   problem = psf.SurfaceProblem(dom, variables="u", namespace={"k": k, "f": rhs})
   problem.add_equation("-div(k*grad(u)) + u = f")
   w = problem.solve()

Triangular patches use the same interface:

.. code-block:: python

   dom = psf.icosphere_tri(n=10, nref=1)
   u = psf.field(lambda x, y, z: x * y * z, dom)

   print(psf.integral(psf.field(1.0, dom)))
   print(psf.norm(psf.lap(u) + 12*u, "inf"))

Reuse a factorization for many right-hand sides:

.. code-block:: python

   L = psf.surfaceop(dom, {"lap": -1.0, "c": 4.0})
   u1 = L.solve(psf.field(lambda x, y, z: x, dom))
   u2 = L.solve(psf.field(lambda x, y, z: y * z, dom))

Coupled reaction--diffusion with a third-order IMEX scheme (see
:doc:`timestepping`):

.. code-block:: python

   ivp = psf.SurfaceIVP(dom, variables={"u": u0, "v": v0}, namespace={"Du": 1e-3, "Dv": 5e-3})
   ivp.add_equation("dt(u) - Du*lap(u) = u - u**3 - v")
   ivp.add_equation("dt(v) - Dv*lap(v) = 0.1*(u - v)")
   state = ivp.build_solver(dt=0.05, scheme="rk443").run(100)

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
