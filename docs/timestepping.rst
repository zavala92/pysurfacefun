Time Stepping
=============

:class:`pysurfacefun.SurfaceIVP` advances

.. math::

   \partial_t u = L u + N(u, t)

where :math:`L` is linear (treated implicitly with the fast direct solver) and
:math:`N` is an arbitrary coupling or reaction term (treated explicitly).  Every
implicit stage solves :math:`(I - \gamma\,\Delta t\,L)\,u = R` with a
scheme-dependent constant :math:`\gamma`, so each operator is factored once and
reused for the whole run.

Schemes
-------

=========  =====  ==============================================================
scheme     order  description
=========  =====  ==============================================================
``sbdf1``  1      IMEX Euler (the default; backward/forward Euler)
``sbdf2``  2      semi-implicit BDF2
``sbdf3``  3      semi-implicit BDF3
``sbdf4``  4      semi-implicit BDF4
``cnab1``  1      Crank--Nicolson / forward Euler
``cnab2``  2      Crank--Nicolson / Adams--Bashforth
``rk111``  1      IMEX Euler as a Runge--Kutta scheme
``rk222``  2      Ascher--Ruuth--Spiteri (2,2,2), L-stable implicit part
``rk443``  3      Ascher--Ruuth--Spiteri (4,4,3), L-stable implicit part
=========  =====  ==============================================================

The multistep ``sbdf`` schemes take their start-up steps with ``rk443`` so
that they reach their full order.  The Runge--Kutta schemes are stiffly
accurate and recover :math:`L U_j` of earlier stages from the stage solves
(exact at the collocation nodes), so no extra differentiation is needed.

All schemes reach their design order; ``tests/test_problems.py`` checks this
on quadrilateral and triangular spheres.  For the Swiss-cheese system of the
examples at :math:`\Delta t = 0.1`, the error at :math:`T = 5` is ``2e-1``
with ``sbdf1``, ``2e-2`` with ``sbdf2``, ``5e-4`` with ``sbdf4``, and ``2e-4``
with ``rk443``.

One field
---------

.. code-block:: python

   def reaction(u, t):
       return u - u**3

   ivp = psf.SurfaceIVP(u0, diffusion=1e-3, reaction=reaction)
   solver = ivp.build_solver(dt=0.05, scheme="rk443").build()
   u = solver.run(200)

``diffusion`` adds ``D lap(u)``, ``linear`` adds ``c u``, and ``operator``
accepts a coefficient dictionary (variable coefficients included).

Coupled fields
--------------

Equations use ``dt(<field>)`` on the left-hand side together with the
implicit linear terms; everything on the right-hand side is explicit:

.. code-block:: python

   ivp = psf.SurfaceIVP(dom, variables={"u": u0, "v": v0}, namespace=parameters)
   ivp.add_equation("dt(u) - Du*lap(u) = alpha*u*(1 - tau1*v**2) + v*(1 - tau2*u)")
   ivp.add_equation("dt(v) - Dv*lap(v) = beta*v + u*(gamma + tau2*v)")

   solver = ivp.build_solver(dt=0.1, scheme="sbdf4")
   state = solver.run(1000)        # {"u": ..., "v": ...}

Equivalently, pass callables:

.. code-block:: python

   ivp = psf.SurfaceIVP(
       dom,
       variables={"u": u0, "v": v0},
       diffusion={"u": Du, "v": Dv},
       reaction=lambda u, v, t: (f(u, v), g(u, v)),
   )

Fields whose linear operators coincide share one factorization and are solved
together.

Output during a run
-------------------

``solver.run(steps, evaluator=evaluator)`` calls an
:class:`pysurfacefun.Evaluator` after every step with a state dictionary that
contains every field, ``t``, ``iteration``, and the solver itself.  The VTK
handler writes one ``.vtu`` file per field (vector fields as a single
three-component array) without third-party packages.
