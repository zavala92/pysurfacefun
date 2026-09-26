Equations and Operators
=======================

Boundary-value problems
-----------------------

:class:`pysurfacefun.SurfaceProblem` (an alias of
:class:`pysurfacefun.SurfaceLBVP`) accepts equations as strings that are linear
in one unknown field.  Names in ``namespace`` are available inside the
equation:

.. code-block:: python

   dom = psf.sphere(n=12, nref=1)
   k = psf.field(lambda x, y, z: 1 + 0.5 * x, dom)       # variable coefficient
   f = psf.field(lambda x, y, z: x * y * z, dom)

   problem = psf.SurfaceProblem(dom, variables="u", namespace={"k": k, "f": f})
   problem.add_equation("-div(k*grad(u)) + u = f")
   u = problem.solve()

The same code works on triangular meshes.  Available operators:

=========================  ====================================================
``lap(u)``                 Laplace--Beltrami operator
``grad(u)``                surface gradient (a vector of three expressions)
``div(V)``                 surface divergence of ``grad(u)``, ``k*grad(u)``, ...
``dot(b, grad(u))``        advection by a vector field or constant 3-vector
``dx(u)``, ``dy(u)``,      tangential Cartesian derivatives (``diffx``, ...
``dz(u)``                  are aliases)
=========================  ====================================================

Coefficients may be numbers, surface fields, or callables ``f(x, y, z)``.
Derivatives of products follow the product rule, so
``div(k*grad(u))`` becomes ``k lap(u) + grad(k).grad(u)``; derivatives of
order higher than two are rejected.  Functions such as ``sin``, ``exp``,
``sqrt`` and ``np`` are available for data terms.

Operator dictionaries
---------------------

The solvers can also be built directly from coefficient dictionaries:

.. code-block:: python

   L = psf.surfaceop(dom, {"lap": -1.0, "dx": 0.3, "c": 2.0}, f)
   u = L.solve()

Recognized keys are ``dxx``, ``dxy``, ``dxz``, ``dyx``, ``dyy``, ``dyz``,
``dzx``, ``dzy``, ``dzz`` (``dxy`` multiplies :math:`D_x D_y u`), ``dx``,
``dy``, ``dz``, ``b``, and the shorthands ``lap``, ``grad``, and ``c``.
Repeated contributions are summed.  :func:`pysurfacefun.apply_operator`
applies such an operator to a field by spectral differentiation; at the
collocation nodes this is exactly the operator the direct solver inverts.

Fields
------

A scalar field stores its values in one contiguous array ``u.data`` of shape
``(P, n, n)`` (quadrilaterals) or ``(P, n (n + 1) / 2)`` (triangles);
``u.vals`` is the list of per-patch views.  Fields support arithmetic with
numbers, arrays, and other fields on the same mesh, NumPy ufuncs
(``np.sin(u)``, ``np.max(u)``), and ``u.real``, ``u.imag``, ``u.conj()``.
Integrals, norms (``1``, ``2``, ``p``, ``"inf"``, ``"H1"``, ``"lap"``),
gradients, divergences, and consistently oriented outward normals
(:func:`pysurfacefun.normal`) work on both mesh types.
