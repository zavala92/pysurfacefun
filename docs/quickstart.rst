Quick Start
===========

Create a high-order quadrilateral sphere and check a Laplace-Beltrami identity:

.. code-block:: python

   import pysurfacefun as psf

   dom = psf.sphere(n=9, nref=1)
   u = psf.surfacefun(lambda x, y, z: x * y * z, dom)

   print(psf.surfacearea(dom))
   print(psf.norm(psf.lap(u) + 12*u, "inf"))

Solve a shifted surface Helmholtz problem:

.. code-block:: python

   alpha = 2.0
   rhs = (alpha - 12.0) * u

   L = psf.surfaceop(dom, {"lap": 1.0, "b": alpha}, rhs)
   uh = L.solve()

   relerr = psf.norm(uh - u, "inf") / psf.norm(u, "inf")
   print(relerr)

Triangular patches use the same style of interface:

.. code-block:: python

   dom = psf.icosphere_tri(n=10, nref=1)
   u = psf.tri_surfacefun(lambda x, y, z: x * y * z, dom)

   print(psf.tri_surfacearea(dom))
   print((psf.tri_lap(u) + 12*u).norm_inf())
