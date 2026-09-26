Installation
============

``pysurfacefun`` requires Python 3.10 or newer and depends only on NumPy.
Install it in editable mode from a clone of the repository:

.. code-block:: bash

   git clone https://github.com/zavala92/pysurfacefun
   cd pysurfacefun
   python -m pip install -e .

Optional extras:

============  =================================================================
extra         enables
============  =================================================================
``plot``      Matplotlib plotting (``plot_surface``, ``wireframe``, ...)
``mat``       MAT-file mesh loading via SciPy/h5py (a NumPy-only reader for
              simple v5 files is built in)
``io``        ``meshio`` conversions (VTU/VTP output itself needs no extras)
``test``      the test suite
``docs``      this documentation
``dev``       everything above plus ``ruff`` and coverage
============  =================================================================

.. code-block:: bash

   python -m pip install -e ".[dev]"
   python -m pytest

Build the documentation
-----------------------

.. code-block:: bash

   python -m pip install -e ".[docs]"
   sphinx-build -b html docs docs/_build/html

The generated site will be available in ``docs/_build/html/index.html``.
