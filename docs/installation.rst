Installation
============

Install the package in editable mode from the Python package directory:

.. code-block:: bash

   cd python_surfacefun
   python -m pip install -e .

For notebooks, plotting, tests, mesh import, and ParaView export:

.. code-block:: bash

   python -m pip install -e ".[dev]"

The core package depends only on NumPy. Optional dependencies are used for
MAT-file mesh loading, VTU export, plotting, and tests.

Build the documentation
-----------------------

Install Sphinx and build the HTML documentation:

.. code-block:: bash

   python -m pip install sphinx
   sphinx-build -b html docs docs/_build/html

The generated site will be available in ``docs/_build/html/index.html``.
