"""Sphinx configuration for the pysurfacefun documentation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pysurfacefun

project = "pysurfacefun"
author = "Gentian Zavalani"
copyright = "2026, Gentian Zavalani"
release = pysurfacefun.__version__
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_format = "short"
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_ivar = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = "pysurfacefun"
html_theme_options = {
    "description": "High-order solvers for surface PDEs",
    "fixed_sidebar": True,
    "page_width": "1180px",
    "sidebar_width": "320px",
    "show_powered_by": False,
}
