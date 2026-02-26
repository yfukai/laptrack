"""Sphinx configuration."""

from datetime import datetime


project = "LapTrack"
author = "Yohsuke T. Fukai"
copyright = f"{datetime.now().year}, {author}"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "sphinx_rtd_theme",
    "sphinxcontrib.autodoc_pydantic",
    "sphinx_gallery.load_style",
    "nbsphinx",
]
napoleon_google_docstring = False
napoleon_numpy_docstring = True
autodoc_typehints = "description"
html_theme = "sphinx_rtd_theme"
autodoc_pydantic_model_show_json = False
exclude_patterns = ["_build"]
