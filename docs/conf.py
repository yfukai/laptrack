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
]
napoleon_google_docstring = False
autodoc_typehints = "description"
html_theme = "sphinx_rtd_theme"
