"""Sphinx configuration for grax."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

project = "grax"
author = "grax contributors"
copyright = "2026, grax contributors"
release = "0.1.1"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"

autosummary_generate = True
autodoc_typehints = "signature"
autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_ivar = True

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

todo_include_todos = False
bibtex_bibfiles = ["references.bib"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

html_theme = "furo"
html_static_path = ["_static"]
html_title = "grax documentation"

latex_engine = "pdflatex"
latex_elements = {
    # Include chapter, section, subsection, and subsubsection in LaTeX TOC.
    "preamble": r"""
\setcounter{tocdepth}{3}
""",
}
latex_documents = [
    ("index", "grax.tex", "grax Documentation", author, "manual"),
]
