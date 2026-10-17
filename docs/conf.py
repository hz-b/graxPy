"""Sphinx configuration for GraxPy."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

project = "GraxPy"
author = "GraxPy contributors"
copyright = "2026, GraxPy contributors"

try:
    release = _pkg_version("graxpy")
except PackageNotFoundError:
    release = "unknown"

version = ".".join(release.split(".")[:2]) if release != "unknown" else "unknown"

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
    "substitution",
]

myst_substitutions = {"release": release}

todo_include_todos = False
bibtex_bibfiles = ["references.bib"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

html_theme = "furo"
html_static_path = ["_static"] if (Path(__file__).resolve().parent / "_static").is_dir() else []
html_js_files = ["site-notice.js"]
html_css_files = ["site-notice.css"]
html_title = "GraxPy documentation"
html_context = {
    "graxpy_release": release,
    "graxpy_contact_name": "Simone Vadilonga",
    "graxpy_contact_email": "simone.vadilonga@helmholtz-berlin.de",
}
html_logo = "_static/GraxPy_logo.svg"
html_favicon = "_static/GraxPy_logo.svg"
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#184e8b",
        "color-brand-content": "#184e8b",
        "color-brand-visited": "#093566",
        "color-link": "#184e8b",
        "color-link--hover": "#093566",
        "color-background-primary": "#ffffff",
        "color-background-secondary": "#f6f8fa",
        "color-background-hover": "#f3f7dc",
        "color-header-background": "#ffffff",
        "color-header-border": "#bcd233",
        "color-admonition-title--note": "#184e8b",
        "color-admonition-title-background--note": "#bcd233",
    },
    "dark_css_variables": {
        "color-brand-primary": "#bcd233",
        "color-brand-content": "#87baff",
        "color-brand-visited": "#6cabe9",
        "color-link": "#87baff",
        "color-link--hover": "#bcd233",
        "color-header-border": "#bcd233",
    },
}
templates_path = ["_templates"]

latex_engine = "pdflatex"
latex_elements = {
    # Include chapter, section, subsection, and subsubsection in LaTeX TOC.
    "preamble": r"""
\setcounter{tocdepth}{3}
""",
}
latex_documents = [
    ("index", "GraxPy.tex", "GraxPy Documentation", author, "manual"),
]
