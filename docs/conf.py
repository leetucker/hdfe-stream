"""Sphinx configuration for the hdfe-stream documentation.

Build locally with

    pip install -e ".[docs]"
    sphinx-build -W --keep-going -b html docs docs/_build/html
"""

import hdfe_stream

project = "hdfe-stream"
author = "Lee Tucker"
copyright = "2026, Lee Tucker"
release = hdfe_stream.__version__
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
# fenced blocks without a language are output and tables, not code
highlight_language = "text"

# Markdown: ::: fences for directives, $...$ math, definition lists, and
# GitHub-style anchors on headings, so [text](#a-heading) links work as they
# do on GitHub.
myst_enable_extensions = ["colon_fence", "dollarmath", "deflist", "attrs_inline"]
myst_heading_anchors = 4

# API reference: the docstrings are numpydoc style. Types are given in the
# Parameters sections, so the signatures show names and defaults only.
autodoc_typehints = "none"
autodoc_member_order = "bysource"
autodoc_class_signature = "separated"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False
napoleon_use_ivar = False
# Attributes as a compact list, like Parameters, under its own heading
napoleon_custom_sections = [("Attributes", "params_style")]
# `name` in a docstring links to the object when it can, and is code either way
default_role = "py:obj"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "polars": ("https://docs.pola.rs/api/python/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

html_theme = "furo"
html_title = f"hdfe-stream {release}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "source_repository": "https://github.com/leetucker/hdfe-stream/",
    "source_branch": "main",
    "source_directory": "docs/",
}

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
