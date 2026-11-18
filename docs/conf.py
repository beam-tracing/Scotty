# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

from importlib.metadata import PackageNotFoundError, version as _version

project = "Scotty"
copyright = "2018 - 2026, Valerian Hall-Chen and the Scotty contributors"
author = "Valerian Hall-Chen"

# The full version, including alpha/beta/rc tags, from the installed package
# (Read the Docs installs it, see .readthedocs.yaml)
try:
    release = _version("scotty-beam-tracing")
except PackageNotFoundError:
    release = "unknown"
version = ".".join(release.split(".")[:2])


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
]

# Add any paths that contain templates here, relative to this directory.
# (There are none: a missing folder here makes Sphinx warn)
templates_path = []

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The default role for text marked up `like this`: a link to the Python object
# of that name if there is one, else code, without a warning (with "any", every
# name that is not an object, such as an argument, gave a warning)
default_role = "py:obj"

# Tell sphinx what the primary language being documented is.
primary_domain = "py"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

html_context = {
    "github_user": "beam-tracing",
    "github_repo": "Scotty",
    "github_version": "master",
    "doc_path": "docs",
}

# Theme options are theme-specific and customise the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = dict(
    # analytics_id=''  this is configured in rtfd.io
    # canonical_url="",
    repository_url="https://github.com/beam-tracing/Scotty",
    repository_branch="master",
    path_to_docs="docs",
    use_edit_page_button=True,
    use_repository_button=True,
    use_issues_button=True,
    home_page_in_toc=False,
    extra_navbar="",
    navbar_footer_text="",
)

# -- Extension configuration -------------------------------------------------

autodoc_typehints = "description"
autodoc_type_aliases = {
    "ArrayLike": "numpy.typing.ArrayLike",
    "ProfileFitLike": "scotty.profile_fit.ProfileFitLike",
}

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}
