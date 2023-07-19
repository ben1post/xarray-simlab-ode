# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------

project = u"xarray-simlab-ode"
copyright = u"2022, Benjamin Post"
author = u"Benjamin Post"

# -- General configuration ---------------------------------------------------

import sys
import os

sys.path.insert(0, os.path.abspath('../src'))

print(f"python exec: {sys.executable}")
print(f"sys.path: {sys.path}")
try:
    import numpy
    print(f"numpy: {numpy.__version__}, {numpy.__file__}")
except ImportError:
    print("no numpy")
try:
    import attr
    print(f"attr: {attr.__version__}, {attr.__file__}")
except ImportError:
    print("no attr")
try:
    import xarray
    print(f"xarray: {xarray.__version__}, {xarray.__file__}")
except ImportError:
    print("no xarray")
try:
    import dask
    print(f"dask: {dask.__version__}, {dask.__file__}")
except ImportError:
    print("no dask")
try:
    import zarr
    print(f"zarr: {zarr.__version__}, {zarr.__file__}")
except ImportError:
    print("no zarr")
try:
    import xsimlab
    print(f"xsimlab: {xsimlab.__version__}, {xsimlab.__file__}")
except ImportError:
    print("no xsimlab")
try:
    import xso
    print(f"xso: {xso.__version__}, {xso.__file__}")
except ImportError:
    print("no xso")

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    "autoapi.extension",
    "myst_nb",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    'sphinx_toolbox.decorators',
]
autoapi_dirs = ["../src"]

autosummary_generate = True

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# for intersphinx

intersphinx_mapping = {
    "xsimlab": ("https://xarray-simlab.readthedocs.io/en/stable/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
}

# We recommend adding the following config value.
# Sphinx defaults to automatically resolve *unresolved* labels using all your Intersphinx mappings.
# This behavior has unintended side-effects, namely that documentations local references can
# suddenly resolve to an external location.
# See also:
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html#confval-intersphinx_disabled_reftypes
intersphinx_disabled_reftypes = ["*"]

