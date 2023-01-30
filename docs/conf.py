# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# set an environment variable for shapely.decorators.requires_geos to see if we
# are in a doc build
import importlib

import os

os.environ["SPHINX_DOC_BUILD"] = "1"

# -- Project information -----------------------------------------------------

project = 'Shapely'
copyright = '2011-2022, Sean Gillies and Shapely contributors'

# The full version, including alpha/beta/rc tags.
import shapely

release = shapely.__version__.split("+")[0]

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'matplotlib.sphinxext.plot_directive',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'numpydoc',
    'sphinx_remove_toctrees'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_css_files = [
    'custom.css',
]

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = False

# -- Options for LaTeX output --------------------------------------------------

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [
    ('index', 'Shapely.tex', 'Shapely Documentation',
     'Sean Gillies', 'manual'),
]

#  --Options for sphinx extensions -----------------------------------------------

# connect docs in other projects
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

plot_rcparams = {
    'savefig.bbox': "tight"
}

#  -- Automatic generation of API reference pages -----------------------------

numpydoc_show_class_members = False
autosummary_generate = True
remove_from_toctrees = ["reference/*"]


def rstjinja(app, docname, source):
    """
    Render our pages as a jinja template for fancy templating goodness.
    """
    # https://www.ericholscher.com/blog/2016/jul/25/integrating-jinja-rst-sphinx/
    # Make sure we're outputting HTML
    if app.builder.format != 'html':
        return
    source[0] = app.builder.templates.render_string(source[0], app.config.html_context)


def get_module_functions(module, exclude=None):
    """Return a list of function names for the given submodule."""
    mod = importlib.import_module(f"shapely.{module}")
    mod_functions = mod.__all__
    # mod_functions = [(f if f in shapely.__dict__ else f"{module}.{f}") for f in mod_functions]
    # assert mod_functions == mod.__all__, module
    return mod_functions


html_context = {
    'get_module_functions': get_module_functions
}

# write dummy _reference.rst with all functions listed to ensure the reference/
# stub pages are crated (the autogeneration of those stub pages by autosummary 
# happens before the jinja rendering is done, and thus at that point the
# autosummary directives do not yet contain the final content

template = """
:orphan:

.. autogenerated file

.. currentmodule:: shapely

.. autosummary::
   :toctree: reference/

"""

modules = [
    "_geometry", "creation", "constructive", "coordinates", "io", "linear",
    "measurement", "predicates", "set_operations"]

functions = [func for mod in modules for func in get_module_functions(mod)]

template += "   " + "\n   ".join(functions)

submodules = ["plotting"]

module_tempalte = """

.. currentmodule:: shapely.{module}

.. autosummary::
   :toctree: reference/

"""
for module in submodules:
    template += module_tempalte.format(module=module)
    functions = get_module_functions(module)
    template += "   " + "\n   ".join(functions)

with open("_reference.rst", "w") as f:
    f.write(template)


def setup(app):
    app.connect("source-read", rstjinja)
