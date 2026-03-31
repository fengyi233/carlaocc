# Configuration file for the Sphinx documentation builder.
import os
import sys

# Add project root to Python path so autodoc can find modules
sys.path.insert(0, os.path.abspath("../.."))

# -- Project information

project = 'CarlaOcc'
copyright = '2025, MIAS Group'
author = 'Yi Feng'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']
html_static_path = ['_static']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

html_context = {
    "display_github": True,
    "github_user": "fengyi233",
    "github_repo": "carlaocc-tutorial",
    "github_version": "main",
    "conf_py_path": "/docs/source/",
    "edit_link": True
}

html_theme_options = {
    # "display_version": True,
    "vcs_pageview_mode": "edit",
}