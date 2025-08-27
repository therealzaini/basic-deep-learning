# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Basic Deep Learning'
copyright = '2025, Diaa Eddine ZAINI'
author = 'Diaa Eddine ZAINI'
release = '0.1.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.mathjax']

templates_path = ['_templates']
exclude_patterns = []

mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"


mathjax_config = {
    'tex': {
        'packages': {'[+]': ['base', 'ams', 'boldsymbol', 'euler']},
        'inlineMath': [['$', '$'], ['\\(', '\\)']],
        'displayMath': [['$$', '$$'], ['\\[', '\\]']],
        'tags': 'ams'
    },
    'loader': {
        'load': ['[tex]/ams', '[tex]/boldsymbol', '[tex]/euler']
    },
    'options': {
        'renderActions': {
            'addMenu': [0, '', '']
        }
    }
}


html_theme_options = {
    'mathjax_config': mathjax_config
}

language = 'English'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
