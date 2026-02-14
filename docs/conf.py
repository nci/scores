# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
from scores import __version__

project = "scores"
copyright = "Licensed under Apache 2.0 - https://www.apache.org/licenses/LICENSE-2.0"
release = "2.5.0"

version = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "nbsphinx",
    "sphinx_gallery.load_style",
]
source_suffix = [".rst", ".md"]

templates_path = ["_templates"]
exclude_patterns = [
    "**/.ipynb_checkpoints",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ["_static"]
html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/nci/scores",
    "use_repository_button": True,
    "show_toc_level": 3,
}

html_baseurl = "https://scores.readthedocs.io/en/stable/"
autodoc_typehints = "description"

nbsphinx_thumbnails = {
    "tutorials/Risk_Matrix_Score": "_images/risk_matrix.png",
    "tutorials/Block_Bootstrapping": "_images/block_bootstrapping_thumbnail.png",
    "tutorials/Isotonic_Regression_And_Reliability_Diagrams": "_images/isotonic_regression_thumbnail.png",
    "tutorials/Binary_Contingency_Scores": "_images/binary_contingency_thumbnail.png",
    "tutorials/Dimension_Handling": "_images/dimension_handling_thumbnail.png",
    "tutorials/Angular_data": "_images/angular_data_thumbnail.png",
    "tutorials/Q-Q_plots": "_images/qq_thumbnail.png",
    "tutorials/NSE": "_images/NSE_thumbnail.png",
    "tutorials/Quantile_Loss": "_images/quantile_loss_thumbnail.png",
    "tutorials/Threshold_Weighted_Scores": "_images/threshold_weighted_scores_thumbnail.png",
    "tutorials/Threshold_Weighted_CRPS_for_Ensembles": "_images/twCRPS_for_ensembles_thumbnail.png",
    "tutorials/Brier_Score": "_images/brier_thumbnail.png",
    "tutorials/Quantile_Interval_And_Interval_Score": "_images/quantile_interval_score_and_interval_score_thumbnail.png",
    "tutorials/Fractions_Skill_Score": "_images/fss_thumbnail.png",
    "tutorials/Diebold_Mariano_Test_Statistic": "_images/diebold_mariano_thumbnail.png",
}

# This is needed to allow linking into auto-generated API documentation
# It means there is a risk that genuine cross-referencing errors will be
# suppressed. Perhaps some way around this could be found in future.
# Uncomment this during testing to reveal potential errors
suppress_warnings = ["myst.xref_missing"]

# -- nbsphinx ---------------------------------------------------------------
# This is processed by Jinja2 and inserted after each notebook
nbsphinx_prolog = r"""
{% set docname = '' + env.doc2path(env.docname, base=False)|string() %}

.. raw:: html

    <div class="admonition note">
      Interactive online version:
      <span style="white-space: nowrap;"><a href="https://mybinder.org/v2/gh/nci/scores/main?labpath={{ docname|e }}"><img alt="Binder badge" src="https://mybinder.org/badge_logo.svg" style="vertical-align:text-bottom"></a>.</span>
      <a href="{{ env.docname.split('/')|last|e + '.ipynb' }}" class="reference download internal" download>Download notebook</a>.
    </div>

.. raw:: latex

    \nbsphinxstartnotebook{\scriptsize\noindent\strut
    \textcolor{gray}{The following section was generated from
    \sphinxcode{\sphinxupquote{\strut {{ docname | escape_latex }}}} \dotfill}}
"""

# This is processed by Jinja2 and inserted after each notebook
nbsphinx_epilog = r"""
{% set docname = 'doc/' + env.doc2path(env.docname, base=None)|string() %}
.. raw:: latex

    \nbsphinxstopnotebook{\scriptsize\noindent\strut
    \textcolor{gray}{\dotfill\ \sphinxcode{\sphinxupquote{\strut
    {{ docname | escape_latex }}}} ends here.}}
"""
