# scores: Verification and Evaluation for Forecasts and Models

[![DOI](https://joss.theoj.org/papers/10.21105/joss.06889/status.svg)](https://doi.org/10.21105/joss.06889) [![CodeQL](https://github.com/nci/scores/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/nci/scores/actions/workflows/github-code-scanning/codeql) [![Coverage Status](https://coveralls.io/repos/github/nci/scores/badge.svg)](https://coveralls.io/github/nci/scores) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/nci/scores/main?labpath=tutorials%2FTutorial_Gallery.ipynb) [![PyPI Version](https://img.shields.io/pypi/v/scores.svg)](https://pypi.org/project/scores/) [![Conda Version](https://img.shields.io/conda/vn/conda-forge/scores.svg)](https://anaconda.org/conda-forge/scores)

> 
> **A list of over 60 metrics, statistical techniques and data processing tools contained in `scores` is [available here](https://scores.readthedocs.io/en/stable/included.html).**

`scores` is a Python package containing mathematical functions for the verification, evaluation and optimisation of forecasts, predictions or models. It supports labelled n-dimensional (multidimensional) data, which is used in many scientific fields and in machine learning. At present, `scores` primarily supports the geoscience communities; in particular, the meteorological, climatological and oceanographic communities.

Documentation: [scores.readthedocs.io](https://scores.readthedocs.io)  
Source code: [github.com/nci/scores](https://github.com/nci/scores)  
Tutorial gallery: [available here](https://scores.readthedocs.io/en/stable/tutorials/Tutorial_Gallery.html)  
Journal article: [*scores: A Python package for verifying and evaluating models and predictions with xarray*](https://doi.org/10.21105/joss.06889)

## Overview

Below is a **curated selection** of the metrics, tools and statistical tests included in `scores`. [(Click here for the full list.)](https://scores.readthedocs.io/en/stable/included.html)

|                       	| **Description** 	| **Selection of Included Functions** 	|
|-----------------------	|-----------------	|--------------	|
| **[Continuous](https://scores.readthedocs.io/en/stable/included.html#continuous)**        	|Scores for evaluating single-valued continuous forecasts.                  	|MAE, MSE, RMSE, Additive Bias, Multiplicative Bias, Percent Bias, Pearson's Correlation Coefficient, Kling-Gupta Efficiency, Flip-Flop Index, Quantile Loss, Quantile Interval Score, Interval Score, Murphy Score, and threshold weighted scores for expectiles, quantiles and Huber Loss.             	|
| **[Probability](https://scores.readthedocs.io/en/stable/included.html#probability)**        |Scores for evaluating forecasts that are expressed as predictive distributions, ensembles, and probabilities of binary events.                   |Brier Score, Continuous Ranked Probability Score (CRPS) for Cumulative Density Functions (CDF) and ensembles (including threshold weighted versions), Receiver Operating Characteristic (ROC), Isotonic Regression (reliability diagrams).               |
| **[Categorical](https://scores.readthedocs.io/en/stable/included.html#categorical)**       	|Scores for evaluating forecasts of categories.                	|17 binary contingency table (confusion matrix) metrics and the FIxed Risk Multicategorical (FIRM) Score.               	|
| **[Spatial](https://scores.readthedocs.io/en/stable/included.html#spatial)** 	|Scores that take into account spatial structure.                 	|Fractions Skill Score.              	|
| **[Statistical Tests](https://scores.readthedocs.io/en/stable/included.html#statistical-tests)** 	|Tools to conduct statistical tests and generate confidence intervals.                 	|Diebold Mariano.              	|
| **[Processing Tools](https://scores.readthedocs.io/en/stable/included.html#processing-tools-for-preparing-data)**        	|Tools to pre-process data.                 	|Data matching, Discretisation, Cumulative Density Function Manipulation.              	|


`scores` not only includes common scores (e.g., MAE, RMSE), it also includes novel scores not commonly found elsewhere (e.g., FIRM, Flip-Flop Index), complex scores (e.g., threshold weighted CRPS), and statistical tests (e.g., the Diebold Mariano test). Additionally, it provides pre-processing tools for preparing data for scores in a variety of formats including cumulative distribution functions (CDF). `scores` provides its own implementations where relevant to avoid extensive dependencies.

`scores` primarily supports xarray datatypes for Earth system data allowing it to work with NetCDF4, HDF5, Zarr and GRIB data formats among others. `scores` uses Dask for scaling and performance. Some metrics work with pandas and we aim to expand this capability. 

All of the scores and metrics in this package have undergone a thorough scientific and software review. Every score has a companion Jupyter Notebook tutorial that demonstrates its use in practice.

## Contributing
To find out more about contributing, see our [contributing guide](https://scores.readthedocs.io/en/stable/contributing.html).

All interactions in discussions, issues, emails and code (e.g., pull requests, code comments) will be managed according to the expectations outlined in the [ code of conduct ](https://github.com/nci/scores/blob/main/CODE_OF_CONDUCT.md) and in accordance with all relevant laws and obligations. This project is an inclusive, respectful and open project with high standards for respectful behaviour and language. The code of conduct is the Contributor Covenant, adopted by over 40,000 open source projects. Any concerns will be dealt with fairly and respectfully, with the processes described in the code of conduct.

## Installation

The [installation guide](https://scores.readthedocs.io/en/stable/installation.html) describes four different use cases for installing, using and working with this package.

**Most users currently want the *all* installation option.** This includes the mathematical functions (scores, metrics, statistical tests etc.), the tutorial dependencies and development libraries.

```bash
# From a local checkout of the Git repository
pip install -e .[all]
```
**To install the mathematical functions ONLY** (no tutorial dependencies, no developer libraries), use the default *minimal* installation option. *minimal* is a stable version with limited dependencies. This can be installed from the [Python Package Index (PyPI)](https://pypi.org/project/scores/) or with [conda](https://anaconda.org/conda-forge/scores).

```bash
# From PyPI
pip install scores
```
```bash
# From conda-forge
conda install conda-forge::scores
```
(Note: at present, only the *minimal* installation option is available from conda. In time, we intend to add more installation options to conda.)

## Using `scores`

Here is a short example of the use of `scores`:

```py
> import scores
> forecast = scores.sample_data.simple_forecast()
> observed = scores.sample_data.simple_observations()
> mean_absolute_error = scores.continuous.mae(forecast, observed)
> print(mean_absolute_error)
<xarray.DataArray ()>
array(2.)
```
[Jupyter Notebook tutorials](https://scores.readthedocs.io/en/stable/tutorials/Tutorial_Gallery.html) are provided for each metric and statistical test in `scores`, as well as for some of the key features of `scores` (e.g., [dimension handling](https://scores.readthedocs.io/en/stable/tutorials/Dimension_Handling.html) and [weighting results](https://scores.readthedocs.io/en/stable/tutorials/Weighting_Results.html)). 

## Finding, Downloading and Working With Data

All metrics, statistical techniques and data processing tools in `scores` work with [xarray](https://xarray.dev). [Some metrics](https://scores.readthedocs.io/en/stable/included.html#pandas) work with [pandas](https://pandas.pydata.org/). As such, `scores` works with any data source for which xarray or pandas can be used. See the [data sources](https://scores.readthedocs.io/en/stable/data.html) page and [this tutorial](https://scores.readthedocs.io/en/stable/tutorials/First_Data_Fetching.html) for more information on finding, downloading and working with different sources of data.

## Acknowledging or Citing `scores`

If you use `scores` for a published work, we would appreciate you citing our [paper](https://doi.org/10.21105/joss.06889):

Leeuwenburg, T., Loveday, N., Ebert, E. E., Cook, H., Khanarmuei, M., Taggart, R. J., Ramanathan, N., Carroll, M., Chong, S., Griffiths, A., & Sharples, J. (2024). scores: A Python package for verifying and evaluating models and predictions with xarray. *Journal of Open Source Software, 9*(99), 6889. [https://doi.org/10.21105/joss.06889](https://doi.org/10.21105/joss.06889)

BibTeX:
```
@article{Leeuwenburg_scores_A_Python_2024,
author = {Leeuwenburg, Tennessee and Loveday, Nicholas and Ebert, Elizabeth E. and Cook, Harrison and Khanarmuei, Mohammadreza and Taggart, Robert J. and Ramanathan, Nikeeth and Carroll, Maree and Chong, Stephanie and Griffiths, Aidan and Sharples, John},
doi = {10.21105/joss.06889},
journal = {Journal of Open Source Software},
month = jul,
number = {99},
pages = {6889},
title = {{scores: A Python package for verifying and evaluating models and predictions with xarray}},
url = {https://joss.theoj.org/papers/10.21105/joss.06889},
volume = {9},
year = {2024}
}
```
