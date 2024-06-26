# scores: Verification and Evaluation for Forecasts and Models

[![CodeQL](https://github.com/nci/scores/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/nci/scores/actions/workflows/github-code-scanning/codeql) [![Coverage Status](https://coveralls.io/repos/github/nci/scores/badge.svg)](https://coveralls.io/github/nci/scores) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/nci/scores/main?labpath=tutorials%2FTutorial_Gallery.ipynb)

> 
> **A list of over 50 metrics, statistical techniques and data processing tools contained in `scores` is [available here](https://scores.readthedocs.io/en/stable/included.html).**

`scores` is a Python package containing mathematical functions for the verification, evaluation and optimisation of forecasts, predictions or models. It supports labelled n-dimensional (multidimensional) data, which is used in many scientific fields and in machine learning. At present, `scores` primarily supports the geoscience communities; in particular, the meteorological, climatological and oceanographic communities.

Documentation is hosted at [scores.readthedocs.io](https://scores.readthedocs.io).  
Source code is hosted at [github.com/nci/scores](https://github.com/nci/scores).  
The tutorial gallery is hosted at [as part of the documentation, here](https://scores.readthedocs.io/en/stable/tutorials/Tutorial_Gallery.html). 

## Overview
Here is a **curated selection** of the metrics, tools and statistical tests included in `scores`:

|                       	| **Description** 	| **Selection of Included Functions** 	|
|-----------------------	|-----------------	|--------------	|
| **[Continuous](https://scores.readthedocs.io/en/stable/included.html#continuous)**        	|Scores for evaluating single-valued continuous forecasts.                  	|Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Additive Bias, Multiplicative Bias, Pearson's Correlation Coefficient, Flip-Flop Index, Quantile Loss, Murphy Score, families of consistent scoring functions for quantiles and expectiles.              	|
| **[Probability](https://scores.readthedocs.io/en/stable/included.html#probability)**       	|Scores for evaluating forecasts that are expressed as predictive distributions, ensembles, and probabilities of binary events.                 	|Brier Score, Continuous Ranked Probability Score (CRPS) for Cumulative Density Function (CDF), Threshold weighted CRPS for CDF, CRPS for ensembles, Receiver Operating Characteristic (ROC), Isotonic Regression (reliability diagrams).              	|
| **[Categorical](https://scores.readthedocs.io/en/stable/included.html#categorical)**       	|Scores (including contingency table metrics) for evaluating forecasts of categories.                	|Probability of Detection (POD), False Alarm Ratio (FAR), Probability of False Detection (POFD), Success Ratio, Accuracy, Peirce's Skill Score, Critical Success Index (CSI), Gilbert Skill Score, Heidke Skill Score, Odds Ratio, Odds Ratio Skill Score, F1 score, Symmetric Extremal Dependence Index, FIxed Risk Multicategorical (FIRM) Score.               	|
| **[Spatial](https://scores.readthedocs.io/en/stable/included.html#spatial)** 	|Scores that take into account spatial structure.                 	|Fractions Skill Score.              	|
| **[Statistical Tests](https://scores.readthedocs.io/en/stable/included.html#statistical-tests)** 	|Tools to conduct statistical tests and generate confidence intervals.                 	|Diebold Mariano.              	|
| **[Processing Tools](https://scores.readthedocs.io/en/stable/included.html#processing-tools-for-preparing-data)**        	|Tools to pre-process data.                 	|Data matching, Discretisation, Cumulative Density Function Manipulation.              	|


`scores` not only includes common scores (e.g. MAE, RMSE), it includes novel scores not commonly found elsewhere (e.g. FIRM, Flip-Flop Index), complex scores (e.g. threshold weighted CRPS), and statistical tests (such as the Diebold Mariano test). Additionally, it provides pre-processing tools for preparing data for scores in a variety of formats including cumulative distribution functions (CDF). `scores` provides its own implementations where relevant to avoid extensive dependencies.

`scores` primarily supports xarray datatypes for Earth system data allowing it to work with NetCDF4, HDF5, Zarr and GRIB data sources among others. `scores` uses Dask for scaling and performance. Some metrics work with pandas and we aim to expand this capability. 

All of the scores and metrics in this package have undergone a thorough scientific review. Every score has a companion Jupyter Notebook tutorial that demonstrates its use in practice.

## Contributing
To find out more about contributing, see our [Contributing Guide](https://scores.readthedocs.io/en/stable/contributing.html).

All interactions in discussions, issues, emails and code (e.g. pull requests, code comments) will be managed according to the expectations outlined in the [ code of conduct ](https://github.com/nci/scores/blob/main/CODE_OF_CONDUCT.md) and in accordance with all relevant laws and obligations. This project is an inclusive, respectful and open project with high standards for respectful behaviour and language. The code of conduct is the Contributor Covenant, adopted by over 40,000 open source projects. Any concerns will be dealt with fairly and respectfully, with the processes described in the code of conduct.

## Using This Package

The [installation guide](https://scores.readthedocs.io/en/stable/installation.html) describes four different use cases for installing, using and working with this package.

Most users currently want the *all* installation option. This includes the mathematical functions (scores, metrics, statistical tests etc.), the tutorial notebooks and development libraries.

From a Local Checkout of the Git Repository

```bash
> pip install -e .[all]
```

Here is a short example of the use of scores:

```py
> import scores
> forecast = scores.sample_data.simple_forecast()
> observed = scores.sample_data.simple_observations()
> mean_absolute_error = scores.continuous.mae(forecast, observed)
> print(mean_absolute_error)
<xarray.DataArray ()>
array(2.)
```

To install the mathematical functions ONLY (no tutorial notebooks, no developer libraries), use the *minimal* installation option. *minimal* is a stable version with limited dependencies and can be installed from the Python Package Index.

```bash
> pip install scores
```

## Finding, Downloading and Working With Data

All metrics, statistical techniques and data processing tools in `scores` work with [xarray](https://xarray.dev). [Some metrics](https://scores.readthedocs.io/en/stable/included.html#pandas) work with [pandas](https://pandas.pydata.org/). As such, `scores` works with any data source for which xarray or pandas can be used. See the [Data Sources](https://scores.readthedocs.io/en/stable/data.html) page and [this tutorial](https://scores.readthedocs.io/en/stable/tutorials/First_Data_Fetching.html) for more information on finding, downloading and working with different sources of data.

## Acknowledging This Work

If you use `scores` for a published work, we would appreciate you citing our arXiv preprint:

Leeuwenburg, T., Loveday, N., Ebert, E. E., Cook, H., Khanarmuei, M., Taggart, R. J., Ramanathan, N., 
Carroll, M., Chong, S., Griffiths, A., & Sharples, J. (2024). *scores: A Python package for verifying and 
evaluating models and predictions with xarray and pandas*. arXiv. [https://doi.org/10.48550/arXiv.2406.07817](https://doi.org/10.48550/arXiv.2406.07817)

