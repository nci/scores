# About the Scores Project!

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/nci/scores/main?labpath=tutorials%2FExplanation.ipynb)

> 
> **A list of over 50 metrics, statistical techniques and data processing tools contained in `scores` is [available here](https://scores.readthedocs.io/en/latest/included.html).**


One-line intro: xarray based verification scores and tools that can scale with Dask if needed. Pandas supported where possible.
Why use it: trusted implementations, novel metrics, performance, one-stop-shop.

Here is a **curated selection** of the metrics and tools included in `scores`:

| Continuous                      | Probability | Categorical      | Statistical Tests |
| ----------                      | ----------- | -----------      | ----------------- |
| MAE, MSE, RMSE, Additive Bias, Multiplicative Bias, Flip-Flop Index, Quantile Score, Murphy score, Pearson's Correlation Coefficient  | Brier Score, CRPS for CDF, CRPS for ensemble, ROC, Isotonic Regression (reliability diagrams)   | Hit Rate, Probability of False Detection, Success Ratio, Peirce's Skill Score, FIRM |  Diebold Mariano (with the Harvey et al. 1997 and the Hering and Genton 2011 modifications) |

**Notice -- This repository is currently undergoing initial construction and maintenance. It is getting much closer to our goals for version one, but there are a few more things to add. This notice will be removed after the first feature release. In the meantime, please feel free to look around, and don't hesitate to get in touch with any questions (see the contributing guide for how).**

Documentation is hosted at [scores.readthedocs.io](https://scores.readthedocs.io).  
Source code is hosted at [github.com/nci/scores](https://github.com/nci/scores).  
The tutorial gallery is hosted at [as part of the documentation, here](https://scores.readthedocs.io/en/latest/tutorials/Explanation.html).

`scores` is a Python package containing mathematical functions for the verification, evaluation, and optimisation of forecasts, predictions or models. It primarily supports the geoscience and Earth system science communities. It also has wide potential application in machine learning, and in other domains.

`scores` includes novel scores not commonly found elsewhere (e.g. FIRM, Flip-Flop Index), complex scores (e.g. threshold weighted CRPS), more common scores (e.g. MAE, RMSE) and statistical tests (such as the Diebold Mariano test). Additionally, it provides pre-processing tools for preparing data for scores in a variety of formats including cumulative distribution functions (CDF). `scores` provides its own implementations where relevant to avoid extensive dependencies.

`scores` is focused on supporting xarray datatypes for Earth system data. It also aims to be compatible with pandas, geopandas, pangeo and work with NetCDF4, hdf5, Zarr and GRIB data sources among others. `scores` uses Dask for scaling and performance.

All of the scores and metrics in this package have undergone a thorough scientific review. Every score has a companion Jupyter Notebook tutorial that demonstrates its use in practice.

## Contributing
To find out more about contributing, see our [contributor's guide](https://github.com/nci/scores/blob/main/docs/contributing.md).

All interactions in discussions, issues, emails and code (e.g. merge requests, code comments) will be managed according to the expectations outlined in the [ code of conduct ](https://github.com/nci/scores/blob/main/CODE_OF_CONDUCT.md) and in accordance with all relevant laws and obligations. This project is an inclusive, respectful and open project with high standards for respectful behaviour and language. The code of conduct is the Contributor Covenant, adopted by over 40, 000 open source projects. Any concerns will be dealt with fairly and respectfully, with the processes described in the code of conduct.

## Using This Package

The [installation guide](https://scores.readthedocs.io/en/latest/installation.html) describes four different use cases for installing, using and working with this package.

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

## Related Packages

There are similar packages which should be acknowledged, in particular [xskillscore](https://xskillscore.readthedocs.io/en/stable/) and [climpred](https://github.com/pangeo-data/climpred). These packages both provide overlapping and similar functionality. `scores` provides an additional option to the community and has additional metrics which are not found in those other packages. `scores` seeks to be self-contained with few dependencies, and so re-implements various metrics which are found in other libraries, so that it can be a simple one-stop-shop for the metrics of interest to its userbase.

## Finding and Downloading Data

Other than very small files to support automated testing, this repository does not contain significant data for tutorials and demonstrations. The tutorials demonstrate how to easily download sample data and generate synthetic data.

## Acknowledging This Work

If you find this work useful, please consider citing or acknowledging it. A citable DOI is coming soon. This section will be updated in the coming weeks to include the correct citation.
