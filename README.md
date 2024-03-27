# About the Scores Project

One-line intro: xarray based verification (accuracy) scoring that can scale with Dask if needed. Pandas supported where possible.
Why use it: trusted implementations, novel metrics, performance, one-stop-shop.

Currently Included Metrics and Tools:

| continuous                      | probability | categorical      | statistical tests |
| ----------                      | ----------- | -----------      | ----------------- |
| MAE, MSE, RMSE, Flip Flop Index, Quantile Score, Isotonic Regression, Pearsons correlation coefficient  | CRPS for CDF, CRPS for ensemble, ROC, Brier score   | FIRM, POD, POFD  |  Diebold Mariano (with the Harvey et al. 1997 and the Hering and Genton 2011 modifications) |

**Notice -- This repository is currently undergoing initial construction and maintenance. It is not yet recommended for use. This notice will be removed after the first feature release. In the meantime, please feel free to look around, and don't hesitate to get in touch with any questions (see the contributing guide for how).**

Documentation is hosted at [scores.readthedocs.io](https://scores.readthedocs.io)

`scores` is a Python package containing mathematical functions for the verification, evaluation, and optimisation of model outputs and predictions. It primarily supports the geoscience and earth system science communities. It also has wide potential application in machine learning, and in domains other than meteorology, geoscience and weather.

`scores` includes novel scores not commonly found elsewhere (e.g. FIRM, FlipFlop Index), complex scores (e.g. CRPS), more common scores (e.g. MAE, RMSE) and statistical tests (such as the Diebold Mariano test). `scores` provides its own implementations where relevant to avoid extensive dependencies.

`scores` is focused on supporting xarray datatypes for earth system data. It also aims to be compatible with pandas, geopandas, pangeo and work with NetCDF4, hdf5, Zarr and GRIB data sources among others. `scores` is designed to utilise Dask for scaling and performance.

All of the scores and metrics in this package have undergone a thorough statistical and scientific review. Every score has a companion Jupyter Notebook tutorial demonstrating its use in practice.

All interactions in discussions, issues, emails and code (e.g. merge requests, code comments) will be managed according to the expectations outlined in the [ code of conduct ](CODE_OF_CONDUCT.md) and in accordance with all relevant laws and obligations. This project is an inclusive, respectful and open project with high standards for respectful behaviour and language. The code of conduct is the Contributor Covenant, adopted by over 40, 000 open source projects. Any concerns will be dealt with fairly and respectfully, with the processes described in the code of conduct.

## Using This Package

The [installation guide](docs/installation.md) contains information on the various ways of installing, using and working with this package.

Installation of the core mathematical API may be performed with:

```py
> pip install scores
```

Here is an example of the use of scores:

```py
> import scores
> forecast = scores.sample_data.simple_forecast()
> observed = scores.sample_data.simple_observations()
> mean_absolute_error = scores.continuous.mae(forecast, observed)
> print(mean_absolute_error)
<xarray.DataArray ()>
array(2.)
```

## Related Packages

There are similar packages which should be acknowledged, in particular [xskillscore](https://xskillscore.readthedocs.io/en/stable/) and [climpred](https://github.com/pangeo-data/climpred). These packages both provide overlapping and similar functionality. `scores` provides an additional option to the community and has additional metrics which are not found in those other packages. `scores` seeks to be self-contained with few dependencies, and so re-implements various metrics which are found in other libraries, so that it can be a simple one-stop-shop for the metrics of interest to its userbase.

## Finding and Downloading Data

Other than very small files to support automated testing, this repository does not contain significant data for tutorials and demonstrations. The tutorials demonstrate how to easily download sample data and generate synthetic data.

## Acknowledging This Work

If you find this work useful, please consider a citation or acknowledgment of it. A citable DOI is coming soon. This section will be updated in the coming weeks to include the correct citation.
