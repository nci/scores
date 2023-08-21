# About the Scores Project

One-line intro: xarray based verification (accuracy) scoring that can scale with Dask if needed. Pandas supported where possible.
Why use it: trusted implementations, novel metrics, performance, one-stop-shop.

**Notice -- This repository is currently undergoing initial construction and maintenance. It is not yet recommended for use. This notice will be removed after the first feature release. In the meantime, please feel free to look around, and don't hesitate to get in touch with any questions (see the contributing guide for how).**

This package is currently in very active development. An addition ten to fifteen scores are expected to be implemented in the coming weeks. Performance testing and dask compatibility will be reviewed during and afterwards. The first 'release' will be tagged at that point.

The documentation is divided into the [user guide](docs/userguide.md), the [contribution guide](docs/contributing.md) (including developer documentation) and [API documentation](docs/api.md).

'scores' is a modular scoring package containing mathematical functions that can be used for the verification, evaluation, and optimisation of models, as well as other statistical functions. It is primarily aiming to support the geoscience and earth system community. Scores is focused on supporting xarray datatypes for earth system data. Other data formats such as Pandas and Iris can be easily be converted to xarray objects to utilise `scores`. It has wider potential application in machine learning and domains other than meteorology, geoscience and weather but primarily supports those fields. It aims to be compatible with geopandas, pangeo and work with NetCDF4, Zarr, and hd5 data sources among others.

All of the scores and metrics in this package have undergone a thorough statistical and scientific review.

All interactions in forums, wikis, issues, emails and code (i.e. merge requests, code comments) will be managed according to the expectations outlined in the [ code of conduct ](CODE_OF_CONDUCT.md) and in accordance with all relevant laws and obligations. This project is an inclusive, respectful and open project with high standards for respectful behaviour and language. The code of conduct is the Contributor Covenant, adopted by over 40, 000 open source projects. Any concerns will be dealt with fairly and respectfully, with the processes described in the code of conduct.

## Getting started using this package.

The [user guide](docs/userguide.md) contains information on installing, using and working with this package. Developers should follow the install steps from the [contributor's guide](docs/contributing.md)

Installation may be performed with:
```
> pip install scores
```

Here is an example of the use of scores:
```
> import scores
> forecast = scores.sample_data.simple_forecast()
> observed = scores.sample_data.simple_observations()
> mean_absolute_error = scores.continuous.mae(forecast, observed)
> print(mean_absolute_error)
<xarray.DataArray ()>
array(2.)
```

## Further Information on Scores and Metrics Included in the Package.

'scores' is a modular scoring package containing verification metrics, error functions, training scores and other statistical functions. It is primarily aiming to support the geoscience and earth system community. Scores is focused on supporting xarray and pandas datatypes for earth system data. It has wider potential application in machine learning and domains other than meteorology, geoscience and weather but primarily supports those fields. It aims to be compatible with geopandas, pangeo and work with NetCDF4 and hdf5 data sources among others.

'scores' includes novel scores not commonly found elsewhere (e.g. FIRM and FlipFlip index), complex scores (CRPS, Diebold Mariano) and more common scores (MAE, RMSE). Scores provides its own implementations where relevant to avoid extensive dependencies, and its roadmap includes a comprehensive implementation of optimised, reviewed and useful set of scoring functions for verification, statistics, optimisation and machine learning.

All of the scores and metrics in this package have undergone a thorough statistical and scientific review.

There are similar packages which should be acknowledged, in particular xskillscore and climpred. These packages both provide overlapping and similar functionality. This package provides an additional option to the community and has additional metrics which are not found in those other packages. The 'scores' package seeks to be self-contained with simple dependencies, and so re-implements various metrics which are found in other libraries, so that it can be a simple one-stop-shop for the metrics of interest to its userbase.

## Finding and Downloading Data

Other than very small files to support automated testing, this repository does not contain significant data for tutorials and demonstrations. The tutorials demonstrate how to easily download sample data and generate synthetic data.

## Acknowledging This Work

If you find this work useful, please consider a citation or acknowledgment of it.

