# Scores: Forecast and Model Verification and Evaluation Software
> 
> **A list of over 50 metrics, statistical techniques and data processing tools contained in `scores` is [available here](https://scores.readthedocs.io/en/latest/included.html).**

`scores` is a Python package containing mathematical functions for the verification, evaluation, and optimisation of forecasts, predictions or models. It primarily supports the meteorological, climatological and geoscience communities. In addition to supporting the Earth system science communities, it also has wide potential application in machine learning and other domains.

**Notice -- This repository is currently undergoing initial construction and maintenance. It is getting much closer to our goals for version one, but there are a few more things to add. This notice will be removed after the first feature release. In the meantime, please feel free to look around, and don't hesitate to get in touch with any questions (see the contributing guide for how).**

Documentation is hosted at [scores.readthedocs.io](https://scores.readthedocs.io).  
Source code is hosted at [github.com/nci/scores](https://github.com/nci/scores).  
The tutorial gallery is hosted at [as part of the documentation, here](https://scores.readthedocs.io/en/latest/tutorials/Explanation.html).

## Overview
Here is a **curated selection** of the metrics, tools and statistical tests included in `scores`:

|                       	| **Description** 	| **Selection of Included Functions** 	|
|-----------------------	|-----------------	|--------------	|
| **[Continuous](https://scores.readthedocs.io/en/latest/included.html#continuous)**        	|Scores for evaluating single-valued continuous forecasts.                  	|Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Additive Bias, Multiplicative Bias, Pearson's Correlation Coefficient, Flip-Flop Index, Quantile loss, Murphy score.              	|
| **[Probability](https://scores.readthedocs.io/en/latest/included.html#probability)**       	|Scores for evaluating forecasts that are expressed as predictive distributions, ensembles, and probabilities of binary events.                 	|Brier Score, Continuous Ranked Probability Score (CRPS) for Cumulative Density Function (CDF), Threshold weighted CRPS for CDF, CRPS for ensembles, Receiver Operating Characteristic (ROC), Isotonic Regression (reliability diagrams).              	|
| **[Categorical](https://scores.readthedocs.io/en/latest/included.html#categorical)**       	|Scores for evaluating forecasts based on categories.                	|Probability of Detection (POD), False Alarm Rate (FAR), Probability of False Detection (POFD), Success Ratio, Accuracy, Peirce's Skill Score, Critical Success Index (CSI), Gilbert Skill Score, Heidke Skill Score, Odds Ratio, Odds Ratio Skill Score, F1 score, FIxed Risk Multicategorical (FIRM) Score.               	|
| **[Statistical Tests](https://scores.readthedocs.io/en/latest/included.html#statistical-tests)** 	|Tools to conduct statistical tests and generate confidence intervals.                 	|Diebold Mariano.              	|
| **[Processing Tools](https://scores.readthedocs.io/en/latest/included.html#processing-tools-for-preparing-data)**        	|Tools to pre-process data.                 	|Data matching, Discretization, Cumulative Density Function Manipulation.              	|


`scores` not only includes common scores (e.g. MAE, RMSE), it includes novel scores not commonly found elsewhere (e.g. FIRM, Flip-Flop Index), complex scores (e.g. threshold weighted CRPS), and statistical tests (such as the Diebold Mariano test). Additionally, it provides pre-processing tools for preparing data for scores in a variety of formats including cumulative distribution functions (CDF). `scores` provides its own implementations where relevant to avoid extensive dependencies.

`scores` primarily supports xarray datatypes for Earth system data allowing it to work with NetCDF4, hdf5, Zarr and GRIB data sources among others. `scores` uses Dask for scaling and performance. Some metrics work with pandas and we will aim to expand this capability. 

All of the scores and metrics in this package have undergone a thorough scientific review. Every score has a companion Jupyter Notebook tutorial that demonstrates its use in practice.

## Contributing
To find out more about contributing, see our [contributor's guide](https://github.com/nci/scores/blob/main/docs/contributing.md).

All interactions in discussions, issues, emails and code (e.g. merge requests, code comments) will be managed according to the expectations outlined in the [ code of conduct ](https://github.com/nci/scores/blob/main/CODE_OF_CONDUCT.md) and in accordance with all relevant laws and obligations. This project is an inclusive, respectful and open project with high standards for respectful behaviour and language. The code of conduct is the Contributor Covenant, adopted by over 40,000 open source projects. Any concerns will be dealt with fairly and respectfully, with the processes described in the code of conduct.

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

## Finding, Downloading and Working With Data

`scores` can be used with a broad variety of data sources. See the [Data Sources](https://scores.readthedocs.io/en/latest/data.html) page and this [tutorial](https://scores.readthedocs.io/en/latest/tutorials/First_Data_Fetching.html) for more information on finding, downloading and working with different sources of data.

## Acknowledging This Work

If you find this work useful, please consider citing or acknowledging it. A citable DOI is coming soon. This section will be updated in the coming weeks to include the correct citation.
