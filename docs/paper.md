---
title: 'scores: A Python package for verifying and evaluating models and predictions with xarray and pandas'
tags:
  - Python
  - verification
  - statistics
  - modelling
  - geoscience
  - earth system science
 
authors:
  - name: Tennessee Leeuwenburg
    orcid: 0009-0008-2024-1967
    affiliation: 1 
    corresponding: true # (This is how to denote the corresponding author)    
  - name: Nicholas Loveday
    orcid: 0009-0000-5796-7069
    affiliation: 1
  - name: Elizabeth E. Ebert
    affiliation: 1    
  - name: Harrison Cook
    orcid: 0009-0009-3207-4876
    affiliation: 1    
  - name: Mohammadreza Khanarmuei
    orcid: 0000-0002-5017-9622
    affiliation: 1    
  - name: Robert J. Taggart
    orcid: 0000-0002-0067-5687
    affiliation: 1
  - name: Nikeeth Ramanathan
    orcid: 0009-0002-7406-7438
    affiliation: 1
  - name: Maree Carroll
    orcid: 0009-0008-6830-8251
    affiliation: 1
  - name: Stephanie Chong
    orcid: 0009-0007-0796-4127
    affiliation: 2
  - name: Aidan Griffiths
    affiliation: 3
  - name: John Sharples
    affiliation: 1

affiliations:
 - name: Bureau of Meteorology, Australia
   index: 1
 - name: Independent Contributor, Australia
   index: 2
 - name: Work undertaken while at the Bureau of Meteorology, Australia
   index: 3


date: 17 May 2024
bibliography: paper.bib 

---

# Summary

`scores` is a Python package containing mathematical functions for the verification, evaluation and optimisation of forecasts, predictions or models. It primarily supports the geoscience communities; in particular, the meteorological, climatological and oceanographic communities. In addition to supporting the Earth system science communities, it also has wide potential application in machine learning and other domains such as economics.

`scores` not only includes common scores (e.g. Mean Absolute Error), it also includes novel scores not commonly found elsewhere (e.g. FIxed Risk Multicategorical (FIRM) score, Flip-Flop Index), complex scores (e.g. threshold-weighted continuous ranked probability score), and statistical tests (such as the Diebold Mariano test). It also contains isotonic regression which is becoming an increasingly important tool in forecast verification and can be used to generate stable reliability diagrams. Additionally, it provides pre-processing tools for preparing data for scores in a variety of formats including cumulative distribution functions (CDF). At the time of writing, `scores` includes over 50 metrics, statistical techniques and data processing tools.

All of the scores and statistical techniques in this package have undergone a thorough scientific and software review. Every score has a companion Jupyter Notebook tutorial that demonstrates its use in practice.

`scores` primarily supports `xarray` datatypes for Earth system data, allowing it to work with NetCDF4, HDF5, Zarr and GRIB data sources among others. `scores` uses Dask for scaling and performance. It has expanding support for `pandas`.  

The software repository can be found at [https://github.com/nci/scores/](https://github.com/nci/scores/).

\pagebreak

# Statement of Need

The purpose of this software is (a) to mathematically verify and validate models and predictions and (b) to foster research into new scores and metrics. 

## Key Benefits of `scores`

In order to meet the needs of researchers and other users, `scores` provides the following key benefits.

**Data Handling**

- Works with n-dimensional data (e.g., geospatial, vertical and temporal dimensions) for both point-based and gridded data. `scores` can effectively handle the dimensionality, data size and data structures commonly used for:
  - gridded Earth system data (e.g. numerical weather prediction models)
  - tabular, point, latitude/longitude or site-based data (e.g. forecasts for specific locations).
- Handles missing data, masking of data and weighting of results.
- Supports `xarray` [@Hoyer:2017] datatypes, and works with NetCDF4 [@NetCDF:2024], HDF5 [@HDF5:2020], Zarr [@zarr:2020] and GRIB [@GRIB:2024] data sources among others.

**Usability**

- A companion Jupyter Notebook [@Jupyter:2024] tutorial for each metric and statistical test that demonstrates its use in practice.
- Novel scores not commonly found elsewhere (e.g. FIRM [@Taggart:2022a], Flip-Flop Index [@Griffiths:2019; @griffiths2021circular]).
- All scores and statistical techniques have undergone a thorough scientific and software review.
- An area specifically to hold emerging scores which are still undergoing research and development. This provides a clear mechanism for people to share, access and collaborate on new scores, and be able to easily re-use versioned implementations of those scores.  

**Compatability**

- Highly modular - provides its own implementations, avoids extensive dependencies and offers a consistent API.
- Easy to integrate and use in a wide variety of environments. It has been used on workstations, servers and in high performance computing (supercomputing) environments. 
- Maintains 100% automated test coverage.
- Uses Dask [@Dask:2016] for scaling and performance.
- Expanding support for `pandas` [@pandas:2024; @McKinney:2010].


## Metrics, Statistical Techniques and Data Processing Tools Included in `scores` 

At the time of writing, `scores` includes over 50 metrics, statistical techniques and data processing tools. For an up to date list, please see the `scores` documentation.

The ongoing development roadmap includes the addition of more metrics, tools, and statistical tests.

Table: A **curated selection** of the metrics, tools and statistical tests currently included in `scores`

|              | **Description** |**A Selection of the Functions Included in `scores`**|
|--------------|-----------------|-----------------------------------------------------|
| **Continuous**        	|Scores for evaluating single-valued continuous forecasts.                  	|Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Additive Bias, Multiplicative Bias, Pearson's Correlation Coefficient, Flip-Flop Index [@Griffiths:2019; @griffiths2021circular], Quantile Loss, Murphy Score [@Ehm:2016].              	  
|
| **Probability**       	|Scores for evaluating forecasts that are expressed as predictive distributions, ensembles, and probabilities of binary events.                 	|Brier Score [@BRIER_1950], Continuous Ranked Probability Score (CRPS) for Cumulative Distribution Functions (CDFs) (including threshold-weighting, see @Gneiting:2011), CRPS for ensembles [@Gneiting_2007; @Ferro_2013], Receiver Operating Characteristic (ROC), Isotonic Regression (reliability diagrams) [@dimitriadis2021stable].              	  
|
| **Categorical**         |Scores for evaluating forecasts of categories.                 |Probability of Detection (POD), Probability of False Detection (POFD), False Alarm Ratio (FAR), Success Ratio, Accuracy, Peirce's Skill Score [@Peirce:1884], Critical Success Index (CSI), Gilbert Skill Score [@gilbert:1884], Heidke Skill Score, Odds Ratio, Odds Ratio Skill Score, F1 Score, Symmetric Extremal Dependence Index [@Ferro:2011], FIxed Risk Multicategorical (FIRM) Score [@Taggart:2022a].        	  
|
| **Spatial**           |Scores that take into account spatial structure.                   |Fractions Skill Score [@Roberts:2008].  
|
| **Statistical Tests** 	|Tools to conduct statistical tests and generate confidence intervals.                 	| Diebold-Mariano [@Diebold:1995] with both the @Harvey:1997 and @Hering:2011 modifications.              	  
|
| **Processing Tools**        	|Tools to pre-process data.                 	|Data matching, discretisation, cumulative density function manipulation. |

## Use in Academic Work

In 2015, the Australian Bureau of Meteorology began developing a new verification system called Jive, which became operational in 2022. For a description of Jive see @loveday2024jive. The Jive verification metrics have been used to support several publications [@Griffiths:2017; @Foley:2020; @Taggart:2022d; @Taggart:2022b; @Taggart:2022c]. `scores` has arisen from the Jive verification system and was created to modularise the Jive verification functions and make them available as an open source package. `scores` also includes additional metrics that Jive does not contain.

`scores` has been used to explore user-focused approaches to evaluating probabilistic and categorical forecasts [@Loveday2024ts]. 

## Related Software Packages

There are multiple open source verification packages in a range of languages. Below is a comparison of `scores` to other open source Python verification packages. None of these include all of the metrics implemented in `scores` (and vice versa).
 
`xskillscore` [@xskillscore] provides many but not all of the same functions as `scores` and does not have direct support for pandas. The Jupyter Notebook tutorials in `scores` cover a wider array of metrics. 

`climpred` [@Brady:2021] uses `xskillscore` combined with data handling functionality, and is focused on ensemble forecasts for climate and weather. `climpred` makes some design choices related to data structure (specifically associated with climate modelling) which may not generalise effectively to broader use cases. Releasing `scores` separately allows the differing design philosophies to be considered by the community.

`METplus` [@Brown:2021] is a substantial verification system used by weather and climate model developers. `METplus` includes a database and a visualisation system, with Python and shell script wrappers to use the `MET` package for the calculation of scores. `MET` is implemented in C++ rather than Python. `METplus` is used as a system rather than providing a modular Python API.

`Verif` [@nipen2023verif] is a command line tool for generating verification plots whereas `scores` provides a Python API for generating numerical scores. 

`Pysteps` [@gmd-12-4185-2019; @Imhoff:2023] is a package for short-term ensemble prediction systems, and includes a significant verification submodule with many useful verification scores. `PySteps` does not provide a standalone verification API. 

`PyForecastTools` [@Morley:2020] is a Python package for model and forecast verification which supports `dmarray` rather than `xarray` data structures and does not include Jupyter Notebook tutorials.

# Acknowledgements

We would like to thank Jason West and Robert Johnson from the Bureau of Meteorology for their feedback on an earlier version of this manuscript.

We would like to thank and acknowledge the National Computational Infrastructure (nci.org.au) for hosting the `scores` repository within their GitHub organisation.

# References
