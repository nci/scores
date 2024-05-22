---
title: 'scores: A Python package for evaluating and verifying forecasts using xarray'
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
    affiliation: 1
affiliations:
 - name: Bureau of Meteorology, Australia
   index: 1
date: 17 May 2024
bibliography: paper.bib 

---

# Summary

`scores` is a Python package containing mathematical functions for the verification, evaluation and optimisation of forecasts, predictions or models. It primarily supports the geoscience communities; in particular, the meteorological, climatological and oceanographic communities. In addition to supporting the Earth system science communities, it also has wide potential application in machine learning and other domains such as economics.

`scores` not only includes common scores (e.g. Mean Absolute Error), it also includes novel scores not commonly found elsewhere (e.g. FIxed Risk Multicategorical (FIRM) score, Flip-Flop Index), complex scores (e.g. threshold-weighted continuous ranked probability score), and statistical tests (such as the Diebold Mariano test). It also contains isotonic regression which is becoming an increasingly important tool in forecast verification and can be used to generate stable reliability diagrams. Additionally, it provides pre-processing tools for preparing data for scores in a variety of formats including cumulative distribution functions (CDF). At the time of writing, `scores` includes over 50 metrics, statistical techniques and data processing tools.

All of the scores and statistical techniques in this package have undergone a thorough scientific and software review. Every score has a companion Jupyter Notebook tutorial that demonstrates its use in practice.

`scores` primarily supports xarray datatypes for Earth system data, allowing it to work with NetCDF4, hdf5, Zarr and GRIB data sources among others. `scores` uses Dask for scaling and performance. It also aims to be compatible with pandas and geopandas. 

The software repository can be found at [https://github.com/nci/scores/](https://github.com/nci/scores/).

# Statement of Need

The research purpose of this software is (a) to mathematically verify and validate scientific research and (b) to foster research into new scores and metrics. 

In order to meet the needs of researchers, `scores` provides the following key benefits.

**Data Handling**

- Works with n-dimensional data (e.g., geospatial, vertical and temporal dimensions) for both point-based and gridded data. `scores` can effectively handle the dimensionality, data size and data structures commonly utilised for:
  - gridded Earth system data (e.g. Numerical Weather Prediction models)
  - tabular, point, latitude/longitude or site-based data (e.g. forecasts for specific locations).
- Handles missing data, masking of data and weighting of results.
- Supports xarray [@Hoyer:2017] datatypes, and works with NetCDF4, hdf5, Zarr and GRIB data sources among others.

**Usability**

- A companion Jupyter Notebook tutorial for each metric and statistical test that demonstrates its use in practice.
- Novel scores not commonly found elsewhere (e.g. FIRM [@Taggart:2022a], Flip-Flop Index [@Griffiths:2019; @griffiths2021circular]).
- All scores and statistical techniques have undergone a thorough scientific and software review.
- An area specifically to hold emerging scores which are still undergoing research and development. This provides a clear mechanism for people to share, access and collaborate on new scores, and be able to easily re-use versioned implementations of those scores.  

**Compatability**

- Highly modular and avoids extensive dependencies by providing its own implementations where relevant.
- Easy to integrate and use in a wide variety of environments. It has been tested and used on workstations, servers and in high performance computing (supercomputing) environments.
- Uses Dask [@Dask:2016] for scaling and performance.
- Aims to be compatible with pandas [@pandas:2024; @McKinney:2010] and geopandas [@geopandas:2024].

## Metrics, Statistical Techniques and Data Processing Tools Included in `scores` 

At the time of writing, `scores` includes **over 50** metrics, statistical techniques and data processing tools. For an up to date list, please see the `scores` [documentation](https://scores.readthedocs.io/en/latest/included.html).

We anticipate more metrics, tools and statistical techniques will be added over time.

Table: A **Curated Selection** of the Metrics, Tools and Statistical Tests Currently Included in `scores`

|              | **Description** |**A Selection of the Functions Included in `scores`**|
|--------------|-----------------|-----------------------------------------------------|
| **[Continuous](https://scores.readthedocs.io/en/latest/included.html#continuous)**        	|Scores for evaluating single-valued continuous forecasts.                  	|Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Additive Bias, Multiplicative Bias, Pearson's Correlation Coefficient, Flip-Flop Index [@Griffiths:2019; @griffiths2021circular], Quantile loss, Murphy score [@Ehm:2016].              	  
|
| **[Probability](https://scores.readthedocs.io/en/latest/included.html#probability)**       	|Scores for evaluating forecasts that are expressed as predictive distributions, ensembles, and probabilities of binary events.                 	|Brier Score, Continuous Ranked Probability Score (CRPS) for Cumulative Distribution Functions (CDFs) (including threshold-weighting, see [@Gneiting:2011]), CRPS for ensembles [@Gneiting_2007; @Ferro_2013], Receiver Operating Characteristic (ROC), Isotonic Regression (reliability diagrams) [@dimitriadis2021stable].              	  
|
| **[Categorical](https://scores.readthedocs.io/en/latest/included.html#categorical)**       	|Scores for evaluating forecasts based on categories.                	|Probability of Detection (POD), False Alarm Rate (FAR), Probability of False Detection (POFD), Success Ratio, Accuracy, Peirce's Skill Score, Critical Success Index (CSI), Gilbert Skill Score, Heidke Skill Score, Odds Ratio, Odds Ratio Skill Score, F1 score, FIxed Risk Multicategorical (FIRM) Score [@Taggart:2022a].               	  
|
| **[Statistical Tests](https://scores.readthedocs.io/en/latest/included.html#statistical-tests)** 	|Tools to conduct statistical tests and generate confidence intervals.                 	| Diebold-Mariano [@Diebold:1995] with both the [@Harvey:1997] and [@Hering:2011] modifications.              	  
|
| **[Processing Tools](https://scores.readthedocs.io/en/latest/included.html#processing-tools-for-preparing-data)**        	|Tools to pre-process data.                 	|Data matching, Discretisation, Cumulative Density Function Manipulation. |

## Use in Academic Work

In 2015, the Australian Bureau of Meteorology began developing a new verification system called Jive. For a description of Jive see @loveday2024jive. The Jive verification metrics have been used to support several publications [@Griffiths:2017; @Foley:2020; @Taggart:2022b; @Taggart:2022c; @Taggart:2022d]. `scores` has arisen from the Jive verification system and was created to modularise the Jive verification functions and make them available as an open source package. 

`scores` has been used to explore user-focused approaches to evaluating probabilistic and categorical forecasts [@loveday2024user].

## Related Software Packages

`climpred` [@Brady:2021] provides some related functionality and provides many of the same scores. `climpred` does not contain some of the novel functions contained within `scores`, and at the same time makes some design choices specifically associated with climate modelling which do not generalise as effectively to broader use cases as may be needed in some circumstances. Releasing `scores` separately allows the differing design philosophies to be considered by the community.

`xskillscore` [@xskillscore] provides many of the same functions as `scores`. `xskillscore` does not contain some of the novel functions contained within `scores` and does not contain the Jupyter Notebook tutorials which provide users with clear guidance on the use of the verification metrics. 

`METplus` [@Brown:2021] provides related functionality. `METplus` includes a database and visualisation system with python wrappers to utilise the `MET` package. Verification scores in `MET` are implemented in C++ rather than Python.  `METplus` does not contain some of the novel functions contained within `scores`.

`Verif` [@nipen2023verif] is a command line tool for forecast verification and is utilised very differently to `scores`. It also does not contain some of the novel metrics in `scores`.

# Acknowledgements

We acknowledge and are grateful for the support of the Australian Bureau of Meteorology in supporting scientific research and the academic process.

We would like to thank and acknowledge the National Computational Infrastructure (nci.org.au) for hosting the `scores` repository within their GitHub organisation.

# References
