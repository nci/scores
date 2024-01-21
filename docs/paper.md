---
title: 'scores: A Python package for verifying accuracy using xarray'
tags:
  - Python
  - geoscience
  - verification
  - science
  - earth system science
  - statistics
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
date: 1 December 2023
bibliography: paper.bib 

---

# Summary

`scores` is a Python package containing mathematical functions for the verification, evaluation and optimisation of forecasts, predictions or models. It primarily supports the geoscience and earth system science communities. It also has wide potential application in machine learning, and in domains other than meteorology, geoscience and weather. 

`scores` includes novel scores not commonly found elsewhere (e.g. FIRM, FlipFlop Index), complex scores (e.g. threshold weighted CRPS), more common scores (e.g. MAE, RMSE) and statistical tests (such as the Diebold Mariano test). 

All of the scores and metrics in this package have undergone a thorough statistical and scientific review. Every score has a companion Jupyter Notebook tutorial demonstrating its use in practice.

`scores` is focused on supporting xarray [@Hoyer:2017] datatypes for earth system data. It also aims to be compatible with pandas and geopandas, and to work with NetCDF4, hdf5, Zarr and GRIB data sources among others. Scores is designed to utilise Dask for scaling and performance.

The software repository can be found at [https://github.com/nci/scores/](https://github.com/nci/scores/).

# Statement of Need

The research purpose of this software is (a) to mathematically verify and validate scientific research and (b) to foster research into new scores and metrics.

## Key Benefits

In order to meet the needs of researchers, `scores`:

- is designed to work with n-dimensional data (e.g., geospatial, vertical and temporal dimensions) for both point-based and gridded data. It has proper treatments for missing data, masking of data and weighting of results.
- includes novel scores not commonly found elsewhere (e.g. FIRM, FlipFlop Index).
- is designed to work effectively with the libraries, data structures and methods commonly used in the meteorology, weather and climate communities. Scores can effectively handle the dimensionality, data size and data structures commonly utilised for: 
  - gridded earth system data (e.g. Numerical Weather Prediction models) 
  - tabular, point, lat/lon or site-based data (e.g. forecasts for specific locations)
  - geospatial, object-based data (e.g. forecasts of thunderstorm tracks or storm cells). 
- includes a companion Jupyter Notebook for each score, metric and statistical test to demonstrate its use in practice
- is highly modular and avoids extensive dependencies by providing its own implementations where relevant.
- is intended to be easy to integrate and utilise in a wide variety of environments. It has been tested and used on workstations, servers and in high performance computing (supercomputing) environments. 
- utilises Dask for scaling and performance

## Roadmap and Future Development

At the time of writing, the scores contained in this package are: Mean Squared Error (MSE), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), the Fixed Risk Multicategorical (FIRM) score [@Taggart:2022a], Continuous Ranked Probability Score (CRPS) for Cumulative Distribution Functions (CDFs) (including threshold-weighting, see [@Gneiting:2011]), the FlipFlop Index [@Griffiths:2019; @griffiths2021circular], Receiver Operating Characteristic (ROC) curves, the quantile score, and the Murphy score [@Ehm:2016]. It also includes the Diebold-Mariano statistical test [@Diebold:1995] with both the [@Harvey:1997] and [@Hering:2011] modifications. Additionally it contains isotonic regression which is becoming an increasingly important tool in forecast verification and can be used to generate stable reliability diagrams [@dimitriadis2021stable].

The `scores` roadmap includes:

- The addition of more scores, metrics and statistical techniques
- Further optimisation and performance improvements
- Increased support for machine learning library integration
- Additional notebooks exploring complex use cases in depth

`scores` has an area specifically to hold emerging scores which are still undergoing research and development. This provides a clear mechanism for people to share, access and collaborate on new scores, and be able to easily re-use versioned implementations of those scores. 

## Use in Academic Work

In 2015, the Australian Bureau of Meteorology began developing a new verification system called Jive. For a description of Jive see [@Loveday:2023].

The Jive verification metrics have been used by [@Griffiths:2017], [@Foley:2020], [@Taggart:2022b], [@Taggart:2022c] and [@Taggart:2022d].

`scores` has arisen from, and now supports, the Jive verification system. `scores` includes mathematical functions from Jive and is intended to modularise these functions and make them available as an open source package. 

`scores` has been used to explore user-focused approaches to evaluating probabilistic and categorical forecasts [@loveday2023userfocused].

## Related Software Packages

`climpred` [@Brady:2021] provides some related functionality and provides many of the same scores. `climpred` does not contain some of the novel functions contained within `scores`, and at the same time makes some design choices specifically associated with climate modelling which do not generalise as effectively to broader use cases as may be needed in some circumstances. Releasing `scores` separately allows the differing design philosophies to be considered by the community.

`xskillscore` [@xskillscore] provides many of the same functions as `scores`. `xskillscore` does not contain some of the novel functions contained within `scores` and does not contain the Jupyter Notebook tutorials which provide users with clear guidance on the use of the verification metrics. 

`METplus` [@Brown:2021] provides related functionality. `METplus` includes a database and visualisation system with python wrappers to utilise the `MET` package. Verification scores in `MET` are implemented in C++ rather than Python.  `METplus` does not contain some of the novel functions contained within `scores`.

`Verif` [@nipen2023verif] is a command line tool for forecast verification and is utilised very differently to `scores`. It also does not contain some of the novel metrics in `scores`.

# Acknowledgements

We acknowledge and are grateful for the support of the Australian Bureau of Meteorology in supporting scientific research and the academic process.

We would like to thank and acknowledge the National Computational Infrastructure (nci.org.au) for hosting the `scores` repository within their GitHub organisation.

# References
