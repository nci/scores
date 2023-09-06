---
title: 'scores: A Python package for verifying accuracy using xarray and pandas'
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
date: 6 September 2023

---

# Summary

`scores` is a package containing mathematical functions for the verification, evaluation, and optimisation of model outputs and predictions. It primarily supports the geoscience and earth system science communities. `scores` is focused on supporting xarray datatypes for earth system data. It has wide potential application in machine learning, and domains other than meteorology, geoscience and weather. It also aims to be compatible with pandas, geopandas, pangeo and work with NetCDF4, Zarr, hdf5 and GRIB data sources among others. 

All of the scores and metrics in this package have undergone a thorough statistical and scientific review. Every score has a companion Jupyter Notebook demonstrating its use in practise.

At the time of writing, the scores contained in this package are: MSE, MAE, RMSE, FIRM [@Taggart:2022], CRPS, the FlipFlop index [@Griffiths:2019] and the Murphy score [@Ehm:2016]. It also includes the Diebold-Mariano statistical test [@Hering:2011] and [@Harvey:1997]. 

# Statement of need

The research purpose of this software is (a) to mathematically verify and validate scientific research and (b) to foster research into new scores and metrics.

`scores` includes novel scores not commonly found elsewhere (e.g. FIRM and FlipFlop index), complex scores (e.g. CRPS), more common scores (e.g. MAE, RMSE) and statistical tests (such as the Diebold Mariano test). Scores provides its own implementations where relevant to avoid extensive dependencies, and its roadmap includes a comprehensive implementation of optimised, reviewed and useful set of scoring functions for verification, statistics, optimisation and machine learning.

`scores` has proper treatments for vertical, horizontal and time dimensionality, as well as point-based data. It has proper treatments for missing data, masking of data and weighting of results.

`scores` was designed to work effectively with the libraries, data structures and methods commonly in use for scoring, verifying and evaluating earth system models including Numerical Weather Prediction (NWP) models, forecasts for specific sites and weather phenomena such as thunderstorms. It can effectively handle the dimensionality, data size and requirements of the modelling community.

`scores` is highly modular and has a minimal set of requirements. It is intented to be easy to integrate and utilise in a wide variety of environments. It has been tested and used on workstations, servers and in high performance computing (supercomputing) environments. 

## Related Works

`climpred` [@Brady2021:2021] provides some related functionality and provides many of the same scores. `climpred` does not contain some of the novel functions contained within `scores`, and at the same time makes some design choices specifically associated with climate modelling which do not generalise as effectively to broader use cases as may be needed in some circumstances. Releasing `scores` separately allows the differing design philosophies to be considered by the community.

`xskillscore` [@xskillscore] provides many of the same functions as `scores`. `xskillscore` does not contain some of the novel functions contained within `scores` and does not contain the Jupyter Notebook tutorials which provide users with clear guidance on the use of the verification metrics. 

# Acknowledgements

We acknowledge and are grateful for the support of the Australian Bureau of Meteorology in supporting scientific research and the academic process.

We would like to thank and acknowledge the National Computational Infrastructure (nci.org.au) for hosting the `scores` repository within their GitHub organisation.

# References
