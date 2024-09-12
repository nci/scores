# Release Notes (What's New)

## Version 1.2.0 (Upcoming Release) 

For a list of all changes in this release, see the [full changelog](https://github.com/nci/scores/compare/1.1.0...develop). Below are the changes we think users may wish to be aware of.

### Features

- Added three new metrics:
	- Percent bias (PBIAS): `scores.continuous.pbias`. See [PR #639](https://github.com/nci/scores/pull/639) and [PR #655](https://github.com/nci/scores/pull/655).
	- Threshold weighted continuous ranked probability score (twCRPS) for ensembles: `scores.probability.tw_crps_for_ensemble`. See [PR #644](https://github.com/nci/scores/pull/644).
	- Tail threshold weighted continuous ranked probability score (twCRPS) for ensembles: `scores.probability.tail_tw_crps_for_ensemble`. See [PR #644](https://github.com/nci/scores/pull/644).
- The FIxed Risk Multicategorical (FIRM) score (`scores.categorical.firm`) can now take a sequence of mulitdimensional arrays (xr.DataArray) of thresholds. This allows the FIRM score to be used with categorical thresholds that vary across the domain. See [PR #661](https://github.com/nci/scores/pull/661).

### Breaking Changes

### Deprecations

### Bug Fixes

### Documentation

- Added information about percent bias to the "Additive Bias and Multiplicative Bias" tutorial. See [PR #639](https://github.com/nci/scores/pull/639) and [PR #656](https://github.com/nci/scores/pull/656). 
- Updated documentation to say there are now over 60 metrics, statistical techniques and data processing tools contained in `scores`. See [PR #659](https://github.com/nci/scores/pull/659).
- In  the "Contributing Guide", updated instructions for installing a conda-based virtual environment. See [PR #654](https://github.com/nci/scores/pull/654). 



### Internal Changes

- Modified automated tests to work with NumPy 2.1. Incorporated a union type of `array` and `generic` in assert statements for Dask operations. See [PR #643](https://github.com/nci/scores/pull/643).

### Contributors to this Release

Durga Shrestha* ([@durgals](https://github.com/durgals)), Maree Carroll ([@mareecarroll](https://github.com/mareecarroll)), Nicholas Loveday ([@nicholasloveday](https://github.com/nicholasloveday)), Tennessee Leeuwenburg ([@tennlee](https://github.com/tennlee)), Stephanie Chong ([@Steph-Chong](https://github.com/Steph-Chong)) and Robert J. Taggart ([@rob-taggart](https://github.com/rob-taggart)). 

\* indicates that this release contains their first contribution to `scores`.

## Version 1.1.0 (August 9, 2024) 

For a list of all changes in this release, see the [full changelog](https://github.com/nci/scores/compare/1.0.0...1.1.0). Below are the changes we think users may wish to be aware of.

### Features

- `scores` is now available on [conda-forge](https://anaconda.org/conda-forge/scores).
- Added five new metrics 
	- threshold weighted squared error: ` scores.continuous.tw_squared_error`
	- threshold weighted absolute error: `scores.continuous.tw_absolute_error`
	- threshold weighted quantile score: `scores.continuous.tw_quantile_score`
	- threshold weighted expectile score: `scores.continuous.tw_expectile_score`
	- threshold weighted Huber loss: `scores.continuous.tw_huber_loss`.  
See [PR #609](https://github.com/nci/scores/pull/609).

### Documentation

- Added "Threshold Weighted Scores" tutorial. See [PR #609](https://github.com/nci/scores/pull/609).
- Removed nbviewer link from documentation. See [PR #615](https://github.com/nci/scores/pull/615).

### Internal Changes

- Modified `numpy.trapezoid` call to work with either NumPy 1 or 2. See [PR #610](https://github.com/nci/scores/pull/610).

### Contributors to this Release

Nicholas Loveday ([@nicholasloveday](https://github.com/nicholasloveday)), Tennessee Leeuwenburg ([@tennlee](https://github.com/tennlee)), Stephanie Chong ([@Steph-Chong](https://github.com/Steph-Chong)) and Robert J. Taggart ([@rob-taggart](https://github.com/rob-taggart)). 

## Version 1.0.0 (July 10, 2024)

We are happy to have reached the point of releasing “Version 1.0.0” of `scores`. While we look forward to many version increments to come, version 1.0.0 represents a milestone. It signifies a stabilisation of the API, and marks a turning point from the initial construction period. We have also published a [paper](https://doi.org/10.21105/joss.06889) in the Journal of Open Source Software (see citation further below).

From this point forward, `scores` will be following the [Semantic Versioning Specification](https://semver.org/) (SemVer) in its release management. 

This is a good moment to acknowledge and thank the contributors that helped us reach this point. They are: Tennessee Leeuwenburg, Nicholas Loveday, Elizabeth E. Ebert, Harrison Cook, Mohammadreza Khanarmuei, Robert J. Taggart, Nikeeth Ramanathan, Maree Carroll, Stephanie Chong, Aidan Griffiths and John Sharples.

Please consider a citation of our paper if you use our code. The citation is:

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

For a list of all changes in this release, see the [full changelog](https://github.com/nci/scores/compare/0.9.3...1.0.0). 

## Version 0.9.3 (July 9, 2024)

For a list of all changes in this release, see the [full changelog](https://github.com/nci/scores/compare/0.9.2...0.9.3). Below are the changes we think users may wish to be aware of.

### Breaking Changes

- Renamed and relocated function `scores.continuous.correlation` to `scores.continuous.correlation.pearsonr`. See [PR #583](https://github.com/nci/scores/pull/583).

### Documentation

- Added "Dimension Handling" tutorial, which describes reducing and preserving dimensions. See [PR #589](https://github.com/nci/scores/pull/589).
- Updated "Detailed Installation Guide" with information on installing kernels in a Jupyter environment. See [PR #586](https://github.com/nci/scores/pull/586) and [PR #587](https://github.com/nci/scores/pull/587).

### Internal Changes

- Introduced pinned versions for dependencies on main. See [PR #580](https://github.com/nci/scores/pull/580).

### Contributors to this Release

Tennessee Leeuwenburg ([@tennlee](https://github.com/tennlee)), Stephanie Chong ([@Steph-Chong](https://github.com/Steph-Chong)) and Nicholas Loveday ([@nicholasloveday](https://github.com/nicholasloveday)).

## Version 0.9.2 (June 26, 2024)

For a list of all changes in this release, see the [full changelog](https://github.com/nci/scores/compare/0.9.1...0.9.2). 

## Version 0.9.1 (June 14, 2024)

For a list of all changes in this release, see the [full changelog](https://github.com/nci/scores/compare/0.9.0...0.9.1). 

## Version 0.9.0 (June 12, 2024)

For a list of all changes in this release, see the [full changelog](https://github.com/nci/scores/compare/0.8.6...0.9.0). 

## Version 0.8.6 (June 11, 2024)

For a list of all changes in this release, see the [full changelog](https://github.com/nci/scores/compare/0.8.5...0.8.6). 

## Version 0.8.5 (June 9, 2024)

For a list of all changes in this release, see the [full changelog](https://github.com/nci/scores/compare/0.8.4...0.8.5). 

## Version 0.8.4 (June 3, 2024)

For a list of all changes in this release, see the [full changelog](https://github.com/nci/scores/compare/0.8.3...0.8.4). 

## Version 0.8.3 (June 2, 2024)

For a list of all changes in this release, see the [full changelog](https://github.com/nci/scores/compare/0.8.2...0.8.3). 

## Version 0.8.2 (May 21, 2024)

For a list of all changes in this release, see the [full changelog](https://github.com/nci/scores/compare/0.8.1...0.8.2). 

## Version 0.8.1 (May 16, 2024)

For a list of all changes in this release, see the [full changelog](https://github.com/nci/scores/compare/0.8...0.8.1). 

## Version 0.8 (May 14, 2024)

For a list of all changes in this release, see the [full changelog](https://github.com/nci/scores/compare/0.7...0.8). 

## Version 0.7 (May 8, 2024)

For a list of all changes in this release, see the [full changelog](https://github.com/nci/scores/compare/v0.6...0.7). 

## Version 0.6 (April 6, 2024)

For a list of all changes in this release, see the [full changelog](https://github.com/nci/scores/compare/v0.5...v0.6). 

Note: version 0.6 was initially tagged as "v0.6" and released on 6th April 2024. On 7th April 2024, an identical version was released with the tag "0.6" (i.e. with the "v" ommitted from the tag).

## Version 0.5 (April 6, 2024)

For a list of all changes in this release, see the [full changelog](https://github.com/nci/scores/compare/v0.4...v0.5). 

## Version 0.4 (September 15, 2023)

For a list of all changes in this release, see the [full changelog](https://github.com/nci/scores/compare/v0.0.2...v0.4). 

## Version 0.0.2 (June 9, 2023)

For a list of all changes in this release, see the [full changelog](https://github.com/nci/scores/commits/v0.0.2/). 

## Version 0.0.1 (January 16, 2023)

Version 0.0.1 was released on PyPI as a placeholder, while very early development and package design was being undertaken.

