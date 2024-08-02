# Release Notes (What's New)

## Version 1.1.0 (Upcoming Release) 

For the full details of all changes in this release, see the [GitHub commit history](https://github.com/nci/scores/compare/1.0.0...develop). Below are the changes we think users may wish to be aware of.

### Features

- Added five new threshold weighted scores 
	- threshold weighted squared error: ` scores.continuous.tw_squared_error`
	- threshold weighted absolute error: `scores.continuous.tw_absolute_error`
	- threshold weighted quantile score: `scores.continuous.tw_quantile_score`
	- threshold weighted expectile score: `scores.continuous.tw_expectile_score`
	- threshold weighted huber loss: `scores.continuous.tw_huber_loss`.  
See [PR #609](https://github.com/nci/scores/pull/609) by [@nicholasloveday](https://github.com/nicholasloveday).

### Breaking Changes

### Deprecations

### Bug Fixes

### Documentation

- Removed nbviewer link from documentation. See [PR #615](https://github.com/nci/scores/pull/615) by [@tennlee](https://github.com/tennlee).

### Internal Changes

- Modified trapezoidal call to work with either NumPy 1 or 2. See [PR #610](https://github.com/nci/scores/pull/610) by [@tennlee](https://github.com/tennlee).

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

For the full details of all changes in this release, see the [GitHub commit history](https://github.com/nci/scores/compare/0.9.3...1.0.0). 

## Version 0.9.3 (July 9, 2024)

For the full details of all changes in this release, see the [GitHub commit history](https://github.com/nci/scores/compare/0.9.2...0.9.3). Below are the changes we think users may wish to be aware of.

### Breaking Changes

- Renamed and relocated function `scores.continuous.correlation` to `scores.continuous.correlation.pearsonr`. See [PR #583](https://github.com/nci/scores/pull/583) by [@nicholasloveday](https://github.com/nicholasloveday). 

### Documentation

- Added "Dimension Handling" tutorial, which describes reducing and preserving dimensions. See [PR #589](https://github.com/nci/scores/pull/589) by [@nicholasloveday](https://github.com/nicholasloveday).
- Updated "Detailed Installation Guide" with information on installing kernels in a Jupyter environment. See [PR #586](https://github.com/nci/scores/pull/586) by [@tennlee](https://github.com/tennlee) and [PR #587](https://github.com/nci/scores/pull/587) by [@Steph-Chong](https://github.com/Steph-Chong).

### Internal Changes

- Introduced pinned versions for dependencies on main. See [PR #580](https://github.com/nci/scores/pull/580)  by [@tennlee](https://github.com/tennlee).

