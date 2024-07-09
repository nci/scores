# Release Notes (What's New)

## Version 1.0.0 (July 10, 2024)

We are happy to have reached the point of releasing “Version 1.0.0” of `scores`. While we look forward to many version increments to come, version 1.0.0 represents a milestone. It signifies a stabilisation of the API, and marks a turning point from the initial construction period. We have also published a paper in the Journal of Open Source Software (see citation further below).

From this point forward, `scores` will be following the Semantic Versioning Specification (SemVer) in its release management. We still intend to release as frequently as the pace of development allows, so there may be periods of rapid change and upgrade. Should a change affect backward compatibility, we will attempt to give a reasonable deprecation period, but we will not be reluctant to proceed to major version upgrades should that be warranted.

This is a good moment to acknowledge and thank the contributors that helped us reach this point. They are: Tennessee Leeuwenburg, Nicholas Loveday, Elizabeth E. Ebert, Harrison Cook, Mohammadreza Khanarmuei, Robert J. Taggart, Nikeeth Ramanathan, Maree Carroll, Stephanie Chong, Aidan Griffiths and John Sharples.

Please consider a citation of our paper if you use our code. The citation is:

Leeuwenburg, T., Loveday, N., Ebert, E. E., Cook, H., Khanarmuei, M., Taggart, R. J., Ramanathan, N., Carroll, M., Chong, S., Griffiths, A., & Sharples, J. (2024). scores: A Python package for verifying and evaluating models and predictions with xarray. Journal of Open Source Software, VolXX(IssueXX), XXX-XXX. https://doi.org/xxxx

For the full details of all changes in this release, see the GitHub commit history. Below are the changes we think users may wish to be aware of.

For the full details of all changes in this release, see the [GitHub commit history](https://github.com/nci/scores/compare/0.9.3...1.0.0). Below are the changes we think users may wish to be aware of.

### Features
### Breaking Changes
### Deprecations
### Bug Fixes
### Documentation
### Internal Changes

## Version 0.9.3 (July 9, 2024)

For the full details of all changes in this release, see the [GitHub commit history](https://github.com/nci/scores/compare/0.9.2...0.9.3). Below are the changes we think users may wish to be aware of.

### Breaking Changes

- Renamed and relocated function `scores.continuous.correlation` to `scores.continuous.correlation.pearsonr`. See [PR #583](https://github.com/nci/scores/pull/583) by [@nicholasloveday](https://github.com/nicholasloveday). 

### Documentation

- Added "Dimension Handling" tutorial, which describes reducing and preserving dimensions. See [PR #589](https://github.com/nci/scores/pull/589) by [@nicholasloveday](https://github.com/nicholasloveday).
- Updated "Detailed Installation Guide" with information on installing kernels in a Jupyter environment. See [PR #586](https://github.com/nci/scores/pull/586) by [@tennlee](https://github.com/tennlee) and [PR #587](https://github.com/nci/scores/pull/587) by [@Steph-Chong](https://github.com/Steph-Chong).

### Internal Changes

- Introduced pinned versions for dependencies on main. See [PR #580](https://github.com/nci/scores/pull/580)  by [@tennlee](https://github.com/tennlee).

