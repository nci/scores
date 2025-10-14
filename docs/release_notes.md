# Release Notes (What's New)

## Version 2.3.0 (October 14, 2025)

For a list of all changes in this release, see the [full changelog](https://github.com/nci/scores/compare/2.2.0...2.3.0). Below are the changes we think users may wish to be aware of.

### Features

- Added a new metric:
	- Percent within X: `scores.continuous.percent_within_x`. See [PR #865](https://github.com/nci/scores/pull/865).
- Added one new metric and two supporting functions. Following the publication of [Taggart & Wilke (2025)](https://doi.org/10.5194/nhess-25-2657-2025), these have been moved from `scores.emerging` to `scores.categorical`:
	- Risk matrix score: `scores.categorical.risk_matrix_score`.
	- Risk matrix score - matrix weights to array: `scores.categorical.matrix_weights_to_array`.
	- Risk matrix score - warning scaling to weight array: `scores.categorical.weights_from_warning_scaling`.  
	***Note:** while removing the functions from `scores.emerging` is technically a breaking change, breaking changes that only impact the "emerging" section of the API do not trigger major releases. This is because the "emerging" section of the API is designed to hold metrics while they are undergoing peer review and it is expected they will be moved out of "emerging" once peer review has concluded.*  
	See [PR #904](https://github.com/nci/scores/pull/904).
- Updated the weighting method used by all `scores` functions that allow the user to supply weights. The updated weighting method normalises the user-supplied weights rather than applying them directly. While both approaches can be valid, the revised approach is more in keeping with general expectations and is conistent with the default approach taken by other libraries. As a part of this change, users can no longer supply weights that contain NaNs (zeroes may be used instead where appropriate). The ["Introduction to weighting and masking" tutorial](https://scores.readthedocs.io/en/stable/tutorials/Weighting_Results.html) has been updated and substantially expanded to explain what the weighting does mathematically. See [PR #899](https://github.com/nci/scores/pull/899).
- Added optional automatic generation of thresholds for the receiver (relative) operating characteristic (ROC) curve (`scores.probability.roc_curve_data`). See [PR #882](https://github.com/nci/scores/pull/882). 


### Bug Fixes

- Updated `scores.continuous.quantile_interval_score` so it now recognises `preserve_dims='all'`. Beforehand, it was not recognising the special case of `preserve_dims='all'` and was raising an error unless a list of dimensions was supplied. (*Note:* the score calculations were not incorrect, it was only that `preserve_dims='all'` was not recognised.) See [PR #893](https://github.com/nci/scores/pull/893).

### Documentation

- Added "Percent Within X" tutorial. See [PR #865](https://github.com/nci/scores/pull/865). 
- Substantially updated and expanded the "Introduction to weighting and masking" tutorial, following changes to the weighting method used by all `scores` functions that allow the user to supply weights. The updated and expanded tutorial explains what the weighting does mathematically. See [PR #899](https://github.com/nci/scores/pull/899).
- Updated the "Quantile-Quantile (Q-Q) Plots for Comparing Forecasts and Observations" tutorial so that the plots render in Read the Docs. See [PR #883](https://github.com/nci/scores/pull/883).
- Updated the description of the second figure in the "Threshold Weighted Continuous Ranked Probability Score (twCRPS) for ensembles" tutorial. See [PR #897](https://github.com/nci/scores/pull/897).
- Updated multiple sections of the documentation following the risk matrix score moving from `scores.emerging` to `scores.categorical`, including:
	- updating docstrings and `docs/included.md`, 
	- updating the tutorial with the new `categorical` methods, and
	- updating references in several sections of the documentation, following the publication of [Taggart & Wilke (2025)](https://doi.org/10.5194/nhess-25-2657-2025).  
	See [PR #904](https://github.com/nci/scores/pull/904).
- In the README, "Detailed Installation Guide" and "Contributing Guide", updated pip install commands to use quotation marks where square brackets are used to specify optional dependencies. This is to ensure compatibility with zsh (the default on macOS) while still working as expected on bash. See [PR #917](https://github.com/nci/scores/pull/917).
- Added thumbnail images to multiple entries in the tutorial gallery. See [PR #874](https://github.com/nci/scores/pull/874), [PR #875](https://github.com/nci/scores/pull/875), [PR #877](https://github.com/nci/scores/pull/877), [PR #879](https://github.com/nci/scores/pull/879), [PR #880](https://github.com/nci/scores/pull/880), [PR #881](https://github.com/nci/scores/pull/881) and [PR #884](https://github.com/nci/scores/pull/884).

### Internal Changes

- In multiple tutorials, added the keyword argument `decode_timedelta=True` to `xarray.open_dataset` for the downloaded files `forecast_grid.nc` and `analysis_grid.nc`. See [PR #894](https://github.com/nci/scores/pull/894).
- Perform input checking earlier in various function calls to improve efficiency, so that error messages can be raised before incurring computational expenses. See [PR #905](https://github.com/nci/scores/pull/905).

### Contributors to this Release

Thomas C. Pagano* ([@thomaspagano](https://github.com/thomaspagano)), Paul R. Smith* ([@prs247au](https://github.com/prs247au)), J. Smallwood* ([@jdgsmallwood](https://github.com/jdgsmallwood)), Tennessee Leeuwenburg ([@tennlee](https://github.com/tennlee)), Nicholas Loveday ([@nicholasloveday](https://github.com/nicholasloveday)), Nikeeth Ramanathan ([@nikeethr](https://github.com/nikeethr)), Stephanie Chong ([@Steph-Chong](https://github.com/Steph-Chong)), Robert J. Taggart ([@rob-taggart](https://github.com/rob-taggart)) and Mohammadreza Khanarmuei ([@reza-armuei](https://github.com/reza-armuei)).

\* indicates that this release contains their first contribution to `scores`.

## Version 2.2.0 (July 26, 2025)

For a list of all changes in this release, see the [full changelog](https://github.com/nci/scores/compare/2.1.0...2.2.0). Below are the changes we think users may wish to be aware of.

### Features

- Added a new metric:
	- Spearman’s correlation coefficient: `scores.continuous.correlation.spearmanr`. See [PR #773](https://github.com/nci/scores/pull/773).
- Added a new function for generating data for diagrams:
    - Quantile-Quantile (QQ) plots: `scores.plotdata.qq`. See [PR #852](https://github.com/nci/scores/pull/852).
- Added new features to the FIxed Risk Multicategorical (FIRM) score (`scores.categorical.firm`):
	- Added support for xr.Datasets in addition to the existing support for xr.DataArrays. See [PR #853](https://github.com/nci/scores/pull/853).
	- Added the optional argument `include_components`. If `include_components` is set to `True` the function will return the overforecast and underforecast penalties along with the FIRM score.
	See See [PR #853](https://github.com/nci/scores/pull/853) and [PR #864](https://github.com/nci/scores/pull/864).
- Added a new `scores.plotdata` section to the API for functions that generate data for verification plots. See [PR #852](https://github.com/nci/scores/pull/852).

### Bug Fixes

- Fixed an issue where `scores.plotdata.roc` didn't add the point (0, 0) in some instances. See [PR #863](https://github.com/nci/scores/pull/863).
- Fixed an issue in `scores.continuous.quantile_interval_score` where broadcasting wasn't being done correctly in some cases. See [PR #867](https://github.com/nci/scores/pull/867).

### Documentation

- Added two new tutorials:
	- "Spearman’s Correlation Coefficient". See [PR #773](https://github.com/nci/scores/pull/773).
	- "Quantile-Quantile (Q-Q) Plots for Comparing Forecasts and Observations". See [PR #852](https://github.com/nci/scores/pull/852).
- Substantially updated "The FIxed Risk Multicategorical (FIRM) Score" tutorial. See [PR #853](https://github.com/nci/scores/pull/853).
- Fixed an error in the formula in the docstring for the quantile interval score (`scores.continuous.quantile_interval_score`). (*Note:* this error was only present in the docstring - the code implemenation of the function was correct and the tutorial listed the correct formula.) See [PR #851](https://github.com/nci/scores/pull/851).
- Updated several "full changelog" URLs in the release notes. See [PR #859](https://github.com/nci/scores/pull/859).

### Internal Changes

- Improved the efficiency of the FIxed Risk Multicategorical (FIRM) score (`scores.categorical.firm`) by moving the call to gather dimensions to earlier within the method. See [PR #853](https://github.com/nci/scores/pull/853).
- Added a new `scores.plotdata` section to the API for functions that generate data for verification plots. See [PR #852](https://github.com/nci/scores/pull/852). The following internal changes were made:
	- Receiver (Relative) Operating Characteristic (ROC):
		- `scores.probability.roc_curve_data` was moved to `scores.plotdata.roc`, but can still be imported as `scores.probability.roc_curve_data`.
	- Murphy Score:
		- `scores.continuous.murphy_score` was moved to `scores.plotdata.murphy_score`, but can still be imported as `scores.continuous.murphy_score` and  `scores.probability.murphy_score`.
		- `scores.continuous.murphy_thetas` was moved to `scores.plotdata.murphy_thetas`, but can still be imported as `scores.continuous.murphy_thetas` and `scores.probability.murphy_thetas`.
- Added an additional CI/CD pipeline for testing without Dask. See [PR #856](https://github.com/nci/scores/pull/856).

### Contributors to this Release

Liam Bluett ([@lbluett](https://github.com/lbluett)), Nicholas Loveday ([@nicholasloveday](https://github.com/nicholasloveday)), Nikeeth Ramanathan ([@nikeethr](https://github.com/nikeethr)), Tennessee Leeuwenburg ([@tennlee](https://github.com/tennlee)), Robert J. Taggart ([@rob-taggart](https://github.com/rob-taggart)), Stephanie Chong ([@Steph-Chong](https://github.com/Steph-Chong)) and Mohammadreza Khanarmuei ([@reza-armuei](https://github.com/reza-armuei)).

## Version 2.1.0 (April 30, 2025)

For a list of all changes in this release, see the [full changelog](https://github.com/nci/scores/compare/2.0.0...2.1.0). Below are the changes we think users may wish to be aware of.

### Features

- Added a new fuction:
	- Block bootstrap: `scores.processing.block_bootstrap`. See [PR #418](https://github.com/nci/scores/pull/418).
- Added two new metrics:
	- Stable equitable error in probability space (SEEPS): `scores.categorical.seeps`. See [PR #809](https://github.com/nci/scores/pull/809) and [PR #833](https://github.com/nci/scores/pull/833).
	- Nash-Sutcliffe model efficiency coefficient (NSE): `scores.continuous.nse`. See [PR #815](https://github.com/nci/scores/pull/815).

### Documentation

- Added "Block Bootstrapping" tutorial. See [PR #418](https://github.com/nci/scores/pull/418).
- Added "Stable Equitable Error in Probability Space (SEEPS)" tutorial. See [PR #809](https://github.com/nci/scores/pull/809).
- Added "Nash-Sutcliffe Efficiency (NSE)" tutorial. See [PR #815](https://github.com/nci/scores/pull/815).
- Updated the "Continuous Ranked Probability Score (CRPS) for Ensembles" tutorial:
	- Labelled dimensions in fcst/obs data.
	- Updated description of the plot to say the area squared corresponds to the CRPS.
	- Added an example with multiple coordinates along a dimension.
	See [PR #805](https://github.com/nci/scores/pull/805).
- Updated "Data Sources":
	- Added links to two additional datasets for gridded global numerical weather prediction.
	- Added links to several additional datasets for point-based data.
	See [PR #823](https://github.com/nci/scores/pull/823) and [PR #831](https://github.com/nci/scores/pull/831).
- Updated references in several sections of the documentation, following the publication of a [preprint](https://doi.org/10.48550/arXiv.2502.08891) for the risk matrix score. See [PR #827](https://github.com/nci/scores/pull/827).

### Internal Changes

- Tested and added compatibility for recent Xarray versions (2025 and onwards) and adjusted dependency specification so new year "major version" rollovers will be permitted by default in future. See [commit #f109f2f](https://github.com/nci/scores/commit/f109f2f434ac684b3d54f447c330466d33703279) and [commit #8428d64](https://github.com/nci/scores/commit/8428d64dcf2a5f5480c61b266284260d4b5078d2).
- In `scores.emerging.weights_from_warning_scaling`, changed the name of the argument `assessment_weights` to  `evaluation_weights`. See [PR #806](https://github.com/nci/scores/issues/806).
***Note:** This is technically a breaking change, but does not trigger a major release as it is contained within the "emerging" section of the API. This area of the API is designated for metrics which are still undergoing peer review and as such are expected to undergo change. Once peer review is concluded, the implementation will be finalised and moved.*
- Add support for developers of `scores` who choose to use the `pixi` tool for environment management. See [PR #835](https://github.com/nci/scores/pull/835), [PR #839](https://github.com/nci/scores/pull/839) and [PR #840](https://github.com/nci/scores/pull/840).

### Contributors to this Release

Dougal T. Squire* ([@dougiesquire](https://github.com/dougiesquire)), Mohammad Mahadi Hasan* ([@engrmahadi](https://github.com/engrmahadi)), Mohammadreza Khanarmuei ([@reza-armuei](https://github.com/reza-armuei)), Nikeeth Ramanathan ([@nikeethr](https://github.com/nikeethr)) Tennessee Leeuwenburg ([@tennlee](https://github.com/tennlee)), Nicholas Loveday ([@nicholasloveday](https://github.com/nicholasloveday)),
Robert J. Taggart ([@rob-taggart](https://github.com/rob-taggart)), Durga Shrestha ([@durgals](https://github.com/durgals)) and Stephanie Chong ([@Steph-Chong](https://github.com/Steph-Chong)).

\* indicates that this release contains their first contribution to `scores`.

## Version 2.0.0 (December 7, 2024)

For a list of all changes in this release, see the [full changelog](https://github.com/nci/scores/compare/1.3.0...2.0.0). Below are the changes we think users may wish to be aware of.

### Breaking Changes

- The function `scores.probability.tw_crps_for_ensemble` previously took an optional (mis-spelled) argument `chainging_func_kwargs`. The spelling has been corrected and the argument is now `chaining_func_kwargs`. See [PR #780](https://github.com/nci/scores/pull/780) and [PR #772](https://github.com/nci/scores/pull/772).
- For those who develop on `scores`, you will need to update your installation of the `scores` package with `pip install -e .[all]`, to get updated versions of `black`, `pylint` and `mypy`. See [PR #768](https://github.com/nci/scores/pull/768), [PR #769](https://github.com/nci/scores/pull/769) and [PR #771](https://github.com/nci/scores/pull/771).

### Features

- Added three new metrics:
	- Brier score for ensembles: `scores.probability.brier_score_for_ensemble`. See [PR #735](https://github.com/nci/scores/pull/735).
	- Negative predictive value: `scores.categorical.BasicContingencyManager.negative_predictive_value`. See [PR #759](https://github.com/nci/scores/pull/759).
	- Positive predictive value: `scores.categorical.BasicContingencyManager.positive_predictive_value`. See [PR #761](https://github.com/nci/scores/pull/761) and [PR #756](https://github.com/nci/scores/pull/756).
- Also added one new emerging metric and two supporting functions:
	- Risk matrix score: `scores.emerging.risk_matrix_scores`.
	- Risk matrix score - matrix weights to array: `scores.emerging.matrix_weights_to_array`.
	- Risk matrix score - warning scaling to weight array: `scores.emerging.weights_from_warning_scaling`.
	See [PR #724](https://github.com/nci/scores/pull/724) and [PR #794](https://github.com/nci/scores/pull/794).
- A new method called `format_table` was added to the class `BasicContingencyManager` to improve visualisation of 2x2 contingency tables. The tutorial `Binary_Contingency_Scores` was updated to demonstrate the use of this function. See [PR #775](https://github.com/nci/scores/pull/775).
- The functions `scores.processing.comparative_discretise`, `scores.processing.binary_discretise` and `scores.processing.binary_discretise_proportion` now accept either a string indicating the choice of operator to be used, or an operator from the [Python core library `operator` module](https://docs.python.org/3/library/operator.html). Using one of the operators from the Python core module is recommended, as doing so is more reliable for a variety of reasons. Support for the use of a string may be removed in future. See [PR #740](https://github.com/nci/scores/pull/740) and [PR #758](https://github.com/nci/scores/pull/758).

### Documentation

- Added "The Risk Matrix Score" tutorial. See [PR #724](https://github.com/nci/scores/pull/724) and [PR #794](https://github.com/nci/scores/pull/794).
- Updated the "Brier Score" tutorial to include a new section about the Brier score for ensembles. See [PR #735](https://github.com/nci/scores/pull/735).
- Updated the "Binary Categorical Scores and Binary Contingency Tables (Confusion Matrices)"
  tutorial:
  - Included "positive predictive value" in the list of binary categorical scores.
  - Included "negative predictive value" in the list of binary categorical scores.
  - Demonstrated the use of the new `format_table` method for visualising 2x2 contingency tables.
  See [PR #759](https://github.com/nci/scores/pull/759) and [PR #775](https://github.com/nci/scores/pull/775).
- Updated the “Contributing Guide”:
	- Added a new section: "Creating Your Own Fork of `scores` for the First Time".
	- Updated the section: "Workflow for Submitting Pull Requests".
	- Added a new section: "Pull Request Etiquette".
	See [PR #787](https://github.com/nci/scores/pull/787).
- Updated the README:
	- Added a link to a video of a PyCon AU 2024 conference presentation about `scores`. See [PR #783](https://github.com/nci/scores/pull/783).
	- Added a link to the archives of `scores` on Zenodo. See [PR #784](https://github.com/nci/scores/pull/784).
- Added `Scoringrules` to "Related Works". See [PR #746](https://github.com/nci/scores/pull/746), [PR #766](https://github.com/nci/scores/pull/766) and [PR #789](https://github.com/nci/scores/pull/789).

### Internal Changes

- Removed scikit-learn as a dependency. `scores` has replaced the use of scikit-learn with a similar function from SciPy (which was an existing `scores` dependency). This change was manually tested and found to be faster. See [PR #774](https://github.com/nci/scores/pull/774).
- Version pinning of dependencies in release files (the wheel and sdist files used by PyPI and conda-forge) is now managed and set by the `hatch_build` script. This allows development versions to be free-floating, while being more specific about dependencies in releases. The previous process also aimed to do this, but was error-prone. A new entry called `pinned_dependencies` was added to pyproject.toml to specify the release dependencies. See [PR #760](https://github.com/nci/scores/pull/760).

### Contributors to this Release

Arshia Sharma* ([@arshiaar](https://github.com/arshiaar)), A.J. Fisher* ([@AJTheDataGuy](https://github.com/AJTheDataGuy)), Liam Bluett* ([@lbluett](https://github.com/lbluett)), Jinghan Fu* ([@JinghanFu](https://github.com/JinghanFu)), Sam Bishop* ([@techdragon](https://github.com/techdragon)), Robert J. Taggart ([@rob-taggart](https://github.com/rob-taggart)), Tennessee Leeuwenburg ([@tennlee](https://github.com/tennlee)), Stephanie Chong ([@Steph-Chong](https://github.com/Steph-Chong)) and
Nicholas Loveday ([@nicholasloveday](https://github.com/nicholasloveday)).

\* indicates that this release contains their first contribution to `scores`.

## Version 1.3.0 (November 15, 2024)

For a list of all changes in this release, see the [full changelog](https://github.com/nci/scores/compare/1.2.0...1.3.0). Below are the changes we think users may wish to be aware of.

### Introduced Support for Python 3.13 and Dropped Support for Python 3.9

- In line with other scientific Python packages, `scores` has dropped support for Python 3.9 in this release.
  `scores` has added support for Python 3.13. See [PR #710](https://github.com/nci/scores/pull/710).

### Features

- Added four new metrics:
	- Quantile Interval Score: `scores.continuous.quantile_interval_score`. See [PR #704](https://github.com/nci/scores/pull/704), [PR #733](https://github.com/nci/scores/pull/733) and [PR #738](https://github.com/nci/scores/pull/738).
	- Interval Score: `scores.continuous.interval_score`. See [PR #704](https://github.com/nci/scores/pull/704), [PR #733](https://github.com/nci/scores/pull/733) and [PR #738](https://github.com/nci/scores/pull/738).
	- Kling-Gupta Efficiency (KGE): `scores.continuous.kge`. See [PR #679](https://github.com/nci/scores/pull/679), [PR #700](https://github.com/nci/scores/pull/700) and [PR #734](https://github.com/nci/scores/pull/734).
	- Interval threshold weighted continuous ranked probability score (twCRPS) for ensembles: `scores.probability.interval_tw_crps_for_ensemble`. See [PR #682](https://github.com/nci/scores/pull/682) and [PR #734](https://github.com/nci/scores/pull/734).
- Added an optional `include_components` argument to several continuous ranked probability score (CRPS) functions for ensembles. If supplied, the `include_components` argument will return the underforecast penalty, the overforecast penalty and the forecast spread term, in addition to the overall CRPS value. This applies to the following CRPS functions:
	- continuous ranked probability score (CRPS) for ensembles: `scores.probability.crps_for_ensemble`
	- threshold weighted continuous ranked probability score (twCRPS) for ensembles: `scores.probability.tw_crps_for_ensemble`
	- tail threshold weighted continuous ranked probability score (twCRPS) for ensembles: `scores.probability.tail_tw_crps_for_ensemble`
	- interval threshold weighted continuous ranked probability score (twCRPS) for ensembles: `scores.probability.interval_tw_crps_for_ensemble`)
	See [PR #708](https://github.com/nci/scores/pull/708) and [PR #734](https://github.com/nci/scores/pull/734).

### Documentation

- Added "Kling–Gupta Efficiency (KGE)" tutorial. See [PR #679](https://github.com/nci/scores/pull/679), [PR #700](https://github.com/nci/scores/pull/700) and [PR #734](https://github.com/nci/scores/pull/734).
- Added "Quantile Interval Score and Interval Score" tutorial. See [PR #704](https://github.com/nci/scores/pull/704), [PR #736](https://github.com/nci/scores/pull/736) and [PR #738](https://github.com/nci/scores/pull/738).
- Added "Threshold Weighted Continuous Ranked Probability Score (twCRPS) for ensembles" tutorial. See [PR #706](https://github.com/nci/scores/pull/706) and [PR #722](https://github.com/nci/scores/pull/722).
- Updated the title in the "Binary Categorical Scores and Binary Contingency Tables (Confusion Matrices)" tutorial and the description for the corresponding thumbnail in the tutorial gallery. See [PR #741](https://github.com/nci/scores/pull/741) and [PR #743](https://github.com/nci/scores/pull/743).
- Updated the pull request template. See [PR #719](https://github.com/nci/scores/pull/719).

### Internal Changes

- Sped up (improved the computational efficiency of) the continuous ranked probability score (CRPS) for ensembles. This also addresses memory issues when a large number of ensemble members are present. See [PR #694](https://github.com/nci/scores/pull/694).

### Contributors to this Release

Mohammadreza Khanarmuei ([@reza-armuei](https://github.com/reza-armuei)), Nicholas Loveday ([@nicholasloveday](https://github.com/nicholasloveday)), Durga Shrestha ([@durgals](https://github.com/durgals)), Tennessee Leeuwenburg ([@tennlee](https://github.com/tennlee)), Stephanie Chong ([@Steph-Chong](https://github.com/Steph-Chong)) and Robert J. Taggart ([@rob-taggart](https://github.com/rob-taggart)).

## Version 1.2.0 (September 13, 2024)

For a list of all changes in this release, see the [full changelog](https://github.com/nci/scores/compare/1.1.0...1.2.0). Below are the changes we think users may wish to be aware of.

### Features

- Added three new metrics:
	- Percent bias (PBIAS): `scores.continuous.pbias`. See [PR #639](https://github.com/nci/scores/pull/639) and [PR #655](https://github.com/nci/scores/pull/655).
	- Threshold weighted continuous ranked probability score (twCRPS) for ensembles: `scores.probability.tw_crps_for_ensemble`. See [PR #644](https://github.com/nci/scores/pull/644).
	- Tail threshold weighted continuous ranked probability score (twCRPS) for ensembles: `scores.probability.tail_tw_crps_for_ensemble`. See [PR #644](https://github.com/nci/scores/pull/644).
- The FIxed Risk Multicategorical (FIRM) score (`scores.categorical.firm`) can now take a sequence of mulitdimensional arrays (xr.DataArray) of thresholds. This allows the FIRM score to be used with categorical thresholds that vary across the domain. See [PR #661](https://github.com/nci/scores/pull/661).

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

For a list of all changes in this release, see the [full changelog](https://github.com/nci/scores/compare/0.6...0.7).

## Version 0.6 (April 6, 2024)

For a list of all changes in this release, see the [full changelog](https://github.com/nci/scores/compare/0.5...0.6).

Note: version 0.6 was initially tagged as "v0.6" and released on 6th April 2024. On 7th April 2024, an identical version was released with the tag "0.6" (i.e. with the "v" ommitted from the tag).

## Version 0.5 (April 6, 2024)

For a list of all changes in this release, see the [full changelog](https://github.com/nci/scores/compare/0.4...0.5).

## Version 0.4 (September 15, 2023)

For a list of all changes in this release, see the [full changelog](https://github.com/nci/scores/compare/0.0.2...0.4).

## Version 0.0.2 (June 9, 2023)

For a list of all changes in this release, see the [full changelog](https://github.com/nci/scores/commits/0.0.2).

## Version 0.0.1 (January 16, 2023)

Version 0.0.1 was released on PyPI as a placeholder, while very early development and package design was being undertaken.
