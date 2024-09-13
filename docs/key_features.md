# Key Features of `scores`

## Data Handling

- Works with labelled, n-dimensional data (e.g., geospatial, vertical and temporal dimensions) for both point-based and gridded data. `scores` can effectively handle the [dimensionality](project:./tutorials/Dimension_Handling.md), data size and data structures commonly used for:
  - gridded Earth system data (e.g., numerical weather prediction models)
  - tabular, point, latitude/longitude or site-based data (e.g., forecasts for specific locations).
- Handles missing data, masking of data and [weighting of results](project:./tutorials/Weighting_Results.md) (e.g. by area, by latitude, by population).
- Supports [xarray](https://xarray.dev/) datatypes, and works with [NetCDF4](https://doi.org/10.5065/D6H70CW6), [HDF5](https://github.com/HDFGroup/hdf5), [Zarr](https://zarr.dev) and [GRIB](https://codes.wmo.int/grib2) data formats among others. 

## Usability

- A companion Jupyter Notebook [tutorial](project:./tutorials/Tutorial_Gallery.md) for each metric and statistical test that demonstrates its use in practice.
- [Over 60 metrics, statistical techniques and data processing tools](included.md), including:
  - commonly-used metrics (e.g., mean absolute error (MAE), root mean squared error (RMSE))
  - novel scores not commonly found elsewhere (e.g., FIxed Risk Multicategorical (FIRM) score ([Taggart et al., 2022](https://doi.org/10.1002/qj.4266)), Flip-Flop Index ([Griffiths et al., 2019](https://doi.org/10.1002/met.1732), [2021](https://doi.org/10.1071/ES21010)))
  - recently developed, user-focused scores and diagnostics including:
      - threshold-weighted scores, such as threshold-weighted continuous ranked probability score (twCRPS) [(Gneiting and Ranjan 2011)](https://doi.org/10.1198/jbes.2010.08110) and threshold-weighted mean squared error (MSE) [(Taggart 2022)](https://doi.org/10.1002/qj.4206)
      - Murphy diagrams [(Ehm et al., 2016)](https://doi.org/10.1111/rssb.12154)
      - isotonic regression for reliability diagrams [(Dimitriadis et al., 2021)](https://doi.org/10.1073/pnas.2016191118)
  - statistical tests (e.g., the Diebold-Mariano ([Diebold & Mariano, 1995](https://doi.org/10.1080/07350015.1995.10524599)) test, with both the Harvey et al. ([1997](https://doi.org/10.1016/S0169-2070(96)00719-4)) and Hering & Genton ([2011](https://doi.org/10.1198/tech.2011.10136)) modifications).
- All scores and statistical techniques have undergone a thorough scientific and software review.
- An area specifically to hold emerging scores which are still undergoing research and development. This provides a clear mechanism for people to share, access and collaborate on new scores, and be able to easily re-use versioned implementations of those scores.  

## Compatibility

- Highly modular - provides its own implementations, avoids extensive dependencies and offers a consistent API.
- Easy to integrate and use in a wide variety of environments. It has been used on workstations, servers and in high performance computing (supercomputing) environments. 
- Maintains 100% automated test coverage.
- Uses [Dask](http://dask.pydata.org) for scaling and performance.
- [Some metrics](included.md#pandas) work with [pandas](https://pandas.pydata.org/) and we aim to expand this capability. 