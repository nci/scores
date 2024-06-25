# Key Features of `scores`

## Data Handling

- Works with labelled, n-dimensional data (e.g., geospatial, vertical and temporal dimensions) for both point-based and gridded data. `scores` can effectively handle the dimensionality, data size and data structures commonly used for:
  - gridded Earth system data (e.g., numerical weather prediction models)
  - tabular, point, latitude/longitude or site-based data (e.g., forecasts for specific locations).
- Handles missing data, masking of data and weighting of results.
- Supports [xarray](https://xarray.dev/) datatypes, and works with [NetCDF4](https://doi.org/10.5065/D6H70CW6), [HDF5](https://github.com/HDFGroup/hdf5), [Zarr](https://zarr.dev) and [GRIB](https://codes.wmo.int/grib2) data sources among others. 

## Usability

- A companion Jupyter Notebook [tutorial](project:./tutorials/Tutorial_Gallery.md) for each metric and statistical test that demonstrates its use in practice.
- Novel scores not commonly found elsewhere (e.g., FIxed Risk Multicategorical (FIRM) score ([Taggart et al., 2022](https://doi.org/10.1002/qj.4266)), Flip-Flop Index ([Griffiths et al., 2019](https://doi.org/10.1002/met.1732), [2021](https://doi.org/10.1071/ES21010))). 
- Complex scores (e.g., threshold-weighted continuous ranked probability score (twCRPS))
- Commonly-used scores are also included, so that `scores` can be used as a standalone package.
- All scores and statistical techniques have undergone a thorough scientific and software review.
- An area specifically to hold emerging scores which are still undergoing research and development. This provides a clear mechanism for people to share, access and collaborate on new scores, and be able to easily re-use versioned implementations of those scores.  

## Compatibility

- Highly modular - provides its own implementations, avoids extensive dependencies and offers a consistent API.
- Easy to integrate and use in a wide variety of environments. It has been used on workstations, servers and in high performance computing (supercomputing) environments. 
- Maintains 100% automated test coverage.
- Uses [Dask](http://dask.pydata.org) for scaling and performance.
- Expanding support for [pandas](https://pandas.pydata.org/).
