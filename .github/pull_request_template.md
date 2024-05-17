Please work through the following checklists. Delete anything that isn't relevant.
## Development for new xarray-based metrics
- [ ] Works with n-dimensional data and includes `reduce_dims`, `preserve_dims`, and `weights` args.
- [ ] Typehints added
- [ ] Docstrings complete and followed Napoleon (google) style
- [ ] Reference to paper/webpage is in docstring
- [ ] Add error handling
- [ ] Imported into the API

## Testing of new xarray-based metrics
- [ ] 100% unit test coverage
- [ ] Test that metric is compatible with dask.
- [ ] Test that metrics work with inputs that contain NaNs
- [ ] Test that broadcasting with xarray works
- [ ] Test both reduce and preserve dims arguments work
- [ ] Test that errors are raised as expected
- [ ] Test that it works with both `xr.Dataarrays` and `xr.Datasets`

## Tutorial notebook 
- [ ] Short introduction to why you would use that metric and what it tells you
- [ ] A link to a reference
- [ ] A "things to try next" section at the end
- [ ] Add notebook to [Explanation.ipynb](https://github.com/nci/scores/blob/develop/tutorials/Explanation.ipynb)
- [ ] Optional - a detailed discussion of how the metric works at the end of the notebook

## Documentation
- [ ] Add the score to the [API documentation](https://github.com/nci/scores/blob/develop/docs/api.md)
- [ ] Add the score to the [included list of metrics and tools](https://github.com/nci/scores/blob/develop/docs/included.md)
