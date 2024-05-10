# Data Sources

Overview of Some Relevant Data Sources

This section suggests how to obtain some 'getting started' weather and climate data for examining the use of the scores and metrics contained in this package. Data referred to here is available under various licenses, and the onus is on the user to understand the conditions of those licenses. The tutorials and walkthroughs in the 'tutorials' directory contain more information and explore the data in more depth.

This page will be improved to provide more specific instructions on downloading and preparing the data in accordance with the scores roadmap. For the moment, it notes a few key datasets which have global coverage and are easily accessible.

## Gridded global numerical weather prediction data
Global weather prediction models are used to generate medium range forecasts and provide the initial and boundary conditions for higher-resolution regional models. Their global coverage makes them a good starting point for demonstrating the application of scoring methods in any region of interest.

The Bureau of Meteorology provides global model data from the ACCESS numerical weather prediction system. See https://geonetwork.nci.org.au/geonetwork/srv/eng/catalog.search#/metadata/f3307_5503_1483_3079 for more information.

Global model data is also available from the NOAA Global Forecast System (GFS). See https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast for more information.

## CliMetLab
ECMWF has developed the CliMetLab python package to simplify access a large range of climate and meteorological datasets. See https://climetlab.readthedocs.io/en/latest/

## NOAA ISD dataset
The NOAA Integrated Surface Database provides hourly point-based (aka in-situ) data globally and is a good starting point for understanding how to work with point-based data. Point-based observations are shared routinely between countries for the purposes of weather modelling.

See https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database for more information.

## Gridded model reanalysis data
Reanalysis data is useful for providing a very long history of atmospheric conditions. The ERA5 dataset is a well known and widely used global reanalysis dataset.

For more information see https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5 and
https://github.com/pangeo-data/WeatherBench

## Gridded satellite (observation) data
Satellite data varies according to the type of orbit and purpose of the mission. It is too complex to quickly address in a demonstration. A guide on working with satellite data may be added in future.

## Gridded radar (observation) data
Radar data also varies according to region and is not a globally standardised data set. Information on Australian based radars can be found at https://www.openradar.io/

## Working with GRIB data
To use `scores` with GRIB data, install [cfgrib](https://github.com/ecmwf/cfgrib) and use `engine='cfgrib'` when opening a grib file with `xarray`.

## Working with NetCDF data
To use `scores` with NetCDF or HDF5 data, install [h5netcdf](https://github.com/h5netcdf/h5netcdf). Opening NetCDF data is demonstrated in the notebook tutorials and the `h5netcdf` library is included in the tutorial dependencies.
