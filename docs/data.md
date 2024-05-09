# Data Sources

## Overview of some relevant data sources

Weather data is often global, big and complex. This brief section suggests how to obtain some 'getting started' data for examining the use of the scores and metrics contained in this package. Data referred to here is available under various licenses, and the onus is on the user to understand the conditions of those licenses. The tutorials and walkthroughs in the 'tutorials' directory contain more information and explore the data in more depth.

This page describes data sets and will be improved to provide more specific instructions on downloading and preparing the data in accordance with the scores roadmap. For the moment, key data sets which have global coverage and are easily accessible are noted.

## Gridded global numerical weather prediction data
Global weather prediction models are used for medium range forecasts and provide the initial and boundary conditions for higher-resolution regional models. Their global coverage makes them a good starting point for demonstrating the application of scoring methods in any region of interest.

The Bureau of Meteorology provides global model data from the ACCESS numerical weather prediction system. See https://geonetwork.nci.org.au/geonetwork/srv/eng/catalog.search#/metadata/f3307_5503_1483_3079 for more information.

Global model data is also available from the NOAA Global Forecast System (GFS). See https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast for more information.

## CliMetLab
ECMWF has developed the CliMetLab python package to simplify access a large range of climate and meteorological datasets. See https://climetlab.readthedocs.io/en/latest/

## NOAA ISD dataset
The NOAA Integrated Surface Database provides hourly point-based (aka in-situ) data globally and is a good starting point for understanding how to work with point-based data. Point-based observations are shared routinely between countries for the purposes of weather modelling.

See https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database for more information.

## Gridded model reanalysis data
Reanalysis data is useful for provide a very long history of data, estimating the atmospheric conditions over history. The ERA5 dataset is a well known and widely used global reanalysis dataset.

https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5
https://github.com/pangeo-data/WeatherBench

## Gridded satellite (observation) data
Satellite data varies according to region, type and age of satellite. It is too complex to quickly address in a demonstration. A guide on working with satellite data may be added in future if it is highlighted in the examples for specific scores.

## Gridded radar (observation) data
Radar data also varies according to region and is not a globally standardised data set. Information on Australian based radars can be found at https://www.openradar.io/

## Working with GRIB data
To use `scores` with GRIB data, install [cfgrib](https://github.com/ecmwf/cfgrib) and use `engine='cfgrib'` when opening a grib file with `xarray`.

## Working with NetCDF data
To use `scores` with NetCDF or HDF5 data, install [h5netcdf](https://github.com/h5netcdf/h5netcdf). Opening NetCDF data is demonstrated in the notebook tutorials and the `h5netcdf` library is included in the tutorial dependencies.
