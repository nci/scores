# Overview of Some Relevant Data Sources

Weather data is often global, big and complex. This extremely brief section serves to point out how to obtain some 'getting started' data for examining the use of the scores and metrics contained in this package. Data referred to here is available under various licenses, and the onus is on the user to understand the conditions of those licenses. The tutorials and walkthroughs in the 'tutorials' directory contain more information and explore the data in more depth.

This page describes data sets and will be improved to provide more specific instructions on downloading and preparing the data in accordance with the roadmap. For the moment, key data sets which have global coverage and are easily accessible are noted.

## Bureau of Meteorology Gridded model (prediction) data
The APS3 ACCESS Numerical Weather Prediction Models provide global model data. Global models are lower resolution and less accurate than those used for short-term weather forecasting. Global provide the initial conditions for higher-resolution regional models. Their global coverage makes them a good starting point for demonstrating the application of scoring methods in any region of interest.

See https://geonetwork.nci.org.au/geonetwork/srv/eng/catalog.search#/metadata/f3307_5503_1483_3079 for more information.

## NOAA Gridded model (prediction) data
The NOAA Global Forecast System provides global scale model data. Global models are lower resolution and less accurate than those used for short-term weather forecasting. Global provide the initial conditions for higher-resolution regional models. Their global coverage makes them a good starting point for demonstrating the application of scoring methods in any region of interest.

See https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast for more information.

## CliMetLab
A python package developed by ECMWF to access a large range of climate and meteorological datasets. See https://climetlab.readthedocs.io/en/latest/

## NOAA ISD dataset
The NOAA Integrated Surface Database provides hourly point-based (aka in-situ) data globally and is a good starting point for understanding how to work with point-based data. Point-based observations are shared routinely between countries for the purposes of weather modelling.

See https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database for more information.

### Gridded model raanalysis data
Reanalysis data is useful for provide a very long history of data, estimating the atmospheric conditions over history. The ERA5 dataset is the best known global reanalysis data set.

https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5
https://github.com/pangeo-data/WeatherBench

## Gridded satellite (observation) data
Satellite data varies according to region, type and age of satellite. It is too complex to quickly address in a demonstration. A guide on working with satellite data may be added in future if it is highlighted in the examples for specific scores.

## Gridded radar (observation) data
Radar data also varies according to region and is not a globally standardised data set. Information on Australian based radars can be found at https://www.openradar.io/


