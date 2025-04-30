# Data Sources

## Introduction

All metrics, statistical techniques and data processing tools in `scores` work with [xarray](https://xarray.dev). [Some metrics](included.md#pandas) work with [pandas](https://pandas.pydata.org/). 

As such, `scores` works with any data source for which xarray or pandas can be used.

Users will need to supply the dataset(s) they wish to work with, as `scores` does not contain datasets.

Data referred to on this page is available under various licenses, and the onus is on the user to understand the conditions of those licenses.

For additional information about downloading and preparing sample data, see [this tutorial](project:./tutorials/First_Data_Fetching.md).

## Working with Different File Formats

### Working with GRIB Data

To use `scores` with [GRIB](https://codes.wmo.int/grib2) data, install [cfgrib](https://github.com/ecmwf/cfgrib) and use `engine='cfgrib'` when opening a GRIB file with xarray.

### Working with NetCDF Data

To use `scores` with [NetCDF](https://doi.org/10.5065/D6H70CW6) or [HDF5](https://github.com/HDFGroup/hdf5) data, install [h5netcdf](https://github.com/h5netcdf/h5netcdf). The h5netcdf library is included in the `scores` ["all"](installation.md#all-dependencies-excludes-some-maintainer-only-packages) and ["tutorial"](installation.md#tutorial-dependencies) installation options. Opening NetCDF data is demonstrated in [this tutorial](project:./tutorials/First_Data_Fetching.md).

## Weather and Climate Data

This section provides a brief overview of some commonly used weather and climate datasets, and software packages for accessing such data. All datasets and software packages listed below are available free of charge.

### Datasets

#### Gridded Global Numerical Weather Prediction Data

Global numerical weather prediction (NWP) models are used to generate medium range forecasts and provide the initial and boundary conditions for higher-resolution regional models. Their global coverage makes them a good starting point for demonstrating the application of scoring methods in any region of interest.

Archived datasets are available for:

- Bureau of Meteorology's Australian Parallel Suite version 3 (APS3) Australian Community Climate and Earth-System Simulator (ACCESS), see [https://doi.org/10.25914/608a993391647](https://doi.org/10.25914/608a993391647).
- [WeatherBench 2](https://weatherbench2.readthedocs.io/en/latest/data-guide.html) contains forecasts with data-driven (AI) and physical NWP models on a common grid. The [twCRPS for ensemble forecasts tutorial](project:./tutorials/Threshold_Weighted_CRPS_for_Ensembles.md) shows how to use this data with `scores`.
- National Oceanic and Atmospheric Administration (NOAA) Global Forecast System (GFS), see [https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast](https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast).
- An archive of AI weather models going back to October 2020 is hosted at [https://noaa-oar-mlwp-data.s3.amazonaws.com/index.html](https://noaa-oar-mlwp-data.s3.amazonaws.com/index.html) as part of the Open Data Dissemination program. It contains, FourCastNet v2-small, Pangu-Weather, and GraphCast Operational data. It is updated twice a day. You can read more about it in their [paper](https://doi.org/10.1175/BAMS-D-24-0057.1).

#### Point-Based Data

Point-based observations (e.g. from weather stations or buoys) are shared routinely between countries for the purposes of weather modelling.

- The NOAA Integrated Surface Database (ISD) provides hourly point-based (*in-situ*) weather station data globally. It is a good starting point for understanding how to work with point-based data. For more information about the NOAA ISD see [https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database](https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database).
- [WeatherReal](https://github.com/microsoft/WeatherReal-Benchmark) contains quality controlled weather station data that uses the ISD. You can read more about WeatherReal in the [pre-print](https://doi.org/10.48550/arXiv.2409.09371).
- The [Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu/) contains a rich variety of datasets. One particularly useful dataset is the [1-minute Automated Surface Observing Network (ASOS) data](https://mesonet.agron.iastate.edu/request/asos/1min.phtml).

#### Gridded Model Reanalysis Data

Reanalysis datasets provide a reliable and detailed reconstruction of past weather and climate conditions, spanning years if not decades.

The ECMWF Reanalysis v5 (ERA5) dataset is a well known and widely used global reanalysis dataset. For more information see [https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5). ERA5 is also included in [WeatherBench 2](https://sites.research.google/weatherbench/), see [this section](https://weatherbench2.readthedocs.io/en/latest/data-guide.html#era5) of their documentation.

#### Gridded Radar (Observation) Data

Radar data provides remotely sensed precipitation estimates at high spatial and temporal resolution. Radar data varies according to region and is not a globally standardised dataset. 

Information on Australian radar data can be found at [https://www.openradar.io/](https://www.openradar.io/).

### Software for Accessing Data

#### CliMetLab

The European Centre for Medium-Range Weather Forecasts (ECMWF) has developed the CliMetLab Python package to simplify access to a large range of climatological and meteorological datasets. See [https://climetlab.readthedocs.io/](https://climetlab.readthedocs.io/).

#### WeatherBench 2

WeatherBench 2 provides a framework for evaluating and comparing a range of machine learning (ML) and physics-based weather forecasting models. It includes ground-truth and baseline [datasets](https://weatherbench2.readthedocs.io/en/latest/data-guide.html) (including ERA5), and code for evaluating models. The [website](https://sites.research.google/weatherbench/) includes scorecards measuring the skill of ML and physics-based models. For more information see [https://sites.research.google/weatherbench/](https://sites.research.google/weatherbench/) and [https://github.com/google-research/weatherbench2](https://github.com/google-research/weatherbench2).



