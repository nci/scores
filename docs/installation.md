# Detailed Installation Guide

This page described common installation patterns. Expert users of pip and conda will note that there are more variations possible, but these are the most common use cases.

It is recommended to use a virtualised Python environment in most settings. `scores` can be easily installed using vence/pip or conda/pip. The base requirements of the package are kept to a minimum to reduce the likelihood of conflicts. This project relies on a relatively recent version of pip, so you might need to upgrade pip within your virtual environment. If this is required, the installation process will automatically prompt you to do so, including the commands required. This is a simple and reliable step which will apply onto to your virtual environment.

Here is a command to create and activate a new virtual environment with *virtualenv*:

```py
python -m venv <path_to_environment>
source <path_to_environment>/bin/activate
```

Here is a command to create and activate a new virtual environment with *conda*
```py
conda create -p <path_to_enviroment> python=3
conda activate <path_to_environment>

```

Most users want the "all" installation. Some people have good reasons to want a simpler and/or more specialised approach. There are 4 different types of supported installation overall:

- all: includes requirements for core, tutorial and development, but excludes maintainer requirements
- minimal: only contains mathematical functions, with simple dependencies
- tutorial: includes jupyter lab and ability to run all the notebooks
- maintainer: includes tools for building the documentation and building for PyPI

## 1. All Dependencies (excludes some maintainer-only packages)

Use this for scores development and general use.

Installs:
* API code and libraries
* Everything needed to run the tutorials
* Testing, static analysis and other developer libraries
* Does **not** install things for making packages and releasing new versions

### From a Local Checkout of the Git Repository

```bash
pip install -e .[all]
```

## 2. Minimal (Mathematical API Functions Only)
Use this to install the scores code into another package or system.

Installs:
* `scores` package
* Only the required core dependencies, nothing extra, no tutorials

### From PyPI

```bash
pip install scores
```

## 3. Tutorial Environment 
Use this to run a tutorial or lab session without allowing changes to scores itself

Installs:
* core dependencies
* Dependencies for running the tutorial notebooks with jupyter lab

### From a Local Git Repository

```bash
pip install .[tutorial]
```

## 4. Maintainer Environment 
Use this to build the docs, create packages and prepare new versions for release.

Installs:
* core dependencies
* Dependencies for building new versions of the `scores` package
* Dependencies for building the documentation as HTML

### From a Local Git Repository

```bash
pip install -e .[maintainer]
```


