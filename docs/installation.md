# Detailed Installation Guide

This page describes the most common installation options for `scores`. Expert users of pip and conda will note that there are more variations possible. A section at the end describes how to work effectively in a Jupyter Notebook environment.

## Setting up a Virtual Environment

In almost all cases, it is recommended to use a virtualised Python environment. 

`scores` can be easily installed using either venv/pip or conda/pip. The requirements of `scores` are kept to a minimum to reduce the likelihood of conflicts. 

`scores` relies on a relatively recent version of pip, so you might need to upgrade pip within your virtual environment. If this is required, the `scores` installation process will automatically prompt you to do so, including the commands required. Upgrading pip within a virtual environment is straightforward, reliable and the pip upgrade will only apply within the virtual environment.

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

## Installation Options

Most users will want the "all" installation option. There are also more specialised options for those who need them.  

The 4 supported installation options are:

- all: contains mathematical functions, tutorials and development libraries. Excludes maintainer requirements.
- minimal: ONLY contains mathematical functions (so has limited dependencies).
- tutorial: ONLY contains mathematical functions and tutorials.
- maintainer: ONLY contains tools for building the documentation and building for PyPI.

### 1. "All" Dependencies (excludes some maintainer-only packages)

Use this for `scores` development and general use.

Installs:
* Mathematical API code and libraries
* Everything needed to run the tutorial notebooks
* Testing, static analysis and other developer libraries
* Does **not** install tools for making packages and releasing new versions

#### From a Local Checkout of the Git Repository

```bash
pip install -e .[all]
```

### 2. "Minimal" Dependencies (Mathematical API Functions Only)
Use this to install the `scores` code into another package or system.

Installs:
* Mathematical API functions and libraries
* Only the required core dependencies. Nothing extra - no tutorials, no developer requirements.
* (Note for high-performance users - dask is not included by default in the minimal install, but will be used if installed into the environment)

#### From PyPI

```bash
pip install scores
```

### 3. "Tutorial" Dependencies
Use this for running tutorials using `scores`, but when you don't need or want developer tools.

Installs:
* Mathematical API functions and libraries
* Jupyter Lab, Plotly, and libraries for reading data, so that the tutorial notebooks can be run

#### From a Local Git Repository

```bash
pip install .[tutorial]
```

### 4. "Maintainer" Dependencies
Use this to build the docs, create packages and prepare new versions for release.

Installs:
* Mathematical API functions and libraries
* Dependencies for building new versions of the `scores` package
* Dependencies for building the documentation as HTML

#### From a Local Git Repository

```bash
pip install -e .[maintainer]
```

## Jupyter Notebooks - Advanced Installation Options

The `scores` ["all"](#all-dependencies-excludes-some-maintainer-only-packages) and ["tutorial"](#tutorial-dependencies) installation options include the JupyterLab software, which can be used (among other things) to run the tutorials. 

Some users may wish to separate the Jupyter environment and the `scores` execution environment. One way to achieve this is by creating a new `scores` virtual environment (using one of the [above options](#setting-up-a-virtual-environment)) and registering it as a new kernel within the Jupyter environment. You can then run the tutorials and/or execute `scores` code within the kernel. Registering the kernel can be done as follows:

1. Determine the "prefix" of the Jupyter environment. 
2. Choose a name to use for a new kernel.
3. Activate the `scores` virtual environment which will be used as the kernel.
4. Execute the registration command.

A sample command to register a new kernel is:

`python -m ipykernel install --user --prefix=<path-to-server-environment> --name=<pick-any-name-here>`

[https://jupyter-tutorial.readthedocs.io/en/24.1.0/kernels/install.html](https://jupyter-tutorial.readthedocs.io/en/24.1.0/kernels/install.html) provides additional technical details regarding the registration of kernels.

