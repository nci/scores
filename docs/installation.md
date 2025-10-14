# Detailed Installation Guide

## Overview

This page describes:

- Setting up a virtual environment.
- The most common installation options for `scores`. (Expert users of pip and conda will note that more variations are possible.)
- An advanced installation option for Jupyter Notebook, for users who wish to separate the Jupyter environment and the `scores` execution environment.

## Setting up a Virtual Environment

In almost all cases, it is recommended to use a virtualised Python environment. 

`scores` can be easily installed using either venv/pip or conda/pip. The requirements of `scores` are kept to a minimum to reduce the likelihood of conflicts. 

`scores` relies on a relatively recent version of pip, so you might need to upgrade pip within your virtual environment. If this is required, the `scores` installation process will automatically prompt you to do so, including the commands required. Upgrading pip within a virtual environment is straightforward, reliable and the pip upgrade will only apply within the virtual environment.

Here is a command to create and activate a new virtual environment with *virtualenv*:

```py
python -m venv <path_to_environment>
source <path_to_environment>/bin/activate
```

Here is a command to create and activate a new virtual environment with *conda*:
```py
conda create --name <my-env>
conda activate <my-env>
```

## Installation Options

There are multiple installation options. Most users currently want the "all" installation option. 

The 4 supported installation options are:

- all: contains mathematical functions, tutorial dependencies and development libraries. 
- minimal: ONLY contains mathematical functions (so has limited dependencies). 
- tutorial: ONLY contains mathematical functions and tutorial dependencies. 
- maintainer: contains tools for building the documentation and building releases. 

Each of the above installation options are available on PyPI. "Minimal" is also available on conda-forge. (In time, we intend to add more installation options to conda-forge.)

### 1. "All" Dependencies (excludes some maintainer-only packages)

Use this for `scores` development and general use.

Installs:
* Mathematical API code and libraries.
* Everything needed to run the tutorial notebooks.
* Testing, static analysis and other developer libraries.
* Does **not** install tools for making packages and releasing new versions.

#### With pip

```bash
# From a local checkout of the Git repository
pip install -e ".[all]"
```

### 2. "Minimal" Dependencies (Mathematical API Functions Only)
Use this to install the `scores` code into another package or system.

Installs:
* Mathematical API functions and libraries.
* Only the required core dependencies. Nothing extra - no tutorials, no developer requirements.
* (Note for high-performance users - dask is not included by default in the minimal install, but will be used if installed into the environment.)

#### From PyPI

```bash
pip install scores
```
#### With conda

```bash
conda install conda-forge::scores
```

(Note: at present, only the "minimal" installation option is available from conda. In time, we intend to add more installation options to conda.)

### 3. "Tutorial" Dependencies
Use this for running tutorials using `scores`, but when you don't need or want developer tools.

Installs:
* Mathematical API functions and libraries.
* JupyterLab, Plotly, and libraries for reading data, so that the tutorial notebooks can be run.

#### With pip 

```bash
# From a local checkout of the Git repository
pip install ".[tutorial]"
```

### 4. "Maintainer" Dependencies
Use this to build the docs, create packages and prepare new versions for release.

Installs:
* Mathematical API functions and libraries.
* Dependencies for building new versions of the `scores` package.
* Dependencies for building the documentation as HTML.

#### With pip

```bash
# From a local checkout of the Git repository
pip install -e ".[maintainer]"
```

## Jupyter Notebook - Advanced Installation Option

The `scores` "all" and "tutorial" installation options include the JupyterLab software, which can be used to run the tutorials and/or execute `scores` code within a Jupyter environment. 

Some users may wish to separate the Jupyter environment and the `scores` execution environment. One way to achieve this is by creating a new `scores` virtual environment (using one of the [above options](#setting-up-a-virtual-environment)) and registering it as a new kernel within the Jupyter environment. You can then run the tutorials and/or execute `scores` code within the kernel. Registering the kernel can be done as follows:

1. Determine the "prefix" of the Jupyter environment. 
2. Choose a name to use for a new kernel.
3. Activate the `scores` virtual environment which will be used as the kernel.
4. Execute the registration command.

A sample command to register a new kernel is:

`python -m ipykernel install --user --prefix=<path-to-server-environment> --name=<pick-any-name-here>`

[https://jupyter-tutorial.readthedocs.io/en/24.1.0/kernels/install.html](https://jupyter-tutorial.readthedocs.io/en/24.1.0/kernels/install.html) provides additional technical details regarding the registration of kernels.

## Using `pixi` for Environment Management (Optional)

An optional, alternative, approach that `scores` supports for installing environments is [`pixi`](https://pixi.sh).
`pixi` is a powerful environment management tool.

It uses a combination of PyPI and Conda channels. `pixi` is configured in `pyproject.toml` in the
root directory of the `scores` GitHub repository. It is configured with some default tasks that a
user can run in ephemeral environments specific for those tasks (see examples below).

`pixi` handles creation, swapping, stacking and cleanup of environments automatically, depending on
the task being run.

```{note}
`scores` currently does not save `pixi.lock` files in its GitHub repository. While `pixi` is
supported in `scores`, it is *not* part of the recommended development toolchain.

`pixi.lock` is intentionally filtered out in `.gitignore`, in order to prevent accidental commits of
the lock file. This may change in the future if there is sufficient adoption.

`pixi` is mainly there for users who *already* are familiar with it, and those who prefer not to
manually deal with python environments.
```

### Installation

`pixi` supports multiple platforms. Its installation process is straightforward and can be found
here: <https://pixi.sh/latest/#installation>.

### Examples

- **As a developer** I want to run some tests.
   - Command: `pixi run pytest-src`
   - Description: this will test the source code in the `dev` environment.
- **As a researcher** I want to launch JupyterLab.
  - Command: `pixi run jupyterlab`
  - Description: this will launch a local JupyterLab server in the `tutorial` environment.
- **As a maintainer** I want to render the docs as html.
  - Command:  `pixi run make-docs`
  - Description: this will render the docs locally to "htmldocs" (similar to what the GitHub
    pipeline currently does).
- **As any user** I want to run a specified command in a particular environment.
  - Command: `pixi run -e <env> <cmd>`, where `<env> = dev | tutorial | maintainer | all` - see
    section on [installation options](#installation-options) above.
  - Description: this will run the command in the specified environment, and return you back to the
    original shell once it has been executed.

