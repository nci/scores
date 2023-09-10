# Installation Guide

It is recommended to use a virtualised Python environment in most settings. The package can be easily installed using pip. The base requirements of the package are kept to a minimum to reduce the likelihood of conflicts.

The basic installation includes only the most essential requirements, so as to not require users to include many complex dependencies.

There are 5 different types of environments which can be installed in `scores`: 

- core: only contains mathematical functions
- tutorial: includes jupyter lab and ability to run all the notebooks
- development: includes pylint, black and other development tools
- maintainer: includes tools for building the documentation
- all: includes requirements for all of the above

## 1. Core Environment 

Installs:
* `scores` package
* Only the required core dependencies, nothing extra, no tutorials.

> **_NOTE:_** Use this environment if you are unsure about what package you require.

### from PyPI

```Bash
pip install scores
```

### from local git repository

```
pip install .
```

## 2. Tutorial Environment 

Installs:
* core environment
* Dependencies for running the tutorial notebooks with jupyter lab.

### PyPI

```Bash
pip install scores[tutorial]
```

### Local git repository

```bash
pip install .[tutorial]
```

## 3. Development Environment 

Installs:
* core environment
* Dependencies for development on the git repository.
  * i.e running tests suite, linters, ect.

### PyPI

```Bash
pip install scores[dev]
```

### Local git repository

```bash
pip install .[dev]
```

## 4. All Non-Maintainer Dependencies 

Installs:
* core dependencies
* develeopment dependencies
* tutorial dependencies

### from PyPI

```Bash
pip install scores[all]
```

### Local git repository

```bash
pip install .[all]
```


## 5. Maintainer Environment 

Installs:
* core environment
* Dependencies for building new versions of the `scores` package,

### PyPI

```Bash
pip install scores[maintainer]
```

### Local git repository

```bash
pip install .[maintainer]
```


