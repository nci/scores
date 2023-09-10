# Installation Guide

It is recommended to use a virtualised Python environment in most settings. The package can be easily installed using pip. The base requirements of the package are kept to a minimum to reduce the likelihood of conflicts.

The basic installation includes only the most essential requirements, so as to not require users to include many complex dependencies.

There are 5 different types of environments which can be installed in `scores`: 

- core: only contains mathematical functions
- tutorial: includes jupyter lab and ability to run all the notebooks
- development: includes pylint, black and other development tools
- maintainer: includes tools for building the documentation
- all: includes requirements for all of the above

#### 1. Core Environment <a name="core-env"></a>

Installs:
* `scores` package
* Only the required core dependencies, nothing extra, no tutorials.

> **_NOTE:_** Use this environment if you are unsure about what package you require.

##### from PyPI

```Bash
pip install scores
```

##### from local git repository

```
pip install .
```

#### 2. Tutorial Environment <a name="tutorial"></a>

Installs:
* [core environment](#core-env)
* Dependencies for running the tutorial notebooks with jupyter lab.

##### PyPI

```Bash
pip install scores[tutorial]
```

##### Local git repository

```bash
pip install .[tutorial]
```

#### 3. Development Environment <a name="dev"></a>

Installs:
* [core environment](#core-env)
* Dependencies for development on the git repository.
  * i.e running tests suite, linters, ect.

##### PyPI

```Bash
pip install scores[dev]
```

##### Local git repository

```bash
pip install .[dev]
```

#### 4. Maintainer Environment <a name="maintainer"></a>

Installs:
* [core environment](#core-env)
* Dependencies for building new versions of the `scores` package,

##### PyPI

```Bash
pip install scores[maintainer]
```

##### Local git repository

```bash
pip install .[maintainer]
```

#### 5. All Dependencies <a name="all"></a>

Installs:
* [core dependencies](#core-env)
* [develeopment dependencies](#dev)
* [tutorial dependencies](#tutorial)

##### from PyPI

```Bash
pip install scores[all]
```

##### Local git repository

```bash
pip install .[all]
```

