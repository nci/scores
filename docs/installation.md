# Installation Guide

It is recommended to use a virtualised Python environment in most settings. The package can be easily installed using pip. The base requirements of the package are kept to a minimum to reduce the likelihood of conflicts.

The basic installation includes only the most essential requirements, so as to not require users to include many complex dependencies.

There are 5 different types of environments which can be installed in `scores`: 

- core: Only contains mathematical functions
- tutorial: Includes jupyter lab and ability to run all the notebooks
- development: Includes pylint, black and other development tools
- maintainer: Includes tools for building the documentation
- all: Includes requirements for all of the above

#### 1. Core environment <a name="base-env"></a>

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

#### 2. Tutorial environment <a name="tutorial"></a>

Installs:
* [base environment](#base-env)
* Dependencies for running the tutorial notebooks with jupyter lab.

##### PyPI

```Bash
pip install scores[tutorial]
```

##### Local git repository

```bash
pip install .[tutorial]
```

#### 3. Development environment <a name="dev"></a>

Installs:
* [base environment](#base-env)
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

#### 4. Maintainter environment <a name="maintainer"></a>

Installs:
* [base environment](#base-env)
* Dependencies for building new versions of the `scores` package,

##### PyPI

```Bash
pip install scores[maintainer]
```

##### Local git repository

```bash
pip install .[maintainer]
```

#### 5. All dependencies <a name="all"></a>

Installs:
* [base dependencies](#base-env)
* [dev dependencies](#dev)
* [tutorial dependencies](#tutorial)

##### from PyPI

```Bash
pip install scores[all]
```

##### Local git repository

```bash
pip install .[all]
```

