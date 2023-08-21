# User Guide

## Installation

It is recommended to use a virtualised Python environment in most settings. The package can be easily installed using pip. The base requirements of the package are kept to a minimum to reduce the likelihood of conflicts.

The basic installation includes only the most essential requirements, so as to not require users to include many complex dependencies.

There are 5 different types of environments which can be installed in `scores`.

#### 1. Base environment <a name="base-env"></a>

Installs:
* `scores` package
* All of the required core dependencies.

> **_NOTE:_** Use this environment if you are unsure about what package you require.

##### from PyPI

```Bash
pip install scores
```

##### from local git repository

```
pip install .
```

#### 2. Development environment <a name="dev"></a>

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

#### 3. Tutorial environment <a name="tutorial"></a>

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

#### 4. maintainter environment <a name="maintainer"></a>

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

#### 4. All environment <a name="all"></a>

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

## Tutorials and Worked Examples

Tutorials are included in the form of Jupyter Notebooks, within the 'tutorials' directory. There is a tutorial for each scoring metric included in the scores package.

A Jupyter Notebook server and plotting dependencies are not included in the package dependencies by default, to keep the package lightweight. To execute the tutorial notebooks, from within your python virtual environment, run the [tutorial installation snippet here](#tutorial).

Users must also set up the expected data files. The process for downloading or generating this sample data is set out in the notebook "First - Data Fetching" and this should be done up-front.

Each score in the package has its own notebook. For very similar scores, this means there may be some repetition, but this approach ensures that each score has clear worked examples which can support users in understanding how to use both the API and the score itself.

To run the `jupyter lab` server, run the command below and follow the prompts:

```bash
jupyter notebook
```
## Scores

Each score is documented in the API documentation [ api.md ](api.md). A simple listing of the currently implemented scores is:

 - Mean Absolute Error
 - Mean Squared Error
 - Root Mean Squared Error
 - Continuous Ranked Probability Score

The following scores are expected to be added shortly
 - Flip Flop Index
 - FIRM

