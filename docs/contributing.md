# Contributing guide

Thank you for considering contributing to the scores project. Contributions of all kinds are welcome from the community!

These guidelines describe how to collaborate effectively.

Types of contributions include bug reports, merge requests and feature requests. Contributions which are in line with the roadmap will be prioritised. The roadmap outlines the intentions for this package.

## Roadmap
1. Addition of more scores, metrics and statistical techniques
2. Further optimisation and performance improvements
3. Increased support for machine learning library integration
4. Additional notebooks exploring complex use cases in depth

## Bug Reports and Feature Requests

Please submit bug requests and feature requests through GitHub as issues. No specific template or approach is requested at this stage. This may evolve, but is currently an open-ended approach.

## Handling Security Concerns

Please see the information provided in [SECURITY.md](SECURITY.md)

## Development Process for a Correction or Improvement

**Note for new contributors:** we are still establishing the smoothest path for new contributors to make code contributions. If you aren't sure where or how to start, please email scores@bom.gov.au and we would be happy to help discuss your goals and work through any issues getting set up.

If you are already confident working with GitHub, please feel free to:
1. create a fork of the `scores` repository, 
2. create a feature branch on your fork,
3. keep your feature branch rebased and up-to-date with the `scores` develop branch,
4. when ready, submit a pull request to the develop branch of `scores`.

Pull requests will undergo both a technical code review and a scientific review to ensure that the merge request maintains or improves the coding and scientific integrity.

The package maintainer may make changes to the code during the merge process or afterwards, such as resolving last-minute conflicts or making any key technical tweaks that are simple to implement.

## Development Process for a New Score or Metric

A new score or metric should be developed on a separate feature branch, rebased against the develop branch. Each merge request should include:

 - The implementation of the new metric or score in xarray, ideally with support for pandas and dask
 - 100% unit test coverage
 - A tutorial notebook showcasing the use of that metric or score, ideally based on the standard sample data
 - API documentation (docstrings) using [Napoleon (google)](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) style, making sure to clearly explain the use of the metric
 - A reference should be added to the API documentation.
   - If there is an authoritative reference (e.g. an academic article that defines the metric, or the specific implementation, being used), please cite that reference.
   - When there is an authoritative reference, please cite it even if it cannot be accessed without payment. In these instances, if there is another suitable reference that is open and available without payment, cite that as well.
   - When there is no authoritative reference, please cite a reference that provides a clear description of the function. Where possible, please cite open references that can be accessed without payment.
   - When available, please include DOIs.
 - Metrics which are still under development or which have not yet had an academic publication will be placed in a holding area within the API until the method has been properly published and peer reviewed (i.e. `scores.emerging`). The 'emerging' area of the API is subject to rapid change, still of sufficient community interest to include, similar to a 'preprint' of a score or metric.
  
Merge requests will undergo both a code review and a science review. The code review will focus on coding style, performance and test coverage. The science review will focus on the mathematical correctness of the implementation and the suitability of the method for inclusion within 'scores'.

## Setting up for Development

There are many ways to set up a development environment, most of which should be okay to adopt. This document provides examples.

### `venv`-based virtual environment

Here is a simple setup process for an individual developer, assuming you have cloned the repository in the current working directory.

```bash
python3 -m venv <specify path to your virtual environment>
source  source <path to virtual environment>/bin/activate
pip install -e .[all]
pytest
```
### `conda`-based virtual environment

```bash
# overwrite default name `scoresenv` with `-n <new-name>` if desired
conda env create -f environment.yml
conda activate scoresenv
pip install -e .[all]
pytest
```

The `environment.yml` file refers to the `pip` package management repositories to utilize the optional dependencies within `pyproject.toml`. Feel free to use `conda` channels, although all required project dependencies will need to be available from `PyPI` sources.

This process should result in an editable installation with all tests passing.

An editable installation is recommended. This is deliberate, to make the process more robust and less prone to 'happy accidents' during import of packages. If you wish to avoid editable installations then refer to the [building a package](#build) section.


### Set up `pre-commit` (optional) <a name="pre-commit"></a>

To automate linter and fixer checks this project uses `pre-commit` which is set up to execute after every local commit. This ensures that code standards are flagged at the development stage rather than in the project CI/CD pipeline. Although it is optional we highly recommended to use the tool before pushing changes to the remote.

```bash
pre-commit install -t pre-commit -t pre-push
```

## Review Processes

Contributions of code through merge requests are welcomed. Prior to developing a merge request, it may be a good idea to create a GitHub issue to capture what the merge request is trying to achieve, any pertinent details, and how it aligns to the roadmap. Otherwise, please explain this in the pull request.

All code will undergo a review process prior to being included in the develop branch in order to ensure both the coding and the statistical and scientific validity of changes. 

The code review process does not differ between contributions from the core team and contributions from the community.

### Scope of a Code Review

A code review is not responsible for scientific correctness; that will be handled by the science review. A code review is responsible for checking the following:

1. Unit test coverage is 100% and unit tests cover functionality and robustness
2. Any security issues are resolved and appropriately handled
3. Documentation and tutorials are written
4. Style guidelines are followed, static analysis and lint checking have been done
5. Code is readable and well-structured
6. Code does not do anything unexpected or beyond the scope of the function
7. Any additional dependencies are justified and do not result in bloat

### Scope of a Science Review

A science review should answer the following questions:

1. Does the function work correctly for several 'central' or intended common use cases?
2. Are edge cases handled correctly (i.e. null data handling, flat/perfect distributions, large/small number of samples, numerical issues)?
3. Does the implementation look correct if read at face value (i.e. detailed knowledge of libraries not required)?
4. Are any issues of scientific interpretation explained (i.e. if multiple mathematical interpretations of a method are reasonable, is this made clear)?
5. Are the examples contained in the tutorial well-explained and relevant?
