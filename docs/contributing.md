# Contributing guide

Thank you for considering contributing to the scores project! Contributions of all kinds are welcome!

Types of contributions include bug reports, merge requests, feature requests, and code reviews. Contributions of all kinds are welcome from the community. Contributions which are in line with the roadmap will be prioritised and the roadmaps outlines the intentions for this package.

These guidelines aim to make it clear how to collaborate effectively.

## Roadmap
1. Addition of more scores, metrics and statistical techniques
2. Further optimisation and performance improvements
3. Increased support for machine learning library integration
4. Additional notebooks exploring complex use cases in depth

## Bug Reports and Feature Requests

Please submit bug requests and feature requests through Github as issues. No specific template or approach is requested at this stage. This may evolve, but is currently an open-ended approach.

## Handling Security Concerns

Please see the information provided in [SECURITY.md](SECURITY.md)

## Development Process for a New Score or Metrics

A new score or metric should be developed on a separate feature branch, rebased against the main branch. Each merge request should include:

 - The implementation of the new metric or score in xarray, ideally with support for pandas and dask
 - 100% unit test coverage
 - A tutorial notebook showcasing the use of that metric or score, ideally based on the standard sample data
 - API documentation (docstrings) using [Napoleon (google)](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) style, making sure to clearly explain the use of the metrics
 - A reference to the paper which described the metrics, added to the API documentation
 - For metrics which do not have a paper reference, an online source or reference should be provided
 - For metrics which are still under development or which have not yet had an academic publication, they will be placed in a holding area within the API until the method has been properly published and peer reviewed (i.e. `scores.emerging`). The 'emerging' area of the API is subject to rapid change, still of sufficient community interest to include, similar to a 'preprint' of a score or metric.

All merge requests should comply with the coding standards outlined in this document. Merge requests will undergo both a code review and a science review. The code review will focus on coding style, performance and test coverage. The science review will focus on the mathematical correctness of the implementation and the suitability of the method for inclusion within 'scores'.

A github ticket should be created explaining the metric which is being implemented and why it is useful.

## Development Process for a Correction or Improvement

Merge requests addressing documentation changes, tutorial improvements, corrections to code or improvements (e.g. performance enhancements, improvements to error messages, greater flexibility) require only a github ticket explaining the goals of the merge request. Merge requests will still undergo both a technical code review and a scientific code review to ensure that the merge request maintains or improves the coding and scientific integrity while achieving the goals set out in the ticket. These are outlined further below.

## Setting up for Development

There are many ways to set up a development environment, most of which should be okay to adopt. This document sets out just one possible way. Conda based virtual environments are also suitable.

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

### Building a package <a name="build"></a>

Feel free to use a non-editable install if you prefer, however you will need to build new versions of the package to test your changes.
To do so run the following command in the root folder of the project:

```bash
# if you don't already have the maintainer deps run the cmd below
pip install scores[maintainer]
python -m build
# install built distribution (.whl)
pip install dist/<my_latest_package>.whl
```

### Setup `pre-commit` (Optional) <a name="pre-commit"></a>

To automate linter and fixer checks this project uses `pre-commit` which is setup to execute after every local commit. This ensures that code standards are flagged at the development stage rather than in the project CI/CD pipeline. Although it is optional we highly recommended to use the tool before pushing changes to the remote.

```bash
pre-commit install -t pre-commit -t pre-push
```

### Coding Practices

Pylint and black should be used at all times to ensure a consistent approach to coding. Isort should be used for the ordering of import statements. All merge requests will be checked prior to acceptance. The project will include configuration files which may be used to capture any overrides to convention that may be adoped.

### Branching and merge requests

`scores` follows the [ git-flow ](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow) branching model. This can be summarised as follows:

- Create a github ticket to describe the goals of your change or feature
- Check out a feature new branch for your work, branching from the 'develop' branch
- When ready, rebase the feature branch against the 'develop'' branch and submit your merge request
- Respond to any feedback left during the review process
- It is the responsibility of the developer to maintain the feature branch against the main branch and resolve any conflicts within the feature branch prior to it being accepted
- When accepted, the feature branch will be merged to the main branch and may be 'squashed' into a single commit in the process
- The package maintainer may make changes to the code during the merge process or afterwards, such as resolving last-minute conflicts or making any key technical tweaks that are simple to implement
- Periodically, a versioned release branch will be taken from the 'develop' branch, and any finalisation work for the release undertaken there
- These release branches will then be finalised, tagged, and merged into main branch. The main branch holds tagged releases.

For most developers, this can be simplified to:

 - Make a ticket to describe the goals of your change or feature
 - Make a feature branch from the 'develop' branch
 - Create a merge request for your feature when you're ready


### Code review processes

Contributions of code through merge requests are welcomed. Prior to developing a merge request, it is a good idea to create an issue in the tracker to capture what the merge request is trying to achieve, any pertinent details, and how it aligns to the roadmap.

All code must undergo a review process prior to being included in the main branch in order to ensure both the coding and the statistical and scientific validity of changes. New metrics will undergo an additional statistical and/or scientific review in addition to the code review. All metrics should reference a specific paper or authoratative source. Novel metrics which have not yet undergone an academic peer-review process, but which offer significant value to the community, will be included in a separate part of the API so that they can be used and discussed, but which also clearly flags their nature as emerging metrics.

The code review process does not differ between the core team and community contributions.

Code will also be reviewed for test coverage, stylistic conventions, thoroughness and correctness by a team of software specialists.

### Scope of a Code Review

A code review is not responsible for scientific correctness, that will be handled by the science review. A code review is responsible for checking the following:

1. Unit test coverage is 100% and unit tests cover functionality and robustness
2. Any security issues are resolved and appropriately handled
3. Documentation and tutorials are written
4. Style guidelines are followed, static analysis and lint checking have been done
5. Code is readable and well-structured
6. Code does not to anything unexpected or beyond the scope of the function
7. Any additional dependencies are justified and do not result in bloat

### Scope of a Science Review

A science review should answer the following questions:

1. Does the function work correctly for several 'central' or intended common use cases?
2. Are edge cases handled correctly (i.e. null data handling, flat/perfect distributions, large/small number of samples, numerical issues)?
3. Does the implementation look correct if read at face value (i.e. detailed knowledge of libraries not required)?
4. Are any issues of scientific interpretation explained (i.e. if multiple mathematical interpretations of a method are reasonable, is this made clear)?
5. Are the examples contained in the tutorial well-explained and relevant?
