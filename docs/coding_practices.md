```{eval-rst}
:orphan:
```

# Coding Practices

The [Contributing Guide](contributing.md) provides (among other things) guidance on the workflows and general expectations associated with contributing a code change to `scores`. This document eschews some of the context to focus on specifying technical information needed when developing code for `scores`.

# Expectations:

1. `black` and `isort` fixers have been run over the code
2. `pylint` and `mypy` linters have been run with no errors
3. Test coverage is 100%
4. All scores/metrics have a demonstration notebook

See the [contributing docs](contributing.md) for a development overview and [`pre-commit` configuration](contributing.md#pre-commit) for automating these steps.

> Note: Please refer to [pyproject.toml dev dependencies](../pyproject.toml#L30) for pinned versions of linters and fixers for this project.

# Type Hinting

Our philosophy is 'we hint what we test'. The functions in `scores` may well function with a broader variety of types than what is hinted. The promise of a type hint is that you know it will work, because it's been tested to work.

# Support for Additional Libraries

This may result in false positives, such as if using a Dask dataframe in place of a Pandas dataframe, or a numpy array in place of a Pandas series. Such things may work, but without being tested and assured by the development team, we don't want to indicate it.
