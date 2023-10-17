# Coding Practises

The [contributing.md](Contributing Guide) provides (among other things) guidance on the workflows and general expectations associated with contributing a code change to `scores`. This document eschews some of the context to focus on specifying technical information needed when developing code for `scores`.

# Expectations:

1. Black has been run over the code
2. Pylint has been run and there are zero lint issues
3. mypy has been run and there are zero mypy issues
4. Test coverage is 100%
5. All scores/metrics have a demonstration notebook

# Type Hinting

Our philosophy is 'we hint what we test'. The functions in `scores` may well function with a broader variety of types than what is hnted. The promise of a type hint is that you know it will work, because it's been tested to work.

# Support for Additional Libraries

This may result in false positives, such as if using a Dask dataframe in place of a Pandas dataframe, or a numpy array in place of a Pandas series. Such things may work, but without being tested and assured by the development team, we don't want to indicate it.
