# Maintainers Notes

Information relevant for package maintenance

## This section covers the process for making a release from the develop branch

1. Run through the notebooks manually
2. Run the unit tests
3. Prepare the merge request in github (do not do a rebase merge to main -- do a regular merge commit)
4. Check readthedocs rebuilds correctly, including manually checking the version number looks right
5. Perform the merge to main
6. Build a new environment and perform basic testing on main
7. Update github with a tagged release
8. Prepare the package files (see below)
9. Prepare the PyPI update (should move to a Github Action)
10. Perform the PyPI update (should move to a Github Action)

### Test the new main branch

1. Create a new virtual environment
2. Make sure you check out the 'main' branch
3. Install scores[all] 
4. Run the tests again, just for surety

### Update GitHub with a tagged release
1. Click on the 'releases' area of the front page
2. Follow the 'create release' workflow, and create a tag at the same time

### Create the package files locally
1. Install scores[maintainer]
2. Run pytest again
3. Run 'hatch build'. This will make the release files (sdist and wheel).

### Update PyPI manually
1. Run python3 -m keyring --disable 
2. Run hatch publish -r test
3. Create a new virtual env, and install the test scores with `python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple scores`
4. Do a little manual testing
5. Run hatch publish
6. Uninstall the test scores, re-install the now-updated package, do a little testing

### Confirm Zenodo correctness

1. Confirm license
2. Confirm authors
3. Scan everything else

## This section covers how to format and prepare release notes

```
# Release Notes (What's New)

## Version X.Y.Z (Month Day, Year) e.g. "Version 0.9.3 (July 9, 2024)"

For a list of all changes in this release, see the [full changelog](https://github.com/nci/scores/compare/(X-1).(Y-1).(Z-1)...X.Y.Z). Below are the changes we think users may wish to be aware of.

### Features
### Breaking Changes
### Deprecations
### Bug Fixes
### Documentation
### Internal Changes
### Contributors to this Release

For each entry: "Brief description. See [PR #ABC](add link)."

For Contributors to this Release: "Name ([@handle](https://github.com/handle))"

When there are new contributors: "\* indicates that this release contains their first contribution to `scores`."
```

**While a version is under development:**
1. Change the date to "(Upcoming Release)"
2. In the full changelog URL, change "X.Y.Z" to "develop", i.e. "(https://github.com/nci/scores/compare/(X-1).(Y-1).(Z-1)...develop)"

**Immediately before making a release:**
1. Change “(Upcoming Release)” to the date of the release (and correctly format the date)
2. In the full changelog URL, change “develop” to the correct version number, i.e (https://github.com/nci/scores/compare/(X-1).(Y-1).(Z-1)...X.Y.Z)
3. Delete any unused headers

## This section covers asking new contributors to add their details to .zenodo.json

```
Thank you very much for your contribution. 

When we release a new version of `scores`, that version is archived on Zenodo. See: https://doi.org/10.5281/zenodo.12697241

As you have contributed to `scores`, would you like to be listed on Zenodo as an author the next time `scores` is archived? 

If so, please open a new pull request. In that pull request please add your details to .zenodo.json (which can be found in the  `scores` root directory).

In .zenodo.json, please add your details at the bottom of the “creators” section. The fields you will need to complete are:

1. “orcid”. This is an optional field. If you don’t have an ORCID, but would like one, you can obtain one here: https://info.orcid.org/researchers/ .
2. “affiliation”. Options include: the name of the institution you are affiliated with, “Independent Researcher” or “Independent Contributor”.
3. “name”. Format: surname, given name(s).
```

## This section gives guidance for maintaining compatibility with old versions of Python and packages

tldr; about 3 years old is OK, longer if painless

https://scientific-python.org/specs/spec-0000/ provides a guide for the scientific Python ecosystem - we should aspire to be at least that compatible with older versions. It describes an approach including outlining when particular packages move out of support.

We have not tested compatibility against all possible package versions which are included in this spec. Conversely, in some cases, it has been fairly straightforward to support packages older than this. 

There is no formal "support" agreement for `scores`. In the context of `scores` package management, maintaining compability means being willing to make reasonable efforts to resolve any issues raised on the issue tracker. If a specific issue arises that would make it impractical to support a version within the compatibility window, then a response will be discussed and agreed on at the time on the basis of practicality.

There is currently no specific testing for older versions of libraries, only older versions of Python (which may or may not intake an older library version). A full matrix test of Python and package versioning would be prohibitively complex, and there would also be no guarantee that pinned older versions wouldn't result in an insecure build (even if only in a test runner). 

The development branch versioning is unpinned, and so any issues arising from newly-released packages should quickly be encountered and then resolved before the next `scores` release. Releases of `scores` use "~=" versioning, which gives flexibility within a range of versions (see https://packaging.python.org/en/latest/specifications/version-specifiers/#id5).

## This section covers how to build the documentation locally 
(Readthedocs should update automatically from a GitHub Action)

### 1. Summary of the tech stack

`scores` utilises:

 - Sphinx, with the myst parts, for making the HTML pages
 - Markdown files as the text source for all the documentation
 - Myst Parser to enable Sphinx to utilise the markdown syntax
 - The 'sphinx book' theme for Sphinx
 - Pandoc should be installed through the OS package manager separately

### 2. Useful information resources are:

 - We follow this recipe: [https://www.youtube.com/watch?v=qRSb299awB0](https://www.youtube.com/watch?v=qRSb299awB0). There isn't a good tutorial written up but the video is excellent.
 - [https://www.sphinx-doc.org/en/master/usage/markdown.html](https://www.sphinx-doc.org/en/master/usage/markdown.html)
 - [https://www.markdownguide.org/cheat-sheet/](https://www.markdownguide.org/cheat-sheet/)

### 3. Process to generate the HTML pages

 - `sphinx-build -b html docs/ htmldocs` is the key command to rebuild the HTML documentation from current sources
 - This requires docs/conf.py to be configured appropriately.

### 4. Generating the markdown for the API documentation

 - Each function must be added explicitly to api.md. The autogeneration tools are not sophisticated enough to process
   the import structure used to define the public API neatly.

### 5. What to update, when and why

|     what                 |     when                 |      why     |
| ------------             | -----------              | ------------ | 
|  README                  |  a new score is added    | in case it deserves a mention
|  api.md                  |  a new function is added | each function must be added individually 
|  included.md             |  a new function is added | each function (and each variation of the function name) must be added individually
|  Tutorial_Gallery.ipynb  |  a new tutorial is added | navigation throughout the docs

## This section covers checking the documentation renders properly in readthedocs

### Tips for working with pull requests from forks

It can be convenient as maintainer to have write access to people's forks to push small fixes into a PR during the process. When doing so, it's a good idea to check out the remote branch as follows (after adding the fork as a remote)

`git checkout -b test <name of remote>/test`

### What documentation needs checking in readthedocs

Each time an existing function is modified or a new function is added to `scores`, the rendering in readthedocs for any modified or newly created documentation must be checked. 

This applies to each of the following documents:

  - included.md
  - API Documentation
  - Tutorials (see also [tutorial rendering](#Tutorial-rendering) further below)
  - (If applicable) README

### Common rendering issues in readthedocs

Frequent issues include:

- Lists (including lists that use bullets, dot points, hyphens, numbers, letters etc.)
  - Check **each** list appears and renders properly
  - Check **all** indented lists/sub-lists for proper indentation
- Figures: check **each** figure appears and renders properly
- Plots: check **each** plot appears and renders properly
- Tables: check **each** table appears and renders properly
- Formulae: check **each** formula appears and renders properly
- API Documentation: in addition to checking the above items, also confirm "Returns" and "Return Type" are rendering as expected

### Tutorial rendering

Things that render well in JupyterLab do not always render properly in readthedocs. Additionally, fixes that work well when built locally, don't always work when merged into the codebase. 

To check the rendering of tutorials in readthedocs:
  - Compare the tutorial in readthedocs against a version running in JupyterLab (as not everything renders in GitHub).
  - Check the entirety of the tutorial (sometimes things will render properly in one section, while not rendering properly in a different section of the same tutorial).
  - If you make any changes to the code cells, re-execute the Notebook in JupyterLab before committing, otherwise some things (e.g. some plots) won't render in readthedocs. Then re-check the tutorial in readthedocs to ensure the tutorial is still rendering properly.


