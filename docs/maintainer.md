# Maintainers Notes

Information relevant for package maintenance

## This section covers the process for making a release from the develop branch

1. Run through the notebooks manually
2. Run the unit tests
3. Prepare the merge request in github (do not do a rebase merge to main -- do a regular merge commit)
4. Check readthedocs rebuilds correctly, including manually checking the version number looks right
5. Prepare the PyPI update (should move to a Github Action)
6. Perform the PyPI update (should move to a Github Action)

### Creating the Package Files Locally

1. Create a new virtual environment
2. Make sure you check out the 'main' branch'
3. Install scores[all] and scores[maintainer]
4. Run the tests again, just for surety
5. Run 'hatch build'. This will make the release files (sdist and wheel).

### Updating PyPI Manually
6. Run python3 -m keyring --disable 
7. Run hatch publish -r test
8. Create a new virtual env, and install the test scores with `python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple scores`
9. Do a little manual testing
10. Run hatch publish
11. Uninstall the test scores, re-install the now-updated package, do a little testing

### Updating GitHub with a tagged release
12. Go click on the "releases" area of the front page
13. Follow the 'create release' workflow, and create a tag at the same time


## This section covers how to build the documentation locally. 
(Readthedocs should update automatically from a GitHub Action)

1. Tech stack for the documentation
2. Information resources
2. Processes for generating the HTML pages for readthedocs
3. Processes for generating the markdown for the api documentation
4. What to update, when and why

## 1. Summary of the tech stack

`scores` utilises:

 - Sphinx, with the myst parts, for making the HTML pages
 - Markdown files as the text source for all the documentation
 - Myst Parser to enable Sphinx to utilise the markdown syntax
 - The 'sphinx book' theme for Sphinx
 - Pandoc should be installed through the OS package manager separately

## 2. Useful information resources are:

 - We follow this recipe: [https://www.youtube.com/watch?v=qRSb299awB0](https://www.youtube.com/watch?v=qRSb299awB0). There isn't a good tutorial written up but the video is excellent.
 - [https://www.sphinx-doc.org/en/master/usage/markdown.html](https://www.sphinx-doc.org/en/master/usage/markdown.html)
 - [https://www.markdownguide.org/cheat-sheet/](https://www.markdownguide.org/cheat-sheet/)

## 3. Process to generate the HTML pages

 - `sphinx-build -b html docs/ htmldocs` is the key command to rebuild the HTML documentation from current sources
 - This requires docs/conf.py to be configured appropriately

## 4. Generating the markdown for the API documentation

 - Each function must be added explicitly to api.md. The autogeneration tools are not sophisticated enough to process
   the import structure used to define the public API neatly

## 5. What to update, when and why

|     what                 |     when                 |      why     |
| ------------             | -----------              | ------------ | 
|  README                  |  a new is score added    | in case it deserves a mention
|  api.md                  |  a new function is added | Each function must be added individually 
|  Explanation.ipynb       |  a new tutorial is added | Navigation throughout the docs