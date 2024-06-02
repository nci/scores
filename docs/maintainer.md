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
7. Prepare the package files (see below)
8. Prepare the PyPI update (should move to a Github Action)
9. Perform the PyPI update (should move to a Github Action)

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
|  README                  |  a new is score added    | in case it deserves a mention
|  api.md                  |  a new function is added | each function must be added individually 
|  included.md             |  a new function is added | each function (and each variation of the function name) must be added individually
|  Tutorial_Gallery.ipynb  |  a new tutorial is added | navigation throughout the docs

## This section covers checking the documentation renders properly in readthedocs

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

### Tutorial rendering

Things that render well in JupyterLab do not always render properly in readthedocs. Additionally, fixes that work well when built locally, don't always work when merged into the codebase. 

To check the rendering of tutorials in readthedocs:
  - Compare the tutorial in readthedocs against a version running in JupyterLab (as not everything renders in GitHub).
  - Check the entirety of the tutorial (sometimes things will render properly in one section, while not rendering properly in a different section of the same tutorial).
  - If you make any changes to the code cells, re-execute the Notebook in JupyterLab before committing, otherwise some things (e.g. some plots) won't render in readthedocs. Then re-check the tutorial in readthedocs to ensure the tutorial is still rendering properly.

Ideally, also check the tutorial renders properly in nbviewer (there is a link at the top of each tutorial page in readthedocs).




