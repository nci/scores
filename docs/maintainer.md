# Maintainers Notes

Information relevant for package maintenance

## This section covers the process for making a release from the develop branch

1. Run through the notebooks manually
2. Run the unit tests
3. Prepare the merge request in github (do not do a rebase merge to main)
4. Check readthedocs rebuilds correctly
5. Prepare the PyPI update (should move to a Github Action)
6. Perform the PyPI update (should move to a Github Action)


## This section covers how to build the documentation

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
|  README                  |  a new is score added    | README does not read from summary_table_of_scores.md
|  summary_table_of_scores |  a new is score added    | The rest of the docs includes this in various spots
|  api.md                  |  a new function is added | Each function must be added individually