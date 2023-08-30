# Maintainers Notes

Information relevant for package maintenance

## This guide covers how to build the documentation

1. Tech stack for the documentation
2. Processes for generating the HTML pages for readthedocs
3. Processes for generating the markdown for the api documentation
4. What to update, when and why

### Summary of the tech stack

`scores` utilises:

 - Sphinx, with the myst parts, for making the HTML pages
 - Markdown files as the text source for all the documentation
 - Myst Parser to enable Sphinx to utilise the markdown syntax
 - The 'sphinx book' theme for Sphinx

#### Useful information resources are:

 - We follow this recipe: [https://www.youtube.com/watch?v=qRSb299awB0](https://www.youtube.com/watch?v=qRSb299awB0). There isn't a good tutorial written up but the video is excellent.
 - [https://www.sphinx-doc.org/en/master/usage/markdown.html](https://www.sphinx-doc.org/en/master/usage/markdown.html)
 - [https://www.markdownguide.org/cheat-sheet/](https://www.markdownguide.org/cheat-sheet/)

### Process to generate the HTML pages

 - `sphinx-build -b html docs/ htmldocs` is the key command to rebuild the HTML documentation from current sources
 - This requires docs/conf.py to be configured appropriately

### Generating the markdown for the API documentation

 - Each function must be added explicitly to api.md. The autogeneration tools are not sophisticated enough to process
   the import structure used to define the public API neatly