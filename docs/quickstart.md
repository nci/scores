# Quick Start Guide (with Tutorials)

Install and run tutorials, from a local checkout

```bash
git clone git@github.com:nci/scores.git
python3 -m venv <path_to_environment>
source <path_to_environment>/bin/activate
# Note - specifying [tutorial] includes dependencies for running the Jupyter Lab server. #
pip install -e .[tutorial]   
cd tutorials
jupyter lab
```

## Tutorials and Worked Examples

Users must set up the required sample data files. The process for downloading or generating this sample data is set out in the notebook "First - Data Fetching" and this should be done up-front.

Each score in the package has its own notebook. For very similar scores, this means there may be some repetition, but this approach ensures that each score has clear worked examples which can support users in understanding how to use both the API and the score itself.

## Included Metrics and Scores

```{include} summary_table_of_scores.md
```

Each score is fully documented in the API documentation [ api.md ](api.md). 