A new score or metric should be developed on a separate feature branch, rebased against the main branch. Each merge request should include:

    The implementation of the new metric or score in xarray, ideally with support for pandas and dask
    100% unit test coverage
    A tutorial notebook showcasing the use of that metric or score, ideally based on the standard sample data
    API documentation (docstrings) using Napoleon (google) style, making sure to clearly explain the use of the metrics
    A reference to the paper which described the metrics, added to the API documentation
    For metrics which do not have a paper reference, an online source or reference should be provided
    For metrics which are still under development or which have not yet had an academic publication, they will be placed in a holding area within the API until the method has been properly published and peer reviewed (i.e. scores.emerging). The 'emerging' area of the API is subject to rapid change, still of sufficient community interest to include, similar to a 'preprint' of a score or metric.
    Add your score to summary_table_of_scores.md in the documentation

All merge requests should comply with the coding standards outlined in this document. Merge requests will undergo both a code review and a science review. The code review will focus on coding style, performance and test coverage. The science review will focus on the mathematical correctness of the implementation and the suitability of the method for inclusion within 'scores'.

A github ticket should be created explaining the metric which is being implemented and why it is useful.
