import scores


def test_plotdata_rank_histogram():
    assert scores.plotdata.rank_histogram is scores.probability.rank_histogram
