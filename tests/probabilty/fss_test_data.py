import numpy as np


def generate(obs_pdf, fcst_pdf, *, seed=42):
    np.random.seed(42)
    h = 400
    w = 600
    obs = np.random.normal(loc=obs_pdf[0], scale=obs_pdf[1], size=(h, w))
    fcst = np.random.normal(loc=fcst_pdf[0], scale=fcst_pdf[1], size=(h, w))
    return (obs, fcst)
