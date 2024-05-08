import numba
import numpy as np
from numba import prange

from scores._rust_experimental import fss as rust_fss


def call_rust_fss(obs, fcst, thr, win):
    return rust_fss(obs, fcst, thr, win)


def generate_sample_data(mu_obs, std_obs, mu_fcst, std_fcst, img_h, img_w):
    obs = np.random.normal(loc=mu_obs, scale=std_obs, size=(img_h, img_w))
    fcst = np.random.normal(loc=mu_fcst, scale=std_fcst, size=(img_h, img_w))
    return (obs, fcst)


def test_rust_fss_basic():
    img_h = 1000
    img_w = 1200
    threshold = 0.5
    window = (100, 100)

    (obs, fcst) = generate_sample_data(
        mu_obs=0.0,
        std_obs=1.0,
        mu_fcst=1.0,
        std_fcst=2.0,
        img_h=img_h,
        img_w=img_w,
    )

    res = call_rust_fss(obs, fcst, threshold, window)
    print(f"fss_rust result: {res}")

    out = 0
    res_numba = fss_numba(fcst, obs, img_w, img_h, threshold, window[0], out, _print=True)


def fss_numba(arr_1, arr_2, w, h, thresh, window_size, out, _print=False):
    out_1 = np.zeros((h, w), dtype=np.int64)
    out_2 = np.zeros((h, w), dtype=np.int64)
    obs_total = np.zeros(h - window_size, dtype=np.int64)
    fcst_total = np.zeros(h - window_size, dtype=np.int64)
    diff_total = np.zeros(h - window_size, dtype=np.int64)
    arr_1_bool = np.array(arr_1 > thresh, dtype=np.int64)
    arr_2_bool = np.array(arr_2 > thresh, dtype=np.int64)
    sum_area_table(arr_1_bool, out_1, w, h)
    sum_area_table(arr_2_bool, out_2, w, h)
    out = fss_parallel(out_1, out_2, w, h, window_size, obs_total, fcst_total, diff_total, out)
    if _print:
        print("fss_numba result: ", out)
    return out


@numba.jit(parallel=True)
def sum_area_table(arr_, out_, w, h):
    out_[0, :] = arr_[0, :]
    out_[:, 0] = arr_[:, 0]
    for i in prange(0, h):
        for j in range(1, w):
            out_[i, j] = out_[i, j - 1] + arr_[i, j]
    for j in prange(0, w):
        for i in range(1, h):
            out_[i, j] = out_[i - 1, j] + out_[i, j]
    return out_


@numba.jit(
    parallel=True,
    locals=dict(
        obs_total_sum=numba.int64,
        fcst_total_sum=numba.int64,
        diff_total_sum=numba.int64,
    ),
)
def fss_parallel(sat_1, sat_2, w, h, window_size, obs_total, fcst_total, diff_total, out):
    obs_total_sum = 0
    fcst_total_sum = 0
    diff_total_sum = 0

    for i in prange(0, h - window_size):
        obs_sum = 0.0
        fcst_sum = 0.0
        diff_sum = 0.0

        for j in range(0, w - window_size):
            obs_a = sat_1[i][j]
            obs_b = sat_1[i][j + window_size]
            obs_c = sat_1[i + window_size][j]
            obs_d = sat_1[i + window_size][j + window_size]

            fcst_a = sat_2[i][j]
            fcst_b = sat_2[i][j + window_size]
            fcst_c = sat_2[i + window_size][j]
            fcst_d = sat_2[i + window_size][j + window_size]

            _obs_sum = obs_d - obs_b - obs_c + obs_a
            _fcst_sum = fcst_d - fcst_b - fcst_c + fcst_a
            _diff_sum = _obs_sum - _fcst_sum
            obs_sum += _obs_sum * _obs_sum
            fcst_sum += _fcst_sum * _fcst_sum
            diff_sum += _diff_sum * _diff_sum

        obs_total[i] = obs_sum
        fcst_total[i] = fcst_sum
        diff_total[i] = diff_sum

    # >>> implicit barrier here?
    # I think when we return from the for loop things get synchronized
    # automatically with an implicit barrier...

    obs_total_sum = obs_total.sum()
    fcst_total_sum = fcst_total.sum()
    diff_total_sum = diff_total.sum()

    out = 1.0 - float(diff_total_sum) / float(obs_total_sum + fcst_total_sum)
    return out


if __name__ == "__main__":
    test_rust_fss_basic()
