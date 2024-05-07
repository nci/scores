use ndarray::prelude::*;
use num_traits::{MulAddAssign, Num};
use rayon::prelude::*;
use time::Instant;

macro_rules! fss_type_bound {
    ($t:ty) => {
        impl FssType for $t {
            // TODO: potential upconversions issues here...
            // but for now we assume everyone uses 64bit
            fn as_float(self: Self) -> f64 {
                self as f64
            }

            fn from(b: bool) -> Self {
                <$t as From<bool>>::from(b)
            }
        }
    };
    ($t:ty, $($ts:ty),+) => {
        fss_type_bound! { $t }
        fss_type_bound! { $($ts),+ }
    };
}

fss_type_bound! {i64, f64}
pub trait FssType: Num + MulAddAssign + Send + Default + Copy + Sync {
    fn as_float(self: Self) -> f64;
    fn from(b: bool) -> Self;
}
type Window = (usize, usize);
type BoundingAreas = (i64, i64, i64, i64);

/// e.g. if input has col alignment, return aligned as | row 1 | row 2 | ... | row ncol |
pub fn transpose_slice_to_vec<T: FssType>(buf: Box<[T]>, nrows: usize, ncols: usize) -> Box<[T]> {
    let mut box_buf = vec![T::default(); buf.len()].into_boxed_slice();
    let mut k = 0;
    for i in 0..ncols {
        for j in 0..nrows {
            let idx = j * ncols + i;
            box_buf[k] = buf[idx];
            k += 1;
        }
    }
    box_buf
}

/// Input is 2D ndarray
/// Note: prefer this to prefix_sum_vec2, since the internal libraries are more optimized
pub fn prefix_sum_array2<T: FssType>(ar: &Array2<T>) -> Array2<T> {
    let mut ar = ar.clone();
    for ax in [0, 1] {
        ar.lanes_mut(Axis(ax))
            .into_iter()
            .par_bridge()
            .for_each(|mut lane| {
                (1..lane.len()).into_iter().for_each(|i| {
                    lane[i] = lane[i].add(lane[i - 1]);
                })
            });
    }
    ar
}

/// Note: For experimental learning purposes only. Preferablly use prefix_sum_array2.
/// Input is a 1D buffer, but represents 2D data.
/// This is purely for supporting non-standard array libs, but the user is responsible for
/// efficiently converting it to a 1D buffer before running this.
pub fn prefix_sum_vec2<T: FssType>(v: &[T], nrows: usize, ncols: usize) -> Array2<T> {
    let mut v = transpose_slice_to_vec(v.into(), nrows, ncols);
    v.par_chunks_mut(nrows).for_each(|row| {
        for i in 1..row.len() {
            row[i] = row[i].add(row[i - 1]);
        }
    });
    let mut v = transpose_slice_to_vec(v, ncols, nrows);
    v.par_chunks_mut(ncols).for_each(|col| {
        for i in 1..col.len() {
            col[i] = col[i].add(col[i - 1]);
        }
    });
    Array2::<T>::from_shape_vec((nrows, ncols), v.to_vec()).unwrap()
}

pub fn apply_threshold<T: FssType, U: FssType>(ar: &ArrayView2<T>, t: U) -> Array2<i64> {
    ar.map(|x| (x.as_float() > t.as_float()) as i64)
}

struct AccResult<T: FssType>(T, T, T); // obs, fcst, abs(obs - fcst)
impl<T: FssType> Default for AccResult<T> {
    fn default() -> AccResult<T> {
        AccResult(T::default(), T::default(), T::default())
    }
}

/// Computes the forecast skill score
/// Reference:
pub fn fss<T: FssType, U: FssType>(
    fcst: ArrayView2<T>,
    obs: ArrayView2<T>,
    thr: U,
    win: Window,
) -> f64 {
    assert_eq!(fcst.shape(), obs.shape(), "Fcst and Obs shape don't match");
    let h = fcst.dim().0;
    let w = fcst.dim().1;
    assert!(
        h > win.0 && w > win.1,
        "Window must be smaller than data shape"
    );
    let _t = Instant::now();
    print_elapsed("start", _t);

    let fcst_th = apply_threshold(&fcst, thr);
    let obs_th = apply_threshold(&obs, thr);

    print_elapsed("apply thresholds", _t);

    // compute prefix sums (Integral image)
    let ps_fcst = prefix_sum_array2(&fcst_th);
    let ps_obs = prefix_sum_array2(&obs_th);

    print_elapsed("prefix sums", _t);

    // accumulate window vals, without exceeding image width
    let mut bounding_areas = Vec::<(BoundingAreas, BoundingAreas)>::new();

    for i in 0..(h - win.0) {
        for j in 0..(w - win.1) {
            let ba_obs: BoundingAreas = (
                ps_obs[(i, j)],
                ps_obs[(i + win.0, j)],
                ps_obs[(i, j + win.1)],
                ps_obs[(i + win.0, j + win.1)],
            );

            let ba_fcst: BoundingAreas = (
                ps_fcst[(i, j)],
                ps_fcst[(i + win.0, j)],
                ps_fcst[(i, j + win.1)],
                ps_fcst[(i + win.0, j + win.1)],
            );
            bounding_areas.push((ba_obs, ba_fcst));
        }
    }

    print_elapsed("bounding areas", _t);

    // accumulate results
    let acc_results: (i64, i64, i64) = bounding_areas
        .into_par_iter()
        .fold(
            || (0, 0, 0),
            |acc: (i64, i64, i64), ba: (BoundingAreas, BoundingAreas)| {
                // Compute areas from integral image
                // X0     X1
                // +------+
                // |      |
                // |      |
                // +------+
                // X2     X3
                // area = X3 - X1 - X2 + X0
                let ba_obs = ba.0;
                let ba_fcst = ba.1;
                let obs_area = ba_obs.3 - ba_obs.1 - ba_obs.2 + ba_obs.0;
                let fcst_area = ba_fcst.3 - ba_fcst.1 - ba_fcst.2 + ba_fcst.0;
                let obs_fcst_diff = obs_area - fcst_area;

                (
                    obs_area * obs_area + acc.0,
                    fcst_area * fcst_area + acc.1,
                    obs_fcst_diff * obs_fcst_diff + acc.2,
                )
            },
        )
        .reduce(|| (0, 0, 0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));

    print_elapsed("fss: ", _t);

    let denom: f64 = acc_results.0.as_float() + acc_results.1.as_float();
    let numer: f64 = acc_results.2.as_float();
    let mut fss_score = 0.0;

    if denom > 0.0 {
        fss_score = 1.0 - numer / denom;
    }

    printlnc!(green: ">>> fss_score: {}", fss_score);
    fss_score
}

fn print_elapsed(title: &str, start: Instant) {
    let time_elapsed = start.elapsed();
    let separator = if title.len() > 0 {
        " (time elapsed): "
    } else {
        ""
    };
    printlnc!(
        yellow: "    {}{}{:3}.{:03}",
        title,
        separator,
        time_elapsed.as_seconds_f64(),
        time_elapsed.subsec_milliseconds()
    );
}
