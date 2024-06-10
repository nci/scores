extern crate ndarray;
extern crate num_traits;
extern crate numpy;
extern crate pyo3;
extern crate rayon;
extern crate time;

#[macro_use]
extern crate colorify;

use numpy::{PyReadonlyArray2};
use pyo3::prelude::*;

use pyo3::{pymodule, types::PyModule, Bound, PyResult, Python};

mod fss_native;

#[pyfunction]
fn fss<'py>(
    _py: Python<'py>,
    pyarray_obs: PyReadonlyArray2<'py, f64>,
    pyarray_fcst: PyReadonlyArray2<'py, f64>,
    threshold: f64,
    window: (usize, usize),
) -> f64 {
    let obs_ar = pyarray_obs.as_array();
    let fcst_ar = pyarray_fcst.as_array();
    fss_native::fss(fcst_ar, obs_ar, threshold, window)
}

#[pymodule]
fn _rust_experimental<'py>(m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fss, m)?)?;
    Ok(())
}
