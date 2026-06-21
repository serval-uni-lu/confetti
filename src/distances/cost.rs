use ndarray::{Array2, ArrayView2};
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

pub fn squared_euclidean_cost_matrix_impl(x: ArrayView2<f64>, y: ArrayView2<f64>) -> Array2<f64> {
    let t1 = x.nrows();
    let t2 = y.nrows();
    let c = x.ncols();
    let mut out = Array2::<f64>::zeros((t1, t2));
    for i in 0..t1 {
        for j in 0..t2 {
            let mut s = 0.0;
            for k in 0..c {
                let d = x[[i, k]] - y[[j, k]];
                s += d * d;
            }
            out[[i, j]] = s;
        }
    }
    out
}

#[pyfunction]
pub fn squared_euclidean_cost_matrix<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray2<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let result = squared_euclidean_cost_matrix_impl(x.as_array(), y.as_array());
    PyArray2::from_owned_array(py, result)
}
