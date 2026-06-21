use ndarray::{Array2, ArrayView2, Axis};
use numpy::{PyArray2, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::distances::cost::squared_euclidean_cost_matrix_impl;

#[inline]
fn softmin3(a: f64, b: f64, c: f64, gamma: f64) -> f64 {
    let min_val = a.min(b).min(c);
    let exp_sum =
        ((min_val - a) / gamma).exp() + ((min_val - b) / gamma).exp() + ((min_val - c) / gamma).exp();
    min_val - gamma * exp_sum.ln()
}

fn soft_dtw_pair(x: ArrayView2<f64>, y: ArrayView2<f64>, gamma: f64) -> f64 {
    let cost = squared_euclidean_cost_matrix_impl(x, y);
    let t1 = cost.nrows();
    let t2 = cost.ncols();

    let mut r = Array2::<f64>::from_elem((t1 + 1, t2 + 1), f64::INFINITY);
    r[[0, 0]] = 0.0;

    for i in 1..=t1 {
        for j in 1..=t2 {
            r[[i, j]] = cost[[i - 1, j - 1]]
                + softmin3(r[[i - 1, j]], r[[i - 1, j - 1]], r[[i, j - 1]], gamma);
        }
    }
    r[[t1, t2]]
}

#[pyfunction]
pub fn soft_dtw(x: PyReadonlyArray2<f64>, y: PyReadonlyArray2<f64>, gamma: f64) -> f64 {
    soft_dtw_pair(x.as_array(), y.as_array(), gamma)
}

#[pyfunction]
pub fn cdist_soft_dtw<'py>(
    py: Python<'py>,
    x: PyReadonlyArray3<f64>,
    y: PyReadonlyArray3<f64>,
    gamma: f64,
) -> Bound<'py, PyArray2<f64>> {
    let x = x.as_array();
    let y = y.as_array();
    let n = x.shape()[0];
    let m = y.shape()[0];

    let x_vecs: Vec<ArrayView2<f64>> = (0..n).map(|i| x.index_axis(Axis(0), i)).collect();
    let y_vecs: Vec<ArrayView2<f64>> = (0..m).map(|j| y.index_axis(Axis(0), j)).collect();

    let flat: Vec<f64> = (0..n)
        .into_par_iter()
        .flat_map(|i| {
            (0..m)
                .map(|j| soft_dtw_pair(x_vecs[i], y_vecs[j], gamma))
                .collect::<Vec<f64>>()
        })
        .collect();

    let result = Array2::from_shape_vec((n, m), flat).unwrap();
    PyArray2::from_owned_array(py, result)
}
