use ndarray::{Array2, ArrayView2, Axis};
use numpy::{PyArray2, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::distances::cost::squared_euclidean_cost_matrix_impl;

fn gak_gram_matrix(x: ArrayView2<f64>, y: ArrayView2<f64>, sigma: f64) -> Array2<f64> {
    let cost = squared_euclidean_cost_matrix_impl(x, y);
    let sigma_sq_2 = 2.0 * sigma * sigma;
    cost.mapv(|c| {
        let g = (-c / sigma_sq_2).exp();
        g * (2.0 - g).recip()
    })
}

fn unnormalized_gak_impl(x: ArrayView2<f64>, y: ArrayView2<f64>, sigma: f64) -> f64 {
    let gram = gak_gram_matrix(x, y, sigma);
    let t1 = gram.nrows();
    let t2 = gram.ncols();

    let mut cum = Array2::<f64>::zeros((t1 + 1, t2 + 1));
    cum[[0, 0]] = 1.0;

    for i in 0..t1 {
        for j in 0..t2 {
            cum[[i + 1, j + 1]] =
                (cum[[i, j + 1]] + cum[[i + 1, j]] + cum[[i, j]]) * gram[[i, j]];
        }
    }
    cum[[t1, t2]]
}

fn gak_pair(x: ArrayView2<f64>, y: ArrayView2<f64>, sigma: f64) -> f64 {
    let k_xy = unnormalized_gak_impl(x, y, sigma);
    let k_xx = unnormalized_gak_impl(x, x, sigma);
    let k_yy = unnormalized_gak_impl(y, y, sigma);
    k_xy / (k_xx * k_yy).sqrt()
}

#[pyfunction]
pub fn gak(x: PyReadonlyArray2<f64>, y: PyReadonlyArray2<f64>, sigma: f64) -> f64 {
    gak_pair(x.as_array(), y.as_array(), sigma)
}

#[pyfunction]
pub fn unnormalized_gak_py(
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray2<f64>,
    sigma: f64,
) -> f64 {
    unnormalized_gak_impl(x.as_array(), y.as_array(), sigma)
}

#[pyfunction]
pub fn cdist_gak<'py>(
    py: Python<'py>,
    x: PyReadonlyArray3<f64>,
    y: PyReadonlyArray3<f64>,
    sigma: f64,
) -> Bound<'py, PyArray2<f64>> {
    let xa = x.as_array();
    let ya = y.as_array();
    let n = xa.shape()[0];
    let m = ya.shape()[0];

    let x_vecs: Vec<ArrayView2<f64>> = (0..n).map(|i| xa.index_axis(Axis(0), i)).collect();
    let y_vecs: Vec<ArrayView2<f64>> = (0..m).map(|j| ya.index_axis(Axis(0), j)).collect();

    let k_xx: Vec<f64> = x_vecs
        .par_iter()
        .map(|xi| unnormalized_gak_impl(*xi, *xi, sigma))
        .collect();
    let k_yy: Vec<f64> = y_vecs
        .par_iter()
        .map(|yj| unnormalized_gak_impl(*yj, *yj, sigma))
        .collect();

    let flat: Vec<f64> = (0..n)
        .into_par_iter()
        .flat_map(|i| {
            (0..m)
                .map(|j| {
                    let k_xy = unnormalized_gak_impl(x_vecs[i], y_vecs[j], sigma);
                    k_xy / (k_xx[i] * k_yy[j]).sqrt()
                })
                .collect::<Vec<f64>>()
        })
        .collect();

    let result = Array2::from_shape_vec((n, m), flat).unwrap();
    PyArray2::from_owned_array(py, result)
}
