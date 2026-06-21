use ndarray::{Array2, ArrayView2, Axis};
use numpy::{PyArray2, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::distances::cost::squared_euclidean_cost_matrix_impl;

fn dtw_accumulated_cost(cost: &Array2<f64>, mask: Option<&Array2<bool>>) -> Array2<f64> {
    let t1 = cost.nrows();
    let t2 = cost.ncols();
    let mut d = Array2::<f64>::from_elem((t1 + 1, t2 + 1), f64::INFINITY);
    d[[0, 0]] = 0.0;

    for i in 0..t1 {
        for j in 0..t2 {
            if let Some(m) = mask {
                if !m[[i, j]] {
                    continue;
                }
            }
            let c = cost[[i, j]];
            let prev = d[[i, j + 1]].min(d[[i + 1, j]]).min(d[[i, j]]);
            d[[i + 1, j + 1]] = c + prev;
        }
    }
    d
}

fn sakoe_chiba_mask(t1: usize, t2: usize, radius: usize) -> Array2<bool> {
    let mut mask = Array2::<bool>::from_elem((t1, t2), false);
    let denom = if t1 > 1 { (t1 - 1) as f64 } else { 1.0 };
    for i in 0..t1 {
        let diag = (t2 as f64 - 1.0) * (i as f64) / denom;
        for j in 0..t2 {
            if (j as f64 - diag).abs() <= radius as f64 {
                mask[[i, j]] = true;
            }
        }
    }
    mask
}

pub fn dtw_pair(x: ArrayView2<f64>, y: ArrayView2<f64>, radius: Option<usize>) -> f64 {
    let cost = squared_euclidean_cost_matrix_impl(x, y);
    let mask = radius.map(|r| sakoe_chiba_mask(cost.nrows(), cost.ncols(), r));
    let d = dtw_accumulated_cost(&cost, mask.as_ref());
    d[[cost.nrows(), cost.ncols()]].sqrt()
}

pub fn dtw_with_path_impl(
    x: ArrayView2<f64>,
    y: ArrayView2<f64>,
    radius: Option<usize>,
) -> (f64, Vec<(usize, usize)>) {
    let cost = squared_euclidean_cost_matrix_impl(x, y);
    let mask = radius.map(|r| sakoe_chiba_mask(cost.nrows(), cost.ncols(), r));
    let d = dtw_accumulated_cost(&cost, mask.as_ref());

    let t1 = cost.nrows();
    let t2 = cost.ncols();
    let mut i = t1;
    let mut j = t2;
    let mut path = Vec::new();
    while i > 0 && j > 0 {
        path.push((i - 1, j - 1));
        let diag = d[[i - 1, j - 1]];
        let up = d[[i - 1, j]];
        let left = d[[i, j - 1]];
        if diag <= up && diag <= left {
            i -= 1;
            j -= 1;
        } else if up <= left {
            i -= 1;
        } else {
            j -= 1;
        }
    }
    path.reverse();
    (d[[t1, t2]].sqrt(), path)
}

#[pyfunction]
pub fn dtw(
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray2<f64>,
    sakoe_chiba_radius: Option<usize>,
) -> f64 {
    dtw_pair(x.as_array(), y.as_array(), sakoe_chiba_radius)
}

#[pyfunction]
pub fn dtw_with_path_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray2<f64>,
    sakoe_chiba_radius: Option<usize>,
) -> (f64, Vec<(usize, usize)>) {
    let _ = py;
    dtw_with_path_impl(x.as_array(), y.as_array(), sakoe_chiba_radius)
}

#[pyfunction]
pub fn cdist_dtw<'py>(
    py: Python<'py>,
    x: PyReadonlyArray3<f64>,
    y: PyReadonlyArray3<f64>,
    sakoe_chiba_radius: Option<usize>,
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
                .map(|j| dtw_pair(x_vecs[i], y_vecs[j], sakoe_chiba_radius))
                .collect::<Vec<f64>>()
        })
        .collect();

    let result = Array2::from_shape_vec((n, m), flat).unwrap();
    PyArray2::from_owned_array(py, result)
}
