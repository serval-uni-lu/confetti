use ndarray::{Array2, ArrayView1};
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

fn gower_pair(x: ArrayView1<f64>, y: ArrayView1<f64>, cat_mask: &[bool], ranges: ArrayView1<f64>) -> f64 {
    let n = x.len();
    let mut sum = 0.0;
    for i in 0..n {
        if cat_mask[i] {
            if (x[i] - y[i]).abs() > f64::EPSILON {
                sum += 1.0;
            }
        } else if ranges[i] > 0.0 {
            sum += (x[i] - y[i]).abs() / ranges[i];
        }
    }
    sum / n as f64
}

#[pyfunction]
pub fn gower(
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
    cat_mask: PyReadonlyArray1<bool>,
    ranges: PyReadonlyArray1<f64>,
) -> f64 {
    let cat_vec: Vec<bool> = cat_mask.as_array().to_vec();
    gower_pair(x.as_array(), y.as_array(), &cat_vec, ranges.as_array())
}

#[pyfunction]
pub fn cdist_gower<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray2<f64>,
    cat_mask: PyReadonlyArray1<bool>,
    ranges: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let xa = x.as_array();
    let ya = y.as_array();
    let n = xa.shape()[0];
    let m = ya.shape()[0];
    let cat_vec: Vec<bool> = cat_mask.as_array().to_vec();
    let ranges_arr = ranges.as_array();

    let x_rows: Vec<ArrayView1<f64>> = (0..n).map(|i| xa.row(i)).collect();
    let y_rows: Vec<ArrayView1<f64>> = (0..m).map(|j| ya.row(j)).collect();

    let flat: Vec<f64> = (0..n)
        .into_par_iter()
        .flat_map(|i| {
            (0..m)
                .map(|j| gower_pair(x_rows[i], y_rows[j], &cat_vec, ranges_arr))
                .collect::<Vec<f64>>()
        })
        .collect();

    let result = Array2::from_shape_vec((n, m), flat).unwrap();
    PyArray2::from_owned_array(py, result)
}
