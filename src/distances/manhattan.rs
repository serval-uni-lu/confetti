use ndarray::{Array2, ArrayView2, Axis};
use numpy::{PyArray2, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use rayon::prelude::*;

fn manhattan_pair(x: ArrayView2<f64>, y: ArrayView2<f64>) -> f64 {
    let mut s = 0.0;
    for i in 0..x.nrows() {
        for k in 0..x.ncols() {
            s += (x[[i, k]] - y[[i, k]]).abs();
        }
    }
    s
}

#[pyfunction]
pub fn manhattan(x: PyReadonlyArray2<f64>, y: PyReadonlyArray2<f64>) -> f64 {
    manhattan_pair(x.as_array(), y.as_array())
}

#[pyfunction]
pub fn cdist_manhattan<'py>(
    py: Python<'py>,
    x: PyReadonlyArray3<f64>,
    y: PyReadonlyArray3<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let xa = x.as_array();
    let ya = y.as_array();
    let n = xa.shape()[0];
    let m = ya.shape()[0];

    let x_vecs: Vec<ArrayView2<f64>> = (0..n).map(|i| xa.index_axis(Axis(0), i)).collect();
    let y_vecs: Vec<ArrayView2<f64>> = (0..m).map(|j| ya.index_axis(Axis(0), j)).collect();

    let flat: Vec<f64> = (0..n)
        .into_par_iter()
        .flat_map(|i| {
            (0..m)
                .map(|j| manhattan_pair(x_vecs[i], y_vecs[j]))
                .collect::<Vec<f64>>()
        })
        .collect();

    let result = Array2::from_shape_vec((n, m), flat).unwrap();
    PyArray2::from_owned_array(py, result)
}
