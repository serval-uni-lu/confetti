use ndarray::{Array2, ArrayView2, Axis};
use numpy::{PyArray2, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Cost matrix
// ---------------------------------------------------------------------------

fn squared_euclidean_cost_matrix_impl(x: ArrayView2<f64>, y: ArrayView2<f64>) -> Array2<f64> {
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
fn squared_euclidean_cost_matrix<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray2<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let x = x.as_array();
    let y = y.as_array();
    let result = squared_euclidean_cost_matrix_impl(x, y);
    PyArray2::from_owned_array(py, result)
}

// ---------------------------------------------------------------------------
// DTW
// ---------------------------------------------------------------------------

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

fn dtw_pair(x: ArrayView2<f64>, y: ArrayView2<f64>, radius: Option<usize>) -> f64 {
    let cost = squared_euclidean_cost_matrix_impl(x, y);
    let mask = radius.map(|r| sakoe_chiba_mask(cost.nrows(), cost.ncols(), r));
    let d = dtw_accumulated_cost(&cost, mask.as_ref());
    d[[cost.nrows(), cost.ncols()]].sqrt()
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

#[pyfunction]
fn dtw(x: PyReadonlyArray2<f64>, y: PyReadonlyArray2<f64>, sakoe_chiba_radius: Option<usize>) -> f64 {
    dtw_pair(x.as_array(), y.as_array(), sakoe_chiba_radius)
}

fn dtw_with_path(
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
fn dtw_with_path_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray2<f64>,
    sakoe_chiba_radius: Option<usize>,
) -> (f64, Vec<(usize, usize)>) {
    let _ = py;
    dtw_with_path(x.as_array(), y.as_array(), sakoe_chiba_radius)
}

#[pyfunction]
fn cdist_dtw<'py>(
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

// ---------------------------------------------------------------------------
// Soft-DTW
// ---------------------------------------------------------------------------

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
            r[[i, j]] = cost[[i - 1, j - 1]] + softmin3(r[[i - 1, j]], r[[i - 1, j - 1]], r[[i, j - 1]], gamma);
        }
    }
    r[[t1, t2]]
}

#[pyfunction]
fn soft_dtw(x: PyReadonlyArray2<f64>, y: PyReadonlyArray2<f64>, gamma: f64) -> f64 {
    soft_dtw_pair(x.as_array(), y.as_array(), gamma)
}

#[pyfunction]
fn cdist_soft_dtw<'py>(
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

// ---------------------------------------------------------------------------
// GAK
// ---------------------------------------------------------------------------

fn gak_gram_matrix(x: ArrayView2<f64>, y: ArrayView2<f64>, sigma: f64) -> Array2<f64> {
    let cost = squared_euclidean_cost_matrix_impl(x, y);
    let sigma_sq_2 = 2.0 * sigma * sigma;
    cost.mapv(|c| {
        let g = (-c / sigma_sq_2).exp();
        g * (2.0 - g).recip()
    })
}

fn unnormalized_gak(x: ArrayView2<f64>, y: ArrayView2<f64>, sigma: f64) -> f64 {
    let gram = gak_gram_matrix(x, y, sigma);
    let t1 = gram.nrows();
    let t2 = gram.ncols();

    let mut cum = Array2::<f64>::zeros((t1 + 1, t2 + 1));
    cum[[0, 0]] = 1.0;

    for i in 0..t1 {
        for j in 0..t2 {
            cum[[i + 1, j + 1]] = (cum[[i, j + 1]] + cum[[i + 1, j]] + cum[[i, j]]) * gram[[i, j]];
        }
    }
    cum[[t1, t2]]
}

fn gak_pair(x: ArrayView2<f64>, y: ArrayView2<f64>, sigma: f64) -> f64 {
    let k_xy = unnormalized_gak(x, y, sigma);
    let k_xx = unnormalized_gak(x, x, sigma);
    let k_yy = unnormalized_gak(y, y, sigma);
    k_xy / (k_xx * k_yy).sqrt()
}

#[pyfunction]
fn gak(x: PyReadonlyArray2<f64>, y: PyReadonlyArray2<f64>, sigma: f64) -> f64 {
    gak_pair(x.as_array(), y.as_array(), sigma)
}

#[pyfunction]
fn unnormalized_gak_py(x: PyReadonlyArray2<f64>, y: PyReadonlyArray2<f64>, sigma: f64) -> f64 {
    unnormalized_gak(x.as_array(), y.as_array(), sigma)
}

#[pyfunction]
fn cdist_gak<'py>(
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

    let k_xx: Vec<f64> = x_vecs.par_iter().map(|xi| unnormalized_gak(*xi, *xi, sigma)).collect();
    let k_yy: Vec<f64> = y_vecs.par_iter().map(|yj| unnormalized_gak(*yj, *yj, sigma)).collect();

    let flat: Vec<f64> = (0..n)
        .into_par_iter()
        .flat_map(|i| {
            (0..m)
                .map(|j| {
                    let k_xy = unnormalized_gak(x_vecs[i], y_vecs[j], sigma);
                    k_xy / (k_xx[i] * k_yy[j]).sqrt()
                })
                .collect::<Vec<f64>>()
        })
        .collect();

    let result = Array2::from_shape_vec((n, m), flat).unwrap();
    PyArray2::from_owned_array(py, result)
}

// ---------------------------------------------------------------------------
// Manhattan
// ---------------------------------------------------------------------------

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
fn manhattan(x: PyReadonlyArray2<f64>, y: PyReadonlyArray2<f64>) -> f64 {
    manhattan_pair(x.as_array(), y.as_array())
}

#[pyfunction]
fn cdist_manhattan<'py>(
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

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

#[pyfunction]
fn rust_available() -> bool {
    true
}

#[pymodule]
fn _rust_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_available, m)?)?;
    m.add_function(wrap_pyfunction!(squared_euclidean_cost_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(dtw, m)?)?;
    m.add_function(wrap_pyfunction!(dtw_with_path_py, m)?)?;
    m.add_function(wrap_pyfunction!(cdist_dtw, m)?)?;
    m.add_function(wrap_pyfunction!(soft_dtw, m)?)?;
    m.add_function(wrap_pyfunction!(cdist_soft_dtw, m)?)?;
    m.add_function(wrap_pyfunction!(gak, m)?)?;
    m.add_function(wrap_pyfunction!(unnormalized_gak_py, m)?)?;
    m.add_function(wrap_pyfunction!(cdist_gak, m)?)?;
    m.add_function(wrap_pyfunction!(manhattan, m)?)?;
    m.add_function(wrap_pyfunction!(cdist_manhattan, m)?)?;
    Ok(())
}
