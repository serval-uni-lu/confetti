use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

fn find_extreme_points(f_front: ArrayView2<f64>, ideal: ArrayView1<f64>) -> Vec<usize> {
    let n_obj = f_front.ncols();
    let n_front = f_front.nrows();
    let mut extreme = vec![0usize; n_obj];

    for j in 0..n_obj {
        let mut best_idx = 0;
        let mut best_asf = f64::INFINITY;
        for i in 0..n_front {
            let mut asf = f64::NEG_INFINITY;
            for k in 0..n_obj {
                let w = if k == j { 1.0 } else { 1e6 };
                let val = (f_front[[i, k]] - ideal[k]) * w;
                if val > asf {
                    asf = val;
                }
            }
            if asf < best_asf {
                best_asf = asf;
                best_idx = i;
            }
        }
        extreme[j] = best_idx;
    }
    extreme
}

fn compute_nadir(extreme_points: ArrayView2<f64>, ideal: ArrayView1<f64>) -> Option<Vec<f64>> {
    let n_dim = extreme_points.ncols();

    let mut translated = nalgebra::DMatrix::<f64>::zeros(n_dim, n_dim);
    for i in 0..n_dim {
        for j in 0..n_dim {
            translated[(i, j)] = extreme_points[[i, j]] - ideal[j];
        }
    }

    let rhs = nalgebra::DVector::<f64>::from_element(n_dim, 1.0);
    let lu = translated.lu();
    let plane = lu.solve(&rhs)?;

    for j in 0..n_dim {
        if plane[j].abs() < 1e-14 {
            return None;
        }
    }

    let mut nadir = Vec::with_capacity(n_dim);
    for j in 0..n_dim {
        let intercept = 1.0 / plane[j];
        if intercept <= 0.0 {
            return None;
        }
        nadir.push(ideal[j] + intercept);
    }
    Some(nadir)
}

fn normalization_update_impl(
    ideal_point: ArrayView1<f64>,
    f: ArrayView2<f64>,
    first_front: &[i64],
    n_dim: usize,
) -> (Vec<f64>, Option<Vec<f64>>) {
    let n_obj = f.ncols();

    let mut new_ideal: Vec<f64> = ideal_point.to_vec();
    for k in 0..n_obj {
        let mut min_val = new_ideal[k];
        for i in 0..f.nrows() {
            if f[[i, k]] < min_val {
                min_val = f[[i, k]];
            }
        }
        new_ideal[k] = min_val;
    }

    let front_indices: Vec<usize> = first_front.iter().map(|&i| i as usize).collect();
    let n_front = front_indices.len();

    if n_front < n_dim {
        let mut nadir = vec![f64::NEG_INFINITY; n_obj];
        for &idx in &front_indices {
            for k in 0..n_obj {
                if f[[idx, k]] > nadir[k] {
                    nadir[k] = f[[idx, k]];
                }
            }
        }
        return (new_ideal, Some(nadir));
    }

    let mut f_front = Array2::<f64>::zeros((n_front, n_obj));
    for (row, &idx) in front_indices.iter().enumerate() {
        for k in 0..n_obj {
            f_front[[row, k]] = f[[idx, k]];
        }
    }

    let ideal_arr = Array1::from_vec(new_ideal.clone());
    let extreme_idx = find_extreme_points(f_front.view(), ideal_arr.view());

    let mut extreme_pts = Array2::<f64>::zeros((n_dim, n_obj));
    for (row, &idx) in extreme_idx.iter().enumerate() {
        for k in 0..n_obj {
            extreme_pts[[row, k]] = f_front[[idx, k]];
        }
    }

    let nadir = compute_nadir(extreme_pts.view(), ideal_arr.view());

    match nadir {
        Some(n) => (new_ideal, Some(n)),
        None => {
            let mut fallback = vec![f64::NEG_INFINITY; n_obj];
            for row in 0..n_front {
                for k in 0..n_obj {
                    if f_front[[row, k]] > fallback[k] {
                        fallback[k] = f_front[[row, k]];
                    }
                }
            }
            (new_ideal, Some(fallback))
        }
    }
}

#[pyfunction]
pub fn normalization_update_py<'py>(
    py: Python<'py>,
    ideal_point: PyReadonlyArray1<f64>,
    f: PyReadonlyArray2<f64>,
    first_front: PyReadonlyArray1<i64>,
    n_dim: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Py<PyAny>)> {
    let front_slice = first_front.as_array();
    let front_vec: Vec<i64> = front_slice.to_vec();
    let (new_ideal, nadir_opt) = normalization_update_impl(
        ideal_point.as_array(),
        f.as_array(),
        &front_vec,
        n_dim,
    );
    let ideal_arr = PyArray1::from_vec(py, new_ideal);
    let nadir_py = match nadir_opt {
        Some(n) => PyArray1::from_vec(py, n).into_any().unbind(),
        None => py.None(),
    };
    Ok((ideal_arr, nadir_py))
}
