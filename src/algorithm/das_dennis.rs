use ndarray::Array2;
use numpy::PyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn das_dennis_impl(n_dim: usize, n_partitions: usize) -> Vec<Vec<f64>> {
    if n_dim == 1 {
        return vec![vec![1.0]];
    }
    if n_partitions == 0 {
        return vec![vec![1.0 / n_dim as f64; n_dim]];
    }

    let mut points: Vec<Vec<f64>> = Vec::new();
    let mut stack: Vec<(Vec<f64>, usize)> = vec![(Vec::new(), n_partitions)];

    while let Some((partial, beta)) = stack.pop() {
        let depth = partial.len();
        if depth == n_dim - 1 {
            let mut point = partial;
            point.push(beta as f64 / n_partitions as f64);
            points.push(point);
        } else {
            for i in (0..=beta).rev() {
                let mut next = partial.clone();
                next.push(i as f64 / n_partitions as f64);
                stack.push((next, beta - i));
            }
        }
    }

    points.sort_by(|a, b| {
        for col in 0..n_dim {
            match a[col].partial_cmp(&b[col]).unwrap() {
                std::cmp::Ordering::Equal => continue,
                other => return other,
            }
        }
        std::cmp::Ordering::Equal
    });

    points
}

#[pyfunction]
pub fn das_dennis_py<'py>(
    py: Python<'py>,
    n_dim: usize,
    n_partitions: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if n_dim < 1 {
        return Err(PyValueError::new_err(format!(
            "n_dim must be >= 1, got {n_dim}"
        )));
    }
    let points = das_dennis_impl(n_dim, n_partitions);
    let n_points = points.len();
    let flat: Vec<f64> = points.into_iter().flatten().collect();
    let arr = Array2::from_shape_vec((n_points, n_dim), flat).unwrap();
    Ok(PyArray2::from_owned_array(py, arr))
}
