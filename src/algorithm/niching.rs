use ndarray::{Array2, ArrayView1, ArrayView2};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

fn perpendicular_distance_impl(n: ArrayView2<f64>, ref_dirs: ArrayView2<f64>) -> Array2<f64> {
    let n_solutions = n.nrows();
    let n_ref = ref_dirs.nrows();
    let n_obj = n.ncols();

    let u_dot_u: Vec<f64> = (0..n_ref)
        .map(|j| {
            let mut s = 0.0;
            for k in 0..n_obj {
                s += ref_dirs[[j, k]] * ref_dirs[[j, k]];
            }
            s.max(1e-14)
        })
        .collect();

    let flat: Vec<f64> = (0..n_solutions)
        .into_par_iter()
        .flat_map(|i| {
            (0..n_ref)
                .map(|j| {
                    let mut v_dot_u = 0.0;
                    for k in 0..n_obj {
                        v_dot_u += n[[i, k]] * ref_dirs[[j, k]];
                    }
                    let scale = v_dot_u / u_dot_u[j];
                    let mut dist_sq = 0.0;
                    for k in 0..n_obj {
                        let d = n[[i, k]] - scale * ref_dirs[[j, k]];
                        dist_sq += d * d;
                    }
                    dist_sq.sqrt()
                })
                .collect::<Vec<f64>>()
        })
        .collect();

    Array2::from_shape_vec((n_solutions, n_ref), flat).unwrap()
}

fn associate_to_niches_impl(
    f: ArrayView2<f64>,
    ref_dirs: ArrayView2<f64>,
    ideal: ArrayView1<f64>,
    nadir: ArrayView1<f64>,
) -> (Vec<i64>, Vec<f64>) {
    let n_solutions = f.nrows();
    let n_obj = f.ncols();

    let mut normalized = Array2::<f64>::zeros((n_solutions, n_obj));
    for i in 0..n_solutions {
        for k in 0..n_obj {
            let denom = (nadir[k] - ideal[k]).max(1e-12);
            normalized[[i, k]] = (f[[i, k]] - ideal[k]) / denom;
        }
    }

    let dist_matrix = perpendicular_distance_impl(normalized.view(), ref_dirs);

    let mut niche_of = Vec::with_capacity(n_solutions);
    let mut dist_to_niche = Vec::with_capacity(n_solutions);
    for i in 0..n_solutions {
        let mut best_j = 0usize;
        let mut best_d = dist_matrix[[i, 0]];
        for j in 1..ref_dirs.nrows() {
            let d = dist_matrix[[i, j]];
            if d < best_d {
                best_d = d;
                best_j = j;
            }
        }
        niche_of.push(best_j as i64);
        dist_to_niche.push(best_d);
    }

    (niche_of, dist_to_niche)
}

#[pyfunction]
pub fn associate_to_niches_py<'py>(
    py: Python<'py>,
    f: PyReadonlyArray2<f64>,
    ref_dirs: PyReadonlyArray2<f64>,
    ideal: PyReadonlyArray1<f64>,
    nadir: PyReadonlyArray1<f64>,
) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>) {
    let (niche_of, dist_to_niche) = associate_to_niches_impl(
        f.as_array(),
        ref_dirs.as_array(),
        ideal.as_array(),
        nadir.as_array(),
    );
    (
        PyArray1::from_vec(py, niche_of),
        PyArray1::from_vec(py, dist_to_niche),
    )
}
