use ndarray::ArrayView2;
use numpy::{PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

fn fast_non_dominated_sort_impl(f: ArrayView2<f64>) -> Vec<Vec<usize>> {
    let n = f.nrows();
    if n == 0 {
        return Vec::new();
    }
    let n_obj = f.ncols();

    let (dominated_by, domination_count_init): (Vec<Vec<usize>>, Vec<usize>) = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut dom_by_i = Vec::new();
            let mut count_i: usize = 0;
            for j in 0..n {
                if i == j {
                    continue;
                }
                let mut i_all_leq = true;
                let mut i_any_lt = false;
                let mut j_all_leq = true;
                let mut j_any_lt = false;
                for k in 0..n_obj {
                    let fi = f[[i, k]];
                    let fj = f[[j, k]];
                    if fi > fj {
                        i_all_leq = false;
                    }
                    if fi < fj {
                        i_any_lt = true;
                    }
                    if fj > fi {
                        j_all_leq = false;
                    }
                    if fj < fi {
                        j_any_lt = true;
                    }
                }
                if i_all_leq && i_any_lt {
                    dom_by_i.push(j);
                }
                if j_all_leq && j_any_lt {
                    count_i += 1;
                }
            }
            (dom_by_i, count_i)
        })
        .unzip();

    let mut domination_count = domination_count_init;
    let mut remaining = vec![true; n];
    let mut fronts: Vec<Vec<usize>> = Vec::new();

    loop {
        let has_remaining = remaining.iter().any(|&r| r);
        if !has_remaining {
            break;
        }

        let front: Vec<usize> = (0..n)
            .filter(|&i| remaining[i] && domination_count[i] == 0)
            .collect();

        if front.is_empty() {
            let last: Vec<usize> = (0..n).filter(|&i| remaining[i]).collect();
            fronts.push(last);
            break;
        }

        for &i in &front {
            remaining[i] = false;
            for &j in &dominated_by[i] {
                if remaining[j] {
                    domination_count[j] = domination_count[j].saturating_sub(1);
                }
            }
        }

        fronts.push(front);
    }

    fronts
}

#[pyfunction]
pub fn fast_non_dominated_sort_py<'py>(
    py: Python<'py>,
    f: PyReadonlyArray2<f64>,
) -> Vec<Bound<'py, PyArray1<i64>>> {
    let fronts = fast_non_dominated_sort_impl(f.as_array());
    fronts
        .into_iter()
        .map(|front| {
            let arr: Vec<i64> = front.into_iter().map(|i| i as i64).collect();
            PyArray1::from_vec(py, arr)
        })
        .collect()
}
