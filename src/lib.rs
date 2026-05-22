mod algorithm;
mod distances;

use pyo3::prelude::*;

#[pyfunction]
fn rust_available() -> bool {
    true
}

#[pymodule]
fn _rust_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_available, m)?)?;

    // distances
    m.add_function(wrap_pyfunction!(distances::cost::squared_euclidean_cost_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(distances::dtw::dtw, m)?)?;
    m.add_function(wrap_pyfunction!(distances::dtw::dtw_with_path_py, m)?)?;
    m.add_function(wrap_pyfunction!(distances::dtw::cdist_dtw, m)?)?;
    m.add_function(wrap_pyfunction!(distances::soft_dtw::soft_dtw, m)?)?;
    m.add_function(wrap_pyfunction!(distances::soft_dtw::cdist_soft_dtw, m)?)?;
    m.add_function(wrap_pyfunction!(distances::gak::gak, m)?)?;
    m.add_function(wrap_pyfunction!(distances::gak::unnormalized_gak_py, m)?)?;
    m.add_function(wrap_pyfunction!(distances::gak::cdist_gak, m)?)?;
    m.add_function(wrap_pyfunction!(distances::gower::gower, m)?)?;
    m.add_function(wrap_pyfunction!(distances::gower::cdist_gower, m)?)?;
    m.add_function(wrap_pyfunction!(distances::manhattan::manhattan, m)?)?;
    m.add_function(wrap_pyfunction!(distances::manhattan::cdist_manhattan, m)?)?;

    // algorithm
    m.add_function(wrap_pyfunction!(algorithm::das_dennis::das_dennis_py, m)?)?;
    m.add_function(wrap_pyfunction!(algorithm::nds::fast_non_dominated_sort_py, m)?)?;
    m.add_function(wrap_pyfunction!(algorithm::niching::associate_to_niches_py, m)?)?;
    m.add_function(wrap_pyfunction!(algorithm::normalization::normalization_update_py, m)?)?;

    Ok(())
}
