from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

def rust_available() -> bool: ...

# --- distances ---

def squared_euclidean_cost_matrix(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> NDArray[np.float64]: ...

def dtw(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    sakoe_chiba_radius: int | None = None,
) -> float: ...

def dtw_with_path_py(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    sakoe_chiba_radius: int | None = None,
) -> tuple[float, list[tuple[int, int]]]: ...

def cdist_dtw(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    sakoe_chiba_radius: int | None = None,
) -> NDArray[np.float64]: ...

def soft_dtw(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    gamma: float,
) -> float: ...

def cdist_soft_dtw(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    gamma: float,
) -> NDArray[np.float64]: ...

def gak(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    sigma: float,
) -> float: ...

def unnormalized_gak_py(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    sigma: float,
) -> float: ...

def cdist_gak(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    sigma: float,
) -> NDArray[np.float64]: ...

def manhattan(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> float: ...

def cdist_manhattan(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> NDArray[np.float64]: ...

# --- algorithm ---

def das_dennis_py(
    n_dim: int,
    n_partitions: int,
) -> NDArray[np.float64]: ...

def fast_non_dominated_sort_py(
    f: NDArray[np.float64],
) -> list[NDArray[np.int64]]: ...

def associate_to_niches_py(
    f: NDArray[np.float64],
    ref_dirs: NDArray[np.float64],
    ideal: NDArray[np.float64],
    nadir: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.float64]]: ...

def normalization_update_py(
    ideal_point: NDArray[np.float64],
    f: NDArray[np.float64],
    first_front: NDArray[np.int64],
    n_dim: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]: ...
