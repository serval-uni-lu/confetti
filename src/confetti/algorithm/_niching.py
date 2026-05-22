"""Reference-direction association and niche-preservation selection for NSGA-III."""

from __future__ import annotations

import numpy as np

try:
    from confetti._rust_core import associate_to_niches_py as _rs_associate

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


def _ensure_f64_c(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr, dtype=np.float64)


def associate_to_niches(
    F: np.ndarray,
    ref_dirs: np.ndarray,
    ideal: np.ndarray,
    nadir: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assign each solution to its closest reference direction.

    Parameters
    ----------
    ``F`` : np.ndarray
        Objective matrix of shape ``(n, n_obj)``.
    ``ref_dirs`` : np.ndarray
        Reference directions of shape ``(n_ref, n_obj)``.
    ``ideal`` : np.ndarray
        Ideal point of shape ``(n_obj,)``.
    ``nadir`` : np.ndarray
        Nadir point of shape ``(n_obj,)``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(niche_of, dist_to_niche)`` where ``niche_of`` is an int array
        of shape ``(n,)`` giving the closest reference-direction index,
        and ``dist_to_niche`` is the corresponding perpendicular distance.
    """
    if _HAS_RUST:
        niche_of, dist_to_niche = _rs_associate(
            _ensure_f64_c(F),
            _ensure_f64_c(ref_dirs),
            _ensure_f64_c(ideal),
            _ensure_f64_c(nadir),
        )
        return np.asarray(niche_of), np.asarray(dist_to_niche)

    # Normalise objectives to [0, 1]
    denom = nadir - ideal
    denom[denom < 1e-12] = 1e-12
    N = (F - ideal) / denom

    # Perpendicular distance from each normalised point to each ref dir
    dist_matrix = _perpendicular_distance(N, ref_dirs)  # (n, n_ref)

    niche_of = np.argmin(dist_matrix, axis=1)
    dist_to_niche = dist_matrix[np.arange(F.shape[0]), niche_of]

    return niche_of, dist_to_niche


def _perpendicular_distance(N: np.ndarray, ref_dirs: np.ndarray) -> np.ndarray:
    """
    Compute perpendicular distances from normalised solutions to every
    reference direction.

    Parameters
    ----------
    ``N`` : np.ndarray
        Normalised objective matrix of shape ``(n, n_obj)``.
    ``ref_dirs`` : np.ndarray
        Reference directions of shape ``(n_ref, n_obj)``.

    Returns
    -------
    np.ndarray
        Distance matrix of shape ``(n, n_ref)``.
    """
    # For each (solution, ref_dir) pair compute the perpendicular distance.
    # Projection of v onto u: proj = (v·u / u·u) * u
    # Perpendicular component: v - proj
    # Distance: ||v - proj||

    # ref_dirs: (n_ref, n_obj)  →  (1, n_ref, n_obj)
    # N:        (n, n_obj)      →  (n, 1,     n_obj)
    u = ref_dirs[np.newaxis, :, :]  # (1, n_ref, n_obj)
    v = N[:, np.newaxis, :]  # (n, 1, n_obj)

    u_dot_u = np.sum(u * u, axis=2, keepdims=True)  # (1, n_ref, 1)
    u_dot_u = np.maximum(u_dot_u, 1e-14)
    proj_scalar = np.sum(v * u, axis=2, keepdims=True)  # (n, n_ref, 1)
    proj = (proj_scalar / u_dot_u) * u  # (n, n_ref, n_obj)

    dist = np.linalg.norm(v - proj, axis=2)  # (n, n_ref)
    return dist


def niching_selection(
    n_remaining: int,
    niche_count: np.ndarray,
    niche_of: np.ndarray,
    dist_to_niche: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Select *n_remaining* individuals from the last front using
    niche-preservation.

    Iteratively picks the least-populated niche, then selects the
    closest candidate (if the niche was empty in earlier fronts) or a
    random candidate (otherwise).

    Parameters
    ----------
    ``n_remaining`` : int
        Number of individuals to select.
    ``niche_count`` : np.ndarray
        Current niche population counts (from earlier fronts).
    ``niche_of`` : np.ndarray
        Niche assignment for each candidate in the last front.
    ``dist_to_niche`` : np.ndarray
        Perpendicular distance for each candidate in the last front.
    ``rng`` : numpy.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Index array (into the last-front candidates) of selected
        individuals.
    """
    n_candidates = len(niche_of)
    mask = np.ones(n_candidates, dtype=bool)
    survivors: list[int] = []
    niche_count = niche_count.copy()

    while len(survivors) < n_remaining:
        # Find niches with minimum count among remaining candidates
        active_niches = np.unique(niche_of[mask])
        counts_for_active = niche_count[active_niches]
        min_count = counts_for_active.min()

        tied_niches = active_niches[counts_for_active == min_count]
        # Shuffle to break ties randomly
        tied_niches = rng.permutation(tied_niches)

        n_select = n_remaining - len(survivors)
        niches_to_fill = tied_niches[:n_select]

        for niche in niches_to_fill:
            if len(survivors) >= n_remaining:
                break

            candidates = np.where(mask & (niche_of == niche))[0]
            if len(candidates) == 0:
                continue

            if niche_count[niche] == 0:
                # Empty niche → pick closest candidate
                chosen = candidates[np.argmin(dist_to_niche[candidates])]
            else:
                # Non-empty niche → pick random candidate
                chosen = candidates[rng.integers(len(candidates))]

            survivors.append(int(chosen))
            mask[chosen] = False
            niche_count[niche] += 1

    return np.array(survivors, dtype=np.intp)
