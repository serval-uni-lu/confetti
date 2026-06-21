# Changelog

All notable changes to CONFETTI are documented here.

## [1.0.0] — 2026-06-21

First stable release. CONFETTI is now a fully self-contained library with zero
dependency on pymoo or tslearn.

### Added

- **Rust-accelerated backend** — distance metrics (DTW, Soft-DTW, GAK, Manhattan)
  and NSGA-III core ported to Rust via PyO3/maturin with Rayon parallelism;
  falls back to pure-numpy when the native extension is unavailable.
- **Custom NSGA-III implementation** — full genetic algorithm in
  `confetti.algorithm` (sampling, crossover, mutation, non-dominated sorting,
  hyperplane normalization, niche-preservation selection).
- **Built-in distance metrics** — pure-numpy DTW, Soft-DTW, GAK, CTW, and
  Manhattan in `confetti.distances`, replacing tslearn.
- **PyTorch adapter** — `TorchModelAdapter` for using PyTorch models alongside
  Keras and scikit-learn classifiers.
- **Visualization theming** — `plot_time_series` and `plot_counterfactual`
  support light/dark themes with improved aesthetics.
- **Class Activation Maps** — `cam()` and `visualize_cam()` for
  feature-importance-weighted perturbations.
- **`py.typed` marker** — PEP 561 compliant for static type checkers.
- **Cross-platform wheels** — pre-built wheels for Linux, macOS (Intel + Apple
  Silicon), and Windows on Python 3.12 and 3.13.

### Changed

- Keras and TensorFlow are now optional dependencies (`pip install confetti-ts[keras]`).
- PyTorch is an optional dependency (`pip install confetti-ts[torch]`).
- Replaced all `print()` calls with `logging`.
- Worker pools now use context managers to prevent resource leaks.

### Removed

- **pymoo** dependency — replaced by `confetti.algorithm`.
- **tslearn** dependency — replaced by `confetti.distances`.
