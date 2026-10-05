# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-05

Initial release.

### Added
- `GCO`: multi-label energy minimization via graph cuts (alpha-expansion, alpha-beta-swap),
  for general and 4-connected grid graphs, with data/smooth/label costs and sparse data costs.
- `BKEnergy`: exact global minimization of submodular pseudo-boolean energies with unary,
  pairwise, and triple terms.
- `BKGraph`: Boykov-Kolmogorov max-flow / min-cut on arbitrary graphs.
- `int32`, `int64`, and `float64` energy/capacity term types, inferred from input arrays
  or given explicitly.
- Type hints throughout, a `py.typed` marker, and a stub file for the compiled extension.
- Thread-safe: `GCO`, `BKEnergy`, and `BKGraph` objects can be used concurrently from
  multiple threads, including on free-threaded (no-GIL) Python builds.
- Keyboard-interruptible `expansion`/`swap` calls, for interactive use in notebooks.
- Tested on Windows, macOS, and Linux, for Python >= 3.10 including free-threaded
  3.13t/3.14t builds.

[Unreleased]: https://github.com/andrewdelong/bkgco/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/andrewdelong/bkgco/releases/tag/v0.1.0
