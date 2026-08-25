# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- `examples/` restructured into nine focused, runnable scripts covering the full
  workflow: loading, preprocessing, IRF removal, fitting, export, spectral
  unmixing (VCA), decomposition (NMF, PCA) and clustering (k-means). Each script
  falls back to synthetic data (`examples/_synthetic_data.py`) when no real
  dataset is configured, so every example runs out of the box.
- A step-by-step, illustrated Tutorial (`docs/tutorial.md`), also mirrored on the
  project's GitLab Wiki, walking through the `examples/` scripts with generated
  figures (`docs/generate_tutorial_images.py`).
- `LICENSE` (BSD 3-Clause) and `NOTICE.md`/`THIRD_PARTY_LICENSES/` documenting the
  RamanSPy- and VCA-derived portions of the codebase.

### Fixed
- `analysis.unmix.NFINDR` called a non-existent `pysptools.eea.nfindr.NFINDR` API
  and passed data in the wrong shape; it now uses `eea.NFINDR().extract(...)` with
  a correctly shaped input cube.

### Changed
- `setup.py`'s license metadata corrected from a placeholder `MIT` classifier to
  `BSD-3-Clause`, matching the added `LICENSE` file.

## [0.1.1]

- Added support for missing data points. The pipeline now warns the user when
  points are missing instead of stopping execution; missing values are treated
  as `None`.

## [0.1]

- Initial upload.
