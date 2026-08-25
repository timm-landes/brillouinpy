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
- `utils.read_meta` and `utils.brillouin_spectral_axis_from_meta`, which read the
  interferometer scan parameters (mirror spacing, scan amplitude, laser
  wavelength) for `brillouin_spectral_axis` from a measurement's `META.json`
  automatically, transparently handling both the current (nested) and legacy
  (flat) metadata schema found across existing datasets.
- `LICENSE` (BSD 3-Clause) and `NOTICE.md`/`THIRD_PARTY_LICENSES/` documenting the
  RamanSPy- and VCA-derived portions of the codebase.
- `export.to_brim`/`export.from_brim`/`export.list_brim_measurements`, for
  reading/writing the [brim](https://github.com/brillouin-imaging/Brillouin-standard-file)
  format (a Zarr-based standard for Brillouin microscopy data, also readable by
  the napari/Fiji brim viewer plugins), via the optional `brimfile` package
  (`pip install brillouinanalyzer[brim]`; needs Python >= 3.11).
- `export.to_brim(..., as_zip=True)`, to write a single `.brim.zip` archive
  instead of a `.brim.zarr` directory.

### Fixed
- `analysis.unmix.NFINDR` called a non-existent `pysptools.eea.nfindr.NFINDR` API
  and passed data in the wrong shape; it now uses `eea.NFINDR().extract(...)` with
  a correctly shaped input cube.
- `export.to_brim` now writes the root `Subtype` attribute explicitly (as
  `'none'`) - `brimfile`'s own `File.create()`/`create_data_group()` never write
  it unless one of the `brimfile.subtypes.*` helpers is used, and while
  `brimfile`'s own reader tolerates that, at least one downstream viewer
  (BrimView, as of `brimfile` 1.7.0) doesn't and fails to open the file with
  `ValueError: Invalid subtype: None`.
- `export.to_brim` no longer writes a `null` pixel size for a single z-slice
  (e.g. a plain `SpectralImage`, where the z-axis is a placeholder to begin
  with) when `z_step_um` isn't given, since that also tripped up BrimView.

### Changed
- `setup.py`'s license metadata corrected from a placeholder `MIT` classifier to
  `BSD-3-Clause`, matching the added `LICENSE` file.

## [0.1.1]

- Added support for missing data points. The pipeline now warns the user when
  points are missing instead of stopping execution; missing values are treated
  as `None`.

## [0.1]

- Initial upload.
