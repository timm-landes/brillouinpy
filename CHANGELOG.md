# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- `Spectrum` / `SpectralImage` / `SpectralVolume` now validate their
  dimensionality at construction (`ndim` 1 / 3 / 4) and raise a clear
  `ValueError` on a mismatch, instead of the class name silently not matching the
  data shape. `core._create_data` still picks the class by `ndim` (base
  `SpectralContainer` for 2D / 5D+); preprocessing and analysis steps are
  unaffected as they deep-copy rather than reconstruct.
- `SpectralContainer.peaks()` is now available on any spectral object, not just
  `Spectrum`; for an object with spatial dimensions it operates on the `mean`
  spectrum.
- `SpectralContainer` (and subclasses) now carry `metadata` and `px_size_um`
  dicts as first-class attributes, settable via the constructor. Both always
  exist on an instance (default: `{}` and `{"x": None, "y": None, "z": None}`),
  and derived objects (spatial slices, `flat`, `mean`, `variance`, `tolist`,
  `from_stack`, `from_image_stack`, `layer`) inherit a copy. Pickles written
  before these attributes existed load with the defaults filled in.
- `io.from_brim` now also attaches the data group's pixel size to the returned
  object as a `px_size_um` dict (`{"x"/"y"/"z": float_or_None}`, in micrometers),
  alongside the existing `metadata` dict.

### Fixed
- `io.to_brim` now always writes an `Experiment.Datetime` metadata field (from the
  new `acquisition_datetime` argument, a value already present in `metadata`, or
  the current local time). Files written without any Experiment metadata made
  BrimView fail with `AttributeError: 'NoneType' object has no attribute 'get'`.
- `io.from_brim` no longer includes empty metadata categories in the returned
  `metadata` dict.

### Changed
- `io.to_brim` reads `metadata` and the x/y/z pixel steps off the passed object
  when the corresponding arguments are omitted, so a
  `from_brim` -> process -> `to_brim` round-trip preserves metadata and pixel
  size without the caller re-supplying them. Explicit arguments still take
  precedence.
- **Reorganised acquisition-side loaders and export/interop functions under a new
  `brillouinpy.io` subpackage**, split by purpose:
  - `io.tfp` - this group's actively used tandem Fabry-Perot loading path
    (`prepare_brillouin_data`, `read_meta`, `brillouin_spectral_axis_from_meta`,
    `extract_coordinates`, `import_DAT_File`).
  - `io.legacy` - superseded loaders kept for reading old datasets
    (`load_spectral_image`, `load_spectral_image_brio2`).
  - `io.multimodal` - loaders for modalities other than the group's primary setup
    (`prepare_raman_data`).
  - `io.export` (formerly the top-level `brillouinpy.export` module) - `brim`/
    `HDF5_BLS` interoperability and TIFF export; the most commonly used functions
    (`to_brim`, `from_brim`, `list_brim_measurements`, `to_hdf5_bls`,
    `from_hdf5_bls`, `list_measurements`) are also re-exported directly on
    `brillouinpy.io` for convenience.

  `brillouinpy.utils` keeps only the generic, setup-independent functions
  (`is_aligned`, `wavelength_to_wavenumber`/`wavenumber_to_wavelength`,
  `raman_spectral_axis`, `fsr`, `brillouin_spectral_axis`, `TFP_IRF_Analysis`).
  The old `brillouinpy.utils.*` names for the moved functions, and the
  `brillouinpy.export` module itself, keep working unchanged for now - both emit
  a `DeprecationWarning` pointing at the new location. These compatibility
  shims may be removed in a future release; update your code to call
  `brillouinpy.io.*` directly when convenient.
- Removed a stray leftover `print(data)` debug statement from
  `prepare_brillouin_data`'s CSV-loading branch (carried over unnoticed from
  `load_spectral_image`; harmless but noisy).

### Added
- A new "Writing your own loader" section in the tutorial (`docs/tutorial.md`),
  explaining the minimal `(intensity_array, spectral_axis)` contract a custom
  loader needs to satisfy to work with the rest of the package, using
  `io.tfp.prepare_brillouin_data` as a worked example.

## [0.2.0] - 2026-08-27

First public release. The project is now developed openly on GitHub, with the
existing GitLab instance kept as an LUH-internal mirror.

### Added
- Public repository at [github.com/timm-landes/brillouinpy](https://github.com/timm-landes/brillouinpy),
  now the canonical, publicly accessible home of the project (mirrored
  internally on `gitlab.uni-hannover.de/phytophotonics/brillouinpy`, reachable
  only from within the LUH network/SSO).
- `.github/workflows/docs.yml`, automatically building and publishing the
  Sphinx documentation to GitHub Pages
  (`https://timm-landes.github.io/brillouinpy/`) on every push to `main`.

### Changed
- **Renamed the package from `BrillouinAnalyzer` to `BrillouinPy`** (import name
  `brillouinpy`). `brillouinanalyzer` remains installable as a thin backwards-
  compatibility shim - `import brillouinanalyzer as bp` still works, but now
  emits a `DeprecationWarning` and re-exports everything from `brillouinpy`.
  Update your imports to `import brillouinpy as bp` when convenient; the shim
  may be removed in a future release. The GitLab project itself (and its URL)
  are unaffected by this change for now.

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
