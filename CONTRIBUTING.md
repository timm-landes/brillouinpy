# Contributing to BrillouinPy

Thanks for your interest in contributing! This document covers the practical setup and
conventions used in this repository.

## Where to contribute

The canonical, publicly accessible repository is
[github.com/timm-landes/brillouinpy](https://github.com/timm-landes/brillouinpy) - open
issues and pull requests there. A mirror also exists on the Leibniz University Hannover
GitLab instance for internal development; it's only reachable from within the LUH
network/SSO, so it isn't usable for external contributions.

## Setting up a development environment

```bash
git clone https://github.com/timm-landes/brillouinpy.git
cd brillouinpy
pip install -e .
pip install pytest ruff
```

Optional extras (`brimfile`, `HDF5_BLS`), needed only if you're touching the code paths
that use them (`brillouinpy.io.export`), can be installed the same way as for regular use
- see the [README](README.md#installation).

## Running the checks locally

These are exactly what CI runs (both the GitHub Actions workflow,
[`.github/workflows/ci.yml`](.github/workflows/ci.yml), and the internal GitLab pipeline,
[`.gitlab-ci.yml`](.gitlab-ci.yml)):

```bash
ruff check .
pytest tests/ -v
```

If you change `docs/tutorial.md`, `docs/`, or any docstrings, also build the docs locally
to check for Sphinx warnings before opening a PR:

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

## Code organization

A quick map, so new code ends up in the right place:

- [`brillouinpy/core.py`](brillouinpy/core.py) - the `SpectralContainer`/`Spectrum`/
  `SpectralImage`/`SpectralVolume` data model.
- [`brillouinpy/preprocessing/`](brillouinpy/preprocessing/) - `PreprocessingStep`/
  `Pipeline` and the built-in steps (denoise, despike, normalise, IRF removal, ...).
- [`brillouinpy/analysis/`](brillouinpy/analysis/) - `AnalysisStep` and the built-in
  methods (decomposition, clustering, unmixing, DHO/Lorentzian fitting).
- [`brillouinpy/io/`](brillouinpy/io/) - reading/writing spectral data, split by purpose:
  - `io.tfp` - this group's actively used tandem Fabry-Perot loading path. If you're
    adding support for reading data from a different setup/lab, model it on this module
    rather than extending it - see the "Writing your own loader" section of the
    [tutorial](docs/tutorial.md) for the minimal contract a loader needs to satisfy.
  - `io.legacy` - superseded loaders kept only for reading old datasets. Don't add new
    functionality here.
  - `io.multimodal` - loaders for modalities other than the group's primary setup
    (currently Raman).
  - `io.export` - interoperability with external formats/packages (`brim`, `HDF5_BLS`,
    OME-TIFF, ...).
- [`brillouinpy/utils.py`](brillouinpy/utils.py) - generic, setup-independent helpers
  only (unit conversions, alignment checks, ...). Anything tied to a specific
  instrument/lab's raw-data layout belongs under `brillouinpy/io/` instead, not here.
- [`brillouinpy/plot/`](brillouinpy/plot/) - plotting helpers (lazily imported by the
  package so that code which doesn't need `matplotlib`, e.g. multiprocessing fit
  workers, doesn't pay for importing it).

## Deprecating/moving something

If you rename or move a public function (as happened with the `brillouinpy.utils`/
`brillouinpy.export` -> `brillouinpy.io` reorganisation), don't just delete the old name:
keep a thin wrapper at the old location that emits a `DeprecationWarning` and forwards to
the new one, so existing user code doesn't break. See `brillouinpy/utils.py`'s
`_deprecated_move_warning` helper and `brillouinpy/export.py` for the two patterns
(function-level wrapper vs. whole-module shim) already used in this codebase, and add a
`### Changed` entry to `CHANGELOG.md` under `[Unreleased]` explaining the move.

## Third-party code

If you're porting or adapting code from another project (as parts of this codebase are,
from [RamanSPy](https://github.com/barahona-research-group/RamanSPy)), you must:

1. Confirm the license is compatible (BSD/MIT-permissive; ask before adding anything
   copyleft like GPL to the core package - see the "Optional runtime dependencies" table
   in [`NOTICE.md`](NOTICE.md) for how LGPL/copyleft *dependencies* are handled instead,
   as opt-in extras rather than copied code).
2. Add an inline attribution comment/docstring at the top of the file (or just above the
   relevant class/function, if only part of the file is derived) naming the source
   project, copyright holder, and license.
3. Add a row to the relevant table in [`NOTICE.md`](NOTICE.md), and the license's full
   text under [`THIRD_PARTY_LICENSES/`](THIRD_PARTY_LICENSES/) if it isn't there already.

## Tests

New functionality should come with tests under [`tests/`](tests/) (pytest). Tests that
need real instrument data should fall back to synthetic data rather than being skipped
outright when none is available - see `examples/_synthetic_data.py` for the pattern used
throughout the `examples/` scripts.

## Changelog

User-facing changes (new features, bug fixes, behaviour changes, deprecations) belong in
[`CHANGELOG.md`](CHANGELOG.md) under `[Unreleased]`, following the existing
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) style (`### Added`/`### Changed`/
`### Fixed` subsections).

## License

By contributing, you agree that your contribution is licensed under the project's
[BSD 3-Clause License](LICENSE).
