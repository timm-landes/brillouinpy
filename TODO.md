# TODO

## Shape validation in SpectralContainer subclasses

`core._create_data()` dispatches purely on `spectral_data.ndim` to decide whether an object
becomes a `Spectrum`, `SpectralImage`, or `SpectralVolume` — for unexpected shapes (2D, 5D+) it
silently falls back to the base `SpectralContainer`. This means the class names don't actually
guarantee the shape they suggest (see the `Deconvoluter_IRF` bug in `03_remove_irf.py`, where a
function silently expected 5D input while operating on a `SpectralImage` (3D) object).

**Proposal**
- Keep public names unchanged (`Spectrum`, `SpectralImage`, `SpectralVolume`, `isinstance` checks,
  constructor calls) — no breaking change for existing users.
- Give each subclass real shape validation in `__init__` (e.g. `SpectralImage` requires
  `ndim == 3`) instead of relying on `ndim`-guessing in `_create_data()`.
- Shape errors then surface immediately at construction with a clear message, instead of
  crashing cryptically deep inside a preprocessing function later.

**Estimate**: ~1 focused work day (core classes + validation: 2-4h, tests: 1-2h, minor doc
updates).

**Known consequence**: Stricter validation could break existing scripts that (like the old
`example.py`) put data of the wrong dimensionality into the "wrong" class — but that would
surface an existing bug rather than introduce a new one.
