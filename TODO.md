# TODO

## Multimodal `SpectralContainer` (`channels`)

See [`docs/design/multimodal_container.md`](docs/design/multimodal_container.md).
Phase 1 (generic `channels` on the container) is the next step; phases 2-4
(camera affine transform / resampling / colocalization) and phase 5
(`to_ramanspy`, `to_xarray`) follow.

## Done

- **Shape validation in `SpectralContainer` subclasses** — `Spectrum`/`SpectralImage`/
  `SpectralVolume` now require `ndim` 1/3/4 and raise a clear `ValueError` at
  construction otherwise. `_create_data()` still dispatches by `ndim` (base
  `SpectralContainer` for 2D/5D+). Preprocessing/analysis steps are unaffected
  (they deep-copy rather than reconstruct).
- **First-class `metadata` / `px_size_um`** on `SpectralContainer`, carried
  through slices/`flat`/`mean`/stacking and round-tripped by `io.from_brim` /
  `io.to_brim`.
