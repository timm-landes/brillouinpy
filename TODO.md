# TODO

## Multimodal `SpectralContainer`

See [`docs/design/multimodal_container.md`](docs/design/multimodal_container.md).
Phase 1 (`channels`, restricted to `SpectralObject`s) is done. Next: phase 2 (`AffineTransform2D` +
camera→scan resampling), phase 3 (colocalization metrics), phase 5
(`to_ramanspy`, `to_xarray`).

## Done

- **Multimodal `channels` (Phase 1)** — `SpectralContainer` carries a
  `channels` dict restricted to `SpectralObject`s (bare arrays without a
  spectral axis go through `as_channel`/`from_channel`). Under indexing and
  `flat`, grid-conformant channels follow the operation and others pass
  through untouched; `mean`/`variance` carry no channels, and stacking drops
  all of them if any input channel is non-conformant. Introspection via
  `channels_grid_conformant` / `repr`; `with_channel`/`drop_channel`/
  `apply_to_channel` add, remove, or transform a channel without mutating
  the original.
- **Shape validation in `SpectralContainer` subclasses** — `Spectrum`/`SpectralImage`/
  `SpectralVolume` now require `ndim` 1/3/4 and raise a clear `ValueError` at
  construction otherwise. `_create_data()` still dispatches by `ndim` (base
  `SpectralContainer` for 2D/5D+). Preprocessing/analysis steps are unaffected
  (they deep-copy rather than reconstruct).
- **First-class `metadata` / `px_size_um`** on `SpectralContainer`, carried
  through slices/`flat`/`mean`/stacking and round-tripped by `io.from_brim` /
  `io.to_brim`.
