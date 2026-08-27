# Design: multimodal `SpectralContainer`

**Status:** planned, not yet implemented (as of 2026-08-27).
**Context:** planned as a short sub-chapter of the author's doctoral thesis, differentiating BrillouinPy from its origin as a fork of [RamanSPy](https://github.com/barahona-research-group/RamanSPy) - see [`NOTICE.md`](../../NOTICE.md) for the existing fork attribution.

## Motivation

BrillouinPy's data model (`SpectralContainer`/`Spectrum`/`SpectralImage`/`SpectralVolume`, [`core.py`](../../brillouinpy/core.py)) is architecturally general enough for arbitrary-dimensional spectral data, but a single instance currently holds exactly one spectral modality. The author's microscope is a combined Brillouin+Raman point-scanning system with an additional brightfield transmission camera (sample positioning) that also takes epi-fluorescence images through a manually swapped filter cube. Supporting this multimodal, correlative use case - without regressing RamanSPy/brimfile/HDF5_BLS compatibility - is the goal of this design.

## Setup and data characteristics

- **Brillouin + Raman**: acquired by the same point-scanner, on the same scan grid. Intrinsically co-registered pixel-for-pixel - no registration needed between these two.
- **Brightfield / fluorescence**: acquired by a separate camera-based optical path (wide-field, not scanned). Different field of view / pixel size than the scanner grid. NOT automatically co-registered with Brillouin/Raman. Multi-color fluorescence is naturally represented the same way as any other spectral data: an array of shape `(x, y, C)` with `C` = number of colors; brightfield is the degenerate `C = 1` case. No bespoke type is needed - a fluorescence/brightfield channel is just another `SpectralObject` whose "spectral axis" holds discrete emission wavelengths/labels instead of a continuous frequency shift.
- Camera and scanner sit at fixed positions in the same microscope, so a **one-time calibrated affine transform** (scale + offset, possibly rotation) between camera pixel coordinates and scan coordinates is sufficient - confirmed realistic for this setup, no per-measurement image-based registration required.

## API sketch

- `SpectralContainer.__init__(spectral_data, spectral_axis, *, channels: dict[str, SpectralObject] | None = None)`
- A `channels` entry is itself a `SpectralObject` (`Spectrum`/`SpectralImage`/`SpectralVolume`).
- Channels sharing the primary object's spatial shape (e.g. Raman on the same scan grid) are "grid-conformant" and participate transparently in spatial operations.
- Channels that don't (e.g. a camera-based brightfield/fluorescence channel) may carry an optional `transform` attribute (an `AffineTransform2D`, scale/offset/rotation) mapping camera pixel coordinates to scan coordinates. Such channels are "non-conformant" until resampled.
- **Non-conformant channel access policy (decided 2026-08-27):** spatial operations that touch `channels` (`__getitem__`, `flat`, `from_stack`/`from_image_stack`) emit a `UserWarning` ("context warning") when they encounter a channel whose spatial shape doesn't match the primary object's, rather than silently passing it through unchanged or raising a hard error. This flags the mismatch to the user without blocking the operation - the non-conformant channel is left untouched (not spatially transformed) and the warning names the channel key and the mismatch.

## Implementation phases

### Phase 0 - API contract
Fix signatures and conventions above before writing code. ~2h, no code.

### Phase 1 - Container core (`brillouinpy/core.py`)
- `__init__`: accept and store `channels`; no shape validation against `spectral_data` (camera channels legitimately differ in shape).
- `_create_data`: pass `channels` through.
- `flat`: flatten grid-conformant channels alongside `spectral_data`; for non-conformant channels, emit the context warning and pass them through unchanged.
- `__getitem__`: apply spatial indexing to grid-conformant channels; emit the context warning and leave non-conformant channels unchanged otherwise.
- `from_stack`/`from_image_stack`: merge `channels` across elements (matching keys/types); emit the context warning for any channel that isn't grid-conformant across the stack.
- `save`/`load` (pickle) already round-trip `channels` losslessly with no changes needed.

Estimate: ~1 day (implementation + docstrings).
Tests: extend `tests/test_core.py` - construction with `channels`, crop/index/flat/stack with a mix of conformant/non-conformant channels (asserting the warning fires), backward compatibility (no `channels` -> unchanged existing behaviour).

### Phase 2 - Camera calibration / registration (`brillouinpy/utils.py`)
- `AffineTransform2D` class (scale, offset, rotation) with `.apply(coords)` / `.inverse()`.
- A calibration helper that estimates the transform from known correspondence points (e.g. 2-3 manually marked landmarks between camera image and scan coordinates) - computed once per setup, then reused (e.g. stored as a constant/config).
- A resampling function that interpolates a camera-based `SpectralObject` onto the scan grid via the transform (e.g. `scipy.ndimage.map_coordinates`), producing a new grid-conformant `SpectralObject`.

Estimate: ~1 day.
Tests: calibration against synthetic correspondence points; resampling accuracy on a synthetic test image.

### Phase 3 - Colocalization / correlation analysis (new module, e.g. `brillouinpy/analysis/colocalize.py`)
- Takes two already grid-conformant `SpectralObject`s (post-resampling) and computes standard colocalization metrics (Pearson/Spearman correlation, possibly Manders coefficients). Can follow the `AnalysisStep` pattern loosely but stands alone given its different input/output shape.

Estimate: ~1 day for core metrics; more if richer analysis is wanted.
Tests: synthetic correlated/uncorrelated images with known expected outcome.

### Phase 4 - Fluorescence workflows
- Existing preprocessing steps (`BackgroundSubtractor`, `denoise.*`, `normalise.*` - see [`misc.py`](../../brillouinpy/preprocessing/misc.py)) are already generic enough to apply directly to fluorescence channels once represented as `SpectralObject`s - **no new code needed**, just documentation/an example showing they apply here too.
- The one genuinely new piece, if wanted: a photobleaching-correction `PreprocessingStep` (time-dependent, no existing equivalent).

Estimate: ~0.5 day (only the bleaching correction is new work).

### Phase 5 - RamanSPy interoperability (`brillouinpy/export.py`)
- `to_ramanspy(spectral_object) -> ramanspy.SpectralContainer` / `from_ramanspy(obj) -> core.SpectralObject`, following the existing `to_brim`/`from_hdf5_bls` pattern (soft optional dependency, see [`export.py`](../../brillouinpy/export.py)).
- `channels` are not representable in RamanSPy's model and are dropped on `to_ramanspy`, with the same kind of caveat already documented for brim/HDF5_BLS.

Estimate: ~0.5 day.
Tests: round-trip test (`brillouinpy -> ramanspy -> brillouinpy`), if `ramanspy` is available as a test dependency.

### Phase 6 - Documentation
- `docs/tutorial.md` plus a new example in `examples/` (multimodal workflow: load Brillouin+Raman+brightfield/fluorescence, calibrate/register, colocalize).
- GitLab wiki sync when `docs/tutorial.md` changes (existing project convention).

Estimate: ~0.5-1 day.

## Ordering and total estimate

Phase 0 -> 1 -> (2 and 5 can run in parallel/independently) -> 3 -> 4 -> 6.
Rough total: **4-5 focused working days**, matching the intended "short sub-chapter" scope.

## Compatibility notes

- **RamanSPy**: duck-typing compatibility (same `spectral_data`/`spectral_axis` attribute shape) is preserved since `channels` is purely additive; the explicit `to_ramanspy`/`from_ramanspy` converters (Phase 5) make this compatibility testable rather than implicit.
- **brimfile / HDF5_BLS**: both are Brillouin-specific standards with no concept of auxiliary channels. `to_brim`/`from_brim` and `to_hdf5_bls`/`from_hdf5_bls` continue to round-trip the primary Brillouin data correctly, but `channels` content is not exported/imported through them. Only the existing pickle-based `save`/`load` round-trips `channels` losslessly (non-interoperably).
