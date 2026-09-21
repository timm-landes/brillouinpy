# Design: multimodal `SpectralContainer`

**Status:** Resolved 2026-09-14 (umbrella issue #3, closed) - narrower than originally planned. `channels` (Phase 1) is implemented; camera calibration/registration (Phase 2) and colocalization (Phase 3) are permanently out of scope; fluorescence workflows (Phase 4) turned out to need no dedicated code; RamanSPy interoperability (Phase 5) is done without explicit converters; the tutorial (Phase 6) is done, minus the parts that depended on Phases 2/3. See "Resolution" below for the full picture and the two threads that continue from here (#23, #24).
**Context:** planned as a short sub-chapter of the author's doctoral thesis, differentiating BrillouinPy from its origin as a fork of [RamanSPy](https://github.com/barahona-research-group/RamanSPy) - see [`NOTICE.md`](../../NOTICE.md) for the existing fork attribution.

## Motivation

BrillouinPy's data model (`SpectralContainer`/`Spectrum`/`SpectralImage`/`SpectralVolume`, [`core.py`](../../brillouinpy/core.py)) is architecturally general enough for arbitrary-dimensional spectral data, but a single instance currently holds exactly one spectral modality. The author's microscope is a combined Brillouin+Raman point-scanning system with an additional brightfield transmission camera (sample positioning) that also takes epi-fluorescence images through a manually swapped filter cube. Supporting this multimodal, correlative use case - without regressing RamanSPy/brimfile/HDF5_BLS compatibility - is the goal of this design.

## Setup and data characteristics

- **Brillouin + Raman**: acquired by the same point-scanner, on the same scan grid. Intrinsically co-registered pixel-for-pixel - no registration needed between these two.
- **Brightfield / fluorescence**: acquired by a separate camera-based optical path (wide-field, not scanned). Different field of view / pixel size than the scanner grid. NOT automatically co-registered with Brillouin/Raman. Multi-color fluorescence is naturally represented the same way as any other spectral data: an array of shape `(x, y, C)` with `C` = number of colors; brightfield is the degenerate `C = 1` case. No bespoke type is needed - a fluorescence/brightfield channel is just another `SpectralObject` whose "spectral axis" holds discrete emission wavelengths/labels instead of a continuous frequency shift.
- Camera and scanner sit at fixed positions in the same microscope, so a **one-time calibrated affine transform** (scale + offset, possibly rotation) between camera pixel coordinates and scan coordinates is sufficient - confirmed realistic for this setup, no per-measurement image-based registration required. **This transform is implemented outside brillouinpy** (acquisition software or a setup-specific script), not by the package - see Phase 2's resolution below for why.

## API

- `SpectralContainer.__init__(spectral_data, spectral_axis, *, metadata=None, px_size_um=None, channels: dict[str, SpectralObject] | None = None)`
- A `channels` entry must be a `SpectralObject` - brillouinpy makes no assumption about the *modality* a channel holds (no reserved names, no per-modality types), but the container enforces the value's *kind*. The primary `spectral_data` stays the Brillouin data; `channels` is a generic side-car.
- **Channel type constraint (#22, done 2026-09-13):** channels are restricted to `SpectralObject`s (`Spectrum`/`SpectralImage`/`SpectralVolume`/`SpectralContainer`), not arbitrary array-likes. A per-pixel map with no spectral axis (classification labels, masks, fit-result maps) is the degenerate case of a `SpectralObject` whose length-1 "spectral axis" holds a label instead of a continuous frequency shift - exactly like a brightfield image's channel. `as_channel`/`from_channel` (`core.py`) wrap and unwrap that case; `SpectralContainer.__init__` and `with_channel` raise `TypeError` for any other value. Rationale: a single type means every future channel-touching feature (`to_xarray` (#24), provenance, export) has one code path instead of dispatching on type at every call site.
- **Per-channel operations (#22):** `SpectralContainer.with_channel(name, obj)` / `.drop_channel(name)` / `.apply_to_channel(name, step)` add, remove, or transform a single named channel, each returning a new object and leaving the original untouched (same copy semantics as the existing spatial operations). `apply_to_channel` only ever calls `step.apply(...)` with no type check on `step`, so a RamanSPy `PreprocessingStep` works as a drop-in `step` too - see Compatibility notes below.
- Channels sharing the primary object's spatial `shape` (e.g. Raman on the same scan grid) are "grid-conformant" and participate transparently in spatial operations.
- Channels that don't (e.g. a camera-based brightfield/fluorescence channel) are "non-conformant" and are passed through spatial operations untouched.
- **Non-conformant channel access policy ("context warning" idea):** `__getitem__` and `flat` act on grid-conformant channels and pass non-conformant ones through **silently, unchanged** - no `UserWarning`. `from_stack`/`from_image_stack` are stricter: they merge `channels` only when every source has the same channel names and each channel is grid-conformant to its own source - otherwise the stacked result carries no channels at all (see Phase 1 below). Rationale: brillouinpy cannot know the intended data layout, so a warning would cry wolf. Visibility instead comes from introspection: `__repr__` lists channel names + shapes, and `SpectralContainer.channels_grid_conformant` returns `{name: bool}`. A future opt-in `warn_on_nonconformant` flag is easy to add if wanted.
- `mean` / `variance` / `tolist` collapse the spatial dimension and carry **no** channels.
- Camera→scan registration (`AffineTransform2D`, resampling) is **permanently out of scope** for the container (Phase 2, closed as out-of-scope - #4): a non-conformant channel has no in-package route to becoming conformant. It is carried alongside the data, passed through spatial operations untouched, and never analysed - registration is the user's step (in acquisition software or a setup-specific script), and a resampled channel simply arrives as a conformant one.

## Resolution

Originally planned as six phases (below, kept for the historical record - each
heading links its issue). All are now resolved via the umbrella issue (#3, closed
2026-09-14): the outcome is a smaller feature than planned, and deliberately so. The
container carries co-acquired modalities and per-pixel maps; everything
modality-specific stays outside the package, reachable through documented extension
points (`channels`, `PreprocessingStep`) rather than built in.

### Phase 1 - Container core (`brillouinpy/core.py`) - DONE (2026-09-03), narrowed (#22, 2026-09-13)
- `__init__` / `_create_data`: accept and store `channels` (always a dict).
- `channels_grid_conformant` property; `__repr__` shows channels.
- `flat`, `__getitem__`: apply the same spatial op to grid-conformant channels; pass non-conformant ones through unchanged (silently).
- `from_stack` / `from_image_stack`: merge `channels` only when every input has the same channel names and each is grid-conformant; otherwise the stacked result carries no channels.
- `save`/`load` (pickle) round-trip `channels`; `__setstate__` fills `{}` for pre-`channels` pickles.
- Also landed alongside: first-class `metadata` / `px_size_um`, and hard `ndim` validation on the `Spectrum`/`SpectralImage`/`SpectralVolume` subclasses.
- **Narrowed by #22:** channels restricted to `SpectralObject`s only (`TypeError` otherwise); `as_channel`/`from_channel` for per-pixel maps with no spectral axis; `with_channel`/`drop_channel`/`apply_to_channel` for per-channel operations. See the API section above.
- Tests: `tests/test_core.py`, `tests/test_ramanspy_interop.py`.

### Phase 2 - Camera calibration / registration - OUT OF SCOPE (#4, closed 2026-09-14)
Not because it's hard - the affine transform described above is realistic for this
setup. Because of where it belongs: co-registration is a property of a particular
microscope (which camera, which optical path, which calibration target), the same
side of the boundary as VIPA spectrum extraction or frequency-axis calibration - steps
that happen before data enters brillouinpy, in acquisition software or a
setup-specific script. Owning it here would mean carrying setup-specific knowledge the
package otherwise avoids. See the API section above for the resulting
non-conformant-channel policy. Reopen #4 if a registration path turns out to be needed
for something the container itself cannot express.

### Phase 3 - Colocalization / correlation analysis - OUT OF SCOPE (#5, closed 2026-09-14)
Which correlation is meaningful between two modalities depends on the question being
asked - Pearson, Spearman, Manders, something spatial - and on the biology. Shipping
one chosen set as *the* supported way would read as a recommendation. Once two
channels are grid-conformant (which requires #4, out of scope), the data is just two
arrays of matching shape: `scipy.stats` covers the standard metrics in a line, and
once `to_xarray` (#24) lands, `xr.corr`/`xr.cov` do it over chosen dimensions directly
with shared coordinates. Does **not** cover #23 (feeding externally derived labels
into `segmented_fit`) - those labels control how many peaks are fitted where, a
capability the package does own; correlation between channels is a different thing.

### Phase 4 - Fluorescence workflows - SUPERSEDED (#6, closed 2026-09-13)
Assumed fluorescence needed its own workflow layer; it doesn't. A fluorescence or
brightfield acquisition is already a `SpectralObject` and goes into `channels`
directly (#22), and the existing preprocessing steps (`BackgroundSubtractor`,
`denoise.*`, `normalise.*`, `Cropper` - see [`misc.py`](../../brillouinpy/preprocessing/misc.py))
operate on the `(*spatial, spectral)` array without knowing the modality, so they
apply unchanged. What remains genuinely fluorescence-specific - photobleaching
correction - doesn't need package code either: `PreprocessingStep` accepts any
callable of the form `(intensity_data, spectral_axis, **kwargs) -> (data, axis)`, so a
correction can be supplied by the user and composed into a pipeline like any built-in
step. Demonstrated in `examples/18_multimodal_channels.py` (#8). The deliberate
consequence: brillouinpy carries no modality-specific domain knowledge (no fluorophore
spectra, no Raman band assignments) - one documented extension point instead.

### Phase 5 - RamanSPy interoperability - DONE WITHOUT CONVERTERS (#7, closed 2026-09-14)
The actual interoperability goal - a brillouinpy `SpectralObject` usable with RamanSPy
- turns out to already hold, with no conversion step: RamanSPy's
`PreprocessingStep.apply()` deep-copies whatever object it's given and only ever
touches `.spectral_data`/`.spectral_axis`, with no `isinstance` check, and brillouinpy
objects (including channel values, see the API section above) carry those same two
attributes untouched. `to_ramanspy`/`from_ramanspy` - what this issue literally asked
for - were deliberately not built, since they aren't needed for the compatibility
claim; the one thing they'd still buy is producing an actual
`ramanspy.SpectralContainer` instance for a RamanSPy-side API that does an
`isinstance` check rather than duck-typing, and no concrete need for that has come up.
If one does, `to_brim`/`from_hdf5_bls`'s soft-optional-dependency pattern (see
[`io/export.py`](../../brillouinpy/io/export.py)) is still the right starting point
for a fresh issue. Verified against the real, installed `ramanspy` package (now a
`test` extra): `tests/test_ramanspy_interop.py`, `examples/17_channel_ramanspy_interop.py`.

### Phase 6 - Documentation - DONE, narrower than planned (#8, closed 2026-09-14)
[`docs/tutorial/multimodal-channels.md`](../tutorial/multimodal-channels.md),
`examples/17_channel_ramanspy_interop.py`, `examples/18_multimodal_channels.py`.
Narrower than originally specified: this phase listed "calibrate/register,
colocalize" as part of the workflow; both are now out of scope (#4, #5), so the page
covers what the package actually does instead - attaching Raman and camera channels,
grid conformance and what it does/doesn't mean, behaviour under spatial operations,
per-pixel maps via `as_channel`, and handing a channel to an external library.

## Continuing work

Two threads continue from here - what's left of the multimodal ambition, tracked as
their own issues rather than under this design document:

- **#23** - feeding externally derived labels into `segmented_fit`. The one
  cross-modal capability the package should own, because it controls a fit rather
  than describing a relationship between channels.
- **#24** - `to_xarray`, exporting a multimodal container as an `xarray.Dataset`. The
  general exit that makes declining Phases 2/3 sustainable: `channels` currently
  survive only in pickle, and no other export carries them.

## Compatibility notes

- **RamanSPy**: duck-typing compatibility (same `spectral_data`/`spectral_axis` attribute shape) is preserved since `channels` is purely additive - verified by test rather than by explicit converters (see Phase 5's resolution above; `to_ramanspy`/`from_ramanspy` were deliberately not built). This applies just as directly to a **channel value** as to the top-level object: a channel is validated (`core._validate_channels`) to be a full `SpectralObject`, so it carries the same untouched attributes and can be handed straight to a RamanSPy `PreprocessingStep` - either directly (`ramanspy_step.apply(container.channels["raman"])`) or via `SpectralContainer.apply_to_channel(name, step)`, which only ever calls `step.apply(...)` with no type check on `step`. No conversion code is needed for this; see `tests/test_ramanspy_interop.py` and `examples/17_channel_ramanspy_interop.py`.
- **brimfile / HDF5_BLS**: both are Brillouin-specific standards with no concept of auxiliary channels. `to_brim`/`from_brim` and `to_hdf5_bls`/`from_hdf5_bls` continue to round-trip the primary Brillouin data correctly, but `channels` content is not exported/imported through them. Only the existing pickle-based `save`/`load` round-trips `channels` losslessly (non-interoperably).
