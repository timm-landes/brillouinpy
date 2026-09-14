# Multimodal data and channels

*Scripts: `examples/18_multimodal_channels.py`, `examples/17_channel_ramanspy_interop.py`*

A correlative setup - this group's combined Brillouin+Raman point-scanner, plus a
camera-based brightfield/epi-fluorescence path for sample positioning - acquires
more than one spectral modality per measurement. `channels` is how `brillouinpy`
represents that: co-acquired data attached alongside a container's primary
`spectral_data`, without inventing a bespoke type per modality. See
[`docs/design/multimodal_container.md`](https://github.com/timm-landes/brillouinpy/blob/main/docs/design/multimodal_container.md)
for the full motivation.

## A quick primer on the container structure

Every object in this tutorial - `Spectrum`, `SpectralImage`, `SpectralVolume` - is a
`SpectralContainer` holding an intensity array of shape `(*spatial, B)`: any number
of leading spatial dimensions, then a trailing spectral axis of length `B` (see
[Writing your own loader](custom-loader.md)). Which concrete class you get is
decided purely by how many dimensions the data has - one spatial dimension gives a
`Spectrum`, two a `SpectralImage`, three a `SpectralVolume`, anything else the base
`SpectralContainer` - so `bp.SpectralImage(data, axis)` and
`bp.core._create_data(data, axis)` (dispatching automatically) are equivalent for
3-D data. `.shape` always reports the *spatial* shape (without the trailing
spectral axis); `.spectral_length` is `B`.

This matters for channels because a channel value is held to exactly the same
standard: it must be one of these same `SpectralObject` types, not a bare array.

## Attaching channels

```python
import brillouinpy as bp
from _synthetic_data import single_peak_image

image = single_peak_image()  # the primary Brillouin data, shape (15, 15)

# A channel value must already be a SpectralObject. Raman, fluorescence and
# brightfield data all qualify as-is - only a spectral-axis-free array (a
# classification map, a mask) needs wrapping first, see below.
raman_channel = bp.SpectralImage(raman_data, raman_axis)

image = image.with_channel("raman", raman_channel)
# equivalently, at construction time:
# image = bp.SpectralImage(data, axis, channels={"raman": raman_channel})
```

`with_channel`/`drop_channel`/`apply_to_channel` never mutate `self` - each returns
a new object, deep-copying every channel (not just the one being touched), so the
original is always safe to keep using.

## Grid-conformant vs. non-conformant channels

The Brillouin+Raman point-scanner acquires both on the same scan grid - a "raman"
channel built that way shares `image`'s spatial `shape` and is **grid-conformant**.
A camera-based brightfield image, acquired over a different field of view/pixel
size, generally isn't - it's **non-conformant**. `brillouinpy` doesn't guess your
intended layout, so it doesn't warn about the mismatch; it just treats the two
cases differently through spatial operations:

```python
image.channels_grid_conformant
# {'raman': True, 'brightfield': False}

pixel = image[0, 0]
pixel.channels["raman"]        # indexed along with the primary object -> a Spectrum
pixel.channels["brightfield"]  # passed through unchanged - still the full image

image.flat.channels["raman"]        # flattened along, shape (225,)
image.flat.channels["brightfield"]  # still unchanged

image.mean.channels  # {} - mean/variance collapse the spatial dimension entirely
                      # and carry no channels (there's nothing left to index)
```

Introspect with `.channels_grid_conformant` and `repr(image)` (which lists every
channel's shape) rather than relying on a warning that would too often be wrong.

## Transforming and removing channels

`apply_to_channel(name, step)` replaces one named channel with `step.apply(...)`
applied to it - `step` only ever needs to expose `.apply(spectral_object)`, with no
type check, so a `brillouinpy.preprocessing` step and a RamanSPy
`PreprocessingStep` are equally valid:

```python
cropped = image.apply_to_channel("raman", bp.preprocessing.misc.Cropper(region=(1000, 2000)))
without_brightfield = image.drop_channel("brightfield")
```

See [`examples/17_channel_ramanspy_interop.py`](https://github.com/timm-landes/brillouinpy/blob/main/examples/17_channel_ramanspy_interop.py):
a channel value is exactly as duck-type compatible with RamanSPy as the top-level
object (same untouched `spectral_data`/`spectral_axis`), so it can be handed
directly to a RamanSPy step too - `apply_to_channel` is just the more ergonomic
entry point for it.

## Labels, masks and fit results as channels

A classification map, a mask, or a per-pixel fit-result map has no spectral axis of
its own - it's the degenerate case of a `SpectralObject` whose length-1 "spectral
axis" holds a label instead of real spectral data. `as_channel`/`from_channel` wrap
and unwrap that case:

```python
label_map = ...  # a plain (*spatial,) array, e.g. from segmented_fit
image = image.with_channel("labels", bp.as_channel(label_map))

recovered = bp.from_channel(image.channels["labels"])  # back to the plain array
```

`from_channel` raises `ValueError` on an object with a real (length > 1) spectral
axis, rather than silently mangling it - pass those (Raman, fluorescence,
brightfield) to `channels` unwrapped instead.

## Persistence

`channels` round-trips losslessly through the existing pickle-based `save`/`load`
(not through `to_brim`/`to_hdf5_bls`, which are Brillouin-specific standards with no
concept of auxiliary channels - see [Exporting results](exporting.md)):

```python
image.save('multimodal_image.pkl', directory='pp_data')
reloaded = bp.SpectralImage.load('pp_data/multimodal_image.pkl')
list(reloaded.channels)  # ['raman', 'brightfield', 'labels']
```
