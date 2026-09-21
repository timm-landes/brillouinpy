# -*- coding: utf-8 -*-
"""
Example 18 - Multimodal data: the ``channels`` API
=====================================================

A single :class:`~brillouinpy.core.SpectralContainer` holds one spectral
modality - the primary ``spectral_data``. A correlative setup (this group's
Brillouin+Raman point-scanner, plus a camera-based brightfield/epi-fluorescence
path) needs more than that: co-acquired data alongside the primary one. That's
what ``channels`` is for - see ``docs/design/multimodal_container.md`` for the
full motivation and roadmap.

This script walks through the ``channels`` API on its own (no real measurement
needed): attaching channels, grid-conformant vs. non-conformant behaviour through
spatial operations, ``apply_to_channel``/``drop_channel``, wrapping a label/mask
map with ``as_channel``/``from_channel``, and the pickle round-trip.
"""
import os

import numpy as np

import brillouinpy as bp
from _synthetic_data import single_peak_image

if __name__ == '__main__':
    # -----------------------------------------------------------------------
    # 1. Attaching channels
    # -----------------------------------------------------------------------
    # The primary object: Brillouin data on a 15x15 scan grid (see 01_load_data.py).
    image = single_peak_image()

    # A grid-conformant channel: Raman, acquired by the same point-scanner, so it
    # shares the primary object's spatial shape. Only its spectral axis differs.
    raman_axis = np.linspace(500, 3000, 40)
    peak = 1.0 * 50.0 ** 2 / ((raman_axis - 1800) ** 2 + 50.0 ** 2)
    raman_channel = bp.SpectralImage(np.tile(peak, image.shape + (1,)), raman_axis)

    # A non-conformant channel: a camera-based brightfield image. Different optical
    # path -> different field of view/pixel size than the scanner grid, so its
    # spatial shape doesn't match. Brightfield is the degenerate case of a
    # SpectralObject with a single "colour" (C=1) - no bespoke type needed.
    brightfield = bp.SpectralImage(np.random.default_rng(0).random((30, 30, 1)), np.array([0.0]))

    # A channel value must already be a SpectralObject - `channels=` (or
    # `with_channel`) raises TypeError for a bare array.
    image = image.with_channel("raman", raman_channel).with_channel("brightfield", brightfield)
    print(f"Container: {image!r}")
    print(f"Grid-conformant: {image.channels_grid_conformant}")

    # -----------------------------------------------------------------------
    # 2. Spatial operations: conformant channels follow along, others pass through
    # -----------------------------------------------------------------------
    pixel = image[0, 0]
    print(f"image[0, 0]: raman channel indexed along -> {pixel.channels['raman']!r},"
          f" brightfield channel shape {pixel.channels['brightfield'].shape} (unchanged)")

    flat = image.flat
    print(f"image.flat: raman channel shape {flat.channels['raman'].shape} (flattened along),"
          f" brightfield channel shape {flat.channels['brightfield'].shape} (unchanged)")

    # mean/variance collapse the spatial dimension entirely and carry no channels.
    print(f"image.mean channels: {image.mean.channels}")

    # -----------------------------------------------------------------------
    # 3. Transforming a single channel: apply_to_channel
    # -----------------------------------------------------------------------
    # apply_to_channel(name, step) only ever calls step.apply(channels[name]), with
    # no type check on `step` - any brillouinpy *or* RamanSPy PreprocessingStep
    # works (see examples/17_channel_ramanspy_interop.py for the RamanSPy case).
    cropper = bp.preprocessing.misc.Cropper(region=(1000, 2000))
    cropped = image.apply_to_channel("raman", cropper)
    print(f"apply_to_channel('raman', Cropper(...)): {cropped.channels['raman'].spectral_length} spectral points "
          f"(original untouched: {image.channels['raman'].spectral_length})")

    # -----------------------------------------------------------------------
    # 4. Removing a channel
    # -----------------------------------------------------------------------
    without_brightfield = image.drop_channel("brightfield")
    print(f"drop_channel('brightfield'): {list(without_brightfield.channels)} "
          f"(original untouched: {list(image.channels)})")

    # -----------------------------------------------------------------------
    # 5. A label/mask map as a channel: as_channel / from_channel
    # -----------------------------------------------------------------------
    # Anything with no spectral axis of its own (a classification map, a mask, a
    # fit-result map) needs as_channel() first - it's the degenerate SpectralObject
    # whose length-1 "spectral axis" holds a label instead of real spectral data.
    per_pixel_mean = np.ma.filled(image.spectral_data, np.nan).mean(axis=-1)
    label_map = (per_pixel_mean > np.nanmean(per_pixel_mean)).astype(int)
    image = image.with_channel("labels", bp.as_channel(label_map))
    recovered = bp.from_channel(image.channels["labels"])
    print(f"as_channel/from_channel round-trip matches: {np.array_equal(recovered, label_map)}")

    # -----------------------------------------------------------------------
    # 6. Channels round-trip through save/load (pickle)
    # -----------------------------------------------------------------------
    os.makedirs('pp_data', exist_ok=True)
    image.save('multimodal_image.pkl', directory='pp_data')
    reloaded = bp.SpectralImage.load(os.path.join('pp_data', 'multimodal_image.pkl'))
    print(f"Reloaded channels: {list(reloaded.channels)}")
