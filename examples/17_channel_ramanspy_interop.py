# -*- coding: utf-8 -*-
"""
Example 17 - Passing a channel to RamanSPy
============================================

The correlative Brillouin+Raman setup described in
``docs/design/multimodal_container.md`` stores the co-registered Raman scan as a
channel on the Brillouin container: ``image.channels["raman"]``. Since a channel is
validated to be a full :class:`~brillouinpy.core.SpectralObject` (not a bare array),
it carries the same untouched ``spectral_data``/``spectral_axis`` attributes as any
top-level brillouinpy object - so it is *already* usable directly with `RamanSPy
<https://github.com/barahona-research-group/RamanSPy>`_ preprocessing steps, with no
conversion step and no library code needed for that.

This script demonstrates both entry points:

- passing ``image.channels["raman"]`` straight into a RamanSPy step's ``.apply()``;
- the equivalent, more ergonomic
  :meth:`~brillouinpy.core.SpectralContainer.apply_to_channel`, which only ever
  calls ``step.apply(...)`` on the named channel with no type check on ``step``, so
  a RamanSPy ``PreprocessingStep`` is a drop-in replacement for a brillouinpy one.

Requires the optional ``ramanspy`` package (``pip install ramanspy``; test-only
extra, not a runtime dependency of brillouinpy) - falls back to a message rather
than crashing if it isn't installed, the same way ``10_export_data.py`` handles the
optional ``HDF5_BLS``/``brimfile`` backends.
"""
import numpy as np

import brillouinpy as bp
from _synthetic_data import single_peak_image

if __name__ == '__main__':
    # The primary Brillouin data (see 01_load_data.py for the real-data path).
    image = single_peak_image()

    # A synthetic Raman-like channel on the same scan grid: a different spectral
    # axis (wavenumber-like), same spatial shape.
    raman_axis = np.linspace(500, 3000, 40)
    peak = 1.0 * 50.0 ** 2 / ((raman_axis - 1800) ** 2 + 50.0 ** 2)  # a single Lorentzian band
    raman_data = np.tile(peak, image.shape + (1,))
    raman_channel = bp.SpectralImage(raman_data, raman_axis)

    image = image.with_channel("raman", raman_channel)
    print(f"Container: {image!r}")

    try:
        import ramanspy

        cropper = ramanspy.preprocessing.misc.Cropper(region=(1000, 2000))

        # 1) Straight into RamanSPy: the channel value is duck-type compatible on
        #    its own, just like the top-level object.
        cropped_channel = cropper.apply(image.channels["raman"])
        print(f"Cropped channel directly: shape {cropped_channel.shape}, "
              f"{cropped_channel.spectral_length} spectral points")

        # 2) Via apply_to_channel: returns a new container, original untouched.
        transformed = image.apply_to_channel("raman", cropper)
        print(f"apply_to_channel result: {transformed.channels['raman'].spectral_length} spectral points "
              f"(original still has {image.channels['raman'].spectral_length})")

    except ImportError as exc:
        print(f"Skipping RamanSPy interop demo ({exc}).")
