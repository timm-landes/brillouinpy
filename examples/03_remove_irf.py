# -*- coding: utf-8 -*-
"""
Example 3 - Removing the Instrument Response Function (IRF)
===============================================================

Real Brillouin spectra typically contain a strong, narrow Instrument Response
Function (IRF)/Rayleigh line in addition to the much weaker Brillouin peak(s) of
interest. Shows how to use
:class:`brillouinpy.preprocessing.misc.IRF_Remover` to detect and blank out
that IRF region.

:class:`~brillouinpy.preprocessing.misc.IRF_Remover` locates the IRF by
starting at the spectrum's global maximum (i.e. it assumes the IRF is the strongest
feature) and scanning outwards for the first exact-zero intensity channel on each
side - which real detector data has around a saturated/blanked IRF region. This
example therefore uses synthetic data crafted with such real zero channels; run it
on your own data only once you've confirmed it has a comparable zero baseline
around the IRF (e.g. by plotting a raw, unprocessed spectrum first).

For a more involved alternative that sharpens the remaining spectrum via
Richardson-Lucy deconvolution before removing the IRF, see
:class:`~brillouinpy.preprocessing.misc.Deconvoluter_IRF` - it requires the
full ``(x, y, z, t, spectral)`` shape returned by
:func:`brillouinpy.utils.prepare_brillouin_data`, not the 3D ``(x, y,
spectral)`` shape used by :class:`~brillouinpy.SpectralImage` here.

Typically, IRF removal happens early in a preprocessing pipeline, before steps
(like normalisation) that would otherwise be skewed by the IRF's much larger
intensity - see ``02_preprocess_data.py`` for how to combine it with other steps.
"""
import matplotlib.pyplot as plt

import brillouinpy as bp
from _synthetic_data import image_with_irf

if __name__ == '__main__':
    brillouin_image, irf_index = image_with_irf()

    # 'offset' is the number of additional channels to blank out on each side of the
    # detected IRF region (e.g. to also remove some stray Rayleigh-scattered light
    # bleeding past the IRF's zero-baseline edges).
    irf_remover = bp.preprocessing.misc.IRF_Remover(offset=3)
    cleaned_image = irf_remover.apply(brillouin_image)

    # Compare a single raw spectrum against the IRF-removed result
    fig = plt.figure(figsize=(10, 5), layout='constrained')
    plt.subplot(121)
    bp.plot.spectra(brillouin_image[0, 0], title='Raw spectrum (with IRF)', yscale='linear')
    plt.subplot(122)
    bp.plot.spectra(cleaned_image[0, 0], title='IRF removed', yscale='linear')
    plt.show()

    # The removed region becomes zero rather than being cropped out of the spectral
    # axis, so 'cleaned_image' keeps the same shape as 'brillouin_image':
    print(f"Raw shape: {brillouin_image.shape + (brillouin_image.spectral_length,)}, "
          f"cleaned shape: {cleaned_image.shape + (cleaned_image.spectral_length,)}")

    # From here on, feed 'cleaned_image' into the rest of the pipeline shown in
    # '02_preprocess_data.py' (denoising, cropping, normalisation, ...).
