# -*- coding: utf-8 -*-
"""
Example 2 - Removing the Instrument Response Function (IRF)
===============================================================

Real Brillouin spectra from Tandem Fabry-Perot interferometers typically contain a
strong, narrow Instrument Response Function (IRF)/Rayleigh line in addition to the
much weaker Brillouin peak(s) of interest. Shows how to use
:class:`~brillouinpy.preprocessing.misc.Deconvoluter_IRF` to sharpen the spectrum
via Richardson-Lucy deconvolution (using the detected IRF as the point-spread
function) and then blank out the (now deconvolved) IRF region.

IRF removal should happen early, before steps (like normalisation or denoising)
that would otherwise be skewed by the IRF's much larger intensity - see
``03_preprocess_data.py`` for the rest of the pipeline, which continues from the
output of this script.

:class:`~brillouinpy.preprocessing.misc.Deconvoluter_IRF` (like
:class:`~brillouinpy.preprocessing.misc.IRF_Remover`) locates the IRF by starting
at the spectrum's global maximum (i.e. it assumes the IRF is the strongest
feature) and scanning outwards for the first exact-zero intensity channel on each
side - which real detector data has around a saturated/blanked IRF region. This
example therefore uses synthetic data crafted with such real zero channels; run it
on your own data only once you've confirmed it has a comparable zero baseline
around the IRF (e.g. by plotting a raw, unprocessed spectrum first).

For the simpler alternative that just crops the IRF away without the
deconvolution step, see :class:`~brillouinpy.preprocessing.misc.IRF_Remover`.

Continues from ``01_load_data.py``: run that example first to get
``pp_data/brillouin_image.pkl``, otherwise this script falls back to synthetic
data on its own.
"""
import os

import matplotlib.pyplot as plt

import brillouinpy as bp
from _synthetic_data import image_with_irf

if __name__ == '__main__':
    brillouin_image_path = os.path.join('pp_data', 'brillouin_image.pkl')
    if os.path.exists(brillouin_image_path):
        brillouin_image = bp.SpectralImage.load(brillouin_image_path)
    else:
        brillouin_image, irf_index = image_with_irf()

    # 'offset' is the number of additional channels to blank out on each side of the
    # detected IRF region (e.g. to also remove some stray Rayleigh-scattered light
    # bleeding past the IRF's zero-baseline edges) - real data with more stray light
    # may need a much larger value here (e.g. 65) than this synthetic example does.
    deconvoluter = bp.preprocessing.misc.Deconvoluter_IRF(offset=3, iterations=4, padding=None)
    cleaned_image = deconvoluter.apply(brillouin_image)

    # Compare a single raw spectrum against the IRF-removed result
    fig = plt.figure(figsize=(10, 5), layout='constrained')
    plt.subplot(121)
    bp.plot.spectra(brillouin_image[0, 0], title='Raw spectrum (with IRF)', yscale='linear')
    plt.subplot(122)
    bp.plot.spectra(cleaned_image[0, 0], title='IRF removed (deconvolved)', yscale='linear')
    plt.show()

    # The removed region becomes NaN rather than being cropped out of the spectral
    # axis, so 'cleaned_image' keeps the same shape as 'brillouin_image'. Downstream
    # steps must be able to handle NaN values (e.g. AnalysisStep drops NaN-containing
    # channels automatically).
    print(f"Raw shape: {brillouin_image.shape + (brillouin_image.spectral_length,)}, "
          f"cleaned shape: {cleaned_image.shape + (cleaned_image.spectral_length,)}")

    # Persist for '03_preprocess_data.py' onwards
    os.makedirs('pp_data', exist_ok=True)
    cleaned_image.save('irf_removed_image.pkl', directory='pp_data')
