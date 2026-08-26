# -*- coding: utf-8 -*-
"""
Example 4 - Classical data analysis: mean & variance
========================================================

Before reaching for a peak-fitting model or a decomposition/clustering method,
it's worth building a "classical" statistical picture of the dataset first: the
mean spectrum (where are the peaks?) and the variance spectrum (which of those
peaks actually change from pixel to pixel, rather than being a constant
background feature or a shared noise floor?).

This matters most for data holding more than one Brillouin doublet, especially
when those doublets overlap closely enough to blend into what looks like a single
peak: the mean spectrum alone can't tell you how many independent modes are
actually mixed into the dataset. The variance spectrum can - each spatially-varying
doublet still shows up as its own pair of variance peaks (Stokes/anti-Stokes),
because that's where the intensity actually changes from pixel to pixel (as the
local mix of the two materials changes), even where the *sum* of both doublets in
the mean spectrum has merged into one indistinguishable peak. This gives a fast,
model-free estimate of how many endmembers/components/modes to look for in the
steps that follow (``05_unmix_vca.py`` through ``09_fit_spectra.py``).

Continues from ``03_preprocess_data.py`` conceptually (nothing to load - this operates
directly on preprocessed data). Unlike the earlier examples, this one uses
``two_material_image()`` rather than ``single_peak_image()``, with its two
materials' frequency shifts moved much closer together than that function's
default (6.0/11.0 GHz) - close enough that the two doublets fully merge in the
mean spectrum, rather than sitting cleanly apart as they do in the later
unmixing/decomposition/clustering examples.
"""
import matplotlib.pyplot as plt

import brillouinpy as bp
from _synthetic_data import two_material_image

if __name__ == '__main__':
    # freq_shift_a/freq_shift_b much closer together than the two_material_image()
    # default (6.0/11.0 GHz) - close enough that the two doublets fully merge in the
    # mean spectrum, so the variance spectrum actually has something extra to show.
    brillouin_image, true_abundances, true_spectra = two_material_image(freq_shift_a=7.0, freq_shift_b=7.5)

    # Denoising matters more here than in earlier steps: the variance spectrum
    # amplifies per-pixel noise (it's a squared quantity), which can otherwise
    # produce spurious extra local maxima next to a real peak.
    preprocessed_image = bp.preprocessing.Pipeline([
        bp.preprocessing.denoise.SavGol(window_length=7, polyorder=2),
        bp.preprocessing.normalise.MaxIntensity(pixelwise=True),
    ]).apply(brillouin_image)

    # .mean and .variance both return a Spectrum: the per-channel mean/variance
    # across every pixel in the image.
    mean_spectrum = preprocessed_image.mean
    variance_spectrum = preprocessed_image.variance

    fig = plt.figure(figsize=(10, 5), layout='constrained')
    plt.subplot(121)
    bp.plot.spectra(mean_spectrum, title='Mean spectrum', yscale='linear')

    # bp.plot.peaks additionally runs scipy.signal.find_peaks and marks the result -
    # a 'prominence' relative to the spectrum's own scale keeps this threshold
    # meaningful regardless of the dataset's absolute intensity units.
    plt.subplot(122)
    _, found_peaks, _ = bp.plot.peaks(
        variance_spectrum, title='Variance spectrum', yscale='linear',
        prominence=variance_spectrum.spectral_data.max() * 0.1, return_peaks=True,
    )
    plt.show()

    # Real Brillouin peaks come in +/- (Stokes/anti-Stokes) pairs, so half the
    # variance-peak count is the number of independent modes ('doublets') worth
    # fitting/decomposing for - here, 2, matching the two materials mixed into
    # this synthetic dataset. Feed this into 'expected_peaks' (fitting, see
    # '09_fit_spectra.py') and 'n_endmembers'/'n_components'/'n_clusters'
    # (unmixing/decomposition/clustering, see '05_unmix_vca.py' onward).
    n_doublets = len(found_peaks) // 2
    print(f"Found {len(found_peaks)} variance peaks at {found_peaks} GHz -> {n_doublets} doublet(s)")
