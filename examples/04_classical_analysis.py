# -*- coding: utf-8 -*-
"""
Example 4 - Classical data analysis: mean & variance
========================================================

Before reaching for a peak-fitting model or a decomposition/clustering method,
it's worth building a "classical" statistical picture of the dataset first: the
mean spectrum (where are the peaks?) and the variance spectrum (which of those
peaks actually change from pixel to pixel, rather than being a constant
background feature or a shared noise floor?).

This matters most for data holding more than one Brillouin doublet: the mean
spectrum alone doesn't tell you how many independent modes are mixed into the
dataset. The variance spectrum does - each spatially-varying doublet shows up as
its own pair of variance peaks (Stokes/anti-Stokes), giving a fast, model-free
estimate of how many endmembers/components/modes to look for in the steps that
follow (``05_unmix_vca.py`` through ``09_fit_spectra.py``).

Continues from ``03_remove_irf.py`` conceptually (nothing to load - this operates
directly on preprocessed data). Unlike the earlier examples, this one uses
``two_material_image()`` rather than ``single_peak_image()``, since it has two
spatially-varying Brillouin doublets to actually tell apart.
"""
import matplotlib.pyplot as plt

import brillouinpy as bp
from _synthetic_data import two_material_image

if __name__ == '__main__':
    brillouin_image, true_abundances, true_spectra = two_material_image()

    preprocessed_image = bp.preprocessing.Pipeline([
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
