# -*- coding: utf-8 -*-
"""
Example 11 - Spectral phasor analysis
=====================================

A model-free alternative to the peak fitting in ``09_fit_spectra.py``. Every
spectrum is projected onto the first harmonic of its DFT, giving one point
``(G, S)`` in the phasor plane (:func:`brillouinpy.analysis.phasor.phasor`).
Spectra with different Brillouin shifts land in different places, so you get
contrast and segmentation without fitting a line shape and without a non-linear
optimisation per pixel - useful for noisy data and for real-time preview.

Method: Elsayad, K. (2019), "Spectral Phasor Analysis for Brillouin
Microspectroscopy", Front. Phys. 7:62. https://doi.org/10.3389/fphy.2019.00062

Uses ``two_region_image``: two materials meeting at a hard edge, each pixel a
pure spectrum (no mixing). See ``12_method_limitations.py`` for how phasor
analysis compares to fitting/unmixing/clustering as the two shifts get closer.
"""
import matplotlib.pyplot as plt
import numpy as np

import brillouinpy as bp
from _synthetic_data import two_region_image

if __name__ == '__main__':
    # Two regions ~0.6 GHz apart - close, but comfortably resolvable at this noise level.
    image, label_map, _ = two_region_image(
        nx=50, ny=50, freq_shift_a=5.6, freq_shift_b=6.2, noise=3e-4, geometry='blob'
    )

    # Restrict to the Stokes side and subtract each spectrum's own baseline: a
    # constant background otherwise drags every phasor toward the origin.
    ph = bp.analysis.phasor.phasor(image, axis_range=(3, 14), background='min')

    # The phasor cloud: two regions -> two clusters.
    bp.plot.phasor(ph, title='Spectral phasor cloud')

    # Fit-free Brillouin-shift map straight from the phasor phase.
    shift_map = bp.analysis.phasor.phase_to_shift(ph)

    # Cursor selection: centre on the region-B cluster (ground truth known here),
    # radius ~1/3 of the distance between the two cluster centroids.
    b = label_map == 1
    center_a = (np.nanmean(ph.G[~b]), np.nanmean(ph.S[~b]))
    center_b = (np.nanmean(ph.G[b]), np.nanmean(ph.S[b]))
    radius = np.hypot(center_b[0] - center_a[0], center_b[1] - center_a[1]) / 3
    region_b_mask = bp.analysis.phasor.phasor_cursor(ph, center=center_b, radius=radius)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4), layout='constrained')
    im0 = axes[0].imshow(shift_map)
    axes[0].set_title('Phasor shift estimate (GHz)')
    fig.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(ph.modulus)
    axes[1].set_title('Phasor modulus (inverse linewidth proxy)')
    fig.colorbar(im1, ax=axes[1])
    axes[2].imshow(region_b_mask)
    axes[2].set_title('Cursor: region B')
    axes[3].imshow(label_map)
    axes[3].set_title('Ground truth')
    plt.show()

    # Or segment the whole map directly in the phasor plane with the existing
    # clustering step (needs scikit-learn):
    # from sklearn.cluster import KMeans
    # labels = KMeans(n_clusters=2, n_init=10).fit_predict(ph.features()).reshape(image.shape)
