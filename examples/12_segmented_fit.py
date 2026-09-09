# -*- coding: utf-8 -*-
"""
Example 12 - Fit with prior classification (``segmented_fit``)
=============================================================

The plain multi-peak DHO fit in ``09_fit_spectra.py`` assumes *every* pixel
holds the same number of peaks. That breaks on a sample where a constant
background/medium spectrum is present everywhere and one or more further
components are only *added* on top in smaller regions (a hydration background
plus a stiff inclusion; a cell's medium, cytoplasm and nucleus): a second or
third DHO peak is rank-deficient wherever its amplitude -> 0, so the fit fails
or plants a spurious spike on noise, and no post-hoc filter on the result
rescues it.

:func:`brillouinpy.analysis.fit.segmented_fit` does the workflow that works:
classify first with a cheap, model-free classifier (k-means on the raw
spectra, ordered by total intensity), then fit exactly as many DHO peaks as
each class actually contains, anchoring the peaks already identified in an
earlier stage. See ``benchmarks/`` for the full evidence.

Uses the synthetic additive-mixture generators; no preprocessed data needed.
"""
import matplotlib.pyplot as plt
import numpy as np

import brillouinpy as bp
from _synthetic_data import additive_blob_image, three_component_image

if __name__ == '__main__':
    # --- Two components: constant background + one added inclusion ------------
    image, label_map, _ = additive_blob_image(
        nx=40, ny=40, background_shift=5.6, component_shift=7.0,
        noise_model='poisson', peak_photons=400, edge='sharp',
    )

    result = bp.analysis.segmented_fit(image, n_components=2)
    # result.labels     : 0 = background only, 1 = background + inclusion
    # result.shift       : per pixel, the *newest* component's fitted shift (GHz)
    #                      - the background shift where labels == 0, the
    #                        inclusion shift where labels == 1
    # result.linewidth / result.amplitude : same, for the newest component
    # result.stage_shift : [median background shift, median inclusion shift]
    print('per-stage shifts (GHz):', np.round(result.stage_shift, 3))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    axes[0].imshow(label_map)
    axes[0].set_title('Ground truth')
    axes[1].imshow(result.labels)
    axes[1].set_title('segmented_fit classification')
    im = axes[2].imshow(result.shift)
    axes[2].set_title('Newest-component shift (GHz)')
    fig.colorbar(im, ax=axes[2])
    plt.show()

    # --- Three nested components: medium / cytoplasm / nucleus ---------------
    cell, cell_labels, true_params = three_component_image(nx=48, ny=48, peak_photons=400)
    cell_result = bp.analysis.segmented_fit(cell, n_components=3)

    print('true nucleus shift:', true_params['nucleus'][0], 'GHz')
    print('fitted nucleus shift (median over nucleus pixels):',
          round(float(np.nanmedian(cell_result.shift[cell_result.labels == 2])), 3), 'GHz')

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    axes[0].imshow(cell_labels)
    axes[0].set_title('Ground truth (0/1/2)')
    axes[1].imshow(cell_result.labels)
    axes[1].set_title('segmented_fit classification')
    im = axes[2].imshow(cell_result.shift)
    axes[2].set_title('Newest-component shift (GHz)')
    fig.colorbar(im, ax=axes[2])
    plt.show()

    # A custom classifier (any callable returning an integer label map ordered
    # fewest -> most components) can be passed instead of the default k-means:
    # def my_classifier(img, n_components): ...; return labels
    # bp.analysis.segmented_fit(image, n_components=2, classifier=my_classifier)
