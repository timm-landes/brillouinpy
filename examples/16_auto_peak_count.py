# -*- coding: utf-8 -*-
"""
Example 16 - Letting the fit choose the number of modes
======================================================

``09_fit_spectra.py`` fixes ``expected_peaks`` up front. When you do not know how
many Brillouin modes a spectrum contains, an information criterion can decide:

* ``fitmodel.estimate_peak_count(image)`` - **one** count for the whole image,
  from the mean spectrum. Cheap and steady; use it when the mode count is a
  property of the sample.
* ``DHO(expected_peaks='auto')`` - a per-pixel choice (fits 1..``max_peaks`` and
  keeps whichever minimises the BIC, provided the added peak is a real, non-tiny
  mode). The result is padded to ``3*max_peaks+2``; ``fitted_peak_count`` reads
  the per-pixel count back.

Both are heuristics: they assume the DHO lineshape is otherwise a good model, so
a residual elastic-scattering wing or an un-deconvolved IRF can bias the count
upward. Raise ``min_peak_fraction`` (or just set ``expected_peaks`` by hand) on
such data.

Uses ``two_mode_image``: every pixel holds two modes - mode A at a fixed 6.0 GHz,
mode B ramping from 10 to 13 GHz across the image.
"""
import matplotlib.pyplot as plt
import numpy as np

import brillouinpy as bp
from _synthetic_data import two_mode_image

if __name__ == '__main__':
    image, true_shift_b, info = two_mode_image(nx=30, ny=30)

    best, scores = bp.analysis.fit.estimate_peak_count(image, return_scores=True)
    print(f"estimate_peak_count (whole image): {best}")
    print("  BIC per candidate:", {k: round(v, 1) for k, v in scores.items()})

    params, _ = bp.analysis.fit.DHO(expected_peaks='auto', max_peaks=3).apply(image)
    counts = bp.analysis.fit.fitted_peak_count(params)
    print(f"per-pixel mode counts: {np.bincount(counts.ravel(), minlength=4)[1:]} (1 / 2 / 3 modes)")

    # Both modes are fitted per pixel: separate them by shift (mode A ~6 GHz is the
    # lower one, mode B ~9-11 GHz the higher).
    two_mode = counts >= 2
    s0, s1 = np.abs(params[..., 1]), np.abs(params[..., 4])
    shift_a = np.where(two_mode, np.minimum(s0, s1), np.nan)
    shift_b = np.where(two_mode, np.maximum(s0, s1), np.nan)
    print(f"mode A shift: {np.nanmedian(shift_a):.2f} GHz (true {info['shift_a']})")
    print(f"mode B shift: recovered {np.nanmin(shift_b):.1f}-{np.nanmax(shift_b):.1f} GHz "
          f"(true {true_shift_b.min():.0f}-{true_shift_b.max():.0f})")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    im0 = axes[0].imshow(counts, vmin=1, vmax=3)
    axes[0].set_title('Fitted number of modes')
    fig.colorbar(im0, ax=axes[0], ticks=[1, 2, 3])
    im1 = axes[1].imshow(true_shift_b)
    axes[1].set_title('True mode-B shift (GHz)')
    fig.colorbar(im1, ax=axes[1])
    im2 = axes[2].imshow(shift_b)
    axes[2].set_title("Recovered mode-B shift (expected_peaks='auto')")
    fig.colorbar(im2, ax=axes[2])
    plt.show()
