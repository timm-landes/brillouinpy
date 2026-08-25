# -*- coding: utf-8 -*-
"""
Example 3 - Fitting Brillouin spectra
======================================

Shows how to fit every spectrum in a :class:`brillouinanalyzer.SpectralImage` with a
peak model - :class:`~brillouinanalyzer.analysis.fitmodel.DHO` (Damped Harmonic
Oscillator, the physically correct lineshape for Brillouin peaks) or
:class:`~brillouinanalyzer.analysis.fitmodel.Lorentzian` - and how to read the
resulting per-pixel parameter maps.

Note: fitting runs on a process pool and is CPU-bound; for a small dataset like the
one used here, most of the wall time is the one-off cost of spawning worker
processes, not the fits themselves.

Continues from ``02_preprocess_data.py``: run that example first to get
``pp_data/preprocessed_image.pkl``, otherwise this script falls back to synthetic
data on its own.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

import brillouinanalyzer as bp
from _synthetic_data import single_peak_image

if __name__ == '__main__':
    pp_data_path = os.path.join('pp_data', 'preprocessed_image.pkl')
    if os.path.exists(pp_data_path):
        preprocessed_image = bp.SpectralImage.load(pp_data_path)
    else:
        preprocessed_image = single_peak_image()

    # Fit a single-peak DHO model to every spectrum.
    # p0 must have length (expected_peaks * 3) + 2: for each peak
    # [Amplitude, Frequency shift (GHz), FWHM (GHz)], followed by [Background, Asymmetry].
    dho_fit = bp.analysis.fitmodel.DHO(
        expected_peaks=1,
        p0=[0.005, 8.5, 1, 0, 0],
        bounds=None,  # give bounds if you get unreasonable results or strongly overlapping peaks
    )
    fitted_parameters, covariances = dho_fit.apply(preprocessed_image)
    # fitted_parameters has shape (x, y, 5): [Amplitude, FreqShift, LineWidth, Background, Asymmetry]

    # A 2-peak fit works the same way, just with a matching p0/bounds length:
    # dho_fit_2peaks = bp.analysis.fitmodel.DHO(
    #     expected_peaks=2,
    #     p0=[0.005, 6.0, 1, 0.005, 11.0, 1, 0, 0],
    #     bounds=None,
    # )
    # fitted_parameters, covariances = dho_fit_2peaks.apply(preprocessed_image)

    # A Lorentzian model can be used the same way instead:
    # lor_fit = bp.analysis.fitmodel.Lorentzian(expected_peaks=1, p0=[1500, 9, 1, 0, 0], bounds=None)
    # fitted_parameters, covariances = lor_fit.apply(preprocessed_image)

    # Visually check the fit against the mean spectrum
    mean_amplitude, mean_shift, mean_linewidth, mean_bg, mean_asym = (
        fitted_parameters[..., i].mean() for i in range(5)
    )
    plt.figure(figsize=(6, 4), layout='constrained')
    bp.plot.mean_spectra(preprocessed_image, title='Fit vs. data', yscale='linear')
    plt.plot(
        preprocessed_image.spectral_axis,
        bp.analysis.fitmodel._DHO_1(
            preprocessed_image.spectral_axis, mean_amplitude, mean_shift, mean_linewidth, mean_bg, mean_asym
        ),
        label='Mean DHO fit', color='red',
    )
    plt.legend()
    plt.show()

    # Plot the fitted parameter maps
    plt.figure(figsize=(15, 5), layout='constrained')
    plt.subplot(131)
    plt.imshow(fitted_parameters[:, :, 0])
    plt.colorbar(label='Amplitude (a.u.)')
    plt.subplot(132)
    plt.imshow(fitted_parameters[:, :, 1])
    plt.colorbar(label='Frequency shift (GHz)')
    plt.subplot(133)
    plt.imshow(fitted_parameters[:, :, 2])
    plt.colorbar(label='Linewidth (GHz)')
    plt.show()

    # Persist the fit result for the export example
    os.makedirs('pp_data', exist_ok=True)
    np.save(os.path.join('pp_data', 'fitted_parameters.npy'), np.ma.filled(fitted_parameters, np.nan))
