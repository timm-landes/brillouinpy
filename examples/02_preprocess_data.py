# -*- coding: utf-8 -*-
"""
Example 2 - Preprocessing Brillouin data
=========================================

Shows how to build and apply a :class:`brillouinpy.preprocessing.Pipeline`
of :class:`~brillouinpy.preprocessing.Step.PreprocessingStep` instances -
despiking, denoising, instrument-response removal, cropping and normalisation.

Continues from ``01_load_data.py``: run that example first (or just execute this
one, it falls back to synthetic data on its own).
"""
import os

import matplotlib.pyplot as plt

import brillouinpy as bp
from _synthetic_data import single_peak_image

if __name__ == '__main__':
    brillouin_image = single_peak_image()

    # Define the processing pipeline. Steps are applied in the order given.
    pipeline = bp.preprocessing.Pipeline([
        # Remove cosmic-ray spikes before anything else touches the intensity values
        bp.preprocessing.despike.WhitakerHayes(kernel_size=3, threshold=8),

        # Smooth out detector noise
        bp.preprocessing.denoise.SavGol(window_length=7, polyorder=2),

        # Keep only the spectral region of interest
        # bp.preprocessing.misc.Cropper(region=(-15, 15)),

        # Normalise every spectrum's intensity to its own maximum
        bp.preprocessing.normalise.MaxIntensity(pixelwise=True),
    ])

    # Instrument Response Function (IRF)/Rayleigh-line removal - only shown here as a
    # snippet, not run on the synthetic data above, since it needs a real zero-baseline
    # IRF region to detect:
    #
    #   # Just crop the IRF channels away, given data loaded via 'load_spectral_image'
    #   # (x, y, spectral) shape:
    #   pipeline.append(bp.preprocessing.misc.IRF_Remover(offset=10))
    #
    #   # Or deconvolve it out via Richardson-Lucy before removing it (see
    #   # '_deconvolute_irf' internals) - this variant needs the full (x, y, z, t,
    #   # spectral) shape returned by 'prepare_brillouin_data', not the 3D
    #   # (x, y, spectral) shape from 'load_spectral_image':
    #   pipeline.append(bp.preprocessing.misc.Deconvoluter_IRF(offset=65, iterations=4, padding=None))

    preprocessed_image = pipeline.apply(brillouin_image)

    # A single preprocessing step can also be applied on its own, without a pipeline:
    # denoised_only = bp.preprocessing.denoise.SavGol(window_length=7, polyorder=2).apply(brillouin_image)

    # Compare the raw and preprocessed mean spectra
    fig = plt.figure(figsize=(10, 5), layout='constrained')
    plt.subplot(121)
    bp.plot.mean_spectra(brillouin_image, title='Raw Brillouin data', yscale='log')
    plt.subplot(122)
    bp.plot.mean_spectra(preprocessed_image, title='Preprocessed Brillouin data', yscale='linear')
    plt.show()

    # Persist the preprocessed data for the next examples
    os.makedirs('pp_data', exist_ok=True)
    preprocessed_image.save('preprocessed_image.pkl', directory='pp_data')
