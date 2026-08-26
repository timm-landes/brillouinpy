# -*- coding: utf-8 -*-
"""
Example 3 - Preprocessing Brillouin data
=========================================

Shows how to build and apply a :class:`brillouinpy.preprocessing.Pipeline`
of :class:`~brillouinpy.preprocessing.Step.PreprocessingStep` instances -
despiking, denoising, cropping and normalisation.

Continues from ``02_remove_irf.py``: run that example first to get
``pp_data/irf_removed_image.pkl``, otherwise this script falls back to
``pp_data/brillouin_image.pkl`` (from ``01_load_data.py``) or synthetic data on
its own. IRF removal deliberately happens before this step (see
``02_remove_irf.py``), since normalisation/denoising here would otherwise be
skewed by the IRF's much larger intensity.
"""
import os

import matplotlib.pyplot as plt

import brillouinpy as bp
from _synthetic_data import single_peak_image

if __name__ == '__main__':
    irf_removed_path = os.path.join('pp_data', 'irf_removed_image.pkl')
    brillouin_image_path = os.path.join('pp_data', 'brillouin_image.pkl')
    if os.path.exists(irf_removed_path):
        brillouin_image = bp.SpectralImage.load(irf_removed_path)
    elif os.path.exists(brillouin_image_path):
        brillouin_image = bp.SpectralImage.load(brillouin_image_path)
    else:
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
