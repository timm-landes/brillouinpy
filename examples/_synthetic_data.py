# -*- coding: utf-8 -*-
"""
Synthetic Brillouin data generator shared by the examples in this folder.

The real workflow starts from :func:`brillouinpy.utils.prepare_brillouin_data`
(see ``01_load_data.py``), which needs a folder of ``.DAT``/``.csv`` measurement
files on disk. So that every example script in this folder can be run as-is,
without any measurement data at hand, this module fabricates a small
:class:`brillouinpy.SpectralImage` from a handful of Damped Harmonic
Oscillator (DHO) peaks placed at known positions - so you always know what the
"correct" fit/unmixing result should look like.
"""
import numpy as np

import brillouinpy as bp


def _dho(x, amplitude, freq_shift, linewidth, background=0.0, asymmetry=0.0):
    return amplitude * 4 * linewidth * freq_shift ** 2 / (
        np.pi * (((x - asymmetry) ** 2 - freq_shift ** 2) ** 2 + 4 * (linewidth * (x - asymmetry)) ** 2)
    ) + background


def _gradient(nx, ny):
    """Abundance map going from 0 to 1 along x."""
    return np.tile(np.linspace(0, 1, nx)[:, None], (1, ny))


def _blob(nx, ny, cx, cy, radius):
    """Abundance map of a soft circular blob centred at (cx, cy)."""
    yy, xx = np.mgrid[0:nx, 0:ny]
    distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return np.clip(1 - distance / radius, 0, 1)


def single_peak_image(nx=15, ny=15, n_channels=200, freq_shift=8.5, linewidth=1.0, noise=5e-4, seed=0):
    """
    A small :class:`~brillouinpy.SpectralImage` where every pixel holds a single
    (slightly noisy) DHO peak, useful for the preprocessing and fitting examples.
    """
    rng = np.random.default_rng(seed)

    spectral_axis = bp.utils.brillouin_spectral_axis(
        mirror_spacing=6e-3, scan_amplitude=480e-9, no_of_channels=n_channels
    )

    amplitude_map = 0.8 + 0.4 * _gradient(nx, ny)  # mild spatial amplitude variation
    spectral_data = np.empty((nx, ny, n_channels))
    for x in range(nx):
        for y in range(ny):
            spectral_data[x, y, :] = _dho(
                spectral_axis, amplitude_map[x, y] * 5e-3, freq_shift, linewidth, background=1e-4
            )
    spectral_data += rng.normal(scale=noise, size=spectral_data.shape)

    return bp.SpectralImage(spectral_data, spectral_axis)


def image_with_irf(nx=15, ny=15, n_channels=250, freq_shift=8.5, linewidth=1.0,
                    irf_width=2, noise=1e-5, seed=2):
    """
    A :class:`~brillouinpy.SpectralImage` holding a genuine Brillouin (DHO) peak
    plus a much stronger, narrow Instrument Response Function (IRF)/Rayleigh peak at
    the centre of the spectral axis (matching real Brillouin spectra, where the
    elastically scattered light sits at zero frequency shift), flanked on both sides
    by exact-zero detector channels - useful for the IRF-removal example.

    :func:`brillouinpy.preprocessing.misc._find_instrumental_response` (used by
    ``IRF_Remover``/``Deconvoluter_IRF``) locates the IRF by starting at the spectrum's
    global maximum and scanning outwards for the first exact-zero channel on each side,
    which is why this generator plants real zeros there rather than just a low value.

    ``noise`` is kept low by default because Richardson-Lucy deconvolution (used by
    ``Deconvoluter_IRF``) amplifies high-frequency noise; at more realistic noise levels
    the deconvolved result rings visibly across the whole spectrum, which is misleading
    as a first illustration of what a well-behaved deconvolution looks like.

    Returns
    -------
    image : SpectralImage
        The spectral image containing both peaks.
    irf_index : int
        The spectral channel index of the (synthetic) IRF's centre.
    """
    rng = np.random.default_rng(seed)

    spectral_axis = bp.utils.brillouin_spectral_axis(
        mirror_spacing=6e-3, scan_amplitude=480e-9, no_of_channels=n_channels
    )

    brillouin_peak = _dho(spectral_axis, 5e-3, freq_shift, linewidth, background=1e-4)

    irf_index = n_channels // 2
    irf_profile = np.exp(-0.5 * ((np.arange(n_channels) - irf_index) / irf_width) ** 2)
    irf_peak = 0.05 * irf_profile  # much stronger than the Brillouin peak, like a real Rayleigh line

    zero_gap = irf_width * 4       # channels between the IRF and its flanking zero band
    zero_band = 3                  # width of the flanking zero band
    left_zero = slice(irf_index - zero_gap - zero_band, irf_index - zero_gap)
    right_zero = slice(irf_index + zero_gap, irf_index + zero_gap + zero_band)

    spectral_data = np.tile(brillouin_peak + irf_peak, (nx, ny, 1))
    spectral_data += rng.normal(scale=noise, size=spectral_data.shape)
    spectral_data = np.clip(spectral_data, 0, None)
    # Punch the flanking zero bands in *after* adding noise, so they stay exact zeros.
    spectral_data[..., left_zero] = 0
    spectral_data[..., right_zero] = 0

    return bp.SpectralImage(spectral_data, spectral_axis), irf_index


def two_material_image(nx=20, ny=20, n_channels=250, noise=3e-4, seed=1,
                        freq_shift_a=6.0, freq_shift_b=11.0):
    """
    A :class:`~brillouinpy.SpectralImage` mixing two "materials" (each a single DHO
    peak at a different frequency shift) with a smooth spatial abundance gradient, useful
    for the unmixing/decomposition examples.

    ``freq_shift_a``/``freq_shift_b`` default to a well-separated pair (6.0/11.0 GHz);
    move them closer together (e.g. for the classical-analysis example) to demonstrate
    variance analysis resolving two doublets that overlap into what looks like a single
    peak in the mean spectrum.

    Returns
    -------
    image : SpectralImage
        The mixed spectral image.
    true_abundances : list[numpy.ndarray]
        The ground-truth (nx, ny) abundance maps used to build the mixture, in the same
        order as ``true_spectra``.
    true_spectra : list[numpy.ndarray]
        The ground-truth pure-material spectra used to build the mixture.
    """
    rng = np.random.default_rng(seed)

    spectral_axis = bp.utils.brillouin_spectral_axis(
        mirror_spacing=6e-3, scan_amplitude=480e-9, no_of_channels=n_channels
    )

    material_a = _dho(spectral_axis, 5e-3, freq_shift_a, 0.8)
    material_b = _dho(spectral_axis, 5e-3, freq_shift_b, 1.2)

    abundance_a = _gradient(nx, ny)
    abundance_b = 1 - abundance_a

    spectral_data = (
        abundance_a[..., None] * material_a[None, None, :]
        + abundance_b[..., None] * material_b[None, None, :]
    )
    spectral_data += rng.normal(scale=noise, size=spectral_data.shape)
    spectral_data = np.clip(spectral_data, 0, None)  # keep intensities non-negative

    image = bp.SpectralImage(spectral_data, spectral_axis)
    return image, [abundance_a, abundance_b], [material_a, material_b]
