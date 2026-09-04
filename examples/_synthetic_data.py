# -*- coding: utf-8 -*-
"""
Synthetic Brillouin data generator shared by the examples in this folder.

The real workflow starts from :func:`brillouinpy.io.tfp.prepare_brillouin_data`
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


def two_region_image(nx=40, ny=40, n_channels=250, freq_shift_a=5.6, freq_shift_b=6.0,
                     linewidth=0.8, noise=3e-4, seed=3, geometry='blob', edge='sharp'):
    """
    A :class:`~brillouinpy.SpectralImage` of two materials meeting at a hard edge -
    every pixel holds exactly one pure DHO spectrum (that of its region) plus
    noise, with no spectral mixing between regions.

    Unlike :func:`two_material_image` (a smooth per-pixel mixture, made for
    unmixing), this matches "two tissues side by side" and is meant for the
    phasor / segmentation examples and the method-limitation assay
    (``12_method_limitations.py``): move ``freq_shift_a``/``freq_shift_b`` close
    together and/or raise ``noise`` to find where each analysis method stops
    resolving the two regions.

    Parameters
    ----------
    freq_shift_a, freq_shift_b : float
        Brillouin shift (GHz) of region A / region B. Default 5.6 / 6.0 GHz -
        deliberately close, like water vs. cytoplasm.
    linewidth : float
        Shared DHO linewidth (GHz).
    noise : float
        Standard deviation of additive Gaussian noise.
    geometry : {'blob', 'half'}
        ``'blob'`` puts region B in a centred disc on a region-A background;
        ``'half'`` splits the image left/right.
    edge : {'sharp', 'soft'}
        ``'sharp'`` (default) is a hard label boundary; ``'soft'`` blends one
        pixel across the border (a mild antialiasing, still not a physical
        mixture).

    Returns
    -------
    image : SpectralImage
        The two-region image.
    label_map : numpy.ndarray of int
        ``(nx, ny)`` ground-truth region labels (0 = A, 1 = B).
    true_spectra : list[numpy.ndarray]
        The two pure-region spectra, in label order.
    """
    rng = np.random.default_rng(seed)

    spectral_axis = bp.utils.brillouin_spectral_axis(
        mirror_spacing=6e-3, scan_amplitude=480e-9, no_of_channels=n_channels
    )

    spectrum_a = _dho(spectral_axis, 5e-3, freq_shift_a, linewidth, background=1e-4)
    spectrum_b = _dho(spectral_axis, 5e-3, freq_shift_b, linewidth, background=1e-4)

    if geometry == 'blob':
        fraction_b = _blob(nx, ny, cx=ny / 2, cy=nx / 2, radius=min(nx, ny) / 3)
    elif geometry == 'half':
        fraction_b = np.tile((np.arange(ny) >= ny / 2).astype(float), (nx, 1))
    else:
        raise ValueError(f"geometry must be 'blob' or 'half', got {geometry!r}.")

    label_map = (fraction_b >= 0.5).astype(int)

    if edge == 'sharp':
        weight_b = label_map.astype(float)
    elif edge == 'soft':
        weight_b = np.clip(fraction_b, 0, 1)
    else:
        raise ValueError(f"edge must be 'sharp' or 'soft', got {edge!r}.")

    spectral_data = (
        (1 - weight_b)[..., None] * spectrum_a[None, None, :]
        + weight_b[..., None] * spectrum_b[None, None, :]
    )
    spectral_data += rng.normal(scale=noise, size=spectral_data.shape)
    spectral_data = np.clip(spectral_data, 0, None)

    return bp.SpectralImage(spectral_data, spectral_axis), label_map, [spectrum_a, spectrum_b]


def additive_blob_image(nx=40, ny=40, n_channels=250, background_shift=5.6, component_shift=6.4,
                        background_linewidth=0.8, component_linewidth=0.8,
                        background_amplitude=5e-3, component_amplitude=2.5e-3,
                        noise=6e-4, seed=4, geometry='blob', edge='soft'):
    """
    A :class:`~brillouinpy.SpectralImage` where one **constant** background
    spectrum is present in every pixel and a second, shifted DHO component is
    *added on top* inside a blob region - so no pixel is ever a pure component
    spectrum.

    This is the additive-mixture counterpart to :func:`two_region_image` (where
    each pixel holds exactly one pure spectrum). It mimics a hydrated sample: the
    water / hydration signal contributes everywhere, and a stiffer inclusion adds
    its own peak on top. VCA's "at least one pure pixel per endmember" assumption
    fails here, and a single-peak DHO fit is biased by the unmodelled background,
    whereas methods keyed to the *change* in spectral shape (phasor, NMF, PCA,
    k-means) can still pick out and separate the blob.

    Parameters
    ----------
    background_shift, component_shift : float
        Brillouin shift (GHz) of the everywhere-background peak and of the peak
        added inside the blob.
    background_linewidth, component_linewidth : float
        DHO linewidths (GHz).
    background_amplitude, component_amplitude : float
        Peak amplitudes. The component is weaker by default (it only ever appears
        added to the background).
    noise : float
        Standard deviation of additive Gaussian noise.
    geometry : {'blob', 'half'}
        ``'blob'`` adds the component in a centred disc; ``'half'`` in the right
        half.
    edge : {'soft', 'sharp'}
        ``'soft'`` (default) ramps the component weight smoothly from 0 to 1
        across the blob (partial mixtures at the rim, like a real interface);
        ``'sharp'`` switches it hard at the boundary.

    Returns
    -------
    image : SpectralImage
        The additive-mixture image.
    label_map : numpy.ndarray of int
        ``(nx, ny)`` ground truth: 0 = background only, 1 = background + component.
    component_weight : numpy.ndarray of float
        The actual per-pixel component weight in ``[0, 1]``.
    """
    rng = np.random.default_rng(seed)

    spectral_axis = bp.utils.brillouin_spectral_axis(
        mirror_spacing=6e-3, scan_amplitude=480e-9, no_of_channels=n_channels
    )

    background = _dho(spectral_axis, background_amplitude, background_shift,
                      background_linewidth, background=1e-4)
    component = _dho(spectral_axis, component_amplitude, component_shift, component_linewidth)

    if geometry == 'blob':
        blob = _blob(nx, ny, cx=ny / 2, cy=nx / 2, radius=min(nx, ny) / 3)
    elif geometry == 'half':
        blob = np.tile((np.arange(ny) >= ny / 2).astype(float), (nx, 1))
    else:
        raise ValueError(f"geometry must be 'blob' or 'half', got {geometry!r}.")

    label_map = (blob >= 0.5).astype(int)
    component_weight = np.clip(blob, 0, 1) if edge == 'soft' else label_map.astype(float)
    if edge not in ('soft', 'sharp'):
        raise ValueError(f"edge must be 'soft' or 'sharp', got {edge!r}.")

    spectral_data = (
        background[None, None, :]
        + component_weight[..., None] * component[None, None, :]
    )
    spectral_data += rng.normal(scale=noise, size=spectral_data.shape)
    spectral_data = np.clip(spectral_data, 0, None)

    return bp.SpectralImage(spectral_data, spectral_axis), label_map, component_weight


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
