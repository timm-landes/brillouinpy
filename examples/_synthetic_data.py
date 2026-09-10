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


def _dho(x, amplitude, freq_shift, linewidth, background=0.0, axis_shift=0.0):
    # Mirrors brillouinpy.analysis.fit.core._dho_line: ``linewidth`` is the HWHM.
    xs = x - axis_shift
    return amplitude * 4 * linewidth * freq_shift ** 2 / (
        np.pi * ((xs ** 2 - freq_shift ** 2) ** 2 + 4 * (linewidth * xs) ** 2)
    ) + background


def _add_noise(clean, rng, *, noise_model, noise, photon_scale):
    """
    Add measurement noise to a clean spectrum/stack.

    ``noise_model='gaussian'`` adds zero-mean Gaussian noise of standard deviation
    ``noise`` (a fixed noise floor, independent of signal level).

    ``noise_model='poisson'`` models photon shot noise: the clean intensity is
    scaled to photon counts by ``photon_scale`` (counts per intensity unit),
    Poisson-sampled, and scaled back - so the noise grows as sqrt(signal) and the
    per-pixel SNR is set by the photon budget, not a constant. ``noise`` is
    ignored.

    Either way the result is clipped to be non-negative.
    """
    clean = np.clip(clean, 0, None)
    if noise_model == 'gaussian':
        out = clean + rng.normal(scale=noise, size=clean.shape)
    elif noise_model == 'poisson':
        out = rng.poisson(clean * photon_scale) / photon_scale
    else:
        raise ValueError(f"noise_model must be 'gaussian' or 'poisson', got {noise_model!r}.")
    return np.clip(out, 0, None)


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


def two_mode_image(nx=30, ny=30, n_channels=250, shift_a=6.0, shift_b_range=(10.0, 13.0),
                   linewidth_a=0.8, linewidth_b=0.5, amplitude_a=5e-3, amplitude_b=4e-3,
                   noise_model='poisson', noise=6e-4, peak_photons=300, seed=8):
    """
    A :class:`~brillouinpy.SpectralImage` where **every** pixel holds *two*
    Brillouin modes: mode A at a fixed shift, and mode B whose frequency shift
    ramps linearly across x (a graded stiffness in a second phase). No pixel is
    single-mode, so it is a clean test for automatic peak-count selection
    (``brillouinpy.analysis.fit.estimate_peak_count`` /
    ``DHO(expected_peaks='auto')``) without the ambiguity a spatially-varying
    number of modes would introduce.

    Returns
    -------
    image : SpectralImage
    shift_b_map : numpy.ndarray of float
        ``(nx, ny)`` ground-truth Brillouin shift of mode B per pixel (GHz).
    info : dict
        ``{'shift_a', 'linewidth_a', 'linewidth_b'}`` - the fixed parameters.
    """
    rng = np.random.default_rng(seed)

    spectral_axis = bp.utils.brillouin_spectral_axis(
        mirror_spacing=6e-3, scan_amplitude=480e-9, no_of_channels=n_channels
    )

    shift_b_map = shift_b_range[0] + _gradient(nx, ny) * (shift_b_range[1] - shift_b_range[0])
    mode_a = _dho(spectral_axis, amplitude_a, shift_a, linewidth_a, background=1e-4)

    spectral_data = np.empty((nx, ny, n_channels))
    for x in range(nx):
        for y in range(ny):
            spectral_data[x, y, :] = mode_a + _dho(
                spectral_axis, amplitude_b, shift_b_map[x, y], linewidth_b
            )

    photon_scale = peak_photons / spectral_data.max()
    spectral_data = _add_noise(spectral_data, rng, noise_model=noise_model,
                               noise=noise, photon_scale=photon_scale)

    info = {'shift_a': shift_a, 'linewidth_a': linewidth_a, 'linewidth_b': linewidth_b}
    return bp.SpectralImage(spectral_data, spectral_axis), shift_b_map, info


def two_region_image(nx=40, ny=40, n_channels=250, freq_shift_a=5.6, freq_shift_b=6.0,
                     linewidth=0.8, noise=3e-4, noise_model='gaussian', peak_photons=200,
                     seed=3, geometry='blob', edge='sharp'):
    """
    A :class:`~brillouinpy.SpectralImage` of two materials meeting at a hard edge -
    every pixel holds exactly one pure DHO spectrum (that of its region) plus
    noise, with no spectral mixing between regions.

    Unlike :func:`two_material_image` (a smooth per-pixel mixture, made for
    unmixing), this matches "two tissues side by side" and is meant for the
    phasor / segmentation examples and the method-comparison assay
    (``benchmarks/``): move ``freq_shift_a``/``freq_shift_b`` close
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
        Standard deviation of additive Gaussian noise (``noise_model='gaussian'``).
    noise_model : {'gaussian', 'poisson'}
        ``'gaussian'`` (default): a constant noise floor of width ``noise``.
        ``'poisson'``: photon shot noise, with the brighter region's peak set to
        ``peak_photons`` counts; ``noise`` is then ignored. See :func:`_add_noise`.
    peak_photons : float
        Photon count at the region peak, for ``noise_model='poisson'``. The
        shot-noise SNR at the peak is roughly ``sqrt(peak_photons)``.
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
    photon_scale = peak_photons / max(spectrum_a.max(), spectrum_b.max())
    spectral_data = _add_noise(spectral_data, rng, noise_model=noise_model,
                               noise=noise, photon_scale=photon_scale)

    return bp.SpectralImage(spectral_data, spectral_axis), label_map, [spectrum_a, spectrum_b]


def additive_blob_image(nx=40, ny=40, n_channels=250, background_shift=5.6, component_shift=6.4,
                        background_linewidth=0.8, component_linewidth=0.8,
                        background_amplitude=5e-3, component_amplitude=2.5e-3,
                        noise=6e-4, noise_model='gaussian', peak_photons=200,
                        seed=4, geometry='blob', edge='soft'):
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
        Standard deviation of additive Gaussian noise (``noise_model='gaussian'``).
    noise_model : {'gaussian', 'poisson'}
        ``'gaussian'`` (default): a constant noise floor of width ``noise``.
        ``'poisson'``: photon shot noise, with the background peak set to
        ``peak_photons`` counts; ``noise`` is then ignored. See :func:`_add_noise`.
    peak_photons : float
        Photon count at the background peak, for ``noise_model='poisson'``.
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
    photon_scale = peak_photons / background.max()
    spectral_data = _add_noise(spectral_data, rng, noise_model=noise_model,
                               noise=noise, photon_scale=photon_scale)

    return bp.SpectralImage(spectral_data, spectral_axis), label_map, component_weight


def three_component_image(nx=40, ny=40, n_channels=250,
                          background_shift=5.6, cytoplasm_shift=6.3, nucleus_shift=7.5,
                          background_linewidth=0.8, cytoplasm_linewidth=0.6, nucleus_linewidth=0.4,
                          background_amplitude=5e-3, cytoplasm_amplitude=4e-3, nucleus_amplitude=3.5e-3,
                          noise_model='poisson', noise=6e-4, peak_photons=200,
                          seed=6, cell_radius_frac=0.38, nucleus_radius_frac=0.16):
    """
    A :class:`~brillouinpy.SpectralImage` mimicking a cell in a hydrated medium:
    a constant **background** (medium) DHO peak everywhere, a second component
    (**cytoplasm**) added on top inside a centred "cell" disc, and a third
    component (**nucleus**) added on top of *both* inside a smaller, concentric
    disc - so nucleus pixels carry all three components additively, cytoplasm
    pixels carry two, and background pixels carry one. All three peaks are
    always present in their respective regions; none of the three domains holds
    a pure single-component spectrum other than the background.

    Ground truth is nested by construction (nucleus is a subset of cell), which
    :func:`brillouinpy.analysis.phasor`-and-friends based methods do not need to
    know, but the DHO-based recommended workflow (see
    ``brillouinpy`` benchmarks' ``dho_fit_segmented_3``) exploits deliberately: a
    cheap classifier first tells background from cytoplasm from nucleus by
    total intensity (background < cytoplasm-only < cytoplasm+nucleus, since
    every extra component only adds signal), then a 1/2/3-peak DHO fit is run
    in exactly the matching region.

    Parameters
    ----------
    background_shift, cytoplasm_shift, nucleus_shift : float
        Brillouin shift (GHz) of each component.
    background_linewidth, cytoplasm_linewidth, nucleus_linewidth : float
        DHO linewidths (GHz) - deliberately distinct per component, so
        linewidth (not just shift) recovery is a meaningful question.
    background_amplitude, cytoplasm_amplitude, nucleus_amplitude : float
        Peak amplitudes.
    noise_model, noise, peak_photons :
        As in :func:`additive_blob_image` - ``noise_model='poisson'`` (default)
        scales the background peak to ``peak_photons`` counts.
    cell_radius_frac, nucleus_radius_frac : float
        Cell and nucleus disc radii, as a fraction of ``min(nx, ny)``. The
        nucleus disc is concentric with and inside the cell disc.

    Returns
    -------
    image : SpectralImage
    label_map : numpy.ndarray of int
        ``(nx, ny)`` ground truth: 0 = background, 1 = cytoplasm, 2 = nucleus.
    true_params : dict[str, tuple[float, float]]
        ``{'background': (shift, linewidth), 'cytoplasm': (...), 'nucleus': (...)}``.
    """
    rng = np.random.default_rng(seed)

    spectral_axis = bp.utils.brillouin_spectral_axis(
        mirror_spacing=6e-3, scan_amplitude=480e-9, no_of_channels=n_channels
    )

    background = _dho(spectral_axis, background_amplitude, background_shift, background_linewidth, background=1e-4)
    cytoplasm = _dho(spectral_axis, cytoplasm_amplitude, cytoplasm_shift, cytoplasm_linewidth)
    nucleus = _dho(spectral_axis, nucleus_amplitude, nucleus_shift, nucleus_linewidth)

    cx, cy = ny / 2, nx / 2
    cell_radius = cell_radius_frac * min(nx, ny)
    nucleus_radius = nucleus_radius_frac * min(nx, ny)
    yy, xx = np.mgrid[0:nx, 0:ny]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    label_map = np.zeros((nx, ny), dtype=int)
    label_map[dist <= cell_radius] = 1
    label_map[dist <= nucleus_radius] = 2

    spectral_data = np.tile(background, (nx, ny, 1))
    spectral_data[label_map >= 1] = spectral_data[label_map >= 1] + cytoplasm
    spectral_data[label_map >= 2] = spectral_data[label_map >= 2] + nucleus

    photon_scale = peak_photons / background.max()
    spectral_data = _add_noise(spectral_data, rng, noise_model=noise_model,
                               noise=noise, photon_scale=photon_scale)

    true_params = {
        'background': (background_shift, background_linewidth),
        'cytoplasm': (cytoplasm_shift, cytoplasm_linewidth),
        'nucleus': (nucleus_shift, nucleus_linewidth),
    }
    return bp.SpectralImage(spectral_data, spectral_axis), label_map, true_params


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


def irf_broadened_image(nx=30, ny=30, n_channels=300, component_shift=6.5,
                        intrinsic_linewidth_background=0.9, intrinsic_linewidth_blob=0.45,
                        irf_fwhm=0.5, irf_shape='lorentzian', elastic_amplitude=0.0,
                        noise_model='poisson', noise=6e-4, peak_photons=250,
                        seed=7, geometry='blob', edge='sharp'):
    """
    A :class:`~brillouinpy.SpectralImage` whose spectra are an intrinsic DHO doublet
    *convolved with a known instrument response function* (IRF) - the case
    :class:`brillouinpy.analysis.fit.core.DHO`'s ``irf=`` argument is built for.

    Every pixel holds one Brillouin doublet at ``component_shift`` GHz. The intrinsic
    (pre-IRF) linewidth is ``intrinsic_linewidth_background`` outside a centred blob and
    ``intrinsic_linewidth_blob`` inside it, so linewidth recovery can be scored against a
    known two-valued map. The clean spectrum is then convolved with an IRF of full width
    ``irf_fwhm`` (:func:`brillouinpy.analysis.fit.core.irf_kernel`), which broadens every
    line - a plain DHO fit sees ``intrinsic + instrumental`` width, an ``irf=``-aware fit
    recovers the intrinsic width.

    Parameters
    ----------
    component_shift : float
        Brillouin shift (GHz) of the doublet (placed symmetrically at +/- this value).
    intrinsic_linewidth_background, intrinsic_linewidth_blob : float
        Intrinsic DHO linewidths (GHz), before IRF convolution, outside / inside the blob.
    irf_fwhm : float
        Full width at half maximum (GHz) of the IRF.
    irf_shape : {'lorentzian', 'gaussian', 'voigt'}
        IRF lineshape. ``'voigt'`` splits ``irf_fwhm`` evenly between its Lorentzian and
        Gaussian parts.
    elastic_amplitude : float
        If > 0, add a narrow elastic / Rayleigh peak at zero shift with this amplitude
        (relative to the Brillouin peak height), so a measured IRF can be extracted from
        the data itself. Its channels are *not* zeroed - crop them with
        :class:`~brillouinpy.preprocessing.misc.IRF_Remover` before fitting.
    noise_model, noise, peak_photons :
        As in :func:`additive_blob_image`.
    geometry : {'blob', 'half'}
    edge : {'sharp', 'soft'}

    Returns
    -------
    image : SpectralImage
    intrinsic_linewidth_map : numpy.ndarray of float
        ``(nx, ny)`` ground-truth intrinsic linewidth per pixel (GHz).
    info : dict
        ``{'irf_fwhm', 'irf_shape', 'component_shift', 'irf_kernel'}`` - ``irf_kernel`` is
        the exact normalised kernel used, ready to pass as ``DHO(..., irf=info['irf_kernel'])``.
    """
    rng = np.random.default_rng(seed)

    spectral_axis = bp.utils.brillouin_spectral_axis(
        mirror_spacing=6e-3, scan_amplitude=480e-9, no_of_channels=n_channels
    )

    if irf_shape == 'voigt':
        irf_spec = ('voigt', irf_fwhm / 2, irf_fwhm / 2)
    elif irf_shape in ('lorentzian', 'gaussian'):
        irf_spec = (irf_shape, irf_fwhm)
    else:
        raise ValueError(f"irf_shape must be 'lorentzian', 'gaussian' or 'voigt', got {irf_shape!r}.")
    kernel = bp.analysis.fit.irf_kernel(irf_spec, spectral_axis)

    def _broadened(linewidth):
        intrinsic = _dho(spectral_axis, 5e-3, component_shift, linewidth)
        pad = kernel.size
        return np.convolve(np.pad(intrinsic, pad, mode='edge'), kernel, mode='same')[pad:-pad]

    if geometry == 'blob':
        blob = _blob(nx, ny, cx=ny / 2, cy=nx / 2, radius=min(nx, ny) / 3)
    elif geometry == 'half':
        blob = np.tile((np.arange(ny) >= ny / 2).astype(float), (nx, 1))
    else:
        raise ValueError(f"geometry must be 'blob' or 'half', got {geometry!r}.")
    inside = blob >= 0.5 if edge == 'sharp' else np.clip(blob, 0, 1)
    if edge not in ('sharp', 'soft'):
        raise ValueError(f"edge must be 'sharp' or 'soft', got {edge!r}.")

    bg_spectrum = _broadened(intrinsic_linewidth_background)
    blob_spectrum = _broadened(intrinsic_linewidth_blob)
    weight = inside.astype(float)
    spectral_data = ((1 - weight)[..., None] * bg_spectrum[None, None, :]
                     + weight[..., None] * blob_spectrum[None, None, :]) + 1e-4

    intrinsic_linewidth_map = (intrinsic_linewidth_background
                               + weight * (intrinsic_linewidth_blob - intrinsic_linewidth_background))

    elastic_zero = None
    if elastic_amplitude > 0:
        # An elastic / Rayleigh line at zero shift that is *the IRF itself* (same
        # kernel that broadened the Brillouin lines), scaled to elastic_amplitude
        # x the Brillouin peak height - so a fit that recovers the kernel from
        # this peak (DHO(irf='auto')) sees the same broadening the lines carry.
        # A finite window (a real detector does not capture infinite Lorentzian
        # wings) flanked by a thin exact-zero band lets
        # brillouinpy.preprocessing.misc._find_instrumental_response locate it
        # (the zeros survive Poisson sampling; with noise_model='gaussian' they
        # don't). Kept well clear of the +/- component_shift Brillouin peaks.
        centre = int(np.argmin(np.abs(spectral_axis)))
        half = kernel.size // 2
        core = kernel / kernel.max()
        lo, hi = centre - half, centre + half + 1
        peak_height = float(spectral_data.max())
        spectral_data[..., lo:hi] += elastic_amplitude * peak_height * core[None, None, :]
        elastic_zero = (slice(max(0, lo - 3), lo), slice(hi, min(n_channels, hi + 3)))
        for band in elastic_zero:
            spectral_data[..., band] = 0.0

    photon_scale = peak_photons / spectral_data.max()
    spectral_data = _add_noise(spectral_data, rng, noise_model=noise_model,
                               noise=noise, photon_scale=photon_scale)
    if elastic_zero is not None:
        for band in elastic_zero:
            spectral_data[..., band] = 0.0

    info = {'irf_fwhm': irf_fwhm, 'irf_shape': irf_shape,
            'component_shift': component_shift, 'irf_kernel': kernel}
    return bp.SpectralImage(spectral_data, spectral_axis), intrinsic_linewidth_map, info


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
