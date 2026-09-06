# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 13:14:03 2025

@author: Timm
"""
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, fftconvolve, peak_prominences, peak_widths
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os

from multiprocessing import shared_memory

from .FitStep import FitStep


# Worker-Globals
_DHO3_DATA = None
_DHO3_SPECTRAL_AXIS = None
_DHO3_EXPECTED_PEAKS = None
_DHO3_P0 = None
_DHO3_BOUNDS = None
_DHO3_FIT_FUNC = None
_DHO3_SHM = None


def _init_dho3_worker(shm_name, shape, dtype, spectral_axis, expected_peaks, p0, bounds, fit_func):
    global _DHO3_DATA, _DHO3_SPECTRAL_AXIS, _DHO3_EXPECTED_PEAKS, _DHO3_P0, _DHO3_BOUNDS, _DHO3_FIT_FUNC, _DHO3_SHM

    _DHO3_SHM = shared_memory.SharedMemory(name=shm_name)
    _DHO3_DATA = np.ndarray(shape, dtype=dtype, buffer=_DHO3_SHM.buf)
    _DHO3_SPECTRAL_AXIS = spectral_axis
    _DHO3_EXPECTED_PEAKS = expected_peaks
    _DHO3_P0 = p0
    _DHO3_BOUNDS = bounds
    _DHO3_FIT_FUNC = fit_func


def _fit_dho3_worker(idx):
    intensity_data_slice = _DHO3_DATA[idx + (slice(None),)]
    return _fitDHO2(
        idx,
        _DHO3_SPECTRAL_AXIS,
        intensity_data_slice,
        _DHO3_EXPECTED_PEAKS,
        _DHO3_P0,
        _DHO3_BOUNDS,
        _DHO3_FIT_FUNC,
    )


def _fit_concurrent_DHO3(intensity_data, spectral_axis, expected_peaks, p0=None, bounds=None):
    variables = int(expected_peaks * 3 + 2)

    if p0 is not None and len(p0) != variables:
        p0 = None
        print("Length of p0 does not match the model. Fallback to p0 = None")

    fit_results = np.full(intensity_data.shape[:-1] + (variables,), np.nan)
    cov_results = np.full(intensity_data.shape[:-1] + (variables, variables), np.nan)

    fit_func = {1: _DHO_1, 2: _DHO_2, 3: _DHO_3}[expected_peaks]
    spectral_axis = spectral_axis[0]

    # Masken im Spektrum behalten, aber als NaN speichern, damit curve_fit sie ignoriert
    data = np.asarray(np.ma.filled(intensity_data, np.nan), dtype=np.float64)
    spatial_shape = data.shape[:-1]
    indices = list(np.ndindex(spatial_shape))

    max_workers = max(1, int(os.cpu_count() * 0.25))
    chunksize = max(1, len(indices) // (max_workers * 4))

    shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
    try:
        shared_data = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
        shared_data[:] = data

        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_dho3_worker,
            initargs=(shm.name, data.shape, data.dtype, spectral_axis, expected_peaks, p0, bounds, fit_func),
        ) as executor:
            for idx, popt, pcov in tqdm(
                executor.map(_fit_dho3_worker, indices, chunksize=chunksize),
                total=len(indices),
                desc="Fitting Spectral data",
            ):
                fit_results[idx] = popt
                cov_results[idx] = pcov

    finally:
        shm.close()
        shm.unlink()

    return fit_results, cov_results

def estimate_p0(spectral_object, expected_peaks):
    """
    Estimates an initial parameter guess (``p0``) for :class:`DHO`/:class:`Lorentzian` fits.

    Runs peak detection once on the mean spectrum of ``spectral_object`` and derives, for each
    expected peak, an amplitude/frequency-shift/linewidth guess from the detected peak's
    height/position/FWHM, plus a shared background and asymmetry guess. Returned in the
    parameter order the fit models expect: ``[I0, freqShift, LineWidth, ...,  Background,
    Asymmetry]``.

    If fewer than ``expected_peaks`` distinct peaks are found (e.g. an overlapping doublet
    showing up as a single broad peak), the remaining slots fall back to evenly spaced
    positions with a width derived from the spectral range, so the estimate degrades
    gracefully instead of raising.

    Parameters
    ----------
    spectral_object : core.SpectralObject
        The data to estimate a p0 for. Its mean spectrum (:attr:`core.SpectralContainer.mean`)
        is used. ``expected_peaks`` is typically known ahead of time, e.g. from the number of
        components found via NMF/VCA.
    expected_peaks : int
        The number of peaks to estimate parameters for (1, 2, or 3).

    Returns
    -------
    list[float]
        The estimated p0, ready to pass into :class:`DHO`/:class:`Lorentzian`.
    """
    if expected_peaks not in (1, 2, 3):
        raise ValueError(f"expected_peaks must be 1, 2 or 3, got {expected_peaks}.")

    mean_spectrum = spectral_object.mean
    axis = mean_spectrum.spectral_axis
    intensity = np.ma.filled(mean_spectrum.spectral_data, np.nan).astype(float)

    if np.all(np.isnan(intensity)):
        raise ValueError("Cannot estimate p0: mean spectrum is entirely masked/NaN.")

    background = float(np.nanpercentile(intensity, 5))
    intensity = np.where(np.isnan(intensity), background, intensity)
    spacing = float(np.mean(np.diff(axis)))

    candidate_peaks, _ = find_peaks(intensity)
    if len(candidate_peaks) > 0:
        prominences = peak_prominences(intensity, candidate_peaks)[0]
        candidate_peaks = candidate_peaks[np.argsort(prominences)[::-1]]

    detected_peaks = list(candidate_peaks[:expected_peaks])
    detected_widths = dict(zip(
        detected_peaks,
        peak_widths(intensity, detected_peaks, rel_height=0.5)[0] if detected_peaks else [],
    ))

    # Pad with evenly spaced fallback positions if too few distinct peaks were detected.
    padded_peaks = []
    if len(detected_peaks) < expected_peaks:
        min_spacing = max(len(intensity) // (2 * expected_peaks), 1)
        remaining = sorted(
            (i for i in range(len(intensity)) if i not in detected_peaks),
            key=lambda i: intensity[i], reverse=True,
        )
        for i in remaining:
            if len(detected_peaks) + len(padded_peaks) >= expected_peaks:
                break
            if all(abs(i - p) > min_spacing for p in detected_peaks + padded_peaks):
                padded_peaks.append(i)
        while len(detected_peaks) + len(padded_peaks) < expected_peaks:
            slot = len(detected_peaks) + len(padded_peaks)
            padded_peaks.append(int(len(intensity) * (slot + 0.5) / expected_peaks))

    fallback_width_samples = len(intensity) / (4 * expected_peaks)

    p0 = []
    for peak_idx in sorted(detected_peaks + padded_peaks):
        width_samples = detected_widths.get(peak_idx, fallback_width_samples)
        I0 = float(max(intensity[peak_idx] - background, spacing))
        freqShift = float(axis[peak_idx])
        LineWidth = float(max(width_samples * spacing, spacing))
        p0.extend([I0, freqShift, LineWidth])

    p0.extend([background, 0.0])  # Background, Asymmetry

    return p0


_FWHM_TO_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


def irf_kernel(irf, spectral_axis):
    """
    Build a normalised, centred convolution kernel from an instrument response function (IRF).

    Pass the result as the ``irf`` argument of :class:`DHO` to fit the *convolved* model
    ``(intrinsic lineshape) * IRF + background`` to the raw spectrum, so the fitted linewidth
    is the intrinsic Brillouin linewidth with the instrumental broadening accounted for in the
    forward model - rather than deconvolving the data first (which amplifies noise). See the
    ``benchmarks/irf_convolution.py`` comparison.

    Parameters
    ----------
    irf : array-like or tuple
        Either

        - a 1-D array: a measured IRF sample (e.g. the background-subtracted elastic /
          Rayleigh / reference-beam peak), on the *same* channel spacing as ``spectral_axis``.
          It is baseline-subtracted, clipped to non-negative, trimmed to a symmetric window
          about its maximum and normalised.
        - ``('lorentzian', fwhm)`` / ``('gaussian', fwhm)`` / ``('voigt', fwhm_lorentz,
          fwhm_gauss)``: a parametric IRF with the given full width(s) at half maximum, in the
          units of ``spectral_axis`` (GHz).

    spectral_axis : array-like
        The (uniformly spaced) spectral axis the model will be evaluated on. A non-uniform
        axis raises ``ValueError`` - resample first.

    Returns
    -------
    numpy.ndarray
        Odd-length 1-D kernel, summing to 1, centred on its middle sample.
    """
    axis = np.asarray(spectral_axis, dtype=float)
    diffs = np.diff(axis)
    dx = float(np.mean(diffs))
    if dx <= 0 or not np.allclose(diffs, dx, rtol=1e-3, atol=0.0):
        raise ValueError(
            "irf_kernel needs a uniformly spaced spectral_axis; got spacing varying by "
            f"{float(np.ptp(diffs)):.3g}. Resample the spectrum onto a regular grid first.")

    if isinstance(irf, (tuple, list)) and len(irf) >= 2 and isinstance(irf[0], str):
        shape = irf[0].lower()
        if shape in ("lorentzian", "gaussian"):
            fwhm = float(irf[1])
            if fwhm <= 0:
                raise ValueError(f"irf fwhm must be positive, got {fwhm}.")
            # Lorentzian tails are heavy - truncating at 5x FWHM drops enough area to
            # bias the renormalised kernel narrow; go wider. A Gaussian is essentially
            # zero by 5 sigma.
            support = 12.0 if shape == "lorentzian" else 5.0
            half_n = max(3, int(np.ceil(support * fwhm / dx)))
            grid = np.arange(-half_n, half_n + 1) * dx
            if shape == "lorentzian":
                kernel = 1.0 / (1.0 + (grid / (fwhm / 2.0)) ** 2)
            else:
                kernel = np.exp(-0.5 * (grid / (fwhm / _FWHM_TO_SIGMA)) ** 2)
        elif shape == "voigt":
            from scipy.special import voigt_profile
            fwhm_l, fwhm_g = float(irf[1]), float(irf[2])
            if fwhm_l <= 0 or fwhm_g <= 0:
                raise ValueError(f"voigt irf fwhms must be positive, got ({fwhm_l}, {fwhm_g}).")
            half_n = max(3, int(np.ceil(5.0 * (fwhm_l + fwhm_g) / dx)))
            grid = np.arange(-half_n, half_n + 1) * dx
            kernel = voigt_profile(grid, fwhm_g / _FWHM_TO_SIGMA, fwhm_l / 2.0)
        else:
            raise ValueError(
                f"parametric irf shape must be 'lorentzian', 'gaussian' or 'voigt', got {irf[0]!r}.")
    else:
        kernel = np.asarray(irf, dtype=float).ravel()
        if kernel.size < 3:
            raise ValueError(f"a measured irf needs at least 3 samples, got {kernel.size}.")
        kernel = np.nan_to_num(kernel, nan=0.0)
        kernel = np.clip(kernel - kernel.min(), 0.0, None)
        peak = int(np.argmax(kernel))
        reach = min(peak, kernel.size - 1 - peak)
        if reach < 1:
            raise ValueError("measured irf peak is at the very edge of the sample; give a wider window.")
        kernel = kernel[peak - reach:peak + reach + 1]

    total = kernel.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("irf kernel has non-positive total; check the input.")
    return kernel / total


class _ConvolvedModel:
    """Picklable wrapper turning a bare lineshape ``f(x, *peak_params, Background, Asymmetry)``
    into ``conv(f(., *peak_params, 0, Asymmetry), kernel) + Background``, evaluated on a stored
    uniform grid and interpolated to whatever sample points ``curve_fit`` passes (which may be
    a subset of the grid, e.g. with the elastic-peak channels dropped)."""

    def __init__(self, base_func, kernel, full_axis):
        self.base_func = base_func
        self.kernel = np.asarray(kernel, dtype=float)
        self.full_axis = np.asarray(full_axis, dtype=float)

    def __call__(self, x, *params):
        x = np.asarray(x, dtype=float)
        background = params[-2]
        peaks = self.base_func(self.full_axis, *params[:-2], 0.0, params[-1])
        n = self.kernel.size
        padded = np.pad(peaks, n, mode="edge")
        convolved = fftconvolve(padded, self.kernel, mode="same")[n:-n]
        model = convolved + background
        if x.shape == self.full_axis.shape and np.array_equal(x, self.full_axis):
            return model
        return np.interp(x, self.full_axis, model)


class DHO(FitStep):
    """
    Fit of one or multiple density distributions of Damped Harmonic Oscillators (DHO)
    I0 * 4 * LineWidth * freqShift**2 /(np.pi*(((x-Asymmetry)**2 - freqShift**2)**2 + 4*(LineWidth*(x-Asymmetry))**2)) + Background

    Parameters
    ----------
    expected_peaks : int
        The number of peaks to fit.
    p0 : list[floats]
        The initial guess
    bounds : list[list[floats], list[floats]] or None
        Boundaries for the plot
    irf : array-like, tuple or None, optional
        Instrument response function. If given, the model fitted to each spectrum is the
        intrinsic DHO lineshape *convolved with the IRF* plus a background, so the fitted
        ``LineWidth`` is the intrinsic Brillouin linewidth with the instrumental broadening
        modelled explicitly - instead of deconvolving the data first (which amplifies noise;
        contrast :class:`~brillouinpy.preprocessing.misc.Deconvoluter_IRF`). Accepts anything
        :func:`irf_kernel` accepts: a measured 1-D IRF sample (e.g. the background-subtracted
        elastic peak) on the spectrum's channel spacing, or a parametric
        ``('lorentzian'|'gaussian', fwhm)`` / ``('voigt', fwhm_l, fwhm_g)`` spec (GHz). The
        elastic-peak channels should still be blanked to ``NaN`` (e.g. via
        :class:`~brillouinpy.preprocessing.misc.IRF_Remover`) so they are excluded from the
        residual; the convolved model itself is evaluated on the full axis. ``None`` (default)
        keeps the plain, unconvolved fit.
    **kwargs :

    """
    def __init__(self, *, expected_peaks, p0, bounds, irf=None):
        super().__init__(_fit_concurrent_DHO2, expected_peaks=expected_peaks, p0=p0, bounds=bounds, irf=irf)


class Lorentzian(FitStep):
    """
    Fit of one or multiple Lorentzian lines
    I0/((((x-Asymmetry)**2 - freqShift**2)**2 + (LineWidth*(x-Asymmetry))**2)) + Background


    Parameters
    ----------
    expected_peaks : int
        The number of peaks to fit.
    p0 : list[floats]
        The initial guess
    bounds : list[list[floats], list[floats]] or None
        Boundaries for the plot
    **kwargs :

    """
    def __init__(self, *, expected_peaks, p0, bounds):
        super().__init__(_fit_concurrent_lorentzian, expected_peaks=expected_peaks, p0=p0, bounds=bounds)


def _fitDHO(x, y, z, t, spectral_axis, intensity_data_slice, expected_peaks, p0, bounds, fit_funcs):
    # Convert masked values to nan
    if np.ma.is_masked(intensity_data_slice):
        # Check if all values are masked
        if np.ma.getmask(intensity_data_slice).all():
            return x, y, z, t, np.full(expected_peaks * 3 + 2, np.nan), np.full((expected_peaks * 3 + 2, expected_peaks * 3 + 2), np.nan)
        intensity_axis = intensity_data_slice.filled(np.nan)
    else:
        intensity_axis = intensity_data_slice.copy()
    intensity_axis[intensity_axis == 0] = np.nan

    # If all values are NaN, return NaN results
    if np.all(np.isnan(intensity_axis)):
        return x, y, z, t, np.full(expected_peaks * 3 + 2, np.nan), np.full((expected_peaks * 3 + 2, expected_peaks * 3 + 2), np.nan)

    if p0 is None:
        p0 = [1] * (expected_peaks * 3 + 2)  # Default initial guesses
    if bounds is None:
        bounds = (0, np.inf)  # No negative values allowed

    try:
        # Use nan_policy='omit' to handle NaN values
        popt, pcov = curve_fit(fit_funcs[expected_peaks], spectral_axis, intensity_axis, p0=p0, bounds=bounds, nan_policy='omit')
        return x, y, z, t, popt, pcov
    except RuntimeError:
        print(f"Fit not successful for pixel ({x},{y},{z},{t})")
        return x, y, z, t, np.full(expected_peaks * 3 + 2, np.nan), np.full((expected_peaks * 3 + 2, expected_peaks * 3 + 2), np.nan)

def _fitDHO2(idx, spectral_axis, intensity_data_slice, expected_peaks, p0, bounds, fit_func):
    # Convert masked values to nan
    if np.ma.is_masked(intensity_data_slice):
        if np.ma.getmask(intensity_data_slice).all():
            return idx, np.full(expected_peaks * 3 + 2, np.nan), np.full((expected_peaks * 3 + 2, expected_peaks * 3 + 2), np.nan)
        intensity_axis = intensity_data_slice.filled(np.nan)
    else:
        intensity_axis = intensity_data_slice.copy()
    intensity_axis[intensity_axis == 0] = np.nan

    if np.all(np.isnan(intensity_axis)):
        return idx, np.full(expected_peaks * 3 + 2, np.nan), np.full((expected_peaks * 3 + 2, expected_peaks * 3 + 2), np.nan)

    if p0 is None:
        p0 = [1] * (expected_peaks * 3 + 2)
    if bounds is None:
        bounds = (0, np.inf)

    try:
        popt, pcov = curve_fit(fit_func, spectral_axis, intensity_axis, p0=p0, bounds=bounds, nan_policy='omit')
        return idx, popt, pcov
    except RuntimeError:
        print(f"Fit not successful for pixel {idx}")
        return idx, np.full(expected_peaks * 3 + 2, np.nan), np.full((expected_peaks * 3 + 2, expected_peaks * 3 + 2), np.nan)

def _fit_concurrent_DHO2(intensity_data, spectral_axis, expected_peaks, p0=None, bounds=None, irf=None):
    variables = int(expected_peaks * 3 + 2)
    spatial_shape = intensity_data.shape[:-1]
    fit_results = np.zeros(spatial_shape + (variables,))
    cov_results = np.empty(spatial_shape + (variables, variables))

    fit_funcs = {1: _DHO_1, 2: _DHO_2, 3: _DHO_3}
    fit_func = fit_funcs[expected_peaks]
    spectral_axis = spectral_axis[0]
    if irf is not None:
        fit_func = _ConvolvedModel(fit_func, irf_kernel(irf, spectral_axis), spectral_axis)
    print('Starting of Multiprocessing can take up to 10 seconds.')
    tasks = []
    with ProcessPoolExecutor(max_workers=max(1,int(os.cpu_count()*0.25))) as executor:
        for idx in np.ndindex(spatial_shape):
            intensity_data_slice = intensity_data[idx + (slice(None),)]
            task = executor.submit(_fitDHO2, idx, spectral_axis, intensity_data_slice, expected_peaks, p0, bounds, fit_func)
            tasks.append(task)

        for future in tqdm(as_completed(tasks), total=len(tasks), desc='Fitting Spectral data'):
            idx, popt, pcov = future.result()
            fit_results[idx] = popt
            cov_results[idx] = pcov

    return fit_results, cov_results

def _fitLorentzian(x, y, X, intensity_data_slice, expected_peaks, p0, bounds, fit_funcs):
    # Convert masked values to nan
    if np.ma.is_masked(intensity_data_slice):
        Y = intensity_data_slice.filled(np.nan)
    else:
        Y = intensity_data_slice.copy()
    Y[Y == 0] = np.nan

    if p0 is None:
        p0 = [1] * (expected_peaks * 3 + 2)  # Default initial guesses
    if bounds is None:
        bounds = (0, np.inf)  # No negative values allowed

    try:
        # Use nan_policy='omit' to handle NaN values
        popt, pcov = curve_fit(fit_funcs[expected_peaks], X, Y, p0=p0, bounds=bounds, nan_policy='omit')
        return x, y, popt, pcov
    except RuntimeError:
        print(f"Fit not successful for pixel ({x},{y})")
        return x, y, np.full(expected_peaks * 3 + 2, np.nan), np.full((expected_peaks * 3 + 2, expected_peaks * 3 + 2), np.nan)


def _fit_concurrent_DHO(intensity_data, spectral_axis, expected_peaks, p0=None, bounds=None):
    variables = int(expected_peaks * 3 + 2)
    if p0 is not None and len(p0) != variables:
        p0 = None
        print('Length of p0 does not match the model. Fallback to p0 = None')
    fit_results = np.zeros((intensity_data.shape[0], intensity_data.shape[1], intensity_data.shape[2], intensity_data.shape[3], variables))
    cov_results = np.empty((intensity_data.shape[0], intensity_data.shape[1], intensity_data.shape[2], intensity_data.shape[3], variables, variables))

    fit_funcs = {1: _DHO_1, 2: _DHO_2, 3: _DHO_3}
    spectral_axis = spectral_axis[0]
    print('Starting of Multiprocessing can take up to 10 seconds.')
    tasks = []
    with ProcessPoolExecutor(max_workers=max(1,int(os.cpu_count()*0.25))) as executor:
        for x in range(intensity_data.shape[0]):
            for y in range(intensity_data.shape[1]):
                for z in range(intensity_data.shape[2]):
                    for t in range(intensity_data.shape[3]):
                            intensity_data_slice = intensity_data[x, y, z, t, :]
                            # Skip if all values are masked
                            # if np.ma.is_masked(intensity_data_slice) and np.ma.getmask(intensity_data_slice).all():
                            #     fit_results[x, y, z, t, :] = np.nan
                            #     cov_results[x, y, z, t, :] = np.nan
                            #     continue
                            task = executor.submit(_fitDHO, x, y, z, t,  spectral_axis, intensity_data_slice, expected_peaks, p0, bounds, fit_funcs)
                            tasks.append(task)

        # Use tqdm to display a progress bar
        for future in tqdm(as_completed(tasks), total=len(tasks), desc='Fitting Spectral data'):
            x, y, z, t, popt, pcov = future.result()
            fit_results[x, y, z, t, :] = popt
            cov_results[x, y, z, t, :, :] = pcov

    return fit_results, cov_results

def _fit_concurrent_lorentzian(intensity_data, spectral_axis, expected_peaks, p0=None, bounds=None):
    variables = int(expected_peaks * 3 + 2)
    if p0 is not None and len(p0) != variables:
        p0 = None
        print('Length of p0 does not match the model. Fallback to p0 = None')
    fit_results = np.zeros((intensity_data.shape[0], intensity_data.shape[1], variables))
    cov_results = np.empty((intensity_data.shape[0], intensity_data.shape[1], variables, variables))

    fit_funcs = {1: _Lorentzian_1, 2: _Lorentzian_2, 3: _Lorentzian_3}
    X = spectral_axis[0]
    print('Starting of Multiprocessing can take up to 10 seconds.')
    tasks = []
    with ProcessPoolExecutor(max_workers=max(1,int(os.cpu_count()*0.25))) as executor:
        for x in range(intensity_data.shape[0]):
            for y in range(intensity_data.shape[1]):
                intensity_data_slice = intensity_data[x, y, :]
                # Skip if all values are masked
                if np.ma.is_masked(intensity_data_slice) and np.ma.getmask(intensity_data_slice).all():
                    fit_results[x, y, :] = np.nan
                    cov_results[x, y, :, :] = np.nan
                    continue
                task = executor.submit(_fitLorentzian, x, y, X, intensity_data_slice, expected_peaks, p0, bounds, fit_funcs)
                tasks.append(task)

        # Use tqdm to display a progress bar
        for future in tqdm(as_completed(tasks), total=len(tasks), desc='Fitting Spectral data'):
            x, y, popt, pcov = future.result()
            fit_results[x, y, :] = popt
            cov_results[x, y, :, :] = pcov

    return fit_results, cov_results


def _DHO_1(x, I0, freqShift, LineWidth, Background, Asymmetry):
    return I0 * 4 * LineWidth * freqShift**2 /(np.pi*(((x-Asymmetry)**2 - freqShift**2)**2 + 4*(LineWidth*(x-Asymmetry))**2)) + Background


def _DHO_2(x, I0, freqShift, LineWidth, I02, freqShift2, LineWidth2, Background, Asymmetry):
    return _DHO_1(x, I0, freqShift, LineWidth, Background, Asymmetry) + _DHO_1(x, I02, freqShift2, LineWidth2, Background, Asymmetry)


def _DHO_3(x, I0, freqShift, LineWidth, I02, freqShift2, LineWidth2, I03, freqShift3, LineWidth3, Background, Asymmetry):
    return _DHO_2(x, I0, freqShift, LineWidth, I02, freqShift2, LineWidth2, Background, Asymmetry) + _DHO_1(x, I03, freqShift3, LineWidth3, Background, Asymmetry)


def _Lorentzian_1(x, I0, freqShift, LineWidth, Background, Asymmetry):
    return I0 * LineWidth /((((x-Asymmetry)**2 - freqShift**2)**2 + (LineWidth*(x-Asymmetry))**2)) + Background


def _Lorentzian_2(x, I0, freqShift, LineWidth, I02, freqShift2, LineWidth2, Background, Asymmetry):
    return _Lorentzian_1(x, I0, freqShift, LineWidth, Background, Asymmetry) + _Lorentzian_1(x, I02, freqShift2, LineWidth2, Background, Asymmetry)


def _Lorentzian_3(x, I0, freqShift, LineWidth, I02, freqShift2, LineWidth2, I03, freqShift3, LineWidth3, Background, Asymmetry):
    return _Lorentzian_2(x, I0, freqShift, LineWidth, I02, freqShift2, LineWidth2, Background, Asymmetry) + _Lorentzian_1(x, I03, freqShift3, LineWidth3, Background, Asymmetry)
