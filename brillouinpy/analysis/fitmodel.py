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
    return _estimate_p0_from_mean(mean_spectrum.spectral_data, mean_spectrum.spectral_axis, expected_peaks)


def _estimate_p0_from_mean(intensity, axis, expected_peaks):
    """Array-based core of :func:`estimate_p0` - takes a 1-D mean-intensity array
    and its axis directly (so callers that already have the mean spectrum, e.g.
    :func:`_fit_concurrent_DHO_auto`, don't rebuild a spectral object)."""
    axis = np.asarray(axis)
    intensity = np.ma.filled(intensity, np.nan).astype(float)

    if np.all(np.isnan(intensity)):
        raise ValueError("Cannot estimate p0: mean spectrum is entirely masked/NaN.")

    background = float(np.nanpercentile(intensity, 5))
    intensity = np.where(np.isnan(intensity), background, intensity)
    spacing = float(np.mean(np.diff(axis)))

    # The DHO model is symmetric about zero (a mode at +f also produces a peak at
    # -f), so on a two-sided Brillouin axis one mode = one Stokes-side peak. Only
    # consider peaks at positive shift, past the central elastic region, so
    # ``expected_peaks`` counts modes rather than visible peaks.
    two_sided = axis.min() < 0 < axis.max()
    stokes_threshold = max(spacing, 0.05 * float(np.nanmax(np.abs(axis)))) if two_sided else -np.inf

    candidate_peaks, _ = find_peaks(intensity)
    candidate_peaks = candidate_peaks[axis[candidate_peaks] > stokes_threshold]
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
            (i for i in range(len(intensity)) if i not in detected_peaks and axis[i] > stokes_threshold),
            key=lambda i: intensity[i], reverse=True,
        )
        for i in remaining:
            if len(detected_peaks) + len(padded_peaks) >= expected_peaks:
                break
            if all(abs(i - p) > min_spacing for p in detected_peaks + padded_peaks):
                padded_peaks.append(i)
        stokes_start = int(np.searchsorted(axis, max(stokes_threshold, axis[0])))
        while len(detected_peaks) + len(padded_peaks) < expected_peaks:
            slot = len(detected_peaks) + len(padded_peaks)
            padded_peaks.append(stokes_start + int((len(intensity) - stokes_start) * (slot + 0.5) / expected_peaks))

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


# --------------------------------------------------------------------------- #
# Automatic peak-count selection (expected_peaks='auto')
# --------------------------------------------------------------------------- #

def _estimate_noise_variance(y):
    """Robust, model-free noise variance from the spectrum's second differences
    (var of the 2nd difference of white noise is 6 sigma^2). Returns ``None`` if
    there is too little data."""
    y = np.asarray(y, dtype=float)
    d = np.diff(y[np.isfinite(y)], n=2)
    if d.size < 4:
        return None
    mad = float(np.median(np.abs(d - np.median(d))))
    sigma = 1.4826 * mad / np.sqrt(6.0)
    return float(sigma ** 2) if sigma > 0 else None


def _information_criterion(rss, n, k, criterion, noise_variance=None):
    """AIC / AICc / BIC, lower is better. With a ``noise_variance`` the fit term
    is a proper chi-square ``RSS / sigma^2`` (so an extra peak that only trims
    RSS by less than the penalty's worth of noise is not selected); without one
    it falls back to the Gaussian ``n * ln(RSS/n)``."""
    fit_term = rss / noise_variance if noise_variance else n * np.log(rss / n)
    if criterion == "aic":
        return fit_term + 2 * k
    if criterion == "aicc":
        return fit_term + 2 * k + (2 * k * (k + 1)) / max(n - k - 1, 1)
    if criterion == "bic":
        return fit_term + k * np.log(n)
    raise ValueError(f"criterion must be 'aic', 'aicc' or 'bic', got {criterion!r}.")


def _dho_peak_heights(popt, n_peaks):
    """Per-peak height ``I0 / (pi * LineWidth)`` - bounded by the data (unlike I0)
    even for a degenerate narrow-spike fit."""
    return [abs(popt[3 * i]) / (np.pi * max(abs(popt[3 * i + 2]), 1e-12)) for i in range(n_peaks)]


def _select_count(entries, min_improvement, min_peak_fraction):
    """From ``{count: (criterion_value, popt)}`` (lower criterion is better), the
    smallest count. A larger count is only accepted if it beats the current best
    by more than ``min_improvement`` *and* every one of its fitted peaks has a
    height of at least ``min_peak_fraction`` of the tallest (so a peak that only
    mops up noise or model mismatch is not counted as a real mode)."""
    counts = sorted(entries)
    best = counts[0]
    for c in counts[1:]:
        score, popt = entries[c]
        if score >= entries[best][0] - min_improvement:
            continue
        heights = _dho_peak_heights(popt, c)
        if min(heights) < min_peak_fraction * max(heights):
            continue
        best = c
    return best


def _auto_bounds(axis, p0, n_peaks, shift_window=3.0):
    """Per-peak bounds for the automatic fit: non-negative amplitude/linewidth,
    and each shift constrained to ``+/- shift_window`` GHz of its ``p0`` value so
    a peak cannot escape to the spectral edge (where it would fit an artefact and
    make the model-selection prefer a spurious extra peak)."""
    axis = np.asarray(axis, dtype=float)
    spacing = float(np.mean(np.abs(np.diff(axis)))) or 1e-3
    max_width = 0.5 * float(np.nanmax(np.abs(axis)))
    lo, hi = [], []
    for i in range(n_peaks):
        s = abs(float(p0[1 + 3 * i]))
        lo += [0.0, max(spacing, s - shift_window), spacing]
        hi += [np.inf, s + shift_window, max_width]
    return lo + [-np.inf, -2.0], hi + [np.inf, 2.0]


def _pad_auto_params(popt, n_peaks, max_peaks):
    """Pad a length-``3n+2`` parameter vector to ``3*max_peaks+2`` with NaN peak
    triplets, keeping the trailing [Background, Asymmetry] last."""
    popt = list(np.asarray(popt, dtype=float))
    return popt[:3 * n_peaks] + [np.nan] * (3 * (max_peaks - n_peaks)) + popt[3 * n_peaks:]


def _pad_auto_cov(pcov, n_peaks, max_peaks):
    variables = 3 * max_peaks + 2
    out = np.full((variables, variables), np.nan)
    src = list(range(3 * n_peaks)) + [3 * n_peaks, 3 * n_peaks + 1]
    dst = list(range(3 * n_peaks)) + [variables - 2, variables - 1]
    out[np.ix_(dst, dst)] = np.asarray(pcov, dtype=float)[np.ix_(src, src)]
    return out


def _fitDHO_auto(idx, spectral_axis, intensity_data_slice, p0_by_count, bounds_by_count,
                 fit_funcs, criterion, min_improvement, min_peak_fraction, max_peaks):
    variables = 3 * max_peaks + 2
    nan_params = np.full(variables, np.nan)
    nan_cov = np.full((variables, variables), np.nan)

    if np.ma.is_masked(intensity_data_slice):
        if np.ma.getmaskarray(intensity_data_slice).all():
            return idx, nan_params, nan_cov
        y = intensity_data_slice.filled(np.nan)
    else:
        y = np.asarray(intensity_data_slice, dtype=float).copy()
    y[y == 0] = np.nan
    if np.all(np.isnan(y)):
        return idx, nan_params, nan_cov

    noise_variance = _estimate_noise_variance(y)
    fits, scores = {}, {}
    for n_peaks, p0 in p0_by_count.items():
        lo, hi = bounds_by_count[n_peaks]
        try:
            popt, pcov = curve_fit(fit_funcs[n_peaks], spectral_axis, y,
                                   p0=list(np.clip(p0, lo, hi)), bounds=(lo, hi), nan_policy="omit")
        except (RuntimeError, ValueError):
            continue
        resid = y - fit_funcs[n_peaks](spectral_axis, *popt)
        ok = np.isfinite(resid)
        n = int(ok.sum())
        k = 3 * n_peaks + 2
        rss = float(np.sum(resid[ok] ** 2))
        if n <= k + 1 or not np.isfinite(rss) or rss <= 0:
            continue
        fits[n_peaks] = (popt, pcov)
        scores[n_peaks] = _information_criterion(rss, n, k, criterion, noise_variance)

    if not scores:
        return idx, nan_params, nan_cov
    n_peaks = _select_count({c: (scores[c], fits[c][0]) for c in scores},
                            min_improvement, min_peak_fraction)
    popt, pcov = fits[n_peaks]
    return idx, np.asarray(_pad_auto_params(popt, n_peaks, max_peaks)), _pad_auto_cov(pcov, n_peaks, max_peaks)


def _fit_concurrent_DHO_auto(intensity_data, spectral_axis, *, max_peaks=3, criterion="bic",
                             min_improvement=6.0, min_peak_fraction=0.25, irf=None):
    if max_peaks not in (2, 3):
        raise ValueError(f"expected_peaks='auto' needs max_peaks in (2, 3), got {max_peaks}.")
    axis = spectral_axis[0]
    data = np.ma.filled(intensity_data, np.nan).astype(float)
    spatial_shape = data.shape[:-1]
    variables = 3 * max_peaks + 2

    base_funcs = {1: _DHO_1, 2: _DHO_2, 3: _DHO_3}
    if irf is not None:
        irf_arr = np.asarray(irf) if not isinstance(irf, (tuple, list, str)) else irf
        if isinstance(irf_arr, np.ndarray) and irf_arr.ndim > 1:
            raise NotImplementedError(
                "expected_peaks='auto' does not support a per-pixel irf yet; pass a shared 1-D "
                "kernel or a parametric ('lorentzian', fwhm) spec, or select the peak count with "
                "fitmodel.estimate_peak_count and fit with a fixed expected_peaks.")
        kernel = irf_kernel(irf_arr, axis)
        fit_funcs = {c: _ConvolvedModel(base_funcs[c], kernel, axis) for c in range(1, max_peaks + 1)}
    else:
        fit_funcs = base_funcs

    mean_intensity = np.nanmean(data.reshape(-1, data.shape[-1]), axis=0)
    p0_by_count, bounds_by_count = {}, {}
    for c in range(1, max_peaks + 1):
        try:
            p0_by_count[c] = _estimate_p0_from_mean(mean_intensity, axis, c)
        except ValueError:
            p0_by_count[c] = [1.0] * (3 * c + 2)
        bounds_by_count[c] = _auto_bounds(axis, p0_by_count[c], c)

    fit_results = np.full(spatial_shape + (variables,), np.nan)
    cov_results = np.full(spatial_shape + (variables, variables), np.nan)
    print("Starting of Multiprocessing can take up to 10 seconds.")
    with ProcessPoolExecutor(max_workers=max(1, int(os.cpu_count() * 0.25))) as executor:
        tasks = [executor.submit(_fitDHO_auto, idx, axis, data[idx + (slice(None),)],
                                 p0_by_count, bounds_by_count, fit_funcs, criterion,
                                 min_improvement, min_peak_fraction, max_peaks)
                 for idx in np.ndindex(spatial_shape)]
        for future in tqdm(as_completed(tasks), total=len(tasks), desc="Fitting Spectral data (auto peak count)"):
            idx, popt, pcov = future.result()
            fit_results[idx] = popt
            cov_results[idx] = pcov
    return fit_results, cov_results


def fitted_peak_count(fit_parameters):
    """Number of peaks actually fitted at each pixel of an ``expected_peaks='auto'``
    result (a NaN peak triplet means that slot was not used)."""
    p = np.asarray(fit_parameters, dtype=float)
    max_peaks = (p.shape[-1] - 2) // 3
    return np.isfinite(p[..., 1:3 * max_peaks:3]).sum(axis=-1)


def estimate_peak_count(spectral_object, *, criterion="bic", candidates=(1, 2, 3),
                        min_improvement=6.0, min_peak_fraction=0.25, bounds=None, return_scores=False):
    """
    Pick the number of Brillouin modes (``expected_peaks``) that best explains the
    *mean* spectrum of ``spectral_object``, by information criterion.

    Much cheaper than ``DHO(expected_peaks='auto')`` (one fit per candidate on a
    single spectrum, not per pixel) and more robust to per-pixel noise - use it
    when the mode count is a property of the sample rather than something that
    varies pixel to pixel.

    This is a heuristic. Information criteria assume the model is otherwise
    correct, so anything the single DHO does *not* capture - a residual elastic /
    Rayleigh wing, an asymmetric or non-Lorentzian real lineshape, an IRF that
    was not deconvolved - reduces the residual when an extra peak is added and
    biases the count upward. On such data, prefer a count from prior knowledge or
    from the variance spectrum (see ``04_classical_analysis.py``), or raise
    ``min_peak_fraction``.

    Parameters
    ----------
    spectral_object : core.SpectralObject
    criterion : {'bic', 'aic', 'aicc'}
        'bic' (default) penalises extra peaks most heavily.
    candidates : iterable of int
        Peak counts to try (each must be 1, 2 or 3).
    min_improvement : float
        An extra peak must improve the criterion by more than this (default 6).
    min_peak_fraction : float
        An extra peak must also have a height of at least this fraction of the
        tallest peak (default 0.25), so a peak that only mops up noise or
        lineshape mismatch is not counted as a real mode.
    bounds : tuple, optional
        Shared ``(lo, hi)`` bounds; if omitted, a default (non-negative
        amplitude/linewidth, shift near each p0) is used per candidate.
    return_scores : bool
        If True, also return ``{count: criterion_value}``.

    Returns
    -------
    int, or (int, dict)
    """
    mean = spectral_object.mean
    axis = np.asarray(mean.spectral_axis)
    y = np.ma.filled(mean.spectral_data, np.nan).astype(float)
    fit_funcs = {1: _DHO_1, 2: _DHO_2, 3: _DHO_3}
    noise_variance = _estimate_noise_variance(y)

    entries, scores = {}, {}
    for c in candidates:
        if c not in (1, 2, 3):
            raise ValueError(f"candidate peak counts must be 1, 2 or 3, got {c}.")
        try:
            p0 = _estimate_p0_from_mean(y, axis, c)
        except ValueError:
            p0 = [1.0] * (3 * c + 2)
        lo, hi = bounds if bounds is not None else _auto_bounds(axis, p0, c)
        try:
            popt, _ = curve_fit(fit_funcs[c], axis, y, p0=list(np.clip(p0, lo, hi)),
                                bounds=(lo, hi), nan_policy="omit")
        except (RuntimeError, ValueError):
            continue
        resid = y - fit_funcs[c](axis, *popt)
        ok = np.isfinite(resid)
        n, k = int(ok.sum()), 3 * c + 2
        rss = float(np.sum(resid[ok] ** 2))
        if n <= k + 1 or not np.isfinite(rss) or rss <= 0:
            continue
        scores[c] = _information_criterion(rss, n, k, criterion, noise_variance)
        entries[c] = (scores[c], popt)

    if not entries:
        raise RuntimeError("no candidate peak count could be fitted to the mean spectrum.")
    best = _select_count(entries, min_improvement, min_peak_fraction)
    return (best, scores) if return_scores else best


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
        nz = np.flatnonzero(kernel > kernel.min())
        if nz.size < 3:
            raise ValueError("measured irf has no usable samples.")
        # Restrict to the detected support (a detected elastic-peak region is
        # otherwise mostly zeros spanning the whole spectrum).
        kernel = kernel[nz[0]:nz[-1] + 1]
        # Subtract a *linear* baseline through the support's two ends: a detected
        # elastic peak sits on the sloping inner wing of the Brillouin doublet,
        # not a flat pedestal, and leaving that in makes the kernel too broad
        # (the fit then over-corrects the linewidth).
        idx = np.arange(kernel.size, dtype=float)
        e = max(1, kernel.size // 8)
        ends_x = np.concatenate([idx[:e], idx[-e:]])
        ends_y = np.concatenate([kernel[:e], kernel[-e:]])
        slope, intercept = np.polyfit(ends_x, ends_y, 1)
        kernel = np.clip(kernel - (slope * idx + intercept), 0.0, None)
        # Drop channels below a small fraction of the peak (residual baseline noise
        # in the shoulders - kept low so genuine Lorentzian wings survive), then
        # trim to a symmetric window about the peak so it is centred.
        kernel[kernel < 0.005 * kernel.max()] = 0.0
        nz = np.flatnonzero(kernel > 0)
        if nz.size < 3:
            raise ValueError("measured irf has no usable samples after baseline subtraction.")
        kernel = kernel[nz[0]:nz[-1] + 1]
        peak = int(np.argmax(kernel))
        reach = min(peak, kernel.size - 1 - peak)
        if reach < 1:
            raise ValueError("measured irf peak is at the very edge of its support; give a wider window.")
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
    expected_peaks : int or ``'auto'``
        The number of Brillouin modes (symmetric peak pairs) to fit. ``'auto'``
        fits 1 .. ``max_peaks`` per pixel and keeps whichever count minimises the
        information ``criterion`` (see below); ``p0``/``bounds`` are then derived
        automatically (from :func:`estimate_p0` on the mean spectrum) and any
        values passed are ignored. The returned parameter array is padded to
        ``3 * max_peaks + 2``, with a NaN ``[I0, freqShift, LineWidth]`` triplet
        for each unused slot - :func:`fitted_peak_count` reads the per-pixel
        count back. Per-pixel model selection is noisy; :func:`estimate_peak_count`
        (one number for the whole image) is the cheaper, steadier option.
    p0 : list[floats] or None
        The initial guess (ignored when ``expected_peaks='auto'``).
    bounds : list[list[floats], list[floats]] or None
        Parameter bounds (ignored when ``expected_peaks='auto'``).
    max_peaks : int, optional
        For ``expected_peaks='auto'``: the largest peak count to consider (2 or 3;
        default 3).
    criterion : {'bic', 'aic', 'aicc'}, optional
        For ``expected_peaks='auto'``: the information criterion to minimise.
        ``'bic'`` (default) penalises extra peaks most heavily.
    min_improvement : float, optional
        For ``expected_peaks='auto'``: an extra peak is accepted only if it
        improves ``criterion`` by more than this (default 6) *and* is at least
        ``min_peak_fraction`` as tall as the biggest peak. Raise to be more
        conservative about adding modes.
    min_peak_fraction : float, optional
        For ``expected_peaks='auto'``: the minimum height (relative to the
        tallest peak) an added peak must have to count as a real mode rather
        than noise / lineshape mop-up. Default 0.25 - raise it (e.g. to 0.4) if a
        known single mode is still being split by residual elastic scattering or
        lineshape mismatch on real data.
    irf : array-like, tuple, ``'auto'`` or None, optional
        Instrument response function. If given, the model fitted to each spectrum is the
        intrinsic DHO lineshape *convolved with the IRF* plus a background, so the fitted
        ``LineWidth`` is the intrinsic Brillouin linewidth with the instrumental broadening
        modelled explicitly - instead of deconvolving the data first (which amplifies noise;
        contrast :class:`~brillouinpy.preprocessing.misc.Deconvoluter_IRF`).

        - ``'auto'``: use the fitted object's ``instrument_response_function`` attribute
          (set by :class:`~brillouinpy.preprocessing.misc.Deconvoluter_IRF` /
          :class:`~brillouinpy.preprocessing.misc.IRF_Remover` with ``store_irf=True``, or
          by :func:`~brillouinpy.preprocessing.misc.assign_irf`). A per-pixel
          ``(*spatial, B)`` IRF gives each pixel its own kernel; a shared 1-D one is used
          everywhere. Falls back to a plain fit (with a warning) if the object carries none.
        - a 1-D array: a measured IRF sample (e.g. the elastic peak) on the spectrum's
          channel spacing.
        - ``('lorentzian'|'gaussian', fwhm)`` / ``('voigt', fwhm_l, fwhm_g)``: a parametric
          IRF, widths in GHz.

        The elastic-peak channels should still be blanked to ``NaN`` (e.g. via
        :class:`~brillouinpy.preprocessing.misc.IRF_Remover`) so they are excluded from the
        residual; the convolved model itself is evaluated on the full axis. ``None`` (default)
        keeps the plain, unconvolved fit.
    **kwargs :

    """
    def __init__(self, *, expected_peaks, p0=None, bounds=None, irf=None,
                 max_peaks=3, criterion='bic', min_improvement=6.0, min_peak_fraction=0.25):
        if isinstance(expected_peaks, str) and expected_peaks == 'auto':
            super().__init__(_fit_concurrent_DHO_auto, max_peaks=max_peaks, criterion=criterion,
                             min_improvement=min_improvement, min_peak_fraction=min_peak_fraction, irf=irf)
        else:
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
    base_func = fit_funcs[expected_peaks]
    spectral_axis = spectral_axis[0]

    # irf may be None, a parametric spec / 1-D kernel (one model for every pixel),
    # or a per-pixel (*spatial, B) array (a kernel built per pixel; a pixel with no
    # detectable IRF falls back to the plain model).
    irf_arr = np.asarray(irf) if (irf is not None and not isinstance(irf, (tuple, list, str))) else irf
    per_pixel_irf = isinstance(irf_arr, np.ndarray) and irf_arr.ndim > 1
    if irf is None:
        shared_fit_func = base_func
    elif per_pixel_irf:
        shared_fit_func = None
    else:
        shared_fit_func = _ConvolvedModel(base_func, irf_kernel(irf_arr, spectral_axis), spectral_axis)

    def _pixel_fit_func(idx):
        if not per_pixel_irf:
            return shared_fit_func
        try:
            return _ConvolvedModel(base_func, irf_kernel(irf_arr[idx], spectral_axis), spectral_axis)
        except ValueError:
            return base_func  # no usable IRF for this pixel

    print('Starting of Multiprocessing can take up to 10 seconds.')
    tasks = []
    with ProcessPoolExecutor(max_workers=max(1,int(os.cpu_count()*0.25))) as executor:
        for idx in np.ndindex(spatial_shape):
            intensity_data_slice = intensity_data[idx + (slice(None),)]
            task = executor.submit(_fitDHO2, idx, spectral_axis, intensity_data_slice, expected_peaks,
                                   p0, bounds, _pixel_fit_func(idx))
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
