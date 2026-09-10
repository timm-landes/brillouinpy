# -*- coding: utf-8 -*-
"""
Hierarchical, additive multi-component peak fitting (DHO or Lorentzian).

Motivation (see ``benchmarks/`` for the full evidence): running a multi-peak
:class:`~brillouinpy.analysis.fit.core.DHO` fit *everywhere* on an image that
mixes background-only and background+component pixels is unreliable - a
second (or third) peak is rank-deficient wherever its own amplitude -> 0, so
the fit either fails outright or plants a spurious narrow spike on a noise
excursion. No post-hoc filter on the result (by amplitude, a robust
threshold, or fit covariance) rescues this: the contamination is a
persistent fraction of pixels overlapping a genuine component's feature
range, not rare outliers a threshold can reject.

:func:`segmented_fit` implements the workflow that does work: classify first
with a cheap, model-free classifier, then fit only as many peaks (DHO by
default, or Lorentzian) as each class actually contains, anchoring peaks
already identified in an earlier
stage (both their shift *and* their amplitude/linewidth, the latter seeding
the newly added peak so the optimiser is not left guessing from a generic,
whole-window amplitude estimate once two or more peaks are already pinned
elsewhere in the spectrum).
"""
import numpy as np

from . import core

__all__ = ["segmented_fit", "SegmentedFitResult"]


def _default_classifier(image, n_components):
    """k-means (``n_components`` clusters) on the raw spectra, with labels
    reordered by ascending total intensity.

    Valid whenever every additional component only *adds* signal (never
    subtracts it) - the nested-additive assumption this whole module relies
    on: class 0 is the fewest-component pixels (e.g. background only), class
    ``n_components - 1`` the most (e.g. background + all inclusions).
    """
    import sklearn.cluster as skcluster  # local: keeps this out of the DHO
    # fit's multiprocessing workers, which never call this function.

    data = np.ma.filled(image.spectral_data, np.nan).astype(float)
    shape = data.shape[:-1]
    flat = data.reshape(-1, data.shape[-1])
    ok = np.all(np.isfinite(flat), axis=1)

    labels_flat = np.full(flat.shape[0], -1, dtype=int)
    labels_flat[ok] = skcluster.KMeans(n_clusters=n_components, n_init=10,
                                       random_state=0).fit_predict(flat[ok])

    totals = np.array([
        np.mean(flat[ok][labels_flat[ok] == k].sum(axis=1)) if np.any(labels_flat[ok] == k) else np.inf
        for k in range(n_components)
    ])
    remap = np.empty(n_components, dtype=int)
    remap[np.argsort(totals)] = np.arange(n_components)

    ordered = np.where(labels_flat >= 0, remap[np.clip(labels_flat, 0, n_components - 1)], -1)
    return ordered.reshape(shape)


class SegmentedFitResult:
    """
    Result of :func:`segmented_fit`.

    Attributes
    ----------
    labels : numpy.ndarray
        Integer class map, ``0`` (fewest components) .. ``n_components - 1``
        (most), or ``-1`` where the classifier declined to label a pixel
        (e.g. fully masked input).
    shift, linewidth, amplitude : numpy.ndarray
        Per pixel, the fitted Brillouin shift (GHz) / linewidth (GHz) /
        amplitude of that pixel's *newest* component - the one its class
        introduces on top of the previous class. ``NaN`` where unset (a
        failed fit, or an unlabelled pixel).
    stage_shift, stage_amplitude, stage_linewidth : list[float]
        Per fitting stage (index ``0`` .. ``n_components - 1``), the *median*
        fitted shift/amplitude/linewidth of that stage's newest peak over its
        own class's pixels - the values propagated as anchors/seeds into
        later stages. Kept here for diagnostics/calibration.
    """

    def __init__(self, labels, shift, linewidth, amplitude, stage_shift, stage_amplitude, stage_linewidth):
        self.labels = labels
        self.shift = shift
        self.linewidth = linewidth
        self.amplitude = amplitude
        self.stage_shift = stage_shift
        self.stage_amplitude = stage_amplitude
        self.stage_linewidth = stage_linewidth


def _fit_stage(image, keep, expected_peaks, anchor_shifts, anchor_seeds, shift_bounds,
               anchor_tolerance, model):
    """Fit ``expected_peaks`` ``model`` peaks (``'dho'`` or ``'lorentzian'``)
    restricted to ``keep`` pixels (all others set to NaN, which the fit already
    skips for free). Peaks
    ``0 .. len(anchor_shifts) - 1`` have their shift bounded to
    ``anchor_shifts[i] +/- anchor_tolerance`` and their amplitude/linewidth
    p0 seeded from ``anchor_seeds[i]``; the last peak is free, seeded from
    half the most recently identified component's amplitude and its own
    linewidth (a generic, whole-window guess is a poor p0 for it once two or
    more other peaks are already pinned nearby - see the module docstring)."""
    data = np.ma.filled(image.spectral_data, np.nan).astype(float)
    masked = data.copy()
    masked[~keep] = np.nan
    restricted = type(image)(masked, image.spectral_axis)

    n_anchored = len(anchor_shifts)
    lo_shift, hi_shift = shift_bounds
    scale = float(np.nanmax(masked))
    lo = [0.0, lo_shift, 0.0] * expected_peaks + [0.0, -2.0]
    hi = [np.inf, hi_shift, 5.0] * expected_peaks + [max(1.0, scale), 2.0]
    for i, a in enumerate(anchor_shifts):
        lo[1 + 3 * i], hi[1 + 3 * i] = a - anchor_tolerance, a + anchor_tolerance

    if expected_peaks == 1:
        p0 = np.asarray(core.estimate_p0(restricted, expected_peaks=1), dtype=float)
        p0[1] = abs(p0[1])  # peak detection may land on the anti-Stokes side
    else:
        stokes_mean = restricted.mean
        inten = np.ma.filled(stokes_mean.spectral_data, np.nan).astype(float)
        axis = np.asarray(stokes_mean.spectral_axis)
        dominant_shift = anchor_shifts[0] if anchor_shifts else float(axis[np.nanargmax(inten)])
        bg_level = float(np.nanpercentile(inten, 10))
        amp_guess = float(np.nanmax(inten) - bg_level)
        p0 = []
        last_placed = dominant_shift
        for i in range(expected_peaks):
            if i < n_anchored:
                shift_i = anchor_shifts[i]
                amp_i, width_i = anchor_seeds[i] if i < len(anchor_seeds) else \
                    ((amp_guess if i == 0 else 0.3 * amp_guess), 0.8)
                last_placed = shift_i
            else:
                shift_i = min(last_placed + 1.5, hi_shift - 0.5)
                if anchor_seeds:
                    prev_amp, prev_width = anchor_seeds[-1]
                    amp_i, width_i = 0.5 * prev_amp, prev_width
                else:
                    amp_i, width_i = 0.3 * amp_guess, 0.8
                last_placed = shift_i
            p0 += [amp_i, shift_i, width_i]
        p0 += [max(bg_level, 0.0), 0.0]
        p0 = np.array(p0)
    p0 = np.clip(p0, lo, hi)

    fit_cls = {"dho": core.DHO, "lorentzian": core.Lorentzian,
               "gaussian": core.Gaussian}[model]
    params, _ = fit_cls(expected_peaks=expected_peaks, p0=list(p0), bounds=(lo, hi)).apply(restricted)
    params = np.asarray(params)
    amps = np.abs(params[..., 0:3 * expected_peaks:3])
    shifts = np.abs(params[..., 1:3 * expected_peaks:3])
    widths = np.abs(params[..., 2:3 * expected_peaks:3])
    return amps, shifts, widths


def segmented_fit(image, n_components, *, classifier='kmeans', model='dho',
                  shift_bounds=(3.0, 14.0), anchor_tolerance=0.4):
    """
    Classify ``image`` into ``n_components`` nested-additive classes, then fit
    exactly the number of :class:`~brillouinpy.analysis.fit.core.DHO` /
    :class:`~brillouinpy.analysis.fit.core.Lorentzian` peaks (see ``model``)
    each class actually contains - the recommended workflow for a sample
    where a constant background/medium spectrum is present everywhere and one
    or more further components are *added* on top in progressively smaller
    regions (e.g. a hydration background plus an inclusion; a cell's medium,
    cytoplasm and nucleus). See the module docstring for why fitting the
    maximum peak count everywhere and filtering afterwards does not work.

    Assumes a *nested additive* geometry: class 0 pixels contain exactly one
    peak (e.g. background only), class 1 pixels contain that peak plus one
    more, ..., class ``n_components - 1`` pixels contain all ``n_components``
    peaks summed - each class strictly contains more signal than the
    previous one, everywhere. The default classifier exploits exactly this
    (ordering k-means clusters by ascending total intensity); pass a custom
    ``classifier`` if that ordering assumption does not hold for your data
    but some other classifier still recovers the right nesting order.

    Parameters
    ----------
    image : core.SpectralObject
        The data to fit (typically a :class:`~brillouinpy.SpectralImage`).
    n_components : int
        Number of additive components, 2 or 3. brillouinpy's DHO fit engine
        only has closed-form models for one to three peaks, so this is the
        practical ceiling; a component count above 3 is not supported.
    classifier : {'kmeans'} or callable, optional
        ``'kmeans'`` (default): the ordered-k-means classifier described
        above. A callable is called as ``classifier(image, n_components)``
        and must return an integer label map (same spatial shape as
        ``image``) with values ``0 .. n_components - 1``, already ordered
        from fewest to most components.
    model : {'dho', 'lorentzian', 'gaussian'}, optional
        Lineshape fitted at each stage. ``'dho'`` (default) uses
        :class:`~brillouinpy.analysis.fit.core.DHO`, ``'lorentzian'``
        :class:`~brillouinpy.analysis.fit.core.Lorentzian`, ``'gaussian'``
        :class:`~brillouinpy.analysis.fit.core.Gaussian`. They share the same
        parameter layout, so ``SegmentedFitResult`` is unchanged; ``linewidth``
        is the HWHM either way.
    shift_bounds : tuple[float, float], optional
        ``(low, high)`` GHz bounds every fitted shift is constrained to - a
        realistic spectrometer range. Default ``(3.0, 14.0)``.
    anchor_tolerance : float, optional
        +/- GHz window each already-identified peak's shift is bounded to in
        later stages. Default ``0.4``.

    Returns
    -------
    SegmentedFitResult

    Examples
    --------
    Two components (a constant background plus one added inclusion)::

        result = brillouinpy.analysis.fit.segmented_fit(image, n_components=2)
        result.labels      # 0 = background-only, 1 = background + inclusion
        result.shift       # background shift where labels==0, inclusion shift where labels==1

    Three nested components (medium / cytoplasm / nucleus)::

        result = brillouinpy.analysis.fit.segmented_fit(image, n_components=3)
    """
    if n_components not in (2, 3):
        raise ValueError(
            f"segmented_fit only supports n_components in (2, 3) - brillouinpy's DHO fit engine "
            f"(brillouinpy.analysis.fit.core) only has closed-form models for one to three peaks, "
            f"got n_components={n_components}."
        )

    if model not in ("dho", "lorentzian", "gaussian"):
        raise ValueError(f"model must be 'dho', 'lorentzian' or 'gaussian', got {model!r}.")

    if classifier == 'kmeans':
        labels = _default_classifier(image, n_components)
    elif callable(classifier):
        labels = np.asarray(classifier(image, n_components))
    else:
        raise ValueError(f"classifier must be 'kmeans' or a callable, got {classifier!r}.")

    shape = labels.shape
    shift = np.full(shape, np.nan)
    linewidth = np.full(shape, np.nan)
    amplitude = np.full(shape, np.nan)
    stage_shift, stage_amplitude, stage_linewidth = [], [], []
    anchor_shifts, anchor_seeds = [], []

    for stage in range(n_components):
        keep = labels == stage
        if np.any(keep):
            amps, shifts, widths = _fit_stage(image, keep, stage + 1, anchor_shifts, anchor_seeds,
                                              shift_bounds, anchor_tolerance, model)
            new_amp, new_shift, new_width = amps[..., stage], shifts[..., stage], widths[..., stage]
            shift = np.where(keep, new_shift, shift)
            linewidth = np.where(keep, new_width, linewidth)
            amplitude = np.where(keep, new_amp, amplitude)
            stage_amp_med = float(np.nanmedian(new_amp[keep]))
            stage_width_med = float(np.nanmedian(new_width[keep]))
            stage_shift_med = float(np.nanmedian(new_shift[keep]))
        else:
            # No pixels in this class (e.g. a small held-out test image) -
            # fall back to a plausible continuation so later stages still
            # have something to anchor to, rather than raising.
            stage_amp_med = anchor_seeds[-1][0] if anchor_seeds else 1.0
            stage_width_med = anchor_seeds[-1][1] if anchor_seeds else 0.8
            stage_shift_med = (anchor_shifts[-1] + 1.0) if anchor_shifts else float(shift_bounds[0])
        stage_shift.append(stage_shift_med)
        stage_amplitude.append(stage_amp_med)
        stage_linewidth.append(stage_width_med)
        anchor_shifts.append(stage_shift_med)
        anchor_seeds.append((stage_amp_med, stage_width_med))

    return SegmentedFitResult(labels, shift, linewidth, amplitude, stage_shift, stage_amplitude, stage_linewidth)
