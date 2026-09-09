# -*- coding: utf-8 -*-
"""
Shared machinery for the benchmark scripts in this folder.

Defines the analysis methods under test, a ground-truth scoring helper, the
parameter sweep loop and the three standard figures. Each driver script
(``method_limitations.py``, ``snr_requirement.py``, ``additive_blob.py``) just
builds a list of ``(parameter_value, image, label_map)`` cases and hands it to
:func:`run_sweep` / :func:`make_figures`.

Not a tutorial module - see ``benchmarks/README.md``.
"""
import os
import sys
import time

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from sklearn.cluster import KMeans as SkKMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

import brillouinpy as bp

# The synthetic-data generators live with the examples; reuse them rather than
# keeping a second copy in sync.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples'))

# Spectral window (GHz) every method is restricted to - tight around the Stokes
# peak: the phasor's sensitivity to a small shift scales with 1 / window width,
# and the symmetric DHO model must not see the anti-Stokes peak.
STOKES_WINDOW = (4.0, 9.0)


def _labels_from_maps(component_maps):
    """argmax across a list/stack of abundance or one-hot maps -> integer label map."""
    return np.argmax(np.stack([np.asarray(m) for m in component_maps], axis=-1), axis=-1)


def stokes_side(image, window=STOKES_WINDOW):
    """Crop an image to a tight window around the Stokes peak."""
    keep = (image.spectral_axis >= window[0]) & (image.spectral_axis <= window[1])
    return bp.SpectralImage(np.asarray(image.spectral_data)[..., keep], image.spectral_axis[keep])


def align_labels(pred, truth):
    """Relabel a predicted 2-cluster map so its labels line up with ``truth`` for
    a like-coloured side-by-side plot: of the two label<->label permutations, keep
    the one that agrees with ``truth`` on more pixels. NaNs pass through. ARI
    itself is label-invariant, so this is cosmetic only."""
    truth = np.asarray(truth).astype(float)
    pred = np.asarray(pred, dtype=float)
    ok = np.isfinite(pred)
    swapped = np.where(ok, 1 - pred, np.nan)
    keep_swapped = np.nanmean(swapped[ok] == truth[ok]) > np.nanmean(pred[ok] == truth[ok])
    return swapped if keep_swapped else pred


def ari(pred, truth):
    """Adjusted Rand index over the pixels a method actually returned a label for."""
    pred = np.asarray(pred, dtype=float).ravel()
    ok = np.isfinite(pred)
    return adjusted_rand_score(np.asarray(truth).ravel()[ok], pred[ok])


# ---------------------------------------------------------------------------
# Methods under test. Each takes a SpectralImage and returns
#   (label_map, shift_map, shift_label, width_map)
# a 2-D predicted cluster segmentation (NaN where it declined to label a
# pixel), plus an optional continuous shift map (with a label for it, expected
# to contain "GHz" when it is a physical shift comparable to ground truth) and
# an optional continuous linewidth map (physical units, comparable to a true
# linewidth the same way). Methods with no physical parameter output (k-means,
# PCA, NMF, VCA) return ``None`` for the last two.
# ---------------------------------------------------------------------------

def _common_background(image, dark_fraction=33):
    """Estimate the spectrum shared by every pixel as the mean of the pixels with
    the lowest Stokes-window intensity (least likely to carry any added signal)."""
    data = np.ma.filled(image.spectral_data, np.nan).astype(float)
    axis = np.asarray(image.spectral_axis)
    win = (axis >= STOKES_WINDOW[0]) & (axis <= STOKES_WINDOW[1])
    stokes_sum = np.nansum(data[..., win], axis=-1).reshape(-1)
    flat = data.reshape(-1, data.shape[-1])
    return np.nanmean(flat[stokes_sum <= np.nanpercentile(stokes_sum, dark_fraction)], axis=0)


def phasor_method(image):
    ph = bp.analysis.phasor.phasor(image, axis_range=STOKES_WINDOW, background='min')
    labels = SkKMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(ph.features())
    return labels.reshape(image.shape), np.asarray(bp.analysis.phasor.phase_to_shift(ph)), \
        'Phasor shift (GHz)', None


def phasor_bgsub_method(image):
    """Spectral phasor after subtracting the estimated common background spectrum
    - so a strong shared Brillouin peak (the hydration signal in ``additive_blob``)
    no longer dominates the transform. Same clustering as :func:`phasor_method`."""
    ph = bp.analysis.phasor.phasor(image, axis_range=STOKES_WINDOW, background=_common_background(image))
    labels = SkKMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(ph.features())
    return labels.reshape(image.shape), np.asarray(bp.analysis.phasor.phase_to_shift(ph)), \
        'Phasor shift (GHz), bg-subtracted', None


def _dho_peak_height(amplitude, linewidth):
    """DHO peak height (~ I0 / (pi * linewidth)) - bounded by the data even when a
    degenerate fit drives I0 huge with a tiny linewidth, unlike I0 itself."""
    return np.abs(amplitude) / (np.pi * np.maximum(np.abs(linewidth), 0.05))


def _fit_dho_raw(image, expected_peaks, anchor_shifts=None, anchor_seeds=None):
    """
    Run the DHO fit and return the raw ``(amps, shifts, widths)`` arrays, each
    shaped ``(..., expected_peaks)``. Shared by every DHO-based method below -
    :func:`dho_fit` turns this into a segmentation; :func:`dho_fit_segmented`
    and :func:`dho_fit_segmented_3` call it once per pre-segmented region
    instead of forcing one fit to cover all of them at once.

    The DHO model is symmetric about zero (peaks at +/-freqShift), so it sees the
    full spectrum, not a one-sided crop. Shift bounds (a realistic spectrometer
    range) plus a p0 clipped into those bounds stop low-SNR fits diverging - the
    phasor needs no such per-dataset tuning.

    ``anchor_shifts``, if given, is a list of up to ``expected_peaks`` floats:
    peak ``i`` (for ``i < len(anchor_shifts)``) has its shift bounded to
    ``anchor_shifts[i] +/- 0.4 GHz`` - i.e. the analyst has already measured
    that many components (background, then background+one-more, ...) and pins
    them there, so the fit spends its one remaining free peak on the newest,
    not-yet-identified component instead of re-splitting an already-known line.
    Peaks beyond ``len(anchor_shifts)`` are free.

    ``anchor_seeds``, if given, is a parallel list of ``(amplitude, linewidth)``
    p0 seeds for those same anchored peaks - e.g. the median amplitude/linewidth
    the stage that first identified them actually fitted. With two or more
    anchored peaks (the 3-peak nucleus stage in :func:`dho_fit_segmented_3`), a
    generic amplitude guess for them is not just imprecise but can send the
    per-pixel optimizer to a spurious local minimum where the anchored peaks
    jointly "absorb" part of the genuinely new component instead of it (this is
    exactly what ``three_component.py``'s recovery figures show below a nucleus
    separation of ~1 GHz). Omit for a single anchored peak, where a generic
    guess is fine.
    """
    anchor_shifts = list(anchor_shifts) if anchor_shifts else []
    anchor_seeds = list(anchor_seeds) if anchor_seeds else []
    n_anchored = len(anchor_shifts)

    # Amplitude / background bounds scale with the data (counts vs. the old
    # arbitrary-unit intensities), so key them off the observed maximum.
    scale = float(np.nanmax(np.ma.filled(image.spectral_data, np.nan)))
    lo = [0.0, 3.0, 0.0] * expected_peaks + [0.0, -2.0]
    hi = [np.inf, 14.0, 5.0] * expected_peaks + [max(1.0, scale), 2.0]
    for i, a in enumerate(anchor_shifts):
        lo[1 + 3 * i], hi[1 + 3 * i] = a - 0.4, a + 0.4

    if expected_peaks == 1 and n_anchored == 0:
        p0 = np.asarray(bp.analysis.fit.estimate_p0(image, expected_peaks=1), dtype=float)
        p0[1] = abs(p0[1])  # peak detection may land on the anti-Stokes side
    else:
        stokes = stokes_side(image).mean
        inten = np.ma.filled(stokes.spectral_data, np.nan).astype(float)
        dominant_shift = anchor_shifts[0] if anchor_shifts else float(stokes.spectral_axis[np.nanargmax(inten)])
        bg_level = float(np.nanpercentile(inten, 10))
        amp_guess = float(np.nanmax(inten) - bg_level)
        p0 = []
        last_placed = dominant_shift
        for i in range(expected_peaks):
            if i < n_anchored:
                shift_i = anchor_shifts[i]
                if i < len(anchor_seeds):
                    amp_i, width_i = anchor_seeds[i]
                else:
                    amp_i, width_i = (amp_guess if i == 0 else 0.3 * amp_guess), 0.8
                last_placed = shift_i
            else:
                shift_i = min(last_placed + 1.5, STOKES_WINDOW[1] - 0.5)
                if anchor_seeds:
                    # A window-wide amp_guess/generic width is a poor p0 for the
                    # free peak once two peaks are already anchored elsewhere in
                    # the spectrum - seed it from the most recently identified
                    # component instead (half its amplitude, its own linewidth).
                    prev_amp, prev_width = anchor_seeds[-1]
                    amp_i, width_i = 0.5 * prev_amp, prev_width
                else:
                    amp_i, width_i = 0.3 * amp_guess, 0.8
                last_placed = shift_i
            p0 += [amp_i, shift_i, width_i]
        p0 += [max(bg_level, 0.0), 0.0]
        p0 = np.array(p0)
    p0 = np.clip(p0, lo, hi)

    params, _ = bp.analysis.fit.DHO(expected_peaks=expected_peaks, p0=list(p0), bounds=(lo, hi)).apply(image)
    params = np.asarray(params)
    amps = np.abs(params[..., 0:3 * expected_peaks:3])       # (..., expected_peaks)
    shifts = np.abs(params[..., 1:3 * expected_peaks:3])
    widths = np.abs(params[..., 2:3 * expected_peaks:3])
    return amps, shifts, widths


def dho_fit(image, expected_peaks=1, anchor_shift=None):
    """
    Fit ``expected_peaks`` DHO peaks per pixel (:func:`_fit_dho_raw`) and segment
    on a fitted parameter.

    - ``expected_peaks=1``: segment on the single fitted shift (the
      ``method_limitations``/``snr_requirement`` case).
    - ``expected_peaks=2`` with ``anchor_shift`` given: the background +
      added-component case, run *everywhere* without pre-segmentation (the
      "raw" 2-peak fit in ``additive_blob.py``). The added peak's **height**
      (``I0 / (pi*width)``, bounded by the data unlike ``I0``) is the
      segmentation feature - ~0 with no component, positive with one; its
      shift is the parameter map. See :func:`dho_fit_segmented` for why this is
      *not* the recommended way to run a multi-peak fit on such a field.
    """
    anchored = expected_peaks >= 2 and anchor_shift is not None
    amps, shifts, widths = _fit_dho_raw(image, expected_peaks, [anchor_shift] if anchored else None)
    heights = _dho_peak_height(amps, widths)                 # robust vs. narrow-spike fits

    if expected_peaks == 1:
        feature = shifts[..., 0]
        param_map, param_label, width_map = feature, 'DHO fitted shift (GHz)', widths[..., 0]
        drop_failed = True
    elif anchored:
        # A 2-peak fit is rank-deficient wherever the 2nd peak's amplitude -> 0
        # (its shift/width stop mattering), so background-only pixels often fail
        # the fit outright, and successful ones can put a tall narrow spike on a
        # noise excursion. Segment on the 2nd peak's *height* (I0 / (pi*width),
        # bounded by the data) and treat a failed fit as "no second peak".
        feature = heights[..., 1]
        param_map, width_map = shifts[..., 1], widths[..., 1]
        param_label = f'DHO {expected_peaks}-peak: added-component shift (GHz)'
        drop_failed = False
    else:
        background_ref = np.nanmedian(shifts)
        component = np.argmax(np.abs(shifts - background_ref), axis=-1, keepdims=True)
        feature = np.take_along_axis(heights, component, axis=-1)[..., 0]
        param_map = np.take_along_axis(shifts, component, axis=-1)[..., 0]
        width_map = np.take_along_axis(widths, component, axis=-1)[..., 0]
        param_label = f'DHO {expected_peaks}-peak: component shift (GHz)'
        drop_failed = False

    return _segment_on_feature(feature, image.shape, param_map, param_label, drop_failed=drop_failed,
                               width_map=width_map)


def _segment_on_feature(feature, shape, param_map, param_label, *, drop_failed=True, width_map=None):
    """GMM(2) on a per-pixel scalar feature. ``drop_failed`` keeps NaN pixels out
    of the fit (and unlabelled); otherwise NaN is treated as feature 0."""
    flat = np.asarray(feature, dtype=float).reshape(-1)
    labels = np.full(flat.shape, np.nan)
    if drop_failed:
        ok = np.isfinite(flat)
        labels[ok] = GaussianMixture(n_components=2, n_init=3, random_state=0).fit_predict(flat[ok, None])
    else:
        filled = np.where(np.isfinite(flat), flat, 0.0)
        labels = GaussianMixture(n_components=2, n_init=3, random_state=0).fit_predict(filled[:, None]).astype(float)
    return labels.reshape(shape), param_map, param_label, width_map


def _kmeans_background_mask(image):
    """
    Cheap, model-free pre-segmentation into "background-only" vs. "has the added
    component": k-means (k=2) on the raw Stokes-window spectra, with the
    lower-total-intensity cluster called background (the added component only
    ever adds signal on top, never removes it). This is exactly
    :func:`raw_kmeans_method`'s clustering, just without the one-hot argmax
    packaging - reused here to decide *where* to spend a second DHO peak.
    """
    cropped = stokes_side(image)
    flat = np.ma.filled(cropped.spectral_data, np.nan).astype(float).reshape(-1, cropped.spectral_data.shape[-1])
    cluster = SkKMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(flat)
    totals = np.nansum(flat, axis=1)
    sample_cluster = int(np.argmax([totals[cluster == c].mean() for c in (0, 1)]))
    return (cluster == sample_cluster).reshape(image.shape)


def dho_fit_segmented(image):
    """
    The recommended way to run a multi-peak DHO fit on a field that mixes
    background-only and background+component pixels: pre-segment first with a
    cheap, model-free classifier, then fit only the number of peaks that is
    physically appropriate in *each* region - one DHO in background-only
    pixels, two in sample pixels, where a second peak is actually present and
    so the fit is well-posed (no amplitude -> 0 degeneracy; contrast
    :func:`dho_fit` with ``anchor_shift`` set, run everywhere without this
    step).

    This is now a thin wrapper over the shipped
    :func:`brillouinpy.analysis.fit.segmented_fit` (``n_components=2``) - so the
    numbers these scripts produce validate the real package feature, not a
    benchmark-only reimplementation. The only benchmark-side specialisation is
    the classifier: :func:`_kmeans_background_mask` clusters the *Stokes-window*
    spectra (``STOKES_WINDOW``), matching the standalone ``k-means (raw)``
    method it is compared against, rather than ``segmented_fit``'s default
    whole-spectrum k-means.

    Returns the pre-segmentation itself as the label map (unchanged by the
    subsequent fit - the point of this method is the parameter maps, not a
    better segmentation) and, as the parameter map, each pixel's background
    shift (background-only) or component shift (sample), for
    :func:`compare_shift_recovery`.
    """
    def classifier(img, n_components):
        # bool mask (True = has the added component = more total signal) -> 0/1,
        # already ordered fewest -> most components as segmented_fit expects.
        return _kmeans_background_mask(img).astype(int)

    result = bp.analysis.segmented_fit(image, n_components=2, classifier=classifier)
    return (result.labels.astype(float), result.shift,
            'DHO (segmented): background/component shift (GHz)', result.linewidth)


def make_dho_fit_2peak_masked_method(anchor_shift, mad_k=3.0):
    """
    A second, *not* recommended way to cope with the everywhere-2-peak fit's
    instability: run it everywhere (as :func:`dho_fit` with ``anchor_shift``
    does), then post-hoc discard (relabel as background) any pixel whose
    second-peak height sits below a robust "this is just noise" cutoff, and
    read the component shift off the surviving pixels.

    The cutoff is ``median(height) + mad_k * MAD(height)`` (MAD scaled to be a
    normal-consistent robust standard deviation) rather than a fraction of the
    field's single largest height - the latter is dominated by whichever pixel
    degenerates worst and was tried first; it is *not* what makes this
    mitigation fail. Even this robust version does not rescue the
    classification: the contamination is not a handful of extreme outliers a
    robust statistic can shrug off, but a persistent several-percent-of-pixels
    fraction of background-only pixels whose degenerate second peak reaches
    the *same* height range as a genuine component peak - there is no gap in
    the distribution for any threshold, robust or not, to land in. See
    ``additive_blob.py`` for the discussion.
    """
    def method(image):
        amps, shifts, widths = _fit_dho_raw(image, expected_peaks=2, anchor_shifts=[anchor_shift])
        heights = _dho_peak_height(amps, widths)
        h1 = heights[..., 1]
        flat = h1.reshape(-1)
        flat = flat[np.isfinite(flat)]
        median = np.median(flat)
        mad = np.median(np.abs(flat - median)) * 1.4826  # -> consistent with a normal std
        threshold = median + mad_k * mad
        is_sample = h1 >= threshold
        shift_map = np.where(is_sample, shifts[..., 1], shifts[..., 0])
        width_map = np.where(is_sample, widths[..., 1], widths[..., 0])
        return is_sample.astype(float), shift_map, 'DHO (2-peak + amplitude mask): shift (GHz)', width_map
    return method


def dho_bgsub_method(image):
    """
    'Measure the buffer, subtract it, fit what's left.' Estimate the constant
    background spectrum from the third of pixels with the lowest Stokes-window
    intensity (least likely to carry the added component) and subtract it. The
    segmentation feature is the integrated positive residual in the Stokes window
    - ~0 where there is no added component, the component's area where there is -
    which needs no fit and cannot be thrown off by a degenerate one. A single DHO
    is still fitted to the residual, only to provide the component-shift map.
    """
    data = np.ma.filled(image.spectral_data, np.nan).astype(float)
    axis = np.asarray(image.spectral_axis)
    win = (axis >= STOKES_WINDOW[0]) & (axis <= STOKES_WINDOW[1])

    stokes_sum = np.nansum(data[..., win], axis=-1).reshape(-1)
    flat_data = data.reshape(-1, data.shape[-1])
    background = _common_background(image)
    residual = data - background

    bright = flat_data[stokes_sum >= np.nanpercentile(stokes_sum, 66)].mean(axis=0) - background
    s0 = abs(float(axis[win][np.nanargmax(bright[win])]))

    residual_area = np.nansum(np.clip(residual[..., win], 0.0, None), axis=-1)

    span = float(np.nanmax(np.abs(residual))) or 1.0
    lo = [0.0, 3.0, 0.0, -span, -2.0]
    hi = [np.inf, 14.0, 5.0, span, 2.0]
    p0 = [max(float(np.nanmax(residual[..., win])), span * 1e-3), s0, 0.8, 0.0, 0.0]
    params, _ = bp.analysis.fit.DHO(expected_peaks=1, p0=p0, bounds=(lo, hi)).apply(
        bp.SpectralImage(residual, axis))
    params = np.asarray(params)

    return _segment_on_feature(
        residual_area, image.shape, np.abs(params[..., 1]),
        'DHO on bg-subtracted residual: shift (GHz)', drop_failed=False, width_map=np.abs(params[..., 2]))


def dho_fit_method(image):
    return dho_fit(image, expected_peaks=1)


def make_dho_fit_2peak_method(anchor_shift):
    """A 2-peak DHO method with peak 0 pinned near ``anchor_shift`` (the known
    background peak). Returns a closure suitable for ``run_sweep``'s methods dict."""
    def dho_fit_2peak_method(image):
        return dho_fit(image, expected_peaks=2, anchor_shift=anchor_shift)
    return dho_fit_2peak_method


def raw_kmeans_method(image):
    projections, _ = bp.analysis.cluster.KMeans(n_clusters=2, n_init=10, random_state=0).apply(stokes_side(image))
    return _labels_from_maps(projections), None, None, None


def pca_method(image):
    # PCA scores are not abundances (they can be negative), so segment by k-means
    # on the first two principal-component score maps rather than argmax.
    projections, _ = bp.analysis.decompose.PCA(n_components=2, random_state=0).apply(stokes_side(image))
    scores = np.stack([np.asarray(p).reshape(-1) for p in projections], axis=1)
    labels = SkKMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(scores)
    return labels.reshape(image.shape), np.asarray(projections[0]), 'PC1 score (a.u.)', None


def nmf_method(image):
    # max_iter well above scikit-learn's default of 200: near-degenerate spectra
    # (small shift / low SNR) converge slowly, and a fair comparison should let
    # NMF actually reach its optimum rather than stop mid-descent.
    projections, _ = bp.analysis.decompose.NMF(
        n_components=2, init='nndsvda', random_state=0, max_iter=5000
    ).apply(stokes_side(image))
    return _labels_from_maps(projections), None, None, None


def vca_method(image):
    # VCA assumes at least one near-pure pixel per endmember. Needs the optional
    # 'pysptools' dependency (imported by brillouinpy.analysis.unmix); skipped
    # automatically by run_sweep if the import fails.
    abundances, _ = bp.analysis.unmix.VCA(n_endmembers=2, abundance_method='ucls').apply(stokes_side(image))
    return _labels_from_maps(abundances), None, None, None


METHODS = {
    'DHO fit (1-peak)': dho_fit_method,
    'Spectral phasor': phasor_method,
    'k-means (raw)': raw_kmeans_method,
    'PCA': pca_method,
    'NMF': nmf_method,
    'VCA': vca_method,
}


# ---------------------------------------------------------------------------
# Three-additive-component methods (``three_component.py`` / experiment D):
# medium (background) + cytoplasm on top + nucleus on top of both. Segmentation
# is a 3-way ARI (label-permutation invariant, so the specific label a
# classifier gives each class does not matter here); the classifiers below are
# the same algorithms as METHODS above, generalised to n_clusters/n_endmembers=3.
# ---------------------------------------------------------------------------

def _kmeans_ordered_mask(image, n_clusters):
    """k-means on the raw Stokes-window spectra, with cluster labels 0..n-1
    assigned in order of ascending total intensity. Meaningful (not just
    cosmetic) when classes are strictly nested by additive components -
    background < background+cytoplasm < background+cytoplasm+nucleus, since
    each extra component only ever adds signal - which is what
    :func:`dho_fit_segmented_3` relies on to know how many peaks each class
    needs."""
    cropped = stokes_side(image)
    flat = np.ma.filled(cropped.spectral_data, np.nan).astype(float).reshape(-1, cropped.spectral_data.shape[-1])
    raw = SkKMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit_predict(flat)
    totals = np.nansum(flat, axis=1)
    order = sorted(range(n_clusters), key=lambda c: totals[raw == c].mean() if np.any(raw == c) else np.inf)
    remap = np.zeros(n_clusters, dtype=int)
    for new, old in enumerate(order):
        remap[old] = new
    return remap[raw].reshape(image.shape)


def three_component_kmeans_method(image):
    labels = _kmeans_ordered_mask(image, 3)
    return labels.astype(float), None, None, None


def three_component_pca_method(image):
    projections, _ = bp.analysis.decompose.PCA(n_components=2, random_state=0).apply(stokes_side(image))
    scores = np.stack([np.asarray(p).reshape(-1) for p in projections], axis=1)
    labels = SkKMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(scores)
    return labels.reshape(image.shape).astype(float), np.asarray(projections[0]), 'PC1 score (a.u.)', None


def three_component_nmf_method(image):
    projections, _ = bp.analysis.decompose.NMF(
        n_components=3, init='nndsvda', random_state=0, max_iter=5000
    ).apply(stokes_side(image))
    return _labels_from_maps(projections).astype(float), None, None, None


def three_component_vca_method(image):
    abundances, _ = bp.analysis.unmix.VCA(n_endmembers=3, abundance_method='ucls').apply(stokes_side(image))
    return _labels_from_maps(abundances).astype(float), None, None, None


def three_component_phasor_method(image):
    ph = bp.analysis.phasor.phasor(image, axis_range=STOKES_WINDOW, background='min')
    labels = SkKMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(ph.features())
    return labels.astype(float).reshape(image.shape), np.asarray(bp.analysis.phasor.phase_to_shift(ph)), \
        'Phasor shift (GHz)', None


def three_component_phasor_bgsub_method(image):
    """Same as :func:`three_component_phasor_method`, with the estimated
    background (medium) spectrum subtracted first - see :func:`phasor_bgsub_method`."""
    ph = bp.analysis.phasor.phasor(image, axis_range=STOKES_WINDOW, background=_common_background(image))
    labels = SkKMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(ph.features())
    return labels.astype(float).reshape(image.shape), np.asarray(bp.analysis.phasor.phase_to_shift(ph)), \
        'Phasor shift (GHz), bg-subtracted', None


def dho_fit_segmented_3(image):
    """
    :func:`dho_fit_segmented` generalised to three additive components. First
    segments into three classes with :func:`_kmeans_ordered_mask` (background <
    cytoplasm-only < cytoplasm+nucleus, by total intensity), then fits exactly
    the physically correct number of DHO peaks in each: one in background-only
    pixels, two (background anchored) in cytoplasm-only pixels, three
    (background and cytoplasm both anchored) in nucleus pixels. Because peaks
    0 and 1 are pinned to the already-identified background/cytoplasm shifts,
    the fit's one remaining free peak (index 2) is always the newest
    component - no need to search for it by distance, unlike the unanchored
    case in :func:`dho_fit`.

    Returns the pre-segmentation as the label map and, as shift/width maps,
    each pixel's *newest* component's shift/linewidth (background's own, for
    background-only pixels; cytoplasm's, for cytoplasm-only pixels; nucleus's,
    for nucleus pixels) - for scoring against each class's true parameters.

    Like :func:`dho_fit_segmented`, this is now a thin wrapper over the shipped
    :func:`brillouinpy.analysis.fit.segmented_fit` (``n_components=3``), with the
    benchmark's Stokes-window k-means (:func:`_kmeans_ordered_mask`, ordered by
    ascending total intensity) passed in as the classifier so the label map
    matches the standalone ``k-means (raw)`` method.
    """
    result = bp.analysis.segmented_fit(
        image, n_components=3,
        classifier=lambda img, n_components: _kmeans_ordered_mask(img, n_components),
    )
    return (result.labels.astype(float), result.shift,
            'DHO (segmented, 3-comp): newest-component shift (GHz)', result.linewidth)


METHODS_D = {
    'DHO fit (segmented, 3-comp)': dho_fit_segmented_3,
    'Spectral phasor': three_component_phasor_method,
    'Spectral phasor (bg-subtracted)': three_component_phasor_bgsub_method,
    'k-means (raw)': three_component_kmeans_method,
    'PCA': three_component_pca_method,
    'NMF': three_component_nmf_method,
    'VCA': three_component_vca_method,
}


def run_sweep_d(cases, snapshot_value, methods=METHODS_D, *, background_truth, cytoplasm_truth, nucleus_linewidth):
    """
    Like :func:`run_sweep`, specialised for the three-additive-component case:

    - ``cases``: ``(nucleus_shift, image, label_map)`` per sweep point,
      ``label_map`` in ``{0, 1, 2}`` (background/cytoplasm/nucleus).
    - Scoring is a single 3-way ARI (adjusted Rand index is label-count
      agnostic and permutation-invariant, so no special-casing is needed there).
    - Parameter recovery is computed *per component*: for background and
      cytoplasm (whose true (shift, linewidth) are fixed - ``background_truth``/
      ``cytoplasm_truth`` tuples) and for nucleus (true shift = the sweep value,
      true linewidth = ``nucleus_linewidth``, fixed). Each is the median
      |fitted - true| over that component's *true* pixels, from whichever
      method exposes a physical shift/width map (only
      ``dho_fit_segmented_3`` here).

    Returns
    -------
    dict with ``values``, ``scores``, ``timings``, ``snapshot``,
    ``snapshot_truth`` (as :func:`run_sweep`) and ``recovery``: name ->
    ``{'background_shift': [...], 'background_width': [...], 'cytoplasm_shift':
    [...], 'cytoplasm_width': [...], 'nucleus_shift': [...], 'nucleus_width':
    [...]}``, each a list of median-|error| or None per sweep point.
    """
    keys = ['background_shift', 'background_width', 'cytoplasm_shift',
            'cytoplasm_width', 'nucleus_shift', 'nucleus_width']
    values = [v for v, _, _ in cases]
    scores = {name: [] for name in methods}
    timings = {name: [] for name in methods}
    recovery = {name: {k: [] for k in keys} for name in methods}
    snapshot, snapshot_truth = {}, None

    for value, image, label_map in cases:
        is_snapshot = np.isclose(value, snapshot_value)
        if is_snapshot:
            snapshot_truth = label_map
        label_map = np.asarray(label_map)
        truths = {
            'background': (background_truth[0], background_truth[1], label_map == 0),
            'cytoplasm': (cytoplasm_truth[0], cytoplasm_truth[1], label_map == 1),
            'nucleus': (value, nucleus_linewidth, label_map == 2),
        }
        print(f"{value:g}")
        for name, fn in methods.items():
            try:
                start = time.perf_counter()
                pred_labels, shift_map, shift_label, width_map = fn(image)
                timings[name].append(time.perf_counter() - start)
            except Exception as exc:
                scores[name].append(np.nan)
                for k in keys:
                    recovery[name][k].append(None)
                print(f"  {name:<28} skipped ({type(exc).__name__}: {exc})")
                continue
            score = ari(pred_labels, label_map)
            scores[name].append(score)
            is_shift = shift_map is not None and shift_label and 'GHz' in shift_label
            for comp, (true_shift, true_width, mask) in truths.items():
                shift_err = _recovery_error(shift_map, true_shift, mask) if is_shift else None
                width_err = _recovery_error(width_map, true_width, mask)
                recovery[name][f'{comp}_shift'].append(shift_err)
                recovery[name][f'{comp}_width'].append(width_err)
            print(f"  {name:<28} ARI = {score:.3f}")
            if is_snapshot:
                snapshot[name] = (pred_labels, shift_map, shift_label, width_map, score)

    return dict(values=values, scores=scores, timings=timings, recovery=recovery,
               snapshot=snapshot, snapshot_truth=snapshot_truth)


def make_figures_d(result, *, xlabel, title, snapshot_value, snapshot_label):
    """
    Build the three standard figures for experiment D (:func:`run_sweep_d`):

    1. 3-way ARI vs. the swept nucleus shift, plus per-method cost.
    2. Predicted 3-class regions vs. ground truth at ``snapshot_value`` (raw
       label maps - no cross-method colour alignment is attempted for 3
       classes, unlike the binary :func:`align_labels`, since ARI itself
       already does not care about label identity).
    3. The continuous shift maps (only methods with one) at ``snapshot_value``.
    """
    values = np.asarray(result['values'])
    scores, timings = result['scores'], result['timings']
    snapshot, snapshot_truth = result['snapshot'], result['snapshot_truth']

    fig1, (ax_ari, ax_time) = plt.subplots(1, 2, figsize=(13, 5.4), layout='constrained')
    for name, series in scores.items():
        if not np.any(np.isfinite(series)):
            continue
        ax_ari.plot(values, series, marker='o', label=name)
    ax_ari.axvline(snapshot_value, color='grey', ls=':', lw=1)
    ax_ari.axhline(0, color='grey', lw=0.5)
    ax_ari.set_xlabel(xlabel)
    ax_ari.set_ylabel('Adjusted Rand index vs. ground truth (3 classes)\n(1 = perfect, 0 = chance level)')
    ax_ari.set_title(title, pad=12)
    ax_ari.set_ylim(-0.1, 1.05)
    ax_ari.legend()

    ran = [n for n in timings if np.any(np.isfinite(scores[n])) and timings[n]]
    samples = {n: (timings[n][1:] if len(timings[n]) > 2 else timings[n]) for n in ran}
    means = np.array([np.mean(samples[n]) for n in ran])
    sems = np.array([
        np.std(samples[n], ddof=1) / np.sqrt(len(samples[n])) if len(samples[n]) > 1 else 0.0
        for n in ran
    ])
    ax_time.bar(ran, means, yerr=sems, capsize=4, error_kw=dict(lw=1))
    ax_time.set_yscale('log')
    ax_time.set_ylabel('Wall time per image (s), steady state\nmean ± standard error, log scale')
    ax_time.set_xlabel(f'n = {len(values)} images per method (first dropped as warm-up)')
    ax_time.set_title('Computational cost per method')
    ax_time.tick_params(axis='x', rotation=20)

    names = [name for name in scores if name in snapshot]
    fig2, axes2 = plt.subplots(1, len(names) + 1, figsize=(4 * (len(names) + 1), 4), layout='constrained')
    fig2.suptitle(f'Predicted regions (raw labels, uncoloured to truth) at {snapshot_label}')
    axes2[0].imshow(snapshot_truth, cmap='viridis', vmin=0, vmax=2)
    axes2[0].set_title('Ground truth\n(0=background, 1=cytoplasm, 2=nucleus)')
    for ax, name in zip(axes2[1:], names):
        pred, _, _, _, score = snapshot[name]
        ax.imshow(pred, cmap='viridis')
        ax.set_title(f'{name}\nARI = {score:.2f}')
    for ax in axes2:
        ax.set_xticks([])
        ax.set_yticks([])

    param_methods = [(name, snapshot[name]) for name in names if snapshot[name][1] is not None]
    fig3, axes3 = plt.subplots(1, len(param_methods), figsize=(5.5 * len(param_methods), 4.5), layout='constrained')
    fig3.suptitle(f'Continuous shift maps at {snapshot_label}')
    axes3 = np.atleast_1d(axes3)
    for ax, (name, (_, shift_map, shift_label, _, _)) in zip(axes3, param_methods):
        vmin, vmax = np.nanpercentile(shift_map, [2, 98])
        im = ax.imshow(shift_map, vmin=vmin, vmax=vmax)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        fig3.colorbar(im, ax=ax, label=shift_label)

    return fig1, fig2, fig3


def make_recovery_figure_d(result, *, xlabel, component, quantity, title):
    """
    One recovery line per method for a single component/quantity from a
    :func:`run_sweep_d` result, e.g. ``component='nucleus', quantity='shift'``.
    Only ``dho_fit_segmented_3`` is expected to have data here (the
    classifiers expose no physical shift/linewidth); it is still a per-method
    loop so a future method with a physical output shows up automatically.
    """
    key = f'{component}_{quantity}'
    unit = 'GHz'
    values = np.asarray(result['values'])
    fig, ax = plt.subplots(figsize=(8, 5.5), layout='constrained')
    for name, series_by_key in result['recovery'].items():
        series = np.asarray([np.nan if v is None else v for v in series_by_key[key]], dtype=float)
        if not np.any(np.isfinite(series)):
            continue
        ax.plot(values, series, marker='o', label=name)
    ax.axhline(0, color='grey', lw=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f'Median |fitted - true| {component} {quantity} ({unit})\nover true {component} pixels')
    ax.set_title(title, pad=12)
    ax.legend()
    return fig


def _recovery_error(value_map, truth, mask):
    """Median absolute error between ``value_map`` and the scalar ``truth``,
    restricted to ``mask`` (a boolean array). ``None`` if nothing to compare."""
    if value_map is None:
        return None
    vals = np.asarray(value_map)[mask]
    vals = vals[np.isfinite(vals)]
    return float(np.median(np.abs(vals - truth))) if vals.size else None


def run_sweep(cases, snapshot_value, methods=METHODS, shift_recovery=False, true_width=None):
    """
    Run every method on every case and score it against ground truth.

    Parameters
    ----------
    cases : list[tuple[float, SpectralImage, ndarray]]
        ``(parameter_value, image, label_map)`` per sweep point. When
        ``shift_recovery`` is set, ``parameter_value`` doubles as the true
        physical shift a method's shift map should recover.
    snapshot_value : float
        The ``parameter_value`` whose predicted maps are kept for
        :func:`make_figures` (matched with ``numpy.isclose``).
    methods : dict[str, callable]
        Overrides the default :data:`METHODS`.
    shift_recovery : bool
        If set, also compute, for every sweep point and every method whose
        shift map is in physical units (its label contains "GHz"), the median
        absolute error between that map (restricted to the *true* sample
        pixels - ground truth, so this isolates shift accuracy from
        segmentation accuracy) and ``parameter_value``. Stored under the
        returned dict's ``recovery`` key (``None`` where a method has no
        shift-valued map, e.g. k-means/PCA/NMF/VCA).
    true_width : float, optional
        If given, the true linewidth (GHz) of the swept component - held fixed
        across the sweep. When set, also computes a ``width_recovery`` series
        the same way as ``recovery``, from each method's width map (unlabelled,
        since there is only one physical quantity it could be).

    Returns
    -------
    dict with keys ``values``, ``scores`` (name -> list of ARI), ``timings``
    (name -> list of per-image seconds), ``snapshot`` (name -> (aligned_labels,
    shift_map, shift_label, width_map, ari)), ``snapshot_truth`` and, if
    ``shift_recovery``/``true_width``, ``recovery``/``width_recovery`` (name ->
    list of median-|error| or None).
    """
    values = [value for value, _, _ in cases]
    scores = {name: [] for name in methods}
    timings = {name: [] for name in methods}
    recovery = {name: [] for name in methods} if shift_recovery else None
    width_recovery = {name: [] for name in methods} if true_width is not None else None
    snapshot, snapshot_truth = {}, None

    for value, image, label_map in cases:
        is_snapshot = np.isclose(value, snapshot_value)
        if is_snapshot:
            snapshot_truth = label_map
        print(f"{value:g}")
        for name, fn in methods.items():
            try:
                start = time.perf_counter()
                pred_labels, param_map, param_label, width_map = fn(image)
                timings[name].append(time.perf_counter() - start)
            except Exception as exc:  # e.g. VCA without the optional pysptools dependency
                scores[name].append(np.nan)
                if shift_recovery:
                    recovery[name].append(None)
                if true_width is not None:
                    width_recovery[name].append(None)
                print(f"  {name:<18} skipped ({type(exc).__name__}: {exc})")
                continue
            score = ari(pred_labels, label_map)
            scores[name].append(score)
            extra = ''
            sample = np.asarray(label_map).astype(bool)
            if shift_recovery:
                # Only a shift map labelled as a GHz shift is comparable to the
                # true shift - e.g. PCA's PC1 score is a useful map (see figure 3)
                # but is not in the same units and must not be scored here.
                is_shift = param_map is not None and param_label and 'GHz' in param_label
                error = _recovery_error(param_map, value, sample) if is_shift else None
                recovery[name].append(error)
                extra += f", shift err = {error:.3f} GHz" if error is not None else ""
            if true_width is not None:
                werror = _recovery_error(width_map, true_width, sample)
                width_recovery[name].append(werror)
                extra += f", width err = {werror:.3f} GHz" if werror is not None else ""
            print(f"  {name:<18} ARI = {score:.3f}{extra}")
            if is_snapshot:
                snapshot[name] = (align_labels(pred_labels, label_map), param_map, param_label, width_map, score)

    result = dict(values=values, scores=scores, timings=timings,
                 snapshot=snapshot, snapshot_truth=snapshot_truth)
    if shift_recovery:
        result['recovery'] = recovery
    if true_width is not None:
        result['width_recovery'] = width_recovery
    return result


def make_figures(result, *, xlabel, title, snapshot_value, snapshot_label,
                 xscale='linear', ylabel='Adjusted Rand index vs. ground truth\n(1 = perfect, 0 = chance level)'):
    """
    Build the three standard benchmark figures from a :func:`run_sweep` result:

    1. ARI vs. the swept parameter, plus per-method computational cost
       (mean +/- standard error of the wall time per image, log y-axis).
    2. Predicted 2-region segmentation vs. ground truth at ``snapshot_value``.
    3. The continuous parameter maps (DHO shift, phasor shift, PC1) at
       ``snapshot_value``.

    ``plt.show()`` is left to the caller.
    """
    values = np.asarray(result['values'])
    scores, timings = result['scores'], result['timings']
    snapshot, snapshot_truth = result['snapshot'], result['snapshot_truth']

    # --- Figure 1a: accuracy vs. the swept parameter ---
    fig1, (ax_ari, ax_time) = plt.subplots(1, 2, figsize=(13, 5.4), layout='constrained')
    for name, series in scores.items():
        if not np.any(np.isfinite(series)):
            continue  # skipped at every sweep point (e.g. VCA without pysptools)
        ax_ari.plot(values, series, marker='o', label=name)
    ax_ari.axvline(snapshot_value, color='grey', ls=':', lw=1)
    ax_ari.axhline(0, color='grey', lw=0.5)
    ax_ari.set_xscale(xscale)
    if xscale == 'log':
        # Readable decade-ish ticks (1, 2, 5 x 10^n) rather than one label per
        # (awkwardly spaced) sweep point; the markers still show every point.
        ax_ari.xaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax_ari.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax_ari.xaxis.set_minor_locator(mticker.NullLocator())
    ax_ari.set_xlabel(xlabel)
    ax_ari.set_ylabel(ylabel)
    ax_ari.set_title(title, pad=12)
    ax_ari.set_ylim(-0.1, 1.05)
    ax_ari.legend()

    # --- Figure 1b: computational cost per method (mean +/- SEM, log y) ---
    # Drop each method's first timing sample: the cold process-pool spawn / import
    # warmup dominates it and is not representative of steady-state cost.
    ran = [n for n in timings if np.any(np.isfinite(scores[n])) and timings[n]]
    samples = {n: (timings[n][1:] if len(timings[n]) > 2 else timings[n]) for n in ran}
    means = np.array([np.mean(samples[n]) for n in ran])
    sems = np.array([
        np.std(samples[n], ddof=1) / np.sqrt(len(samples[n])) if len(samples[n]) > 1 else 0.0
        for n in ran
    ])
    ax_time.bar(ran, means, yerr=sems, capsize=4, error_kw=dict(lw=1))
    ax_time.set_yscale('log')
    ax_time.set_ylabel('Wall time per image (s), steady state\nmean ± standard error, log scale')
    ax_time.set_xlabel(f'n = {len(values)} images per method (first dropped as warm-up)')
    ax_time.set_title('Computational cost per method')
    ax_time.tick_params(axis='x', rotation=20)

    # --- Figure 2 ---
    names = [name for name in scores if name in snapshot]
    fig2, axes2 = plt.subplots(1, len(names) + 1, figsize=(4 * (len(names) + 1), 4), layout='constrained')
    fig2.suptitle(f'Predicted regions at {snapshot_label}')
    axes2[0].imshow(snapshot_truth, cmap='coolwarm', vmin=0, vmax=1)
    axes2[0].set_title('Ground truth')
    for ax, name in zip(axes2[1:], names):
        aligned, _, _, _, score = snapshot[name]
        ax.imshow(aligned, cmap='coolwarm', vmin=0, vmax=1)
        ax.set_title(f'{name}\nARI = {score:.2f}')
    for ax in axes2:
        ax.set_xticks([])
        ax.set_yticks([])

    # --- Figure 3 ---
    param_methods = [(name, snapshot[name]) for name in names if snapshot[name][1] is not None]
    fig3, axes3 = plt.subplots(1, len(param_methods), figsize=(5.5 * len(param_methods), 4.5), layout='constrained')
    fig3.suptitle(f'Continuous parameter maps at {snapshot_label}')
    axes3 = np.atleast_1d(axes3)
    for ax, (name, (_, param_map, param_label, _, _)) in zip(axes3, param_methods):
        vmin, vmax = np.nanpercentile(param_map, [2, 98])
        im = ax.imshow(param_map, vmin=vmin, vmax=vmax)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        fig3.colorbar(im, ax=ax, label=param_label)

    return fig1, fig2, fig3


def make_recovery_figure(result, *, xlabel, title, key='recovery',
                         ylabel='Median |fitted shift - true shift| (GHz)\nover true sample pixels, per sweep point'):
    """
    Build a parameter-recovery figure from a :func:`run_sweep` result computed
    with ``shift_recovery=True`` (``key='recovery'``, the default) or
    ``true_width=...`` (``key='width_recovery'``): one line per method whose
    map is in physical units (methods with no such map, e.g. k-means/PCA/NMF/
    VCA, are skipped). Lower is better; this is the accuracy question ("how
    close is the recovered parameter to the truth?"), independent of whether a
    method also segments the field correctly (see the accompanying ARI figure
    for that).
    """
    values = np.asarray(result['values'])
    fig, ax = plt.subplots(figsize=(8, 5.5), layout='constrained')
    for name, series in result[key].items():
        series = np.asarray([np.nan if v is None else v for v in series], dtype=float)
        if not np.any(np.isfinite(series)):
            continue
        ax.plot(values, series, marker='o', label=name)
    ax.axhline(0, color='grey', lw=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=12)
    ax.legend()
    return fig


def save_figures(figs, save_dir, prefix, dpi=120, suffixes=None):
    """Save a tuple of figures under ``save_dir`` as ``<prefix>__<suffix>.png``.
    ``suffixes`` defaults to the three :func:`make_figures` panels
    (``accuracy_and_cost``/``snapshot_segmentation``/``snapshot_parameter_maps``);
    pass a matching-length list to save other figures (e.g. from
    :func:`make_recovery_figure`). Returns the paths."""
    if suffixes is None:
        suffixes = ['accuracy_and_cost', 'snapshot_segmentation', 'snapshot_parameter_maps']
    os.makedirs(save_dir, exist_ok=True)
    paths = []
    for fig, suffix in zip(figs, suffixes):
        path = os.path.join(save_dir, f'{prefix}__{suffix}.png')
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        paths.append(path)
    return paths
