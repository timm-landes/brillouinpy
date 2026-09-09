# -*- coding: utf-8 -*-
"""
Monte Carlo reproduction of Elsayad (2019, Front. Phys. 7:62) Fig. 3 and Fig. 5
================================================================================

Experiments A-D ask a *spatial segmentation* question: given an image made of
several regions, does a method draw the right boundary. This script asks a
different, non-spatial question, the one Elsayad's Fig. 3/5 actually ask: for a
single, PURE, one-peak spectrum, how much does a method's parameter estimate
scatter from repeated independent noisy measurements of the *same* spectrum -
and does that scatter let two classes stay distinguishable under noise where it
otherwise wouldn't. No image, no segmentation, no additive mixing: each
"sample" here is one full spectrum with its own noise draw.

- **Fig. 3 reproduction**: three of Elsayad's own pure single-peak spectra -
  water (7.45 GHz, 0.80 GHz linewidth), cytoplasm (7.70 GHz, 1.00 GHz), nucleus
  (8.10 GHz, 1.10 GHz) - each repeated ``N_REPS`` times at one moderate noise
  level. Plots the phasor (G, S) cloud against the fitted (shift, linewidth)
  cloud, one colour per class, to see whether the phasor keeps the three
  classes visually separated where the fit's point cloud starts to blur
  together.
- **Fig. 5 reproduction**: two very close peaks, 7.8 vs. 7.9 GHz (Elsayad's own
  pair, 0.1 GHz apart) - a two-sample test (Mann-Whitney U) between their
  estimated shifts, swept over noise level, comparing the phasor against TWO
  fit variants: the same informatively-bounded single-DHO fit used throughout
  this benchmark suite (``_assay._fit_dho_raw``, p0 estimated from the data,
  *not* the true value - Elsayad's own comparison starts the fit at the exact
  ground truth, which this deliberately does not), and a naive fit with
  wide-open bounds and a generic initial guess. This separates "does fitting
  lose to the phasor" from "does an *uninformed* fit lose to the phasor",
  which experiments A-D already suggest are different questions.

A first version of this script swept the Poisson shot-noise photon budget for
Fig. 5, which is not what Elsayad's Fig. 5 actually sweeps: shot noise is his
*baseline* there; the swept quantity is the strength of an **additional,
additive, signal-independent** noise term (his "detector noise" / "1/f noise"
scenarios) layered on top of it - a fundamentally different kind of
degradation from shot noise (which scales with the signal) and one a gradient-
based fit is plausibly far more sensitive to than a Fourier-projection method,
independent of how well-posed the underlying peak-finding problem is. This
version adds that term (see :func:`make_realizations`'s ``detector_noise_w``)
and sweeps it instead, while deliberately keeping this benchmark suite's own
choices elsewhere: the DHO line shape (not Elsayad's Lorentzian), this
suite's own finer spectral resolution (not his ~150 MHz/67-channel axis), and
a data-estimated (not ground-truth) fit initial guess. Elsayad's own
normalisation for the noise-strength parameter is not fully recoverable from
the paper text, so the ``detector_noise_w`` scale here (a fraction of the
clean peak height) is this script's own calibration, not a literal unit
match - read the resulting curves as "does the qualitative claim hold under
this kind of noise", not a numerical reproduction of his exact values.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit

import brillouinpy as bp

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples'))
from _assay import _fit_dho_raw  # noqa: E402  (path set up above)
from _synthetic_data import _add_noise, _dho  # noqa: E402

N_CHANNELS = 250
N_REPS = 300
NOISE_MODEL = 'poisson'
STOKES_WINDOW = (4.0, 9.5)
AMPLITUDE = 5e-3

# The additive "detector noise" term (see module docstring) is not i.i.d. per
# channel: Elsayad describes it as normally distributed with a width of 1 GHz,
# which only makes sense as a *correlation length* along the frequency axis,
# not a per-channel amplitude - i.e. a slowly undulating baseline wobble, not
# per-channel jitter. Generated as Gaussian-smoothed white noise below.
DETECTOR_NOISE_CORR_GHZ = 1.0

# A representative "shot-noise-limited" photon budget: high enough that shot
# noise alone is a mild baseline (matching Elsayad's Panel B), so that sweeping
# the detector-noise term afterwards isolates its own effect.
PEAK_PHOTONS_BASE = 400

# Elsayad (2019) Fig. 3's own parameter choices.
COMPONENTS = {
    'water':     (7.45, 0.80),
    'cytoplasm': (7.70, 1.00),
    'nucleus':   (8.10, 1.10),
}
FIG3_PEAK_PHOTONS = 120  # one moderate/hard shot-noise-only level (top row)
FIG3_DETECTOR_W = 0.30   # representative "harsh" detector-noise level (bottom row), on top of PEAK_PHOTONS_BASE
FIG3_SEED = 10

# Elsayad (2019) Fig. 5's close pair - linewidth not restated there; using a
# value in the middle of the three components' range above.
CLOSE_SHIFT_A, CLOSE_SHIFT_B = 7.8, 7.9
CLOSE_LINEWIDTH = 0.9
DETECTOR_NOISE_SWEEP = np.array([0.0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.7])
FIG5_SEED = 20


def _axis():
    return bp.utils.brillouin_spectral_axis(mirror_spacing=6e-3, scan_amplitude=480e-9,
                                            no_of_channels=N_CHANNELS)


def _smooth_noise(rng, shape, axis, corr_width_ghz=DETECTOR_NOISE_CORR_GHZ):
    """A smooth, spectrally-correlated unit-scale noise field: white noise
    low-pass filtered along the channel axis with a Gaussian kernel of width
    ``corr_width_ghz`` - a slowly wobbling baseline distortion, not per-channel
    jitter. Renormalised to unit std regardless of the smoothing width so the
    caller's scale factor has a stable meaning."""
    channel_spacing = float(np.mean(np.diff(axis)))
    sigma_channels = corr_width_ghz / channel_spacing
    white = rng.normal(size=shape)
    smooth = gaussian_filter1d(white, sigma=sigma_channels, axis=-1)
    return smooth / smooth.std()


def make_realizations(shift, linewidth, amplitude, n_reps, peak_photons, seed, detector_noise_w=0.0):
    """``n_reps`` independent noisy realizations of one pure DHO spectrum, as an
    ``(n_reps, 1)``-pixel :class:`~brillouinpy.SpectralImage` so the rest of
    brillouinpy's per-pixel API (phasor, DHO fit) applies unchanged.

    ``detector_noise_w``, if nonzero, adds a smooth, spectrally-correlated
    baseline distortion (see :func:`_smooth_noise`) on top of the Poisson shot
    noise, scaled to ``detector_noise_w`` times the clean peak height - the
    signal-independent noise term Elsayad's Fig. 5 actually sweeps (see the
    module docstring)."""
    axis = _axis()
    clean = _dho(axis, amplitude, shift, linewidth, background=1e-4)
    rng = np.random.default_rng(seed)
    data = np.tile(clean, (n_reps, 1))
    photon_scale = peak_photons / clean.max()
    noisy = _add_noise(data, rng, noise_model=NOISE_MODEL, noise=0.0, photon_scale=photon_scale)
    if detector_noise_w:
        noisy = np.clip(noisy + detector_noise_w * clean.max() * _smooth_noise(rng, noisy.shape, axis), 0, None)
    return bp.SpectralImage(noisy.reshape(n_reps, 1, N_CHANNELS), axis)


def phasor_estimates(image):
    ph = bp.analysis.phasor.phasor(image, axis_range=STOKES_WINDOW, background='min')
    shift = np.asarray(bp.analysis.phasor.phase_to_shift(ph)).ravel()
    modulus = np.asarray(ph.modulus).ravel()
    return shift, modulus


def bounded_fit_estimates(image):
    """The informatively-bounded, informed-p0 single-DHO fit used throughout
    this benchmark suite (shift bounded to a realistic spectrometer range, p0
    from :func:`brillouinpy.analysis.fit.core.estimate_p0`)."""
    _, shifts, widths = _fit_dho_raw(image, expected_peaks=1)
    return shifts[..., 0].ravel(), widths[..., 0].ravel()


def naive_fit_estimates(image):
    """A deliberately *uninformed* fit: wide-open bounds (0 to the full axis
    range), a generic p0 that does not know roughly where the peak sits or how
    sharp it should be. This is the condition Elsayad's Fig. 5 argument about
    fitting losing to the phasor under noise implicitly assumes; the rest of
    this benchmark suite never fits this way."""
    axis = np.asarray(image.spectral_axis)
    data = np.asarray(image.spectral_data)[:, 0, :]
    n = data.shape[0]
    shifts = np.full(n, np.nan)
    widths = np.full(n, np.nan)
    lo = [0.0, 0.0, 0.0, -np.inf, -2.0]
    hi = [np.inf, float(axis.max()), 10.0, np.inf, 2.0]
    for i in range(n):
        y = data[i]
        shift_guess = abs(float(axis[np.argmax(y)]))
        p0 = np.clip([max(y.max() - np.median(y), 1e-6), shift_guess, 1.0, max(np.median(y), 0.0), 0.0], lo, hi)
        try:
            popt, _ = curve_fit(bp.analysis.fit.core._DHO_1, axis, y, p0=list(p0), bounds=(lo, hi), maxfev=5000)
            shifts[i] = abs(popt[1])
            widths[i] = abs(popt[2])
        except Exception:
            pass
    return shifts, widths


# --------------------------------------------------------------------------- #
# Fig. 3 reproduction: three-class parameter-estimate clouds at fixed noise
# --------------------------------------------------------------------------- #
def _fig3_row(axes, peak_photons, detector_noise_w, seed_offset):
    colors = {'water': 'tab:blue', 'cytoplasm': 'tab:orange', 'nucleus': 'tab:green'}
    ax_phasor, ax_fit, ax_naive = axes
    row_data = {}
    for i, (name, (shift, lw)) in enumerate(COMPONENTS.items()):
        image = make_realizations(shift, lw, AMPLITUDE, N_REPS, peak_photons,
                                  seed=seed_offset + i, detector_noise_w=detector_noise_w)
        ph_shift, ph_mod = phasor_estimates(image)
        fit_shift, fit_width = bounded_fit_estimates(image)
        naive_shift, naive_width = naive_fit_estimates(image)
        row_data[name] = dict(ph_shift=ph_shift, ph_mod=ph_mod, fit_shift=fit_shift, fit_width=fit_width,
                              naive_shift=naive_shift, naive_width=naive_width, true_shift=shift, true_lw=lw)
        ax_phasor.scatter(ph_shift, ph_mod, s=8, alpha=0.4, color=colors[name], label=name)
        ax_fit.scatter(fit_shift, fit_width, s=8, alpha=0.4, color=colors[name], label=name)
        ax_naive.scatter(naive_shift, naive_width, s=8, alpha=0.4, color=colors[name], label=name)
    for ax, ylabel in ((ax_phasor, 'Phasor modulus'), (ax_fit, 'Fitted linewidth (GHz)'),
                       (ax_naive, 'Fitted linewidth (GHz)')):
        ax.set_xlabel('Estimated shift (GHz)')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
    return row_data


def run_fig3(save_dir=None):
    """Two rows: shot-noise-only (Elsayad's Panel B) vs. shot noise **plus** a
    representative smooth detector-noise wobble (his Panels C-E) - reproducing
    his staged comparison directly, rather than only one noise condition."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), layout='constrained')
    row1 = _fig3_row(axes[0], FIG3_PEAK_PHOTONS, 0.0, FIG3_SEED)
    row2 = _fig3_row(axes[1], PEAK_PHOTONS_BASE, FIG3_DETECTOR_W, FIG3_SEED + 100)

    for ax, title in zip(axes[0], ('Spectral phasor', 'Bounded DHO fit (data-estimated p0)',
                                   'Naive DHO fit (wide-open, generic p0)')):
        ax.set_title(title)
    axes[0, 0].annotate(f'Shot noise only\n(peak ≈ {FIG3_PEAK_PHOTONS:g} photons, '
                        f'SNR ≈ {FIG3_PEAK_PHOTONS ** 0.5:.0f})',
                        xy=(0, 0.5), xytext=(-axes[0, 0].yaxis.labelpad - 45, 0), xycoords='axes fraction',
                        textcoords='offset points', size=10, ha='right', va='center', rotation=90)
    axes[1, 0].annotate(f'Shot (peak ≈ {PEAK_PHOTONS_BASE:g} ph.) + smooth\ndetector noise, W = {FIG3_DETECTOR_W:g}',
                        xy=(0, 0.5), xytext=(-axes[1, 0].yaxis.labelpad - 45, 0), xycoords='axes fraction',
                        textcoords='offset points', size=10, ha='right', va='center', rotation=90)
    fig.suptitle(f'Reproduction of Elsayad (2019) Fig. 3 - {N_REPS} noisy realizations per class per row')

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, 'phasor_paper_reproduction__fig3_clouds.png'),
                   dpi=120, bbox_inches='tight')
    return dict(shot_only=row1, shot_plus_detector=row2), fig


# --------------------------------------------------------------------------- #
# Fig. 5 reproduction: two-sample separation of a 0.1 GHz-close pair vs. noise
# --------------------------------------------------------------------------- #
def run_fig5(save_dir=None):
    """Sweeps the smooth detector-noise strength ``W`` at a fixed, mildly
    shot-noise-limited baseline (``PEAK_PHOTONS_BASE``) - this is the axis
    Elsayad's Fig. 5 actually sweeps (see the module docstring), not the
    shot-noise level itself."""
    p_phasor, p_bounded, p_naive = [], [], []
    for j, w in enumerate(DETECTOR_NOISE_SWEEP):
        img_a = make_realizations(CLOSE_SHIFT_A, CLOSE_LINEWIDTH, AMPLITUDE, N_REPS, PEAK_PHOTONS_BASE,
                                  seed=FIG5_SEED + 2 * j, detector_noise_w=w)
        img_b = make_realizations(CLOSE_SHIFT_B, CLOSE_LINEWIDTH, AMPLITUDE, N_REPS, PEAK_PHOTONS_BASE,
                                  seed=FIG5_SEED + 2 * j + 1, detector_noise_w=w)

        ph_a, _ = phasor_estimates(img_a)
        ph_b, _ = phasor_estimates(img_b)
        p_phasor.append(stats.mannwhitneyu(ph_a, ph_b, alternative='two-sided').pvalue)

        fit_a, _ = bounded_fit_estimates(img_a)
        fit_b, _ = bounded_fit_estimates(img_b)
        p_bounded.append(stats.mannwhitneyu(fit_a, fit_b, alternative='two-sided').pvalue)

        naive_a, _ = naive_fit_estimates(img_a)
        naive_b, _ = naive_fit_estimates(img_b)
        ok_a, ok_b = np.isfinite(naive_a), np.isfinite(naive_b)
        p_naive.append(stats.mannwhitneyu(naive_a[ok_a], naive_b[ok_b], alternative='two-sided').pvalue
                       if ok_a.any() and ok_b.any() else np.nan)
        print(f'detector_noise_w={w:5.2f}  p(phasor)={p_phasor[-1]:.2e}  '
             f'p(bounded fit)={p_bounded[-1]:.2e}  p(naive fit)={p_naive[-1]:.2e}')

    fig, ax = plt.subplots(figsize=(7.5, 5.4), layout='constrained')
    ax.plot(DETECTOR_NOISE_SWEEP, p_phasor, marker='o', label='Spectral phasor')
    ax.plot(DETECTOR_NOISE_SWEEP, p_bounded, marker='o', label='DHO fit (bounded, data-estimated p0)')
    ax.plot(DETECTOR_NOISE_SWEEP, p_naive, marker='o', label='DHO fit (naive, wide-open)')
    ax.axhline(0.05, color='grey', ls=':', lw=1, label='p = 0.05')
    ax.set_yscale('log')
    ax.invert_yaxis()
    ax.set_xlabel(f'Smooth detector-noise strength W (fraction of peak height, '
                 f'{DETECTOR_NOISE_CORR_GHZ:g} GHz correlation length)\n'
                 f'shot noise fixed at peak ≈ {PEAK_PHOTONS_BASE:g} photons')
    ax.set_ylabel('Mann-Whitney p-value, shift estimates\n(7.8 GHz vs. 7.9 GHz, lower = better separated)')
    ax.set_title(f'Reproduction of Elsayad (2019) Fig. 5 - {N_REPS} realizations/class/point')
    ax.legend()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, 'phasor_paper_reproduction__fig5_pvalue.png'),
                   dpi=120, bbox_inches='tight')
    return dict(detector_noise_w=DETECTOR_NOISE_SWEEP, p_phasor=p_phasor,
               p_bounded=p_bounded, p_naive=p_naive), fig


def run(save_dir=None):
    print('=== Fig. 3 reproduction (three-class parameter clouds) ===')
    fig3_data, fig3 = run_fig3(save_dir=save_dir)
    print('=== Fig. 5 reproduction (close-pair separation vs. noise) ===')
    fig5_data, fig5 = run_fig5(save_dir=save_dir)
    return dict(fig3=fig3_data, fig5=fig5_data)


if __name__ == '__main__':
    run()
    plt.show()
