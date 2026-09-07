# -*- coding: utf-8 -*-
"""
IRF handling: deconvolve-then-fit vs. fit-the-convolved-model (experiment E)
===========================================================================

Study script, not a tutorial - see ``benchmarks/README.md``.

Every real Brillouin spectrum is the intrinsic lineshape convolved with the
spectrometer's instrument response function (IRF). To get the *intrinsic*
linewidth (the mechanically meaningful quantity) the instrumental broadening
has to be removed. Two ways:

- **Deconvolve the data, then fit** (``brillouinpy.preprocessing.misc.
  Deconvoluter_IRF``): Richardson-Lucy deconvolution with the measured elastic
  peak as the point-spread function, then a plain DHO fit. RL is iterative,
  non-linear and amplifies high-frequency noise.
- **Fit the convolved model** (``brillouinpy.analysis.fitmodel.DHO(irf=...)``,
  added alongside this script): leave the data untouched and fit
  ``(DHO * IRF) + background``, so the fitted ``LineWidth`` is the intrinsic
  linewidth with the instrumental broadening in the forward model.

This script compares, on synthetic doublets with a known intrinsic linewidth
convolved with a known IRF (``examples/_synthetic_data.irf_broadened_image``):

- ``naive``       - plain 1-peak DHO fit on the broadened data (no correction).
- ``rl_deconv``   - Richardson-Lucy (true IRF as PSF, ``RL_ITERS`` iterations),
                    then a plain DHO fit.
- ``conv_fit``    - ``DHO(irf=<true IRF kernel>)`` on the broadened data.

Two sweeps: (1) IRF width relative to the intrinsic linewidth, at fixed SNR;
(2) photon budget (SNR), at a fixed IRF/linewidth ratio. Metric: median
absolute error between the fitted and the true intrinsic linewidth over all
pixels (both blob and background regions, scored against the ground-truth
linewidth map). Shift error and wall time are reported alongside.
"""
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from skimage.restoration import richardson_lucy

import brillouinpy as bp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples'))
from _synthetic_data import irf_broadened_image  # noqa: E402

INTRINSIC_LINEWIDTH_BG = 0.9      # GHz, outside the blob
INTRINSIC_LINEWIDTH_BLOB = 0.45   # GHz, inside the blob
COMPONENT_SHIFT = 6.5             # GHz
NX = NY = 22
N_CHANNELS = 300
RL_ITERS = 15

IRF_RATIOS = np.array([0.15, 0.3, 0.5, 0.8, 1.1, 1.5, 2.0])   # irf_fwhm / mean intrinsic linewidth
RATIO_FOR_SNR_SWEEP = 0.8
PEAK_PHOTONS_SWEEP = np.array([60, 120, 250, 500, 1000, 2000])
PHOTONS_FOR_RATIO_SWEEP = 500

_MEAN_INTRINSIC = 0.5 * (INTRINSIC_LINEWIDTH_BG + INTRINSIC_LINEWIDTH_BLOB)


def _p0_bounds():
    p0 = [3e-3, COMPONENT_SHIFT, 0.8, 1e-4, 0.0]
    lo = [0.0, COMPONENT_SHIFT - 1.5, 0.05, -1e-2, -1.0]
    hi = [1.0, COMPONENT_SHIFT + 1.5, 5.0, 1e-1, 1.0]
    return p0, (lo, hi)


def _fit_plain(image):
    p0, bounds = _p0_bounds()
    params, _ = bp.analysis.fitmodel.DHO(expected_peaks=1, p0=p0, bounds=bounds).apply(image)
    return np.asarray(params)


def _fit_conv(image, kernel):
    p0, bounds = _p0_bounds()
    params, _ = bp.analysis.fitmodel.DHO(expected_peaks=1, p0=p0, bounds=bounds, irf=kernel).apply(image)
    return np.asarray(params)


def _rl_deconvolve(image, kernel):
    data = np.ma.filled(image.spectral_data, np.nan).astype(float)
    out = np.empty_like(data)
    kernel = kernel / kernel.sum()
    for idx in np.ndindex(data.shape[:-1]):
        spec = data[idx]
        scale = np.nanmax(spec) or 1.0
        rl = richardson_lucy(np.clip(spec / scale, 0, 1), kernel, num_iter=RL_ITERS, clip=False)
        out[idx] = rl * scale
    return bp.SpectralImage(out, image.spectral_axis)


def _errors(params, linewidth_map):
    shift = np.abs(params[..., 1])
    width = np.abs(params[..., 2])
    residual = width - linewidth_map
    lw_err = float(np.nanmedian(np.abs(residual)))            # headline: median abs error
    lw_bias = float(np.abs(np.nanmedian(residual)))           # systematic offset
    lw_scatter = float(1.4826 * np.nanmedian(np.abs(residual - np.nanmedian(residual))))
    shift_err = float(np.nanmedian(np.abs(shift - COMPONENT_SHIFT)))
    return dict(linewidth_error=lw_err, linewidth_bias=lw_bias,
                linewidth_scatter=lw_scatter, shift_error=shift_err)


def _one_case(irf_fwhm, peak_photons, seed):
    image, linewidth_map, info = irf_broadened_image(
        nx=NX, ny=NY, n_channels=N_CHANNELS, component_shift=COMPONENT_SHIFT,
        intrinsic_linewidth_background=INTRINSIC_LINEWIDTH_BG,
        intrinsic_linewidth_blob=INTRINSIC_LINEWIDTH_BLOB,
        irf_fwhm=irf_fwhm, irf_shape='lorentzian',
        noise_model='poisson', peak_photons=peak_photons, seed=seed,
    )
    kernel = info['irf_kernel']

    out = {}
    for name, fn in (
        ('naive', lambda: _fit_plain(image)),
        ('rl_deconv', lambda: _fit_plain(_rl_deconvolve(image, kernel))),
        ('conv_fit', lambda: _fit_conv(image, kernel)),
    ):
        t0 = time.perf_counter()
        params = fn()
        dt = time.perf_counter() - t0
        rec = _errors(params, linewidth_map)
        rec['seconds'] = dt
        out[name] = rec
    return out


def run(save_dir=None):
    methods = ['naive', 'rl_deconv', 'conv_fit']

    ratio_res = {m: [] for m in methods}
    for i, ratio in enumerate(IRF_RATIOS):
        res = _one_case(ratio * _MEAN_INTRINSIC, PHOTONS_FOR_RATIO_SWEEP, seed=10 + i)
        for m in methods:
            ratio_res[m].append(res[m])
        print(f'ratio {ratio:.2f}: ' + ', '.join(
            f'{m} dLW={res[m]["linewidth_error"]:.3f}' for m in methods))

    snr_res = {m: [] for m in methods}
    for i, photons in enumerate(PEAK_PHOTONS_SWEEP):
        res = _one_case(RATIO_FOR_SNR_SWEEP * _MEAN_INTRINSIC, int(photons), seed=40 + i)
        for m in methods:
            snr_res[m].append(res[m])
        print(f'photons {photons}: ' + ', '.join(
            f'{m} dLW={res[m]["linewidth_error"]:.3f}' for m in methods))

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5), layout='constrained')
    for m in methods:
        axes[0].plot(IRF_RATIOS, [r['linewidth_error'] for r in ratio_res[m]], 'o-', label=m)
    axes[0].set_xlabel('IRF FWHM / mean intrinsic linewidth')
    axes[0].set_ylabel('Median |fitted - true intrinsic linewidth| (GHz)')
    axes[0].set_title(f'Linewidth recovery vs. IRF width\n(peak ≈ {PHOTONS_FOR_RATIO_SWEEP} photons)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    snr = np.sqrt(PEAK_PHOTONS_SWEEP)
    for m in methods:
        axes[1].plot(snr, [r['linewidth_error'] for r in snr_res[m]], 'o-', label=m)
    axes[1].set_xlabel('Shot-noise SNR at the peak (≈ sqrt photons)')
    axes[1].set_ylabel('Median |fitted - true intrinsic linewidth| (GHz)')
    axes[1].set_title(f'Linewidth recovery vs. SNR\n(IRF / linewidth = {RATIO_FOR_SNR_SWEEP})')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    for m in methods:
        axes[2].plot(IRF_RATIOS, [r['shift_error'] for r in ratio_res[m]], 'o-', label=m)
    axes[2].set_xlabel('IRF FWHM / mean intrinsic linewidth')
    axes[2].set_ylabel('Median |fitted - true shift| (GHz)')
    axes[2].set_title('Shift recovery vs. IRF width\n(symmetric IRF: should not bias the shift)')
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, 'irf_convolution__recovery.png'), dpi=110)

    return dict(
        ratios=IRF_RATIOS.tolist(), photons=PEAK_PHOTONS_SWEEP.tolist(),
        ratio_sweep={m: ratio_res[m] for m in methods},
        snr_sweep={m: snr_res[m] for m in methods},
    )


if __name__ == '__main__':
    result = run(save_dir=os.path.join(os.path.dirname(__file__), 'report', 'figures'))

    for sweep, label in ((result['ratio_sweep'], 'IRF/linewidth sweep'),
                         (result['snr_sweep'], 'SNR sweep')):
        for key in ('linewidth_error', 'linewidth_bias', 'linewidth_scatter'):
            print(f'\n=== {label} - {key} (GHz) ===')
            for m in ('naive', 'rl_deconv', 'conv_fit'):
                print(f'{m:10s}', [round(r[key], 3) for r in sweep[m]])
    print('\n=== wall time per image (s), IRF/linewidth sweep ===')
    for m in ('naive', 'rl_deconv', 'conv_fit'):
        print(f'{m:10s}', [round(r["seconds"], 2) for r in result['ratio_sweep'][m]])
