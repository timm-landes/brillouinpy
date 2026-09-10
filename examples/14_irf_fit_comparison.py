# -*- coding: utf-8 -*-
"""
Example 14 - Plain fit vs. IRF-convolution fit
==============================================

Compares the two ways of getting the *intrinsic* Brillouin linewidth out of an
IRF-broadened spectrum, on synthetic data (known ground truth) and, if you have
it, on the real measurement from ``01_load_data.py``:

* **plain fit** - ``DHO(expected_peaks=1)`` straight on the (elastic-peak
  cropped) data. Its ``LineWidth`` is the *observed* width: intrinsic +
  instrumental broadening.
* **IRF-convolution fit** - ``DHO(expected_peaks=1, irf='auto')`` on the same
  data, with the elastic peak detected and kept as the per-pixel
  ``instrument_response_function`` by ``IRF_Remover(store_irf=True)``. Its
  ``LineWidth`` is the intrinsic width, with the instrumental broadening in the
  forward model.

See ``docs/tutorial/irf-fit-comparison.md`` for the discussion and
``benchmarks/irf_convolution.py`` for a fuller synthetic sweep.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

import brillouinpy as bp
from _synthetic_data import irf_broadened_image

# Point this at the same measurement folder as 01_load_data.py to run the
# comparison on real data as well; otherwise only the synthetic part runs.
PROJECT_PATH = r'C:\Users\Timm\Desktop\Leipzig\Zygo_2_4'


def _fit(image, p0, bounds, irf=None):
    return np.asarray(bp.analysis.fit.DHO(
        expected_peaks=1, p0=p0, bounds=bounds, **({} if irf is None else {'irf': irf}),
    ).apply(image)[0])


def _prepare(image, denoise=False):
    """Detect + crop the elastic peak (keeping it as the per-pixel IRF). Optionally
    denoise afterwards - useful for noisy real data, skip it for clean synthetic
    data where SavGol would slightly narrow the lines and bias the comparison."""
    cropped = bp.preprocessing.misc.IRF_Remover(offset=6, store_irf=True).apply(image)
    if denoise:
        cropped = bp.preprocessing.denoise.SavGol(window_length=9, polyorder=3).apply(cropped)
    return cropped


def _irf_fwhm(irf, axis):
    dx = float(np.mean(np.diff(axis)))
    out = []
    for row in irf.reshape(-1, irf.shape[-1]):
        row = row - row.min()
        if row.max() <= 0:
            continue
        above = np.flatnonzero(row >= row.max() / 2)
        out.append((above[-1] - above[0]) * dx)
    return float(np.median(out))


if __name__ == '__main__':
    # ------------------------------------------------------------------ #
    # Synthetic: intrinsic 0.9 GHz linewidth, broadened by a 0.8 GHz IRF #
    # ------------------------------------------------------------------ #
    irf_fwhm = 0.8
    image, true_lw_map, info = irf_broadened_image(
        nx=40, ny=40, component_shift=6.5, irf_fwhm=irf_fwhm, irf_shape='gaussian',
        intrinsic_linewidth_background=0.9, intrinsic_linewidth_blob=0.45,
        elastic_amplitude=1.5, noise_model='poisson', peak_photons=500,
    )
    p0 = [3e-3, 6.5, 1.0, 1e-4, 0.0]
    bounds = ([0, 5, 0.05, -1e-2, -1], [1, 8, 5, 1e-1, 1])
    prepared = _prepare(image, denoise=False)
    plain = _fit(prepared, p0, bounds)
    param = _fit(prepared, p0, bounds, irf=('gaussian', irf_fwhm))  # calibrated IRF
    auto = _fit(prepared, p0, bounds, irf='auto')                   # IRF detected from the elastic peak

    bg = true_lw_map > 0.7
    print(f'SYNTHETIC (Gaussian IRF FWHM {irf_fwhm} GHz; true intrinsic linewidth: bg 0.90, blob 0.45 GHz)')
    print(f"  detected IRF FWHM: {_irf_fwhm(prepared.instrument_response_function, image.spectral_axis):.3f} GHz")
    for name, arr in (('plain fit', plain), (f"irf=('gaussian', {irf_fwhm})", param), ("irf='auto'", auto)):
        lw = np.abs(arr[..., 2])
        print(f"  {name:26s}  bg {np.nanmedian(lw[bg]):.3f}   blob {np.nanmedian(lw[~bg]):.3f}")

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), layout='constrained')
    for ax, m, t in ((axes[0], true_lw_map, 'True intrinsic'),
                     (axes[1], np.abs(plain[..., 2]), 'Plain fit'),
                     (axes[2], np.abs(param[..., 2]), f"irf=('gaussian', {irf_fwhm})"),
                     (axes[3], np.abs(auto[..., 2]), "irf='auto' (detected)")):
        im = ax.imshow(m, vmin=0.3, vmax=1.4)
        ax.set_title(f'{t}\nlinewidth (GHz)')
        fig.colorbar(im, ax=ax)
    fig.suptitle('Synthetic: the plain fit keeps the instrumental broadening; irf= removes it')
    plt.show()

    # ------------------------------------------------------------------ #
    # Real data from 01_load_data.py (skipped if not present)            #
    # ------------------------------------------------------------------ #
    real = None
    pkl = os.path.join('pp_data', 'brillouin_image.pkl')
    try:
        data = bp.io.legacy.load_spectral_image(PROJECT_PATH, 'Brillouin')
        axis = bp.utils.brillouin_spectral_axis(mirror_spacing=6e-3, scan_amplitude=480e-9,
                                                no_of_channels=data.shape[-1])
        real = bp.SpectralImage(data, axis)
    except (FileNotFoundError, NotADirectoryError):
        if os.path.exists(pkl):
            real = bp.SpectralImage.load(pkl)

    if real is None:
        print('\nNo real measurement found - run 01_load_data.py first, or set PROJECT_PATH.')
    else:
        rp0 = [float(np.nanmax(real.spectral_data)) * 0.1, 8.0, 1.2, 20.0, 0.0]
        rb = ([0, 5, 0.1, -1e4, -1], [1e7, 12, 6, 1e4, 1])
        rprepared = _prepare(real, denoise=True)  # real data is noisy - denoise before fitting
        rplain = _fit(rprepared, rp0, rb)
        rauto = _fit(rprepared, rp0, rb, irf='auto')
        sh = np.abs(rauto[..., 1])
        ok = np.isfinite(sh) & (sh > 5) & (sh < 12)
        lw_p, lw_c = np.abs(rplain[..., 2]), np.abs(rauto[..., 2])
        irf_fwhm = _irf_fwhm(rprepared.instrument_response_function, real.spectral_axis)
        print(f'\nREAL DATA {real.shape}  (no ground truth)')
        print(f"  detected IRF FWHM: {irf_fwhm:.3f} GHz")
        print(f"  shift:      plain {np.nanmedian(np.abs(rplain[..., 1])[ok]):.3f}   "
              f"irf='auto' {np.nanmedian(sh[ok]):.3f} GHz  (agree - the IRF is symmetric)")
        print(f"  linewidth:  plain {np.nanmedian(lw_p[ok]):.3f}   irf='auto' {np.nanmedian(lw_c[ok]):.3f} GHz  "
              f"(-{np.nanmedian(lw_p[ok] - lw_c[ok]):.3f}: small, the IRF is narrow next to the "
              f"~{np.nanmedian(lw_p[ok]):.1f} GHz linewidth)")

        fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
        axes[0].plot(real.spectral_axis, rprepared.instrument_response_function.reshape(
            -1, real.spectral_length)[::41].T, alpha=0.4)
        axes[0].set_xlim(-3, 3)
        axes[0].set_title(f'Detected per-pixel IRF (FWHM {irf_fwhm:.2f} GHz)')
        axes[0].set_xlabel('Brillouin shift (GHz)')
        for ax, m, t in ((axes[1], lw_p, 'Plain fit linewidth'), (axes[2], lw_c, "irf='auto' linewidth")):
            im = ax.imshow(np.where(ok, m, np.nan))
            ax.set_title(f'{t} (GHz)')
            fig.colorbar(im, ax=ax)
        fig.suptitle('Real data: same shift, a small linewidth correction (narrow IRF)')
        plt.show()
