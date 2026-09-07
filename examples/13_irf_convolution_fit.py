# -*- coding: utf-8 -*-
"""
Example 13 - Fitting with the instrument response function (IRF) in the model
============================================================================

A measured Brillouin spectrum is the *intrinsic* lineshape convolved with the
spectrometer's instrument response function. The fitted linewidth from a plain
DHO fit (``09_fit_spectra.py``) therefore includes the instrumental broadening
- it is not the mechanically meaningful intrinsic linewidth.

The usual fix is to deconvolve the data first
(``brillouinpy.preprocessing.misc.Deconvoluter_IRF``, Richardson-Lucy). This
example shows the alternative: keep the data and fit the *convolved* model
``(DHO * IRF) + background`` by passing ``irf=`` to
:class:`brillouinpy.analysis.fitmodel.DHO`. The fitted ``LineWidth`` is then the
intrinsic linewidth, with the instrumental broadening in the forward model - no
deconvolution, no iteration count, and stable at low photon counts. See
``benchmarks/irf_convolution.py`` for the full comparison.

Two ways to get the IRF to the fit:

1. **From the data** - a tandem Fabry-Perot records the elastic peak on every
   scan. ``IRF_Remover(store_irf=True)`` crops it away *and* stashes it on the
   object as a per-pixel ``instrument_response_function``; ``DHO(irf='auto')``
   then picks it up.
2. **Measured separately** (e.g. a VIPA IRF taken once, or a few times for drift
   correction) - attach it with
   ``brillouinpy.preprocessing.misc.assign_irf(image, irf_measurements, ...)``.

``irf=`` also accepts an explicit array or a parametric
``('lorentzian'|'gaussian', fwhm)`` / ``('voigt', fwhm_l, fwhm_g)`` spec (GHz).
"""
import matplotlib.pyplot as plt
import numpy as np

import brillouinpy as bp
from _synthetic_data import irf_broadened_image

if __name__ == '__main__':
    # An intrinsic doublet (0.9 GHz linewidth outside a blob, 0.45 GHz inside),
    # convolved with a 0.6 GHz Gaussian IRF, with a measurable elastic peak.
    image, intrinsic_linewidth_map, info = irf_broadened_image(
        nx=30, ny=30, component_shift=6.5,
        intrinsic_linewidth_background=0.9, intrinsic_linewidth_blob=0.45,
        irf_fwhm=0.6, irf_shape='gaussian', elastic_amplitude=1.5,
        noise_model='poisson', peak_photons=400,
    )

    p0 = [3e-3, 6.5, 0.8, 1e-4, 0.0]
    bounds = ([0, 5, 0.05, -1e-2, -1], [1, 8, 5, 1e-1, 1])

    # (1) Detect + crop the elastic peak, keeping it as the per-pixel IRF.
    pp = bp.preprocessing.misc.IRF_Remover(offset=3, store_irf=True).apply(image)
    print("per-pixel IRF attached:", pp.instrument_response_function.shape)

    # (a) Plain fit on the cropped (still IRF-broadened) data - overestimates linewidth.
    plain_params, _ = bp.analysis.fitmodel.DHO(expected_peaks=1, p0=p0, bounds=bounds).apply(pp)

    # (b) Fit the convolved model, IRF taken from the object.
    auto_params, _ = bp.analysis.fitmodel.DHO(
        expected_peaks=1, p0=p0, bounds=bounds, irf='auto',
    ).apply(pp)

    # (c) For reference: the exact parametric IRF (the best case).
    param_params, _ = bp.analysis.fitmodel.DHO(
        expected_peaks=1, p0=p0, bounds=bounds, irf=('gaussian', info['irf_fwhm']),
    ).apply(pp)

    plain_lw = np.abs(plain_params[..., 2])
    auto_lw = np.abs(auto_params[..., 2])
    param_lw = np.abs(param_params[..., 2])
    bg = intrinsic_linewidth_map > 0.7
    print("true intrinsic linewidth:  bg 0.90, blob 0.45 GHz")
    for name, lw in (("plain fit", plain_lw), ("irf='auto'", auto_lw), ("irf=parametric", param_lw)):
        print(f"  {name:16s}  bg {np.nanmedian(lw[bg]):.3f}   blob {np.nanmedian(lw[~bg]):.3f}")

    # (2) If the IRF was measured separately instead:
    #     irf_scan = bp.io.prepare_brillouin_data(irf_path, 'Brillouin').squeeze()
    #     pp = bp.preprocessing.misc.assign_irf(pp, irf_scan.mean)          # measured once
    #     pp = bp.preprocessing.misc.assign_irf(pp, [irf_before, irf_after],  # drift-corrected
    #                                           at=[0, pp.flat.shape[0] - 1], method='linear')

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
    for ax, data, title in (
        (axes[0], intrinsic_linewidth_map, 'True intrinsic linewidth (GHz)'),
        (axes[1], plain_lw, 'Plain DHO fit linewidth (GHz)'),
        (axes[2], auto_lw, "DHO(irf='auto') linewidth (GHz)"),
    ):
        im = ax.imshow(data, vmin=0.3, vmax=1.3)
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
    plt.show()
