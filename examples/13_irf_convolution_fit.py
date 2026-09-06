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
example shows the alternative: leave the data alone and fit the *convolved*
model ``(DHO * IRF) + background`` by passing ``irf=`` to
:class:`brillouinpy.analysis.fitmodel.DHO`. The fitted ``LineWidth`` is then the
intrinsic linewidth, with the instrumental broadening accounted for in the
forward model - no deconvolution, no iteration count to tune, and stable at low
photon counts. See ``benchmarks/irf_convolution.py`` for the full comparison.

``irf=`` accepts either a measured 1-D IRF sample (e.g. the elastic peak) on the
spectrum's channel spacing, or a parametric ``('lorentzian'|'gaussian', fwhm)``
/ ``('voigt', fwhm_l, fwhm_g)`` spec in GHz.
"""
import matplotlib.pyplot as plt
import numpy as np

import brillouinpy as bp
from _synthetic_data import irf_broadened_image

if __name__ == '__main__':
    # An intrinsic doublet (0.9 GHz linewidth outside a blob, 0.45 GHz inside)
    # convolved with a 0.5 GHz Lorentzian IRF.
    image, intrinsic_linewidth_map, info = irf_broadened_image(
        nx=30, ny=30, component_shift=6.5,
        intrinsic_linewidth_background=0.9, intrinsic_linewidth_blob=0.45,
        irf_fwhm=0.5, irf_shape='lorentzian',
        noise_model='poisson', peak_photons=300,
    )

    p0 = [3e-3, 6.5, 0.8, 1e-4, 0.0]
    bounds = ([0, 5, 0.05, -1e-2, -1], [1, 8, 5, 1e-1, 1])

    # (a) Plain fit on the raw (IRF-broadened) data - overestimates the linewidth.
    plain_params, _ = bp.analysis.fitmodel.DHO(expected_peaks=1, p0=p0, bounds=bounds).apply(image)

    # (b) Fit the convolved model. Here with the known parametric IRF; in practice
    #     you would extract the elastic / reference-beam peak from the measurement
    #     (blank its channels with IRF_Remover so they stay out of the residual) and
    #     pass that 1-D sample as irf=, or build the kernel explicitly:
    #         kernel = bp.analysis.fitmodel.irf_kernel(elastic_peak_samples, image.spectral_axis)
    #         DHO(..., irf=kernel)
    conv_params, _ = bp.analysis.fitmodel.DHO(
        expected_peaks=1, p0=p0, bounds=bounds, irf=('lorentzian', info['irf_fwhm']),
    ).apply(image)

    plain_lw = np.abs(plain_params[..., 2])
    conv_lw = np.abs(conv_params[..., 2])
    print(f"true intrinsic linewidth:      {np.unique(intrinsic_linewidth_map.round(2))} GHz")
    print(f"plain fit, median linewidth:   {np.nanmedian(plain_lw):.3f} GHz  (broadened)")
    print(f"irf= fit,  median linewidth:   {np.nanmedian(conv_lw):.3f} GHz  (intrinsic)")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
    vmin, vmax = 0.3, 1.3
    for ax, data, title in (
        (axes[0], intrinsic_linewidth_map, 'True intrinsic linewidth (GHz)'),
        (axes[1], plain_lw, 'Plain DHO fit linewidth (GHz)'),
        (axes[2], conv_lw, 'DHO(irf=...) fit linewidth (GHz)'),
    ):
        im = ax.imshow(data, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
    plt.show()
