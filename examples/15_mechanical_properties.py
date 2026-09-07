# -*- coding: utf-8 -*-
"""
Example 15 - Mechanical quantities from a fit
============================================

A Brillouin fit gives a shift and a linewidth per pixel (``09_fit_spectra.py``).
``analysis.mechanics`` turns those into physical quantities, grouped by how much
you need to know about the sample:

* **loss tangent** ``tan(delta) = linewidth / shift`` - needs *nothing* else; it
  is the ratio of the loss and storage moduli directly.
* **hypersound velocity** - needs the refractive index, laser wavelength and
  scattering angle.
* **storage / loss modulus, longitudinal viscosity** - additionally need the
  density.

The refractive index, density, wavelength and scattering angle are **required**
(no defaults): a Brillouin measurement does not contain them, and a wrong guess
would give a plausible-looking but wrong number.
"""
import matplotlib.pyplot as plt
import numpy as np

import brillouinpy as bp
from _synthetic_data import single_peak_image

if __name__ == '__main__':
    image = single_peak_image(nx=25, ny=25, freq_shift=7.5, linewidth=0.7)

    fit = bp.analysis.fitmodel.DHO(expected_peaks=1, p0=[5e-3, 7.5, 0.7, 0, 0], bounds=None)
    params, covariances = fit.apply(image)

    # Everything computable, in one call. loss_tangent always; the rest because
    # the material constants are supplied; uncertainties because covariances is.
    props = bp.analysis.mechanics.from_dho_fit(
        params, covariances,
        refractive_index=1.35,     # e.g. a hydrated biological sample
        density=1050.0,            # kg/m^3
        wavelength=532e-9,         # m
        scattering_angle=180.0,    # backscattering
    )

    print(f"loss tangent:          {np.nanmedian(props.loss_tangent):.4f}")
    print(f"hypersound velocity:   {np.nanmedian(props.hypersound_velocity):.0f} m/s")
    print(f"storage modulus M':    {np.nanmedian(props.storage_modulus) / 1e9:.3f} GPa "
          f"+- {np.nanmedian(props.storage_modulus_uncertainty) / 1e9:.3f}")
    print(f"loss modulus M'':      {np.nanmedian(props.loss_modulus) / 1e6:.1f} MPa")
    print(f"longitudinal viscosity: {np.nanmedian(props.longitudinal_viscosity) * 1e3:.2f} mPa*s")

    # Only the loss tangent, straight from shift + linewidth, no constants:
    lt = bp.analysis.mechanics.loss_tangent(params[..., 1], params[..., 2])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    for ax, m, t in ((axes[0], lt, 'Loss tangent'),
                     (axes[1], props.storage_modulus / 1e9, "Storage modulus M' (GPa)"),
                     (axes[2], props.longitudinal_viscosity * 1e3, 'Longitudinal viscosity (mPa*s)')):
        im = ax.imshow(m)
        ax.set_title(t)
        fig.colorbar(im, ax=ax)
    plt.show()
