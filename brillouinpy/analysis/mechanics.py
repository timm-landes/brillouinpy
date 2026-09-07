# -*- coding: utf-8 -*-
"""
Mechanical quantities derived from a fitted Brillouin peak.

Every function here turns the output of a :class:`~brillouinpy.analysis.fitmodel.DHO`
fit - a Brillouin shift and linewidth - into a physical quantity. They are
grouped by how much you have to know about the sample and the setup:

* :func:`loss_tangent` needs **nothing** beyond the fit itself: ``tan(delta) =
  linewidth / shift`` is the ratio of the loss and storage moduli directly, and
  is independent of refractive index, density and scattering geometry.
* :func:`hypersound_velocity` needs the **refractive index**, laser
  **wavelength** and **scattering angle**.
* :func:`storage_modulus`, :func:`loss_modulus` and
  :func:`longitudinal_viscosity` additionally need the **density**.

The refractive index, density, wavelength and scattering angle are **required**
keyword arguments with no defaults - a Brillouin measurement does not contain
them, and a silently-wrong assumption (a backscattering geometry, a
water-like refractive index) would produce a plausible-looking but wrong number.

All shift/linewidth inputs are in **GHz** (the fit's own units); ``wavelength``
is in **metres**, ``scattering_angle`` in **degrees**, ``density`` in
**kg/m^3**. Results: velocity in m/s, moduli in Pa, viscosity in Pa*s.

``linewidth`` is expected to be the :class:`~brillouinpy.analysis.fitmodel.DHO`
fit's ``LineWidth``, i.e. the **half width at half maximum** (HWHM). Where a
physical linewidth enters (loss tangent, loss modulus, viscosity) it is
converted to the FWHM (``Gamma = 2 * LineWidth``) internally, so
``loss_tangent = Gamma / shift = tan(delta)`` directly.

Optional first-order uncertainty propagation: pass ``shift_uncertainty`` /
``linewidth_uncertainty`` (e.g. the square root of the fit covariance's
diagonal) and the function returns ``(value, uncertainty)`` instead of just
``value``. The material constants are treated as exact.
"""
from dataclasses import dataclass

import numpy as np

__all__ = [
    "loss_tangent", "hypersound_velocity", "storage_modulus", "loss_modulus",
    "longitudinal_viscosity", "from_dho_fit", "MechanicalProperties",
]


_HWHM_TO_FWHM = 2.0


def _q_factor(refractive_index, wavelength, scattering_angle):
    """``lambda / (2 n sin(theta/2))`` - the shift-to-velocity conversion factor
    (metres), i.e. ``V = shift_in_Hz * this``."""
    theta = np.deg2rad(scattering_angle)
    return wavelength / (2.0 * refractive_index * np.sin(theta / 2.0))


def _maybe_with_uncertainty(value, rel_variance_terms):
    """Return ``value``, or ``(value, sqrt(sum(terms)) * |value|)`` if any term
    is not None. ``rel_variance_terms`` are the ``(sigma_x / x)`` contributions."""
    terms = [t for t in rel_variance_terms if t is not None]
    if not terms:
        return value
    rel = np.sqrt(sum(np.square(t) for t in terms))
    return value, np.abs(value) * rel


def loss_tangent(shift, linewidth, *, shift_uncertainty=None, linewidth_uncertainty=None):
    """
    ``tan(delta) = Gamma / shift`` with ``Gamma = 2 * linewidth`` the FWHM - the
    loss tangent of the longitudinal modulus (``M'' / M'``) for a
    damped-harmonic-oscillator lineshape. ``linewidth`` is the DHO fit's
    ``LineWidth`` (HWHM).

    Needs no material constants or scattering geometry: the shift and linewidth
    scale identically with all of them, so their ratio is invariant. This is the
    one mechanical quantity a bare Brillouin fit gives you unambiguously.
    """
    shift = np.abs(np.asarray(shift, dtype=float))
    fwhm = _HWHM_TO_FWHM * np.abs(np.asarray(linewidth, dtype=float))
    value = fwhm / shift
    return _maybe_with_uncertainty(value, [
        None if linewidth_uncertainty is None else _HWHM_TO_FWHM * np.asarray(linewidth_uncertainty) / fwhm,
        None if shift_uncertainty is None else np.asarray(shift_uncertainty) / shift,
    ])


def hypersound_velocity(shift, *, refractive_index, wavelength, scattering_angle,
                        shift_uncertainty=None):
    """
    Longitudinal (hypersound) phase velocity ``V = shift * lambda / (2 n sin(theta/2))``,
    in m/s. Needs the refractive index, laser wavelength and scattering angle;
    not the density.
    """
    shift = np.abs(np.asarray(shift, dtype=float))
    value = shift * 1e9 * _q_factor(refractive_index, wavelength, scattering_angle)
    return _maybe_with_uncertainty(value, [
        None if shift_uncertainty is None else np.asarray(shift_uncertainty) / shift,
    ])


def storage_modulus(shift, *, density, refractive_index, wavelength, scattering_angle,
                    shift_uncertainty=None):
    """
    Longitudinal storage modulus ``M' = rho * V^2`` (real part of the complex
    longitudinal modulus), in Pa.
    """
    shift = np.abs(np.asarray(shift, dtype=float))
    v = shift * 1e9 * _q_factor(refractive_index, wavelength, scattering_angle)
    value = density * v ** 2
    return _maybe_with_uncertainty(value, [
        None if shift_uncertainty is None else 2.0 * np.asarray(shift_uncertainty) / shift,
    ])


def loss_modulus(shift, linewidth, *, density, refractive_index, wavelength, scattering_angle,
                 shift_uncertainty=None, linewidth_uncertainty=None):
    """
    Longitudinal loss modulus ``M'' = M' * tan(delta) = rho * V^2 * Gamma / shift``
    with ``Gamma = 2 * linewidth`` the FWHM (imaginary part of the complex
    longitudinal modulus), in Pa. ``linewidth`` is the DHO fit's ``LineWidth`` (HWHM).
    """
    shift = np.abs(np.asarray(shift, dtype=float))
    fwhm = _HWHM_TO_FWHM * np.abs(np.asarray(linewidth, dtype=float))
    v = shift * 1e9 * _q_factor(refractive_index, wavelength, scattering_angle)
    value = density * v ** 2 * fwhm / shift
    return _maybe_with_uncertainty(value, [
        None if shift_uncertainty is None else np.asarray(shift_uncertainty) / shift,
        None if linewidth_uncertainty is None else _HWHM_TO_FWHM * np.asarray(linewidth_uncertainty) / fwhm,
    ])


def longitudinal_viscosity(linewidth, *, density, refractive_index, wavelength, scattering_angle,
                           linewidth_uncertainty=None):
    """
    Apparent longitudinal (bulk + shear) viscosity
    ``eta_L = rho * Gamma * lambda^2 / (8 pi n^2 sin^2(theta/2))`` with
    ``Gamma = 2 * linewidth`` the FWHM, in Pa*s. ``linewidth`` is the DHO fit's
    ``LineWidth`` (HWHM).

    From ``eta_L = rho * Gamma_omega / q^2`` with the scattering wavevector
    ``q = 4 pi n sin(theta/2) / lambda``; for fixed material constants and
    geometry it depends only on the linewidth (the shift dependence cancels), so
    the shift is not an argument here.
    """
    fwhm = _HWHM_TO_FWHM * np.abs(np.asarray(linewidth, dtype=float))
    k = _q_factor(refractive_index, wavelength, scattering_angle)
    value = density * k ** 2 * (fwhm * 1e9) / (2.0 * np.pi)
    return _maybe_with_uncertainty(value, [
        None if linewidth_uncertainty is None else _HWHM_TO_FWHM * np.asarray(linewidth_uncertainty) / fwhm,
    ])


@dataclass
class MechanicalProperties:
    """
    Bundle of mechanical quantities from :func:`from_dho_fit`. ``loss_tangent`` is
    always present; the rest are ``None`` unless the needed material constants
    were supplied. Each ``*_uncertainty`` is ``None`` unless ``covariances`` was
    supplied.
    """
    loss_tangent: np.ndarray
    loss_tangent_uncertainty: np.ndarray = None
    hypersound_velocity: np.ndarray = None
    hypersound_velocity_uncertainty: np.ndarray = None
    storage_modulus: np.ndarray = None
    storage_modulus_uncertainty: np.ndarray = None
    loss_modulus: np.ndarray = None
    loss_modulus_uncertainty: np.ndarray = None
    longitudinal_viscosity: np.ndarray = None
    longitudinal_viscosity_uncertainty: np.ndarray = None


def from_dho_fit(fit_parameters, covariances=None, *, peak=0,
                 refractive_index=None, density=None, wavelength=None, scattering_angle=None):
    """
    Compute every available mechanical quantity from a
    :meth:`brillouinpy.analysis.fitmodel.DHO.apply` result.

    Parameters
    ----------
    fit_parameters : numpy.ndarray
        The fitted parameters, shape ``(..., expected_peaks * 3 + 2)`` with the
        ``[I0, freqShift, LineWidth, ...]`` layout the DHO fit returns.
    covariances : numpy.ndarray, optional
        The matching covariance array, shape ``(..., n_params, n_params)``. If
        given, every quantity gets a first-order uncertainty from the fitted
        shift/linewidth variances (material constants treated as exact).
    peak : int, optional
        Which peak of a multi-peak fit to use (0-based). Default 0.
    refractive_index, density, wavelength, scattering_angle :
        As in the module-level functions - each quantity is computed only if the
        constants it needs are all provided. ``loss_tangent`` needs none of them.

    Returns
    -------
    MechanicalProperties
    """
    params = np.asarray(fit_parameters, dtype=float)
    s_idx, w_idx = 1 + 3 * peak, 2 + 3 * peak
    shift = np.abs(params[..., s_idx])
    linewidth = np.abs(params[..., w_idx])

    s_unc = w_unc = None
    if covariances is not None:
        # DHO.apply on an image returns the covariance as a list of per-row arrays;
        # np.asarray stacks that back to (..., n_params, n_params).
        cov = np.asarray(covariances, dtype=float)
        s_unc = np.sqrt(np.abs(cov[..., s_idx, s_idx]))
        w_unc = np.sqrt(np.abs(cov[..., w_idx, w_idx]))

    def split(result):
        return result if isinstance(result, tuple) else (result, None)

    lt, lt_u = split(loss_tangent(shift, linewidth, shift_uncertainty=s_unc,
                                  linewidth_uncertainty=w_unc))
    out = MechanicalProperties(loss_tangent=lt, loss_tangent_uncertainty=lt_u)

    geom = dict(refractive_index=refractive_index, wavelength=wavelength,
                scattering_angle=scattering_angle)
    if None not in geom.values():
        out.hypersound_velocity, out.hypersound_velocity_uncertainty = split(
            hypersound_velocity(shift, shift_uncertainty=s_unc, **geom))
        if density is not None:
            out.storage_modulus, out.storage_modulus_uncertainty = split(
                storage_modulus(shift, density=density, shift_uncertainty=s_unc, **geom))
            out.loss_modulus, out.loss_modulus_uncertainty = split(
                loss_modulus(shift, linewidth, density=density, shift_uncertainty=s_unc,
                             linewidth_uncertainty=w_unc, **geom))
            out.longitudinal_viscosity, out.longitudinal_viscosity_uncertainty = split(
                longitudinal_viscosity(linewidth, density=density,
                                       linewidth_uncertainty=w_unc, **geom))

    return out
