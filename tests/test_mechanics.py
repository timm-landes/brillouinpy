import numpy as np
import pytest

from brillouinpy.analysis import mechanics

# Water at ~20 C, 532 nm laser, backscattering (theta = 180 deg).
WATER = dict(refractive_index=1.33, wavelength=532e-9, scattering_angle=180.0)
RHO_WATER = 998.0
SHIFT_WATER = 7.46      # GHz, measured
LINEWIDTH_WATER = 0.5   # GHz HWHM (the DHO fit's LineWidth); FWHM = 1.0


def test_loss_tangent_uses_the_fwhm():
    # loss_tangent = (2 * linewidth) / shift, linewidth being the HWHM
    assert mechanics.loss_tangent(8.0, 1.0) == pytest.approx(0.25)
    # invariant to shift/linewidth scaling by the same factor
    assert mechanics.loss_tangent(16.0, 2.0) == pytest.approx(0.25)


def test_loss_tangent_uncertainty_propagation():
    val, unc = mechanics.loss_tangent(8.0, 1.0, shift_uncertainty=0.08, linewidth_uncertainty=0.05)
    assert val == pytest.approx(0.25)
    # relative error = sqrt((0.05/1)^2 + (0.08/8)^2), scale-invariant under HWHM->FWHM
    assert unc == pytest.approx(0.25 * np.hypot(0.05, 0.01))


def test_hypersound_velocity_of_water():
    v = mechanics.hypersound_velocity(SHIFT_WATER, **WATER)
    assert v == pytest.approx(1490, abs=15)  # m/s


def test_storage_modulus_of_water():
    m_prime = mechanics.storage_modulus(SHIFT_WATER, density=RHO_WATER, **WATER)
    assert m_prime == pytest.approx(2.2e9, rel=0.05)  # ~2.2 GPa


def test_loss_modulus_equals_storage_times_loss_tangent():
    m_prime = mechanics.storage_modulus(SHIFT_WATER, density=RHO_WATER, **WATER)
    m_double = mechanics.loss_modulus(SHIFT_WATER, LINEWIDTH_WATER, density=RHO_WATER, **WATER)
    td = mechanics.loss_tangent(SHIFT_WATER, LINEWIDTH_WATER)
    assert m_double == pytest.approx(m_prime * td, rel=1e-9)


def test_longitudinal_viscosity_of_water_and_linear_in_linewidth():
    eta = mechanics.longitudinal_viscosity(LINEWIDTH_WATER, density=RHO_WATER, **WATER)
    assert eta == pytest.approx(6e-3, rel=0.4)  # a few mPa*s
    assert mechanics.longitudinal_viscosity(2 * LINEWIDTH_WATER, density=RHO_WATER, **WATER) \
        == pytest.approx(2 * eta, rel=1e-9)


def test_material_constants_are_mandatory():
    with pytest.raises(TypeError):
        mechanics.hypersound_velocity(7.0)
    with pytest.raises(TypeError):
        mechanics.storage_modulus(7.0, density=1000.0, refractive_index=1.33, wavelength=532e-9)


def _fake_fit(shift=7.46, linewidth=0.6, shift_var=0.01 ** 2, lw_var=0.02 ** 2, shape=(3, 4)):
    params = np.zeros(shape + (5,))
    params[..., 0] = 1.0
    params[..., 1] = shift
    params[..., 2] = linewidth
    cov = np.zeros(shape + (5, 5))
    cov[..., 1, 1] = shift_var
    cov[..., 2, 2] = lw_var
    return params, cov


def test_from_dho_fit_only_loss_tangent_without_constants():
    params, _ = _fake_fit()
    result = mechanics.from_dho_fit(params)

    assert result.loss_tangent.shape == (3, 4)
    np.testing.assert_allclose(result.loss_tangent, 2 * 0.6 / 7.46)
    assert result.storage_modulus is None
    assert result.hypersound_velocity is None
    assert result.loss_tangent_uncertainty is None


def test_from_dho_fit_full_with_covariance():
    params, cov = _fake_fit()
    result = mechanics.from_dho_fit(
        params, cov, refractive_index=1.33, density=998.0, wavelength=532e-9, scattering_angle=180.0)

    for field in ('loss_tangent', 'hypersound_velocity', 'storage_modulus',
                  'loss_modulus', 'longitudinal_viscosity'):
        assert getattr(result, field) is not None
        assert getattr(result, field).shape == (3, 4)
        assert getattr(result, field + '_uncertainty') is not None

    # matches the standalone functions
    np.testing.assert_allclose(
        result.storage_modulus,
        mechanics.storage_modulus(7.46, density=998.0, **WATER))
    # velocity uncertainty is purely relative to the shift error
    np.testing.assert_allclose(
        result.hypersound_velocity_uncertainty / result.hypersound_velocity,
        0.01 / 7.46, rtol=1e-6)


def test_from_dho_fit_peak_selection():
    params = np.zeros((2, 2, 8))  # 2-peak fit: [I0,s,w, I0,s2,w2, bg, asym]
    params[..., 1], params[..., 2] = 7.0, 0.5
    params[..., 4], params[..., 5] = 9.0, 1.0

    assert mechanics.from_dho_fit(params, peak=0).loss_tangent[0, 0] == pytest.approx(2 * 0.5 / 7.0)
    assert mechanics.from_dho_fit(params, peak=1).loss_tangent[0, 0] == pytest.approx(2 * 1.0 / 9.0)
