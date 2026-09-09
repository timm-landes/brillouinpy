import numpy as np
import pytest

from brillouinpy.analysis.fit.core import _DHO_1
from brillouinpy.analysis.phasor import phasor, phase_to_shift, phasor_cursor
from brillouinpy.core import Spectrum, SpectralImage


def _dho_spectrum(freq_shift, linewidth=0.6, axis=None):
    axis = np.linspace(0.5, 20, 400) if axis is None else axis
    return axis, _DHO_1(axis, I0=5.0, freqShift=freq_shift, LineWidth=linewidth,
                        Background=0.05, axis_shift=0.0)


def test_phasor_shape_matches_input():
    axis, intensity = _dho_spectrum(8.0)
    data = np.broadcast_to(intensity, (6, 5, axis.size)).copy()
    image = SpectralImage(data, axis)

    result = phasor(image)

    assert result.G.shape == (6, 5)
    assert result.S.shape == (6, 5)
    assert np.isfinite(result.modulus).all()


def test_modulus_within_unit_circle():
    axis, intensity = _dho_spectrum(8.0)
    result = phasor(Spectrum(intensity, axis), background='min')

    assert result.modulus.ravel()[0] <= 1.0 + 1e-9


def test_phase_increases_with_peak_position():
    axis = np.linspace(0.5, 20, 400)
    low = phasor(Spectrum(_dho_spectrum(5.0, axis=axis)[1], axis), background='min')
    high = phasor(Spectrum(_dho_spectrum(12.0, axis=axis)[1], axis), background='min')

    assert high.phase.ravel()[0] > low.phase.ravel()[0]


def test_phase_to_shift_recovers_peak():
    axis = np.linspace(0.5, 20, 600)
    _, intensity = _dho_spectrum(9.0, linewidth=0.4, axis=axis)
    result = phasor(Spectrum(intensity, axis), background='min')

    assert phase_to_shift(result).ravel()[0] == pytest.approx(9.0, abs=0.6)


def test_phase_to_shift_rejects_higher_harmonics():
    axis, intensity = _dho_spectrum(8.0)
    result = phasor(Spectrum(intensity, axis), harmonic=2)

    with pytest.raises(ValueError):
        phase_to_shift(result)


def test_two_populations_separate_in_phasor_plane():
    axis = np.linspace(0.5, 20, 400)
    a = _dho_spectrum(6.0, axis=axis)[1]
    b = _dho_spectrum(13.0, axis=axis)[1]
    data = np.stack([np.tile(a, (4, 1)), np.tile(b, (4, 1))])  # (2, 4, B)
    result = phasor(SpectralImage(data, axis), background='min')

    cursor = phasor_cursor(result, center=(result.G[0].mean(), result.S[0].mean()), radius=0.05)

    assert cursor[0].all()
    assert not cursor[1].any()


def test_masked_pixel_yields_nan_phasor():
    axis, intensity = _dho_spectrum(8.0)
    data = np.ma.masked_array(np.tile(intensity, (2, 1)), mask=False)
    data[1] = np.ma.masked
    result = phasor(SpectralImage(data[:, None, :], axis))

    assert np.isfinite(result.G[0, 0])
    assert np.isnan(result.G[1, 0])


def test_rejects_too_few_channels():
    axis = np.linspace(0.5, 20, 400)
    _, intensity = _dho_spectrum(8.0, axis=axis)

    with pytest.raises(ValueError):
        phasor(Spectrum(intensity, axis), axis_range=(8.0, 8.05))
