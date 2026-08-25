import numpy as np
import pytest

from brillouinpy.analysis.fitmodel import (
    _DHO_1,
    _DHO_2,
    _DHO_3,
    _Lorentzian_1,
    DHO,
)
from brillouinpy.core import Spectrum


def test_dho1_peaks_near_freq_shift():
    # The model is symmetric about Asymmetry, so it peaks at both +freqShift and -freqShift;
    # restrict to the positive half to check the location of one of them.
    x = np.linspace(0, 20, 2001)
    y = _DHO_1(x, I0=2.0, freqShift=5.0, LineWidth=0.3, Background=1.0, Asymmetry=0.0)

    assert x[np.argmax(y)] == pytest.approx(5.0, abs=0.05)


def test_dho1_approaches_background_far_from_peak():
    y_far = _DHO_1(np.array([1000.0]), I0=2.0, freqShift=5.0, LineWidth=0.3, Background=1.0, Asymmetry=0.0)

    assert y_far[0] == pytest.approx(1.0, abs=1e-3)


def test_dho2_is_sum_of_two_dho1():
    x = np.linspace(-20, 20, 50)
    params = dict(I0=1.0, freqShift=5.0, LineWidth=0.3, I02=2.0, freqShift2=8.0, LineWidth2=0.5,
                  Background=0.5, Asymmetry=0.1)

    combined = _DHO_2(x, **params)
    expected = (
        _DHO_1(x, params["I0"], params["freqShift"], params["LineWidth"],
               params["Background"], params["Asymmetry"])
        + _DHO_1(x, params["I02"], params["freqShift2"], params["LineWidth2"],
                 params["Background"], params["Asymmetry"])
    )

    assert np.allclose(combined, expected)


def test_dho3_is_dho2_plus_dho1():
    x = np.linspace(-20, 20, 50)
    params = dict(I0=1.0, freqShift=5.0, LineWidth=0.3, I02=2.0, freqShift2=8.0, LineWidth2=0.5,
                  I03=0.5, freqShift3=12.0, LineWidth3=0.2, Background=0.5, Asymmetry=0.0)

    combined = _DHO_3(x, **params)
    expected = (
        _DHO_2(x, params["I0"], params["freqShift"], params["LineWidth"], params["I02"], params["freqShift2"],
               params["LineWidth2"], params["Background"], params["Asymmetry"])
        + _DHO_1(x, params["I03"], params["freqShift3"], params["LineWidth3"],
                 params["Background"], params["Asymmetry"])
    )

    assert np.allclose(combined, expected)


def test_lorentzian1_peaks_near_freq_shift():
    # Same symmetry consideration as the DHO model above.
    x = np.linspace(0, 20, 2001)
    y = _Lorentzian_1(x, I0=2.0, freqShift=5.0, LineWidth=0.3, Background=1.0, Asymmetry=0.0)

    assert x[np.argmax(y)] == pytest.approx(5.0, abs=0.05)


def test_dho_fit_recovers_known_parameters():
    true_params = dict(I0=5.0, freqShift=6.0, LineWidth=0.4, Background=1.0, Asymmetry=0.0)
    axis = np.linspace(-20, 20, 400)
    intensity = _DHO_1(axis, **true_params)

    spectrum = Spectrum(intensity, axis)

    fit = DHO(expected_peaks=1, p0=[1, 5, 1, 0.5, 0], bounds=(0, np.inf))
    popt, _ = fit.apply(spectrum)

    assert popt[0] == pytest.approx(true_params["I0"], rel=0.05)
    assert popt[1] == pytest.approx(true_params["freqShift"], rel=0.05)
    assert popt[2] == pytest.approx(true_params["LineWidth"], rel=0.05)
    assert popt[3] == pytest.approx(true_params["Background"], abs=0.1)
