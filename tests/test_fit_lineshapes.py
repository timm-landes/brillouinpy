import numpy as np
import pytest

from brillouinpy.analysis import fit as fitmodel
from brillouinpy.analysis.fit.core import _DHO_1, _DHO_2, _Lorentzian_1
from brillouinpy.analysis.fit.lineshapes import (
    LINESHAPES,
    assemble_model,
    model_funcs,
    register_lineshape,
)
from brillouinpy.core import SpectralImage


def _image(axis, spectrum, shape=(2, 2), noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(np.prod(shape))
    data = np.stack([spectrum + rng.normal(0, noise, axis.size) for _ in range(n)])
    return SpectralImage(data.reshape(*shape, -1), axis)


def test_builtin_families_registered():
    for name in ("dho", "lorentzian", "gaussian",
                 "dho_elastic", "lorentzian_elastic", "gaussian_elastic"):
        assert name in LINESHAPES


def test_model_funcs_returns_handwritten_for_dho_and_lorentzian():
    assert model_funcs("dho")[1] is _DHO_1
    assert model_funcs("lorentzian")[1] is _Lorentzian_1


def test_model_funcs_unknown_name_raises():
    with pytest.raises(ValueError, match="unknown lineshape"):
        model_funcs("not_a_model")


def test_assembled_dho_matches_handwritten():
    x = np.linspace(-15, 15, 400)
    p = (1.3, 7.0, 0.5, 0.4, 9.2, 0.6, 0.1, 0.05)
    assert np.allclose(assemble_model(LINESHAPES["dho"], 2)(x, *p), _DHO_2(x, *p))


def test_assembled_model_is_picklable():
    import pickle
    m = assemble_model(LINESHAPES["gaussian"], 1)
    assert pickle.loads(pickle.dumps(m))(np.array([0.0]), 1.0, 5.0, 0.5, 0.0, 0.0) is not None


def test_gaussian_fit_recovers_parameters():
    axis = np.linspace(-15, 15, 600)
    truth = dict(I0=2.0, freqShift=7.5, LineWidth=0.8, Background=0.3, axis_shift=0.0)
    core = LINESHAPES["gaussian"].core
    spectrum = core(axis, truth["I0"], truth["freqShift"], truth["LineWidth"]) + truth["Background"]
    image = _image(axis, spectrum, noise=2e-3)

    p0 = fitmodel.estimate_p0(image, expected_peaks=1, model="gaussian")
    params, _ = fitmodel.Gaussian(expected_peaks=1, p0=p0, bounds=(0, np.inf)).apply(image)

    assert params[0, 0, 1] == pytest.approx(truth["freqShift"], abs=0.05)
    assert params[0, 0, 2] == pytest.approx(truth["LineWidth"], abs=0.1)


def test_gaussian_elastic_recovers_slope():
    axis = np.linspace(-15, 15, 600)
    core = LINESHAPES["gaussian"].core
    slope = 0.01
    spectrum = core(axis, 2.0, 7.5, 0.8) + 0.3 + slope * axis
    image = _image(axis, spectrum, noise=2e-3)

    p0 = fitmodel.estimate_p0(image, expected_peaks=1, model="gaussian_elastic")
    assert len(p0) == 6
    lo = [0, 0, 0, -np.inf, -2, -np.inf]
    hi = [np.inf] * 6
    params, _ = fitmodel.Gaussian(expected_peaks=1, elastic=True, p0=p0, bounds=(lo, hi)).apply(image)

    assert params[0, 0, -1] == pytest.approx(slope, abs=3e-3)


def test_elastic_family_rejects_irf():
    axis = np.linspace(-15, 15, 400)
    image = _image(axis, np.zeros_like(axis) + 1.0)
    with pytest.raises(NotImplementedError, match="irf convolution is not supported"):
        fitmodel.DHO(expected_peaks=1, elastic=True, irf=("lorentzian", 0.5),
                     p0=[1, 7, 0.5, 0, 0, 0], bounds=(0, np.inf)).apply(image)


def test_dho_auto_rejects_elastic():
    with pytest.raises(NotImplementedError, match="does not support elastic"):
        fitmodel.DHO(expected_peaks="auto", elastic=True)


def _pseudo_voigt_core(xs, I0, freqShift, LineWidth):
    g = LINESHAPES["gaussian"].core(xs, I0, freqShift, LineWidth)
    lo = LINESHAPES["lorentzian"].core(xs, I0, freqShift, LineWidth)
    return 0.5 * g + 0.5 * lo


def test_register_and_fit_custom_lineshape():
    register_lineshape("pseudo_voigt_test", core=_pseudo_voigt_core)
    try:
        axis = np.linspace(-15, 15, 600)
        spectrum = _pseudo_voigt_core(axis, 2.0, 7.5, 0.8) + 0.3
        image = _image(axis, spectrum, noise=2e-3)
        p0 = fitmodel.estimate_p0(image, expected_peaks=1, model="pseudo_voigt_test")
        params, _ = fitmodel.PeakFit(model="pseudo_voigt_test", expected_peaks=1,
                                     p0=p0, bounds=(0, np.inf)).apply(image)
        assert params[0, 0, 1] == pytest.approx(7.5, abs=0.05)
    finally:
        LINESHAPES.pop("pseudo_voigt_test", None)
