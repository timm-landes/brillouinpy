"""
Smoke tests for brillouinpy.plot: every public plotting function is called once
with minimal valid data on the non-interactive Agg backend and must return
without raising. These modules (plot/plot.py, plot/_core.py) had no test coverage
at all - the first user to run examples/01 hit a hard error (plt.cm.get_cmap,
removed in matplotlib 3.9) that a smoke test like this catches immediately.
"""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from brillouinpy import plot
from brillouinpy.analysis.phasor import phasor
from brillouinpy.core import SpectralImage, Spectrum


def _dho(x, freq_shift=8.0, linewidth=1.0):
    return 5e-3 * 4 * linewidth * freq_shift ** 2 / (
        np.pi * ((x ** 2 - freq_shift ** 2) ** 2 + 4 * (linewidth * x) ** 2)
    ) + 1e-4


@pytest.fixture
def axis():
    return np.linspace(-20, 20, 80)


@pytest.fixture
def spectrum(axis):
    return Spectrum(_dho(axis), axis)


@pytest.fixture
def spectral_image(axis):
    data = np.stack([[_dho(axis, 8.0 + 0.1 * i) for i in range(3)] for _ in range(4)])
    return SpectralImage(data, axis)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_spectra_single_spectrum(spectrum):
    assert plot.spectra(spectrum, yscale="linear") is not None


def test_spectra_multiple_groups(spectrum):
    out = plot.spectra([spectrum, spectrum], label=["a", "b"], yscale="linear")
    assert out is not None


@pytest.mark.parametrize("plot_type", ["single", "separate", "stacked", "single stacked"])
def test_spectra_every_plot_type(spectrum, plot_type):
    assert plot.spectra([spectrum, spectrum], plot_type=plot_type, yscale="linear") is not None


def test_mean_spectra(spectral_image):
    assert plot.mean_spectra(spectral_image, yscale="linear") is not None


def test_peaks(spectrum):
    ax, peaks, props = plot.peaks(spectrum, yscale="linear", return_peaks=True)
    assert ax is not None
    assert len(peaks) >= 1


def test_peak_dist(spectral_image):
    assert plot.peak_dist(spectral_image, 8.0) is not None


def test_image_from_array():
    assert plot.image(np.random.default_rng(0).random((5, 5))) is not None


def test_image_multiple_slices():
    rng = np.random.default_rng(0)
    out = plot.image([rng.random((5, 5)), rng.random((5, 5))], title=["A", "B"])
    assert len(out) == 2


def test_volume_from_array():
    assert plot.volume(np.random.default_rng(0).random((4, 4, 4))) is not None


def test_phasor(spectral_image):
    result = phasor(spectral_image, background="min")
    assert plot.phasor(result) is not None


def test_show_is_noop_on_agg():
    plot.show()
