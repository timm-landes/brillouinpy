import numpy as np

from brillouinpy.preprocessing.misc import Deconvoluter_IRF
from brillouinpy.core import Spectrum, SpectralImage, SpectralContainer


def _spectrum_with_irf(n_channels=100, irf_index=50, irf_width=2):
    axis = np.linspace(-20, 20, n_channels)
    peak = np.exp(-0.5 * ((np.arange(n_channels) - (irf_index - 20)) / 3) ** 2)
    irf = np.exp(-0.5 * ((np.arange(n_channels) - irf_index) / irf_width) ** 2)
    data = 0.1 * peak + irf
    data[irf_index - 10:irf_index - 7] = 0
    data[irf_index + 7:irf_index + 10] = 0
    return data, axis


def test_deconvoluter_irf_blanks_irf_region_on_spectrum():
    data, axis = _spectrum_with_irf()
    spectrum = Spectrum(data, axis)

    result = Deconvoluter_IRF(offset=2, iterations=4, padding=None).apply(spectrum)

    assert np.all(np.isnan(result.spectral_data[45:55]))
    assert not np.any(np.isnan(result.spectral_data[:40]))


def test_deconvoluter_irf_works_on_spectral_image():
    data, axis = _spectrum_with_irf()
    image_data = np.tile(data, (3, 3, 1))
    image = SpectralImage(image_data, axis)

    result = Deconvoluter_IRF(offset=2, iterations=4, padding=None).apply(image)

    assert isinstance(result, SpectralImage)
    assert result.spectral_data.shape == image_data.shape
    assert np.all(np.isnan(result.spectral_data[1, 1, 45:55]))


def test_deconvoluter_irf_works_on_raw_5d_shape():
    data, axis = _spectrum_with_irf()
    volume_data = np.tile(data, (2, 2, 1, 1, 1))
    container = SpectralContainer(volume_data, axis)

    result = Deconvoluter_IRF(offset=2, iterations=4, padding=None).apply(container)

    assert result.spectral_data.shape == volume_data.shape
    assert np.all(np.isnan(result.spectral_data[0, 1, 0, 0, 45:55]))
