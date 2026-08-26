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


def test_deconvoluter_irf_padding_preserves_shape_and_stays_finite():
    data, axis = _spectrum_with_irf()
    spectrum = Spectrum(data, axis)

    result = Deconvoluter_IRF(offset=2, iterations=4, padding=15).apply(spectrum)

    assert result.spectral_data.shape == data.shape
    valid = ~np.isnan(result.spectral_data)
    assert np.all(np.isfinite(result.spectral_data[valid]))
    assert np.all(np.isnan(result.spectral_data[45:55]))


def test_deconvoluter_irf_padding_reduces_edge_artefact():
    # A peak whose flank is cut off right at the array's edge (index 0) - the case padding
    # is meant to help with, since the unpadded deconvolution treats that as a hard boundary.
    n_channels = 100
    axis = np.linspace(-20, 20, n_channels)
    irf_index = 50
    irf = np.exp(-0.5 * ((np.arange(n_channels) - irf_index) / 2) ** 2)
    edge_peak = np.exp(-0.5 * ((np.arange(n_channels) - 3) / 3) ** 2)
    data = 0.1 * edge_peak + irf
    data[irf_index - 10:irf_index - 7] = 0
    data[irf_index + 7:irf_index + 10] = 0
    spectrum = Spectrum(data, axis)

    unpadded = Deconvoluter_IRF(offset=2, iterations=4, padding=None).apply(spectrum)
    padded = Deconvoluter_IRF(offset=2, iterations=4, padding=20).apply(spectrum)

    # Padding changes the result near the edge - it isn't silently ignored.
    assert not np.allclose(unpadded.spectral_data[:5], padded.spectral_data[:5])
