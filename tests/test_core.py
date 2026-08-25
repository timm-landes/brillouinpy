import numpy as np
import pytest

from brillouinpy.core import (
    SpectralContainer,
    Spectrum,
    SpectralImage,
    SpectralVolume,
    _create_data,
)


def test_init_sorts_by_spectral_axis():
    data = np.array([3.0, 1.0, 2.0])
    axis = np.array([30.0, 10.0, 20.0])

    spectrum = Spectrum(data, axis)

    assert np.array_equal(spectrum.spectral_axis, [10.0, 20.0, 30.0])
    assert np.array_equal(spectrum.spectral_data, [1.0, 2.0, 3.0])


def test_init_raises_on_length_mismatch():
    with pytest.raises(ValueError):
        SpectralContainer(np.zeros((5, 3)), np.arange(4))


@pytest.mark.parametrize(
    "shape, expected_type",
    [
        ((10,), Spectrum),
        ((4, 4, 10), SpectralImage),
        ((4, 4, 3, 10), SpectralVolume),
        ((2, 4, 4, 3, 10), SpectralContainer),
    ],
)
def test_create_data_dispatches_by_dimensionality(shape, expected_type):
    data = np.random.rand(*shape)
    axis = np.linspace(-20, 20, shape[-1])

    obj = _create_data(data, axis)

    assert type(obj) is expected_type


def test_spectral_length_and_shape():
    image = SpectralImage(np.random.rand(4, 5, 10), np.linspace(-20, 20, 10))

    assert image.spectral_length == 10
    assert image.shape == (4, 5)


def test_shape_for_single_spectrum_is_one():
    spectrum = Spectrum(np.random.rand(10), np.linspace(-20, 20, 10))

    assert spectrum.shape == (1,)


def test_mean_and_variance():
    axis = np.linspace(-20, 20, 10)
    data = np.stack([np.ones(10), 3 * np.ones(10)])
    image = SpectralContainer(data, axis)

    assert np.allclose(image.mean.spectral_data, 2 * np.ones(10))
    assert np.allclose(image.variance.spectral_data, np.ones(10))


def test_band_returns_closest_slice():
    axis = np.array([0.0, 1.0, 2.0, 3.0])
    data = np.array([[10.0, 20.0, 30.0, 40.0]])
    container = SpectralContainer(data, axis)

    result = container.band(1.9)

    assert np.array_equal(result, [30.0])


def test_band_out_of_bounds_raises():
    axis = np.array([0.0, 1.0, 2.0])
    container = SpectralContainer(np.zeros((1, 3)), axis)

    with pytest.raises(ValueError):
        container.band(10.0)


def test_getitem_spatial_indexing_on_image():
    axis = np.linspace(-20, 20, 10)
    image = SpectralImage(np.random.rand(4, 4, 10), axis)

    row = image[0]

    assert type(row) is SpectralContainer  # a single row is 2D -> falls back to the base container type
    assert np.array_equal(row.spectral_data, image.spectral_data[0])


def test_getitem_on_single_spectrum_raises():
    spectrum = Spectrum(np.random.rand(10), np.linspace(-20, 20, 10))

    with pytest.raises(ValueError):
        spectrum[0]


def test_tolist_returns_spectrum_objects():
    axis = np.linspace(-20, 20, 10)
    image = SpectralImage(np.random.rand(2, 3, 10), axis)

    spectra = image.tolist()

    assert len(spectra) == 6
    assert all(isinstance(s, Spectrum) for s in spectra)


def test_from_stack_combines_aligned_spectra():
    axis = np.linspace(-20, 20, 10)
    spectra = [Spectrum(np.random.rand(10), axis) for _ in range(3)]

    stacked = SpectralContainer.from_stack(spectra)

    assert stacked.shape == (3,)
    assert np.array_equal(stacked.spectral_axis, axis)


def test_from_image_stack_builds_volume():
    axis = np.linspace(-20, 20, 10)
    images = [SpectralImage(np.random.rand(3, 3, 10), axis) for _ in range(2)]

    volume = SpectralVolume.from_image_stack(images)

    assert volume.shape == (3, 3, 2)


def test_volume_layer_returns_image():
    axis = np.linspace(-20, 20, 10)
    volume = SpectralVolume(np.random.rand(3, 3, 2, 10), axis)

    layer = volume.layer(0)

    assert isinstance(layer, SpectralImage)
    assert np.array_equal(layer.spectral_data, volume.spectral_data[..., 0, :])


def test_save_and_load_roundtrip(tmp_path):
    axis = np.linspace(-20, 20, 10)
    spectrum = Spectrum(np.random.rand(10), axis)

    spectrum.save("spectrum.pkl", directory=str(tmp_path))
    loaded = Spectrum.load(str(tmp_path / "spectrum.pkl"))

    assert np.array_equal(loaded.spectral_data, spectrum.spectral_data)
    assert np.array_equal(loaded.spectral_axis, spectrum.spectral_axis)
