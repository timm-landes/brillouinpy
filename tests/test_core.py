import pickle

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


@pytest.mark.parametrize(
    "cls, bad_shape",
    [
        (Spectrum, (4, 10)),
        (SpectralImage, (4, 10)),
        (SpectralImage, (4, 4, 3, 10)),
        (SpectralVolume, (4, 4, 10)),
    ],
)
def test_subclass_rejects_wrong_dimensionality(cls, bad_shape):
    axis = np.linspace(-20, 20, bad_shape[-1])
    with pytest.raises(ValueError, match="dimensional"):
        cls(np.random.rand(*bad_shape), axis)


def test_base_container_accepts_any_dimensionality():
    axis = np.linspace(-20, 20, 10)
    obj = SpectralContainer(np.random.rand(2, 3, 4, 5, 10), axis)
    assert obj.shape == (2, 3, 4, 5)


def test_peaks_on_image_uses_mean_spectrum():
    axis = np.linspace(-10, 10, 41)
    bump = np.exp(-((axis - 3) ** 2))
    image = SpectralImage(np.tile(bump, (3, 3, 1)), axis)

    peaks, _ = image.peaks(prominence=0.2)
    spectrum_peaks, _ = Spectrum(bump, axis).peaks(prominence=0.2)

    assert np.array_equal(peaks, spectrum_peaks)
    assert axis[peaks[0]] == pytest.approx(3, abs=0.5)


def test_channels_default_empty_and_grid_conformance():
    axis = np.linspace(-20, 20, 10)
    raman = SpectralImage(np.random.rand(4, 3, 6), np.linspace(0, 100, 6))
    camera = SpectralImage(np.random.rand(64, 48, 1), np.array([550.0]))
    image = SpectralImage(np.random.rand(4, 3, 10), axis,
                          channels={"raman": raman, "camera": camera})

    assert SpectralImage(np.random.rand(4, 3, 10), axis).channels == {}
    assert image.channels_grid_conformant == {"raman": True, "camera": False}


def test_grid_conformant_channel_follows_spatial_ops():
    axis = np.linspace(-20, 20, 10)
    raman = SpectralImage(np.arange(4 * 3 * 6).reshape(4, 3, 6), np.linspace(0, 100, 6))
    camera = SpectralImage(np.random.rand(8, 8, 1), np.array([550.0]))
    image = SpectralImage(np.random.rand(4, 3, 10), axis,
                          channels={"raman": raman, "camera": camera})

    sliced = image[1:3, 0:2]
    assert sliced.channels["raman"].shape == (2, 2)
    assert sliced.channels["camera"].shape == (8, 8)  # non-conformant: untouched
    assert np.array_equal(sliced.channels["raman"].spectral_data,
                          raman.spectral_data[1:3, 0:2])

    assert image.flat.channels["raman"].shape == (12,)
    assert image.flat.channels["camera"].shape == (8, 8)

    # derived channels are copies
    sliced.channels["raman"].spectral_data[:] = 0
    assert raman.spectral_data.sum() != 0

    # mean/variance carry no channels
    assert image.mean.channels == {}


def test_from_image_stack_merges_conformant_channels_only():
    axis = np.linspace(-20, 20, 10)

    def img(with_bad=False):
        chans = {"raman": SpectralImage(np.random.rand(3, 3, 6), np.linspace(0, 100, 6))}
        if with_bad:
            chans["cam"] = SpectralImage(np.random.rand(5, 5, 1), np.array([550.0]))
        return SpectralImage(np.random.rand(3, 3, 10), axis, channels=chans)

    merged = SpectralVolume.from_image_stack([img(), img()])
    assert merged.channels["raman"].shape == (3, 3, 2)

    dropped = SpectralVolume.from_image_stack([img(with_bad=True), img(with_bad=True)])
    assert dropped.channels == {}


def test_channels_survive_pickle_and_legacy_setstate():
    axis = np.linspace(-20, 20, 10)
    raman = SpectralImage(np.random.rand(3, 3, 6), np.linspace(0, 100, 6))
    image = SpectralImage(np.random.rand(3, 3, 10), axis, channels={"raman": raman})
    assert list(pickle.loads(pickle.dumps(image)).channels) == ["raman"]

    legacy = SpectralImage(np.random.rand(3, 3, 10), axis)
    del legacy.__dict__["channels"]
    assert pickle.loads(pickle.dumps(legacy)).channels == {}


def test_metadata_and_px_size_default_to_empty():
    spectrum = Spectrum(np.random.rand(10), np.linspace(-20, 20, 10))

    assert spectrum.metadata == {}
    assert spectrum.px_size_um == {"x": None, "y": None, "z": None}


def test_metadata_and_px_size_carry_through_slicing_and_reductions():
    axis = np.linspace(-20, 20, 10)
    md = {"Experiment": {"Sample": ("water", None)}}
    image = SpectralImage(np.random.rand(4, 3, 10), axis,
                          metadata=md, px_size_um={"x": 0.5, "y": 0.25})

    sliced = image[0]
    assert sliced.metadata == md
    assert sliced.px_size_um["x"] == 0.5

    # derived objects get a copy, not a shared reference
    sliced.metadata["Experiment"]["Sample"] = ("changed", None)
    assert image.metadata["Experiment"]["Sample"] == ("water", None)

    assert image.mean.metadata == md
    assert image.flat.metadata == md
    assert image.tolist()[0].metadata == md


def test_instrument_response_function_validation_and_axis_sort():
    axis = np.array([20.0, 0.0, -20.0])  # unsorted
    irf_1d = np.array([1.0, 2.0, 3.0])
    img = SpectralImage(np.random.rand(2, 2, 3), axis,
                        instrument_response_function=irf_1d)
    # sorted alongside the spectral axis
    assert np.array_equal(img.spectral_axis, [-20.0, 0.0, 20.0])
    assert np.array_equal(img.instrument_response_function, [3.0, 2.0, 1.0])

    with pytest.raises(ValueError):  # wrong channel count
        SpectralImage(np.random.rand(2, 2, 3), np.linspace(-1, 1, 3),
                      instrument_response_function=np.ones(4))
    with pytest.raises(ValueError):  # per-pixel shape mismatch
        SpectralImage(np.random.rand(2, 2, 3), np.linspace(-1, 1, 3),
                      instrument_response_function=np.ones((3, 3, 3)))


def test_per_pixel_irf_follows_spatial_operations():
    axis = np.linspace(-20, 20, 8)
    irf = np.random.rand(4, 3, 8)
    image = SpectralImage(np.random.rand(4, 3, 8), axis, instrument_response_function=irf)

    assert image[1].instrument_response_function.shape == (3, 8)
    np.testing.assert_array_equal(image[1].instrument_response_function, irf[1])
    assert image.flat.instrument_response_function.shape == (12, 8)
    # mean/variance collapse the per-pixel IRF to the mean IRF
    np.testing.assert_allclose(image.mean.instrument_response_function,
                               irf.reshape(-1, 8).mean(axis=0))
    assert image.tolist()[0].instrument_response_function.shape == (8,)

    volume = SpectralVolume.from_image_stack([image, image])
    assert volume.instrument_response_function.shape == (4, 3, 2, 8)
    assert volume.layer(0).instrument_response_function.shape == (4, 3, 8)


def test_shared_irf_passes_through_unchanged_and_survives_pickle():
    axis = np.linspace(-20, 20, 8)
    kernel = np.hanning(8)
    image = SpectralImage(np.random.rand(4, 3, 8), axis, instrument_response_function=kernel)

    np.testing.assert_array_equal(image[0].instrument_response_function, kernel)
    np.testing.assert_array_equal(image.flat.instrument_response_function, kernel)

    restored = pickle.loads(pickle.dumps(image))
    np.testing.assert_array_equal(restored.instrument_response_function, kernel)

    legacy = SpectralImage(np.random.rand(3, 3, 8), axis)
    del legacy.__dict__["instrument_response_function"]
    assert pickle.loads(pickle.dumps(legacy)).instrument_response_function is None


def test_metadata_survives_pickle_and_legacy_pickle_without_attrs(tmp_path):
    axis = np.linspace(-20, 20, 10)
    image = SpectralImage(np.random.rand(3, 3, 10), axis, metadata={"a": 1})
    image.save("img.pkl", directory=str(tmp_path))
    assert SpectralImage.load(str(tmp_path / "img.pkl")).metadata == {"a": 1}

    # simulate a pickle written before metadata/px_size_um existed
    legacy = SpectralImage(np.random.rand(3, 3, 10), axis)
    del legacy.__dict__["metadata"]
    del legacy.__dict__["px_size_um"]
    restored = pickle.loads(pickle.dumps(legacy))
    assert restored.metadata == {}
    assert restored.px_size_um == {"x": None, "y": None, "z": None}


def test_save_and_load_roundtrip(tmp_path):
    axis = np.linspace(-20, 20, 10)
    spectrum = Spectrum(np.random.rand(10), axis)

    spectrum.save("spectrum.pkl", directory=str(tmp_path))
    loaded = Spectrum.load(str(tmp_path / "spectrum.pkl"))

    assert np.array_equal(loaded.spectral_data, spectrum.spectral_data)
    assert np.array_equal(loaded.spectral_axis, spectrum.spectral_axis)
